"""The agent execution loop: hard-bounded, reproducible given a seed, every
termination explicit.

Runs through a `paretoguard.runtime.Runtime`, not a raw `Provider` —
transient transport/provider failures for a single step are `Runtime`'s job
(retry/backoff/budget/timeout), exactly as for any other request in this
codebase. This executor only adds the *agent-loop*-level hard limits
(steps, tool calls, cumulative cost, wall time) `Runtime` has no concept of.
See `paretoguard.recovery`'s module docstring for the next layer up: what
happens *across* a step's terminal failure (switch model, fall back,
escalate) is that module's job, not this one's.

**No unbounded loop is possible**: the `for` loop below is bounded by
`AgentLimits.max_steps`, and every `return` inside it is an explicit,
typed `TerminationReason` — there is no code path that falls through
without one.

**Chaos integration (Commit 29, extended Phase E.5)**: an optional
`fault_injector` applies tool-level faults immediately before each
`tool.run()` call, recording the fault's id on the resulting `AgentStep` —
the one injection point this executor uniquely owns (`chaos.injector`'s
module docstring names `AgentExecutor` as its intended caller for tool
faults). Provider-level faults are *applied* with no executor-side
integration at all: wrap the `Provider` given to this executor's `Runtime`
in `chaos.injector.FaultyProvider`, and every step already carries a
`chaos_step` (`= step_index`) in its request metadata for it to key on,
matching `ClosedLoopExecutor`'s convention for recovery attempts (see
`routing.execution`). Phase E.5 closes the remaining gap — *attributing* a
provider fault back to the `AgentStep` it hit: the same `fault_injector`
(when supplied) is additionally queried, read-only, for a PROVIDER-category
fault at each step before the `Runtime.run()` call, purely to label the
resulting `AgentStep.fault_id`. `FaultInjector.faults_for` is a pure
function of its inputs (same `(seed, task_id, step, fault_id)` always
yields the same decision — see `chaos.injector`'s own docstring), so this
observation never influences, duplicates, or races the real application a
`FaultyProvider`-wrapped provider performs independently inside
`Runtime.run()`; it only requires the caller to pass the *same* injector to
both places, or the observed and applied fault ids will disagree (see
`routing.escalation_benchmark`/`structured_agent_v1` callers for the
pattern). `AgentStep.recovery_action` remains unset in Phase E.5 — see
docs/LIMITATIONS.md for why populating it meaningfully is a real new
capability (mid-trajectory recovery), not a field-population task, and was
deliberately left for a future phase rather than forced in here.
"""

import time
from dataclasses import dataclass
from typing import Any

from paretoguard.agents.protocol import Tool, ToolArgumentError
from paretoguard.agents.state import AgentStep, AgentTrajectory, TerminationReason
from paretoguard.chaos.faults import FaultCategory, ToolFaultKind
from paretoguard.chaos.injector import FaultInjector, apply_tool_fault, check_tool_fault
from paretoguard.core.ids import deterministic_request_id
from paretoguard.core.models import FinishReason, InferenceRequest, Message, Role, ToolResult
from paretoguard.runtime import Runtime

_NO_UNDERLYING_CALL_FAULTS = frozenset(
    {ToolFaultKind.EXCEPTION, ToolFaultKind.TIMEOUT, ToolFaultKind.TEMPORARY_UNAVAILABLE}
)
"""Tool faults that never call the underlying tool at all (see
`chaos.injector.apply_tool_fault`'s docstring) — every other `ToolFaultKind`
corrupts a *real* result, so the tool must still be run first."""


@dataclass(frozen=True)
class AgentLimits:
    """Every field here is a hard cap the executor enforces — see CLAUDE.md:
    "No unbounded agent loops (agents enforces max steps/tool calls/cost/
    wall time)"."""

    max_steps: int = 10
    max_tool_calls: int = 10
    max_cost_usd: float | None = None
    max_wall_time_s: float | None = 30.0

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.max_tool_calls < 0:
            raise ValueError("max_tool_calls must be >= 0")
        if self.max_cost_usd is not None and self.max_cost_usd < 0:
            raise ValueError("max_cost_usd must be >= 0")
        if self.max_wall_time_s is not None and self.max_wall_time_s <= 0:
            raise ValueError("max_wall_time_s must be > 0")


class AgentExecutor:
    """Drives one task through `runtime` and `tools` to an explicit
    termination. Stateless across calls — safe to reuse for many tasks."""

    def __init__(
        self,
        runtime: Runtime,
        tools: list[Tool[Any]],
        *,
        limits: AgentLimits | None = None,
        fault_injector: FaultInjector | None = None,
    ) -> None:
        self._runtime = runtime
        self._tools: dict[str, Tool[Any]] = {tool.name: tool for tool in tools}
        self._limits = limits or AgentLimits()
        self._fault_injector = fault_injector

    async def run(
        self,
        *,
        task_id: str,
        provider: str,
        model: str,
        user_prompt: str,
        system_prompt: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> AgentTrajectory:
        messages: list[Message] = []
        if system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=system_prompt))
        messages.append(Message(role=Role.USER, content=user_prompt))

        tool_specs = [tool.spec() for tool in self._tools.values()]
        steps: list[AgentStep] = []
        tool_call_count = 0
        total_cost_usd = 0.0
        start = time.perf_counter()

        for step_index in range(self._limits.max_steps):
            if self._exceeded_wall_time(start):
                return self._finish(task_id, provider, model, steps, TerminationReason.TIMEOUT)

            request = InferenceRequest(
                request_id=deterministic_request_id(task_id, str(step_index), provider, model),
                task_id=task_id,
                provider=provider,
                model=model,
                messages=list(messages),
                tools=tool_specs,
                # `chaos_step=step_index`: lets a `chaos.FaultyProvider`
                # wrapping this executor's own `Runtime` provider draw an
                # independent, reproducible fault decision at each step
                # (see `routing.execution.ClosedLoopExecutor`'s identical
                # `chaos_step` convention for recovery attempts).
                metadata={**(metadata or {}), "chaos_step": step_index},
            )
            provider_fault_id = self._observe_provider_fault(task_id, step_index, provider, model)
            response = await self._runtime.run(request)
            step_cost = response.cost.total_cost_usd if response.cost else None

            if not response.succeeded:
                steps.append(
                    AgentStep(
                        step_index=step_index,
                        fault_id=provider_fault_id,
                        latency_ms=response.latency.total_latency_ms,
                        cost_usd=step_cost,
                        total_tokens=response.token_usage.total_tokens,
                    )
                )
                return self._finish(
                    task_id, provider, model, steps, TerminationReason.UNRECOVERABLE_FAILURE
                )

            total_cost_usd += step_cost or 0.0
            if self._limits.max_cost_usd is not None and total_cost_usd > self._limits.max_cost_usd:
                steps.append(
                    AgentStep(
                        step_index=step_index,
                        model_output_text=response.output_text,
                        fault_id=provider_fault_id,
                        latency_ms=response.latency.total_latency_ms,
                        cost_usd=step_cost,
                        total_tokens=response.token_usage.total_tokens,
                    )
                )
                return self._finish(
                    task_id, provider, model, steps, TerminationReason.BUDGET_EXCEEDED
                )

            if response.finish_reason != FinishReason.TOOL_CALLS or not response.tool_calls:
                steps.append(
                    AgentStep(
                        step_index=step_index,
                        model_output_text=response.output_text,
                        fault_id=provider_fault_id,
                        latency_ms=response.latency.total_latency_ms,
                        cost_usd=step_cost,
                        total_tokens=response.token_usage.total_tokens,
                    )
                )
                return self._finish(
                    task_id,
                    provider,
                    model,
                    steps,
                    TerminationReason.FINAL_ANSWER,
                    final_output_text=response.output_text,
                    final_structured_output=response.structured_output,
                )

            # One tool call handled per step (matches this repo's existing
            # tool_use_v1 suite design: single-tool-call scenarios).
            call = response.tool_calls[0]
            tool_call_count += 1  # noqa: SIM113 - conditional accumulator, not the loop index
            if tool_call_count > self._limits.max_tool_calls:
                steps.append(
                    AgentStep(
                        step_index=step_index,
                        requested_tool=call.name,
                        requested_arguments=call.arguments,
                        fault_id=provider_fault_id,
                        latency_ms=response.latency.total_latency_ms,
                        cost_usd=step_cost,
                        total_tokens=response.token_usage.total_tokens,
                    )
                )
                return self._finish(
                    task_id, provider, model, steps, TerminationReason.TOOL_CALL_LIMIT
                )

            tool = self._tools.get(call.name)
            if tool is None:
                steps.append(
                    AgentStep(
                        step_index=step_index,
                        requested_tool=call.name,
                        requested_arguments=call.arguments,
                        arguments_valid=False,
                        fault_id=provider_fault_id,
                        latency_ms=response.latency.total_latency_ms,
                        cost_usd=step_cost,
                        total_tokens=response.token_usage.total_tokens,
                    )
                )
                return self._finish(task_id, provider, model, steps, TerminationReason.INVALID_TOOL)

            try:
                validated_args = tool.validate_arguments(call.arguments)
            except ToolArgumentError:
                steps.append(
                    AgentStep(
                        step_index=step_index,
                        requested_tool=call.name,
                        requested_arguments=call.arguments,
                        arguments_valid=False,
                        fault_id=provider_fault_id,
                        latency_ms=response.latency.total_latency_ms,
                        cost_usd=step_cost,
                        total_tokens=response.token_usage.total_tokens,
                    )
                )
                return self._finish(
                    task_id, provider, model, steps, TerminationReason.INVALID_TOOL_ARGUMENTS
                )

            result, tool_fault_id = self._run_tool(
                tool, validated_args, task_id, step_index, call.name
            )
            result = result.model_copy(update={"tool_call_id": call.id})
            steps.append(
                AgentStep(
                    step_index=step_index,
                    requested_tool=call.name,
                    requested_arguments=call.arguments,
                    arguments_valid=True,
                    tool_result=result,
                    # A tool fault and a provider fault could in principle
                    # both fire on the same step (the provider fault hit
                    # the response that *led to* this tool call, e.g. a
                    # latency spike that doesn't corrupt tool_calls
                    # content); the tool fault takes precedence in that rare
                    # case since it's the one actually attached to this
                    # step's tool_result.
                    fault_id=tool_fault_id or provider_fault_id,
                    latency_ms=response.latency.total_latency_ms,
                    cost_usd=step_cost,
                    total_tokens=response.token_usage.total_tokens,
                )
            )

            messages.append(
                Message(
                    role=Role.ASSISTANT, content=response.output_text or f"[tool call: {call.name}]"
                )
            )
            observation = (
                str(result.output) if not result.is_error else f"ERROR: {result.error_message}"
            )
            messages.append(Message(role=Role.TOOL, content=observation, tool_call_id=call.id))

        return self._finish(task_id, provider, model, steps, TerminationReason.STEP_LIMIT)

    def _observe_provider_fault(
        self, task_id: str, step_index: int, provider: str, model: str
    ) -> str | None:
        """Read-only: asks `self._fault_injector` (if set) whether a
        PROVIDER-category fault would fire for this exact step, purely to
        attribute it on the resulting `AgentStep.fault_id`. Never applies
        anything — `FaultInjector.faults_for` is a pure function of its
        inputs, so calling it here changes nothing about what a
        `FaultyProvider`-wrapped `Runtime` provider independently decides
        and applies for the same step (see this module's docstring for why
        that requires the caller to pass the same injector to both)."""
        if self._fault_injector is None:
            return None
        events = self._fault_injector.faults_for(
            task_id=task_id, step=step_index, provider=provider, model=model
        )
        provider_events = [e for e in events if e.category == FaultCategory.PROVIDER]
        return provider_events[0].fault_id if provider_events else None

    def _run_tool(
        self, tool: Tool[Any], args: Any, task_id: str, step_index: int, tool_name: str
    ) -> tuple[ToolResult, str | None]:
        """Runs `tool`, consulting `self._fault_injector` first (if set) via
        the same `check_tool_fault`/`apply_tool_fault` pair
        `chaos.injector`'s module docstring names `AgentExecutor` as the
        intended caller of. Returns `(result, fault_id)` — `fault_id` is
        `None` when no fault fired, for `AgentStep.fault_id`."""
        if self._fault_injector is None:
            return tool.run(args), None
        event = check_tool_fault(
            self._fault_injector, task_id=task_id, step=step_index, tool_name=tool_name
        )
        if event is None:
            return tool.run(args), None
        real_result = (
            None if ToolFaultKind(event.kind) in _NO_UNDERLYING_CALL_FAULTS else tool.run(args)
        )
        return apply_tool_fault(event, real_result), event.fault_id

    def _exceeded_wall_time(self, start: float) -> bool:
        if self._limits.max_wall_time_s is None:
            return False
        return (time.perf_counter() - start) > self._limits.max_wall_time_s

    @staticmethod
    def _finish(
        task_id: str,
        provider: str,
        model: str,
        steps: list[AgentStep],
        reason: TerminationReason,
        *,
        final_output_text: str | None = None,
        final_structured_output: dict[str, object] | None = None,
    ) -> AgentTrajectory:
        return AgentTrajectory(
            task_id=task_id,
            provider=provider,
            model=model,
            steps=steps,
            termination_reason=reason,
            final_output_text=final_output_text,
            final_structured_output=final_structured_output,
        )
