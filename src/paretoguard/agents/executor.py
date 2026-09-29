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
"""

import time
from dataclasses import dataclass
from typing import Any

from paretoguard.agents.protocol import Tool, ToolArgumentError
from paretoguard.agents.state import AgentStep, AgentTrajectory, TerminationReason
from paretoguard.core.ids import deterministic_request_id
from paretoguard.core.models import FinishReason, InferenceRequest, Message, Role
from paretoguard.runtime import Runtime


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
        self, runtime: Runtime, tools: list[Tool[Any]], *, limits: AgentLimits | None = None
    ) -> None:
        self._runtime = runtime
        self._tools: dict[str, Tool[Any]] = {tool.name: tool for tool in tools}
        self._limits = limits or AgentLimits()

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
                metadata=metadata or {},
            )
            response = await self._runtime.run(request)
            step_cost = response.cost.total_cost_usd if response.cost else None

            if not response.succeeded:
                steps.append(
                    AgentStep(
                        step_index=step_index,
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
                        latency_ms=response.latency.total_latency_ms,
                        cost_usd=step_cost,
                        total_tokens=response.token_usage.total_tokens,
                    )
                )
                return self._finish(
                    task_id, provider, model, steps, TerminationReason.INVALID_TOOL_ARGUMENTS
                )

            result = tool.run(validated_args).model_copy(update={"tool_call_id": call.id})
            steps.append(
                AgentStep(
                    step_index=step_index,
                    requested_tool=call.name,
                    requested_arguments=call.arguments,
                    arguments_valid=True,
                    tool_result=result,
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
