"""Deterministic, seed-controlled fault injection.

`FaultInjector` only *decides* whether a fault fires — seeded on stable
identifiers (task id, step, fault id), never a random UUID or wall-clock
time (see the Phase D audit's determinism bug and
`core.ids.deterministic_request_id`, the same discipline applied here).
Applying a decided fault is a separate, explicit step (`FaultyProvider` for
providers; `check_tool_fault`/`apply_tool_fault` for tools) — nothing here
monkey-patches global state; every wrapper is a plain composition object the
caller constructs and controls.
"""

from collections.abc import Callable
from dataclasses import dataclass
from random import Random

from paretoguard.chaos.faults import (
    FaultCategory,
    FaultEvent,
    ProviderFaultKind,
    ToolFaultKind,
)
from paretoguard.chaos.policies import FaultPolicy
from paretoguard.core.models import (
    ErrorInfo,
    FailureCategory,
    FinishReason,
    InferenceRequest,
    InferenceResponse,
    LatencyRecord,
    ToolResult,
)
from paretoguard.providers.base import Provider

_INJECTION_POINT_BY_CATEGORY: dict[FaultCategory, str] = {
    FaultCategory.PROVIDER: "provider_call",
    FaultCategory.TOOL: "tool_execution",
    FaultCategory.CONTEXT: "task_construction",
}


class FaultInjector:
    """Holds a fixed set of `FaultPolicy`s and a run seed. `faults_for` is a
    pure function of its arguments plus that seed — the same
    `(task_id, step, target)` always yields the same decision, across
    process runs, given the same seed."""

    def __init__(self, policies: list[FaultPolicy], *, seed: int) -> None:
        self._policies = policies
        self._seed = seed

    def _rng_for(self, task_id: str, step: int, fault_id: str) -> Random:
        return Random(f"{self._seed}:{task_id}:{step}:{fault_id}")

    def faults_for(
        self,
        *,
        task_id: str,
        step: int,
        provider: str | None = None,
        model: str | None = None,
        tool: str | None = None,
    ) -> list[FaultEvent]:
        """Every policy whose target matches and whose seeded draw at this
        step fires. Usually 0 or 1 in practice, but nothing prevents
        multiple policies targeting the same call from both firing."""
        fired: list[FaultEvent] = []
        for policy in self._policies:
            if not self._targets_match(policy, provider, model, tool):
                continue
            probability = policy.schedule.probability_at(step)
            rng = self._rng_for(task_id, step, policy.fault_id)
            if rng.random() < probability:
                fired.append(
                    FaultEvent(
                        fault_id=policy.fault_id,
                        category=policy.category,
                        kind=policy.kind,
                        target_provider=policy.target_provider,
                        target_model=policy.target_model,
                        target_tool=policy.target_tool,
                        injection_point=_INJECTION_POINT_BY_CATEGORY[policy.category],
                        seed=self._seed,
                        step=step,
                        parameters=dict(policy.parameters),
                        expected_recoverability=policy.expected_recoverability,
                    )
                )
        return fired

    @staticmethod
    def _targets_match(
        policy: FaultPolicy, provider: str | None, model: str | None, tool: str | None
    ) -> bool:
        if policy.target_provider is not None and policy.target_provider != provider:
            return False
        if policy.target_model is not None and policy.target_model != model:
            return False
        return not (policy.target_tool is not None and policy.target_tool != tool)


def _error_response(
    request: InferenceRequest, category: FailureCategory, message: str, *, retryable: bool
) -> InferenceResponse:
    from paretoguard.core.models import TokenUsage

    return InferenceResponse(
        request_id=request.request_id,
        provider=request.provider,
        model=request.model,
        finish_reason=FinishReason.ERROR,
        token_usage=TokenUsage(input_tokens=0, output_tokens=0),
        latency=LatencyRecord(total_latency_ms=0.0),
        error=ErrorInfo(category=category, message=message, retryable=retryable),
    )


_PROVIDER_FAULT_CATEGORY: dict[ProviderFaultKind, FailureCategory] = {
    ProviderFaultKind.TIMEOUT: FailureCategory.TIMEOUT,
    ProviderFaultKind.RATE_LIMIT: FailureCategory.RATE_LIMIT,
    ProviderFaultKind.SERVER_ERROR: FailureCategory.PROVIDER_FAILURE,
}


FaultCallback = Callable[[FaultEvent], None]


@dataclass
class FaultyProvider(Provider):
    """Wraps `inner`, consulting `injector` before every call. The step
    index is read from `request.metadata["chaos_step"]` (default `0`) —
    the caller (a benchmark loop, `AgentExecutor`, ...) owns tracking it,
    keeping this wrapper itself stateless and reusable across many requests.
    """

    inner: Provider
    injector: FaultInjector
    on_fault: FaultCallback | None = None

    def __post_init__(self) -> None:
        self.name = self.inner.name

    async def complete(self, request: InferenceRequest) -> InferenceResponse:
        step = int(request.metadata.get("chaos_step", 0))
        task_id = request.task_id or str(request.request_id)
        events = self.injector.faults_for(
            task_id=task_id, step=step, provider=request.provider, model=request.model
        )
        provider_events = [e for e in events if e.category == FaultCategory.PROVIDER]
        if not provider_events:
            return await self.inner.complete(request)

        event = provider_events[0]
        if self.on_fault is not None:
            self.on_fault(event)
        return await self._apply(request, event)

    async def _apply(self, request: InferenceRequest, event: FaultEvent) -> InferenceResponse:
        kind = ProviderFaultKind(event.kind)
        if kind in _PROVIDER_FAULT_CATEGORY:
            return _error_response(
                request,
                _PROVIDER_FAULT_CATEGORY[kind],
                f"chaos: simulated {kind.value} (fault_id={event.fault_id!r})",
                retryable=True,
            )

        response = await self.inner.complete(request)
        if not response.succeeded:
            return response

        if kind == ProviderFaultKind.LATENCY_SPIKE:
            multiplier = float(event.parameters.get("multiplier", 5.0))
            return response.model_copy(
                update={
                    "latency": response.latency.model_copy(
                        update={"total_latency_ms": response.latency.total_latency_ms * multiplier}
                    )
                }
            )
        if kind == ProviderFaultKind.MALFORMED_STRUCTURED_OUTPUT and response.structured_output:
            drop_key = event.parameters.get("drop_key")
            corrupted = dict(response.structured_output)
            if drop_key in corrupted:
                del corrupted[drop_key]
            else:
                corrupted["__malformed__"] = True
            return response.model_copy(update={"structured_output": corrupted})
        if kind == ProviderFaultKind.TRUNCATED_OUTPUT and response.output_text:
            ratio = float(event.parameters.get("truncate_ratio", 0.5))
            cut = max(1, int(len(response.output_text) * ratio))
            return response.model_copy(
                update={
                    "output_text": response.output_text[:cut],
                    "finish_reason": FinishReason.LENGTH,
                }
            )
        return response


def check_tool_fault(
    injector: FaultInjector, *, task_id: str, step: int, tool_name: str
) -> FaultEvent | None:
    """The tool-side equivalent of `FaultyProvider`'s decision step — call
    this immediately before `tool.run(...)`, and use `apply_tool_fault`
    instead of the real result if it returns an event."""
    events = injector.faults_for(task_id=task_id, step=step, tool=tool_name)
    tool_events = [e for e in events if e.category == FaultCategory.TOOL]
    return tool_events[0] if tool_events else None


def apply_tool_fault(event: FaultEvent, real_result: ToolResult | None) -> ToolResult:
    """Turns a decided `FaultEvent` into the `ToolResult` a caller should
    use instead of the tool's real output. `real_result` is required for
    faults that corrupt a real result (`STALE_RESULT`/`MISSING_FIELD`/
    `PARTIAL_RESULT`/`SCHEMA_MISMATCH`) — pass `None` only for faults that
    never call the underlying tool at all (`EXCEPTION`/`TIMEOUT`/
    `TEMPORARY_UNAVAILABLE`)."""
    kind = ToolFaultKind(event.kind)
    if kind == ToolFaultKind.EXCEPTION:
        return ToolResult(
            tool_call_id="",
            is_error=True,
            error_message=f"chaos: simulated tool exception ({event.fault_id})",
        )
    if kind == ToolFaultKind.TIMEOUT:
        return ToolResult(
            tool_call_id="",
            is_error=True,
            error_message=f"chaos: simulated tool timeout ({event.fault_id})",
        )
    if kind == ToolFaultKind.TEMPORARY_UNAVAILABLE:
        return ToolResult(
            tool_call_id="",
            is_error=True,
            error_message=f"chaos: tool temporarily unavailable ({event.fault_id})",
        )

    if real_result is None or real_result.is_error or not isinstance(real_result.output, dict):
        return (
            real_result if real_result is not None else ToolResult(tool_call_id="", is_error=True)
        )

    output = dict(real_result.output)
    if kind == ToolFaultKind.STALE_RESULT:
        output["_stale"] = True
        return real_result.model_copy(update={"output": output})
    if kind == ToolFaultKind.MISSING_FIELD:
        drop_key = event.parameters.get("drop_key") or next(iter(output), None)
        output.pop(drop_key, None)
        return real_result.model_copy(update={"output": output})
    if kind == ToolFaultKind.SCHEMA_MISMATCH:
        return real_result.model_copy(update={"output": {"unexpected_shape": str(output)}})
    if kind == ToolFaultKind.PARTIAL_RESULT:
        keys = list(output)[: max(1, len(output) // 2)]
        return real_result.model_copy(update={"output": {k: output[k] for k in keys}})
    return real_result
