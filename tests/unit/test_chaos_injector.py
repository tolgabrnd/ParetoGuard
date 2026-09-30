"""Unit tests for FaultInjector, FaultyProvider, and tool-fault application."""

import pytest

from paretoguard.chaos.faults import (
    ContextFaultKind,
    FaultCategory,
    FaultEvent,
    ProviderFaultKind,
    ToolFaultKind,
)
from paretoguard.chaos.injector import (
    FaultInjector,
    FaultyProvider,
    apply_tool_fault,
    check_tool_fault,
)
from paretoguard.chaos.policies import ConstantProbability, FaultPolicy
from paretoguard.chaos.scenarios import apply_context_fault, provider_degradation_schedule
from paretoguard.core.models import (
    FinishReason,
    InferenceRequest,
    InferenceResponse,
    LatencyRecord,
    Message,
    Role,
    ToolResult,
)
from paretoguard.providers.mock import MockProvider


def _policy(fault_id: str = "p1", probability: float = 1.0, **kwargs) -> FaultPolicy:
    return FaultPolicy(
        fault_id=fault_id,
        category=kwargs.pop("category", FaultCategory.PROVIDER),
        kind=kwargs.pop("kind", ProviderFaultKind.TIMEOUT.value),
        schedule=ConstantProbability(probability),
        **kwargs,
    )


def _request(step: int = 0, provider: str = "mock", model: str = "mock-strong") -> InferenceRequest:
    return InferenceRequest(
        task_id="task-1",
        provider=provider,
        model=model,
        messages=[Message(role=Role.USER, content="hi")],
        metadata={"chaos_step": step},
    )


def test_same_seed_and_stable_ids_produce_identical_faults_across_fresh_injectors() -> None:
    policy = _policy(probability=0.5)
    injector_a = FaultInjector([policy], seed=7)
    injector_b = FaultInjector([policy], seed=7)
    results_a = [
        injector_a.faults_for(task_id="t1", step=s, provider="mock", model="m") for s in range(50)
    ]
    results_b = [
        injector_b.faults_for(task_id="t1", step=s, provider="mock", model="m") for s in range(50)
    ]
    assert [len(r) for r in results_a] == [len(r) for r in results_b]


def test_different_seed_can_produce_different_faults() -> None:
    policy = _policy(probability=0.5)
    injector_a = FaultInjector([policy], seed=1)
    injector_b = FaultInjector([policy], seed=2)
    fired_a = [
        bool(injector_a.faults_for(task_id="t1", step=s, provider="mock", model="m"))
        for s in range(50)
    ]
    fired_b = [
        bool(injector_b.faults_for(task_id="t1", step=s, provider="mock", model="m"))
        for s in range(50)
    ]
    assert fired_a != fired_b


def test_deterministic_schedule_fires_exact_fault_in_the_configured_window() -> None:
    policy = provider_degradation_schedule(
        target_provider="mock",
        target_model="m",
        degraded_start=100,
        recovered_start=200,
        baseline_probability=0.0,
        degraded_probability=1.0,
    )
    injector = FaultInjector([policy], seed=0)
    assert injector.faults_for(task_id="t", step=50, provider="mock", model="m") == []
    assert len(injector.faults_for(task_id="t", step=150, provider="mock", model="m")) == 1
    assert injector.faults_for(task_id="t", step=250, provider="mock", model="m") == []


def test_fault_policy_does_not_affect_a_different_target() -> None:
    policy = _policy(probability=1.0, target_provider="provider-x")
    injector = FaultInjector([policy], seed=0)
    assert injector.faults_for(task_id="t", step=0, provider="provider-y", model="m") == []
    assert len(injector.faults_for(task_id="t", step=0, provider="provider-x", model="m")) == 1


def test_fault_event_metadata_is_recorded() -> None:
    policy = _policy(
        probability=1.0, target_provider="mock", target_model="m", parameters={"k": "v"}
    )
    injector = FaultInjector([policy], seed=3)
    events = injector.faults_for(task_id="t", step=0, provider="mock", model="m")
    assert len(events) == 1
    event = events[0]
    assert event.fault_id == "p1"
    assert event.category == FaultCategory.PROVIDER
    assert event.kind == ProviderFaultKind.TIMEOUT.value
    assert event.seed == 3
    assert event.step == 0
    assert event.parameters == {"k": "v"}
    assert event.injection_point == "provider_call"


async def test_faulty_provider_short_circuits_on_timeout() -> None:
    policy = _policy(probability=1.0, kind=ProviderFaultKind.TIMEOUT.value)
    injector = FaultInjector([policy], seed=0)
    faulty = FaultyProvider(MockProvider(), injector)
    response = await faulty.complete(_request())
    assert not response.succeeded
    assert response.error is not None
    assert response.error.category.value == "timeout"


async def test_faulty_provider_passes_through_when_no_fault_fires() -> None:
    policy = _policy(probability=0.0)
    injector = FaultInjector([policy], seed=0)
    faulty = FaultyProvider(MockProvider(), injector)
    response = await faulty.complete(_request())
    assert response.succeeded


async def test_faulty_provider_calls_on_fault_callback() -> None:
    policy = _policy(probability=1.0)
    injector = FaultInjector([policy], seed=0)
    seen: list[FaultEvent] = []
    faulty = FaultyProvider(MockProvider(), injector, on_fault=seen.append)
    await faulty.complete(_request())
    assert len(seen) == 1
    assert seen[0].fault_id == "p1"


async def test_faulty_provider_truncates_output() -> None:
    policy = _policy(probability=1.0, kind=ProviderFaultKind.TRUNCATED_OUTPUT.value)
    injector = FaultInjector([policy], seed=0)
    faulty = FaultyProvider(MockProvider(), injector)
    response = await faulty.complete(_request())
    assert response.succeeded
    assert response.finish_reason == FinishReason.LENGTH


async def test_faulty_provider_inflates_latency_on_spike() -> None:
    """Uses a fixed-latency stub provider rather than MockProvider's real
    (sub-millisecond, wall-clock-measured) latency — comparing two
    separately-timed real measurements at that scale is too noisy to assert
    a multiplier against reliably."""

    class _FixedLatencyProvider(MockProvider):
        async def complete(self, request: InferenceRequest) -> InferenceResponse:
            response = await super().complete(request)
            return response.model_copy(update={"latency": LatencyRecord(total_latency_ms=100.0)})

    policy = _policy(
        probability=1.0, kind=ProviderFaultKind.LATENCY_SPIKE.value, parameters={"multiplier": 10.0}
    )
    injector = FaultInjector([policy], seed=0)
    faulty = FaultyProvider(_FixedLatencyProvider(), injector)
    response = await faulty.complete(_request())
    assert response.latency.total_latency_ms == pytest.approx(1000.0)


async def test_faulty_provider_uses_chaos_step_from_metadata() -> None:
    policy = provider_degradation_schedule(
        target_provider="mock",
        target_model="mock-strong",
        degraded_start=10,
        recovered_start=20,
        baseline_probability=0.0,
        degraded_probability=1.0,
    )
    injector = FaultInjector([policy], seed=0)
    faulty = FaultyProvider(MockProvider(), injector)
    early = await faulty.complete(_request(step=5))
    degraded = await faulty.complete(_request(step=15))
    assert early.succeeded
    assert not degraded.succeeded


def test_check_tool_fault_returns_none_for_unmatched_target() -> None:
    policy = FaultPolicy(
        fault_id="tool-fault",
        category=FaultCategory.TOOL,
        kind=ToolFaultKind.EXCEPTION.value,
        schedule=ConstantProbability(1.0),
        target_tool="calculator",
    )
    injector = FaultInjector([policy], seed=0)
    assert check_tool_fault(injector, task_id="t", step=0, tool_name="inventory_lookup") is None
    event = check_tool_fault(injector, task_id="t", step=0, tool_name="calculator")
    assert event is not None
    assert event.category == FaultCategory.TOOL


def test_apply_tool_fault_exception_needs_no_real_result() -> None:
    event = FaultEvent(
        fault_id="f",
        category=FaultCategory.TOOL,
        kind=ToolFaultKind.EXCEPTION.value,
        injection_point="tool_execution",
        seed=0,
    )
    result = apply_tool_fault(event, None)
    assert result.is_error


def test_apply_tool_fault_stale_result_marks_output() -> None:
    event = FaultEvent(
        fault_id="f",
        category=FaultCategory.TOOL,
        kind=ToolFaultKind.STALE_RESULT.value,
        injection_point="tool_execution",
        seed=0,
    )
    real = ToolResult(tool_call_id="c1", output={"quantity": 5})
    result = apply_tool_fault(event, real)
    assert result.output["_stale"] is True
    assert result.output["quantity"] == 5


def test_apply_tool_fault_missing_field_drops_a_key() -> None:
    event = FaultEvent(
        fault_id="f",
        category=FaultCategory.TOOL,
        kind=ToolFaultKind.MISSING_FIELD.value,
        injection_point="tool_execution",
        seed=0,
        parameters={"drop_key": "quantity"},
    )
    real = ToolResult(tool_call_id="c1", output={"sku": "SKU-1", "quantity": 5})
    result = apply_tool_fault(event, real)
    assert "quantity" not in result.output
    assert result.output["sku"] == "SKU-1"


def test_apply_tool_fault_partial_result_truncates_fields() -> None:
    event = FaultEvent(
        fault_id="f",
        category=FaultCategory.TOOL,
        kind=ToolFaultKind.PARTIAL_RESULT.value,
        injection_point="tool_execution",
        seed=0,
    )
    real = ToolResult(tool_call_id="c1", output={"a": 1, "b": 2, "c": 3, "d": 4})
    result = apply_tool_fault(event, real)
    assert len(result.output) < 4


def test_apply_context_fault_distractor_injection_appends_text() -> None:
    from random import Random

    text = apply_context_fault("original prompt", ContextFaultKind.DISTRACTOR_INJECTION, Random(0))
    assert "original prompt" in text
    assert len(text) > len("original prompt")


def test_apply_context_fault_contradictory_stale_record_preserves_authoritative_note() -> None:
    from random import Random

    text = apply_context_fault(
        "authoritative fact",
        ContextFaultKind.CONTRADICTORY_STALE_RECORD,
        Random(0),
        contradictory_value="old wrong fact",
    )
    assert "authoritative fact" in text
    assert "old wrong fact" in text
    assert "authoritative" in text.lower()


def test_apply_context_fault_rejects_unknown_kind() -> None:
    from random import Random

    with pytest.raises(ValueError):
        apply_context_fault("x", "not-a-real-kind", Random(0))  # type: ignore[arg-type]
