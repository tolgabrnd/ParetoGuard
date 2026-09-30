"""Unit tests for ClosedLoopExecutor: the Phase E automatic closed-loop
control plane (routing -> execution -> recovery -> feedback)."""

from uuid import uuid4

from paretoguard.chaos.faults import ExpectedRecoverability, FaultCategory, ProviderFaultKind
from paretoguard.chaos.injector import FaultInjector, FaultyProvider
from paretoguard.chaos.policies import FaultPolicy, StepRangeSchedule
from paretoguard.core.models import (
    ErrorInfo,
    FailureCategory,
    FinishReason,
    InferenceRequest,
    InferenceResponse,
    LatencyRecord,
    Message,
    ModelSpec,
    Role,
    RoutingDecision,
    TokenUsage,
)
from paretoguard.evals.models import EvalCase, GraderConfig, GraderKind, GroundTruth
from paretoguard.providers.base import Provider
from paretoguard.recovery.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState
from paretoguard.recovery.fallback import FallbackPolicy
from paretoguard.recovery.policy import RecoveryPolicy
from paretoguard.recovery.retry import RecoveryRetryPolicy
from paretoguard.routing.escalation import EscalationRouter
from paretoguard.routing.execution import ClosedLoopExecutor
from paretoguard.routing.protocol import Router
from paretoguard.routing.static import StaticRouter
from paretoguard.routing.types import RoutingRequest
from paretoguard.runtime import RetryPolicy
from paretoguard.storage import ExperimentStore
from paretoguard.telemetry.health import HealthTracker


class _KeyedScriptedProvider(Provider):
    """Returns a scripted response sequence per (provider, model), by call
    count for that key — lets tests exercise fail-then-succeed chains
    across different fallback candidates deterministically."""

    def __init__(self, name: str, scripts: dict[str, list[InferenceResponse]]) -> None:
        self.name = name
        self._scripts = scripts
        self._counts: dict[str, int] = {}

    async def complete(self, request: InferenceRequest) -> InferenceResponse:
        key = request.model
        count = self._counts.get(key, 0)
        script = self._scripts[key]
        response = script[min(count, len(script) - 1)]
        self._counts[key] = count + 1
        return response.model_copy(
            update={
                "request_id": request.request_id,
                "provider": request.provider,
                "model": request.model,
            }
        )


def _ok(text: str = "answer") -> InferenceResponse:
    return InferenceResponse(
        request_id=uuid4(),
        provider="mock",
        model="m",
        output_text=text,
        finish_reason=FinishReason.STOP,
        token_usage=TokenUsage(input_tokens=2, output_tokens=2),
        latency=LatencyRecord(total_latency_ms=1.0),
    )


def _fail(category: FailureCategory = FailureCategory.TIMEOUT) -> InferenceResponse:
    return InferenceResponse(
        request_id=uuid4(),
        provider="mock",
        model="m",
        finish_reason=FinishReason.ERROR,
        token_usage=TokenUsage(input_tokens=1, output_tokens=0),
        latency=LatencyRecord(total_latency_ms=1.0),
        error=ErrorInfo(category=category, message="simulated failure", retryable=True),
    )


def _case(case_id: str = "c1") -> EvalCase:
    return EvalCase(
        case_id=case_id,
        messages=[Message(role=Role.USER, content="hi there friend")],
        grader=GraderConfig(kind=GraderKind.EXACT_MATCH),
        ground_truth=GroundTruth(text="answer"),
    )


def _candidates() -> list[ModelSpec]:
    return [
        ModelSpec(name="model-a", provider="mock", context_window=100_000),
        ModelSpec(name="model-b", provider="mock", context_window=100_000),
        ModelSpec(name="model-c", provider="mock", context_window=100_000),
    ]


# Runtime's own default RetryPolicy (3 attempts, real backoff delay) would
# otherwise silently consume a scripted fail-then-succeed sequence before
# the recovery layer ever saw a failure, and would slow every test down
# with real sleeps. Every ClosedLoopExecutor below disables it explicitly
# so each test isolates the *recovery*-layer behavior it's named for.
_NO_RUNTIME_RETRY = RetryPolicy(max_attempts=1)


async def test_succeeds_on_first_attempt_and_updates_health() -> None:
    provider = _KeyedScriptedProvider("mock", {"model-a": [_ok("answer")]})
    tracker = HealthTracker()
    executor = ClosedLoopExecutor(
        StaticRouter(provider="mock", model="model-a"),
        _candidates(),
        {"mock": provider},
        health_tracker=tracker,
        retry_policy=_NO_RUNTIME_RETRY,
    )
    result, outcome = await executor.execute(_case())
    assert result.succeeded
    assert outcome.succeeded
    assert outcome.attempt_count == 1
    assert outcome.recovery_actions == []
    health = tracker.get("mock", "model-a")
    assert health is not None
    assert health.success_count == 1


async def test_recovers_via_fallback_after_transient_failure() -> None:
    provider = _KeyedScriptedProvider("mock", {"model-a": [_fail()], "model-b": [_ok("answer")]})
    router = StaticRouter(provider="mock", model="model-a")
    policy = RecoveryPolicy(retry_policy=RecoveryRetryPolicy(max_same_candidate_attempts=0))
    executor = ClosedLoopExecutor(
        router,
        _candidates(),
        {"mock": provider},
        recovery_policy=policy,
        retry_policy=_NO_RUNTIME_RETRY,
    )
    result, outcome = await executor.execute(_case())
    assert result.succeeded
    assert outcome.final_model == "model-b"
    assert outcome.attempt_count == 2
    assert outcome.recovery_actions == ["fallback_model"]


async def test_retries_same_candidate_before_falling_back() -> None:
    provider = _KeyedScriptedProvider("mock", {"model-a": [_fail(), _ok("answer")]})
    router = StaticRouter(provider="mock", model="model-a")
    policy = RecoveryPolicy(retry_policy=RecoveryRetryPolicy(max_same_candidate_attempts=1))
    executor = ClosedLoopExecutor(
        router,
        _candidates(),
        {"mock": provider},
        recovery_policy=policy,
        retry_policy=_NO_RUNTIME_RETRY,
    )
    result, outcome = await executor.execute(_case())
    assert result.succeeded
    assert outcome.final_model == "model-a"
    assert outcome.attempt_count == 2
    assert outcome.recovery_actions == ["retry_same"]


async def test_terminal_abstain_when_all_candidates_fail() -> None:
    provider = _KeyedScriptedProvider(
        "mock",
        {"model-a": [_fail()], "model-b": [_fail()], "model-c": [_fail()]},
    )
    router = StaticRouter(provider="mock", model="model-a")
    policy = RecoveryPolicy(
        retry_policy=RecoveryRetryPolicy(max_same_candidate_attempts=0),
        fallback_policy=FallbackPolicy(max_fallback_depth=3),
    )
    executor = ClosedLoopExecutor(
        router,
        _candidates(),
        {"mock": provider},
        recovery_policy=policy,
        max_attempts=10,
        retry_policy=_NO_RUNTIME_RETRY,
    )
    result, outcome = await executor.execute(_case())
    assert not result.succeeded
    assert not outcome.succeeded
    assert outcome.recovery_actions[-1] == "abstain"
    # never exceeds the configured fallback depth's worth of distinct candidates
    assert outcome.attempt_count <= 3


async def test_max_attempts_hard_caps_the_loop_even_with_unlimited_retry_budget() -> None:
    """Safety net: even a pathological RecoveryPolicy that always says
    RETRY_SAME must not loop forever — max_attempts is the executor's own
    hard limit, independent of what any policy decides."""
    provider = _KeyedScriptedProvider("mock", {"model-a": [_fail()]})
    router = StaticRouter(provider="mock", model="model-a")
    policy = RecoveryPolicy(retry_policy=RecoveryRetryPolicy(max_same_candidate_attempts=1_000_000))
    executor = ClosedLoopExecutor(
        router,
        _candidates(),
        {"mock": provider},
        recovery_policy=policy,
        max_attempts=4,
        retry_policy=_NO_RUNTIME_RETRY,
    )
    result, outcome = await executor.execute(_case())
    assert not result.succeeded
    assert outcome.attempt_count == 4


async def test_escalation_router_feedback_loop_closes_automatically() -> None:
    """The whole point of Commit 28: EscalationRouter.record_outcome() is
    never called by the test — ClosedLoopExecutor must call it."""

    class _AlwaysConfidentRouter(Router):
        def __init__(self) -> None:
            self.name = "inner"

        def route(self, routing_request: RoutingRequest) -> RoutingDecision:
            return RoutingDecision(
                selected_model="model-a", predicted_success=0.95, explanation="confident"
            )

    router = EscalationRouter(_AlwaysConfidentRouter())
    provider = _KeyedScriptedProvider("mock", {"model-a": [_ok("answer")]})
    executor = ClosedLoopExecutor(
        router, _candidates(), {"mock": provider}, retry_policy=_NO_RUNTIME_RETRY
    )
    await executor.execute(_case())

    records = router.escalation_log()
    assert len(records) == 1
    assert records[0].succeeded is True  # set by ClosedLoopExecutor, not the test


async def test_circuit_breaker_updates_automatically_and_blocks_retry_same() -> None:
    provider = _KeyedScriptedProvider(
        "mock", {"model-a": [_fail(), _fail()], "model-b": [_ok("answer")]}
    )
    router = StaticRouter(provider="mock", model="model-a")
    breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1))
    policy = RecoveryPolicy(
        retry_policy=RecoveryRetryPolicy(max_same_candidate_attempts=5), circuit_breaker=breaker
    )
    executor = ClosedLoopExecutor(
        router,
        _candidates(),
        {"mock": provider},
        recovery_policy=policy,
        circuit_breaker=breaker,
        retry_policy=_NO_RUNTIME_RETRY,
    )
    result, outcome = await executor.execute(_case())
    assert result.succeeded
    assert outcome.final_model == "model-b"
    # circuit for model-a must have recorded the failure and opened
    assert breaker.state_for("mock:model-a") == CircuitState.OPEN


async def test_persists_through_the_normal_experiment_pipeline() -> None:
    provider = _KeyedScriptedProvider("mock", {"model-a": [_fail()], "model-b": [_ok("answer")]})
    router = StaticRouter(provider="mock", model="model-a")
    policy = RecoveryPolicy(retry_policy=RecoveryRetryPolicy(max_same_candidate_attempts=0))
    with ExperimentStore(":memory:") as store:
        executor = ClosedLoopExecutor(
            router,
            _candidates(),
            {"mock": provider},
            recovery_policy=policy,
            store=store,
            retry_policy=_NO_RUNTIME_RETRY,
        )
        await executor.execute(_case(), run_id="run-1")

        eval_results = store.get_eval_results("run-1")
        assert len(eval_results) == 1
        assert eval_results[0].succeeded

        decisions_df = store.routing_decisions_df(run_id="run-1")
        assert decisions_df.height == 1  # only the *initial* routing decision

        requests_df = store.requests_df(run_id="run-1")
        assert requests_df.height == 2  # both attempts persisted

        trace_df = store.trace_events_df(run_id="run-1")
        assert trace_df.height >= 1  # at least the recovery-action event

        outcome_events = store.get_outcome_events("run-1")
        assert len(outcome_events) == 1
        assert outcome_events[0].succeeded
        assert outcome_events[0].attempt_count == 2
        assert outcome_events[0].final_model == "model-b"


async def test_chaos_step_advances_per_attempt_so_a_retry_can_escape_a_fault() -> None:
    """Regression test for the Commit 29 `chaos_step` fix: each attempt's
    request must carry `chaos_step = attempt_number - 1`, not a fixed value
    — otherwise a retry of the same candidate would replay the identical
    fault decision as the attempt it's retrying and could never succeed."""
    inner = _KeyedScriptedProvider("mock", {"model-a": [_ok("answer")]})
    injector = FaultInjector(
        [
            FaultPolicy(
                fault_id="first-attempt-only",
                category=FaultCategory.PROVIDER,
                kind=ProviderFaultKind.SERVER_ERROR.value,
                # Fires only at chaos_step == 0 (the first attempt).
                schedule=StepRangeSchedule(((0, 1.0), (1, 0.0))),
                target_provider="mock",
                expected_recoverability=ExpectedRecoverability.RECOVERABLE,
            )
        ],
        seed=0,
    )
    provider = FaultyProvider(inner=inner, injector=injector)
    router = StaticRouter(provider="mock", model="model-a")
    policy = RecoveryPolicy(retry_policy=RecoveryRetryPolicy(max_same_candidate_attempts=1))
    executor = ClosedLoopExecutor(
        router,
        _candidates(),
        {"mock": provider},
        recovery_policy=policy,
        retry_policy=_NO_RUNTIME_RETRY,
    )
    result, outcome = await executor.execute(_case())
    assert result.succeeded
    assert outcome.attempt_count == 2
    assert outcome.final_model == "model-a"  # recovered via retry, never fell back


async def test_rejects_candidate_with_no_configured_provider() -> None:
    import pytest

    candidates = [ModelSpec(name="orphan", provider="ghost", context_window=1000)]
    with pytest.raises(ValueError, match="ghost"):
        ClosedLoopExecutor(
            StaticRouter(provider="mock", model="orphan"),
            candidates,
            {"mock": _KeyedScriptedProvider("mock", {})},
        )
