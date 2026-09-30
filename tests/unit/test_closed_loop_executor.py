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
from paretoguard.recovery.context import RecoveryContext, RecoveryDecision
from paretoguard.recovery.fallback import FallbackPolicy
from paretoguard.recovery.health import RecoveryHealthPolicy
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


async def test_quality_failure_is_detected_and_recovered_via_fallback() -> None:
    """The Phase E.5 fix: a transport-successful-but-wrong-answer response
    must not be treated as terminal success — it should be graded in-loop,
    classified as a quality failure, and trigger recovery."""
    provider = _KeyedScriptedProvider(
        "mock", {"model-a": [_ok("wrong-answer")], "model-b": [_ok("answer")]}
    )
    router = StaticRouter(provider="mock", model="model-a")
    policy = RecoveryPolicy(
        retry_policy=RecoveryRetryPolicy(max_same_candidate_attempts=0),
        fallback_policy=FallbackPolicy(max_fallback_depth=2),
        allow_escalation=False,
    )
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
    assert outcome.final_model == "model-b"
    assert outcome.recovery_actions == ["fallback_model"]


async def test_quality_failure_escalates_when_escalation_enabled() -> None:
    provider = _KeyedScriptedProvider(
        "mock", {"model-a": [_ok("wrong-answer")], "model-b": [_ok("answer")]}
    )
    router = StaticRouter(provider="mock", model="model-a")
    policy = RecoveryPolicy(
        retry_policy=RecoveryRetryPolicy(max_same_candidate_attempts=0),
        fallback_policy=FallbackPolicy(max_fallback_depth=2),
        allow_escalation=True,
    )
    executor = ClosedLoopExecutor(
        router,
        _candidates(),
        {"mock": provider},
        recovery_policy=policy,
        retry_policy=_NO_RUNTIME_RETRY,
    )
    result, outcome = await executor.execute(_case())
    assert result.succeeded
    assert outcome.recovery_actions == ["escalate"]


async def test_quality_failure_counts_as_a_failure_for_health_tracking() -> None:
    provider = _KeyedScriptedProvider(
        "mock", {"model-a": [_ok("wrong-answer")], "model-b": [_ok("answer")]}
    )
    router = StaticRouter(provider="mock", model="model-a")
    policy = RecoveryPolicy(retry_policy=RecoveryRetryPolicy(max_same_candidate_attempts=0))
    tracker = HealthTracker()
    executor = ClosedLoopExecutor(
        router,
        _candidates(),
        {"mock": provider},
        recovery_policy=policy,
        health_tracker=tracker,
        retry_policy=_NO_RUNTIME_RETRY,
    )
    await executor.execute(_case())
    health = tracker.get("mock", "model-a")
    assert health is not None
    assert health.success_count == 0  # the wrong answer must not count as a success
    assert health.request_count == 1


async def test_validate_quality_false_restores_pre_e5_behavior() -> None:
    """With validate_quality=False, a wrong-but-transport-successful answer
    is treated as terminal success (the pre-Phase-E.5 behavior) — recovery
    is never invoked, even though the final EvalResult still correctly
    grades it as failed (grading happens unconditionally at the end)."""
    provider = _KeyedScriptedProvider("mock", {"model-a": [_ok("wrong-answer")]})
    router = StaticRouter(provider="mock", model="model-a")
    policy = RecoveryPolicy(retry_policy=RecoveryRetryPolicy(max_same_candidate_attempts=0))
    executor = ClosedLoopExecutor(
        router,
        _candidates(),
        {"mock": provider},
        recovery_policy=policy,
        retry_policy=_NO_RUNTIME_RETRY,
        validate_quality=False,
    )
    result, outcome = await executor.execute(_case())
    assert outcome.attempt_count == 1  # never recovered: treated as success in-loop
    assert outcome.recovery_actions == []
    assert not result.succeeded  # the final EvalResult still grades it correctly


class _ContextSpyPolicy(RecoveryPolicy):
    """Captures the `RecoveryContext` each `decide()` call receives, then
    delegates to a real policy — lets a test inspect what the executor
    actually built without needing its own RecoveryPolicy subclass logic."""

    captured: list[RecoveryContext]

    def __init__(self, delegate: RecoveryPolicy) -> None:
        object.__setattr__(self, "_delegate", delegate)
        object.__setattr__(self, "captured", [])

    def decide(self, context, candidates) -> RecoveryDecision:  # type: ignore[no-untyped-def]
        self.captured.append(context)
        return self._delegate.decide(context, candidates)  # type: ignore[attr-defined]


async def test_latency_drift_excluded_from_recovery_context_by_default() -> None:
    """Regression test for a real determinism bug found via Phase E.5's
    larger-scale sustained_outage run: HealthTracker's latency EMA is
    derived from real wall-clock timing (even against MockProvider), so
    feeding it into RecoveryContext.latency_drift by default made recovery
    decisions non-deterministic across identical runs. Must stay empty
    unless include_latency_drift_in_recovery=True."""
    provider = _KeyedScriptedProvider("mock", {"model-a": [_fail()], "model-b": [_ok("answer")]})
    tracker = HealthTracker()
    # Seed enough latency samples that latency_drift_ratio is not None.
    for _ in range(5):
        tracker.record_outcome("mock", "model-b", succeeded=True, latency_ms=10.0)
    router = StaticRouter(provider="mock", model="model-a")
    spy = _ContextSpyPolicy(
        RecoveryPolicy(retry_policy=RecoveryRetryPolicy(max_same_candidate_attempts=0))
    )
    executor = ClosedLoopExecutor(
        router,
        _candidates(),
        {"mock": provider},
        recovery_policy=spy,
        health_tracker=tracker,
        retry_policy=_NO_RUNTIME_RETRY,
    )
    await executor.execute(_case())
    assert len(spy.captured) == 1
    assert spy.captured[0].latency_drift == {}


async def test_latency_drift_populated_when_explicitly_enabled() -> None:
    provider = _KeyedScriptedProvider("mock", {"model-a": [_fail()], "model-b": [_ok("answer")]})
    tracker = HealthTracker()
    for _ in range(5):
        tracker.record_outcome("mock", "model-b", succeeded=True, latency_ms=10.0)
    router = StaticRouter(provider="mock", model="model-a")
    spy = _ContextSpyPolicy(
        RecoveryPolicy(retry_policy=RecoveryRetryPolicy(max_same_candidate_attempts=0))
    )
    executor = ClosedLoopExecutor(
        router,
        _candidates(),
        {"mock": provider},
        recovery_policy=spy,
        health_tracker=tracker,
        retry_policy=_NO_RUNTIME_RETRY,
        include_latency_drift_in_recovery=True,
    )
    await executor.execute(_case())
    assert len(spy.captured) == 1
    assert "mock:model-b" in spy.captured[0].latency_drift


async def test_recently_degraded_candidate_is_less_likely_chosen_as_fallback() -> None:
    """End-to-end health-aware recovery proof (Phase E.5): model-b is
    earlier in candidate order than model-c and *would* succeed if picked,
    but its pre-seeded rolling health is badly degraded — the executor must
    prefer the healthier model-c instead, purely from automatic health
    feedback, with no explicit instruction to avoid model-b."""
    provider = _KeyedScriptedProvider(
        "mock",
        {"model-a": [_fail()], "model-b": [_ok("answer")], "model-c": [_ok("answer")]},
    )
    tracker = HealthTracker()
    for _ in range(10):
        tracker.record_outcome(
            "mock",
            "model-b",
            succeeded=False,
            latency_ms=1.0,
            error_category=FailureCategory.PROVIDER_FAILURE,
        )
    router = StaticRouter(provider="mock", model="model-a")
    policy = RecoveryPolicy(
        retry_policy=RecoveryRetryPolicy(max_same_candidate_attempts=0),
        health_policy=RecoveryHealthPolicy(),
    )
    executor = ClosedLoopExecutor(
        router,
        _candidates(),
        {"mock": provider},
        recovery_policy=policy,
        health_tracker=tracker,
        retry_policy=_NO_RUNTIME_RETRY,
    )
    result, outcome = await executor.execute(_case())
    assert result.succeeded
    assert outcome.final_model == "model-c"


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
