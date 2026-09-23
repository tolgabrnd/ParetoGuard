"""Unit tests for ReliabilityAwareRouter."""

from paretoguard.core.models import InferenceRequest, Message, ModelSpec, Role
from paretoguard.routing import (
    CandidateProfile,
    ReliabilityAwareRouter,
    ReliabilityThresholds,
    RoutingRequest,
    candidate_key,
    extract_task_features,
)
from paretoguard.telemetry.health import HealthTracker


def _request() -> InferenceRequest:
    return InferenceRequest(
        provider="mock", model="mock-strong", messages=[Message(role=Role.USER, content="hi")]
    )


def _model(name: str, *, provider: str = "mock") -> ModelSpec:
    return ModelSpec(name=name, provider=provider, context_window=100_000)


def _health_map(tracker: HealthTracker, pairs: list[tuple[str, str]]) -> dict:
    return {
        candidate_key(provider, model): health
        for provider, model in pairs
        if (health := tracker.get(provider, model)) is not None
    }


def _routing_request(candidates, health, profiles=None) -> RoutingRequest:
    request = _request()
    return RoutingRequest(
        request=request,
        features=extract_task_features(request),
        candidates=candidates,
        health=health,
        profiles=profiles or {},
    )


def test_prefers_healthy_candidate_over_degraded_one() -> None:
    tracker = HealthTracker(ema_alpha=0.3)
    for _ in range(20):
        tracker.record_outcome("mock", "healthy", succeeded=True, latency_ms=10.0)
    for _ in range(20):
        tracker.record_outcome("mock", "degraded", succeeded=False, latency_ms=10.0)

    router = ReliabilityAwareRouter()
    candidates = [_model("healthy"), _model("degraded")]
    health = _health_map(tracker, [("mock", "healthy"), ("mock", "degraded")])
    decision = router.route(_routing_request(candidates, health))

    assert decision.selected_model == "healthy"
    assert "degraded" in decision.explanation


def test_degraded_candidate_is_not_hard_excluded() -> None:
    tracker = HealthTracker(ema_alpha=0.3)
    for _ in range(20):
        tracker.record_outcome("mock", "degraded", succeeded=False, latency_ms=10.0)

    router = ReliabilityAwareRouter()
    candidates = [_model("degraded")]
    health = _health_map(tracker, [("mock", "degraded")])
    decision = router.route(_routing_request(candidates, health))

    # Only candidate available: still selected despite being unhealthy, not excluded.
    assert decision.selected_model == "degraded"
    assert decision.excluded_candidates == {}
    assert "least-bad" in decision.explanation


def test_unknown_candidate_with_no_health_history_is_not_penalized() -> None:
    router = ReliabilityAwareRouter()
    candidates = [_model("new-model"), _model("degraded")]
    tracker = HealthTracker(ema_alpha=0.3)
    for _ in range(20):
        tracker.record_outcome("mock", "degraded", succeeded=False, latency_ms=10.0)
    health = _health_map(tracker, [("mock", "degraded")])

    decision = router.route(_routing_request(candidates, health))
    assert decision.selected_model == "new-model"


def test_cold_start_guard_treats_few_samples_as_unknown() -> None:
    thresholds = ReliabilityThresholds(min_requests_for_health=100)
    router = ReliabilityAwareRouter(thresholds)
    tracker = HealthTracker(ema_alpha=0.9)
    # Only 2 failures recorded — far below the cold-start threshold of 100.
    tracker.record_outcome("mock", "few-samples", succeeded=False, latency_ms=10.0)
    tracker.record_outcome("mock", "few-samples", succeeded=False, latency_ms=10.0)
    health = _health_map(tracker, [("mock", "few-samples")])

    candidates = [_model("few-samples")]
    decision = router.route(_routing_request(candidates, health))
    # Still selected (only candidate) but its explanation must not claim "healthy".
    assert decision.selected_model == "few-samples"
    assert "unknown" in decision.explanation


def test_healthy_status_beats_unknown_status() -> None:
    router = ReliabilityAwareRouter()
    tracker = HealthTracker(ema_alpha=0.3)
    for _ in range(20):
        tracker.record_outcome("mock", "healthy", succeeded=True, latency_ms=10.0)
    health = _health_map(tracker, [("mock", "healthy")])

    candidates = [_model("healthy"), _model("unknown-model")]
    decision = router.route(_routing_request(candidates, health))
    assert decision.selected_model == "healthy"


def test_falls_back_to_profile_predicted_success_when_unknown() -> None:
    router = ReliabilityAwareRouter()
    candidates = [_model("a"), _model("b")]
    profiles = {
        candidate_key("mock", "a"): CandidateProfile(predicted_success=0.9),
        candidate_key("mock", "b"): CandidateProfile(predicted_success=0.2),
    }
    decision = router.route(_routing_request(candidates, {}, profiles))
    assert decision.selected_model == "a"


def test_probes_a_never_selected_degraded_candidate_periodically() -> None:
    thresholds = ReliabilityThresholds(probe_interval_calls=3)
    router = ReliabilityAwareRouter(thresholds)
    tracker = HealthTracker(ema_alpha=0.3)
    for _ in range(20):
        tracker.record_outcome("mock", "healthy", succeeded=True, latency_ms=10.0)
    for _ in range(20):
        tracker.record_outcome("mock", "degraded", succeeded=False, latency_ms=10.0)

    candidates = [_model("healthy"), _model("degraded")]
    health = _health_map(tracker, [("mock", "healthy"), ("mock", "degraded")])
    routing_request = _routing_request(candidates, health)

    selections = [router.route(routing_request).selected_model for _ in range(6)]
    # Healthy wins every ranked call, but the degraded candidate must surface
    # at least once as a probe within `probe_interval_calls` calls.
    assert selections.count("healthy") >= 3
    assert "degraded" in selections


def test_is_deterministic() -> None:
    router = ReliabilityAwareRouter()
    tracker = HealthTracker(ema_alpha=0.3)
    for _ in range(20):
        tracker.record_outcome("mock", "healthy", succeeded=True, latency_ms=10.0)
    health = _health_map(tracker, [("mock", "healthy")])
    candidates = [_model("healthy"), _model("other")]
    routing_request = _routing_request(candidates, health)
    decisions = {router.route(routing_request).selected_model for _ in range(5)}
    assert decisions == {"healthy"}
