"""Unit tests for HealthTracker / ModelHealth."""

from uuid import uuid4

import pytest

from paretoguard.core.models import (
    FailureCategory,
    FinishReason,
    InferenceRequest,
    LatencyRecord,
    Message,
    Role,
    TokenUsage,
)
from paretoguard.core.models.inference import ErrorInfo, InferenceResponse
from paretoguard.providers.mock import SCENARIO_METADATA_KEY, MockProvider, MockScenario
from paretoguard.runtime import RetryPolicy, Runtime
from paretoguard.telemetry import HealthTracker


def _response(
    *,
    provider: str = "mock",
    model: str = "mock-strong",
    succeeded: bool = True,
    latency_ms: float = 10.0,
) -> InferenceResponse:
    return InferenceResponse(
        request_id=uuid4(),
        provider=provider,
        model=model,
        output_text="ok" if succeeded else None,
        finish_reason=FinishReason.STOP if succeeded else FinishReason.ERROR,
        token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        latency=LatencyRecord(total_latency_ms=latency_ms),
        error=None
        if succeeded
        else ErrorInfo(category=FailureCategory.RATE_LIMIT, message="rl", retryable=True),
    )


def test_record_outcome_tracks_counts_and_success_rate() -> None:
    tracker = HealthTracker()
    tracker.record_outcome("mock", "mock-strong", succeeded=True, latency_ms=10.0)
    tracker.record_outcome(
        "mock",
        "mock-strong",
        succeeded=False,
        latency_ms=20.0,
        error_category=FailureCategory.TIMEOUT,
    )

    health = tracker.get("mock", "mock-strong")
    assert health is not None
    assert health.request_count == 2
    assert health.success_count == 1
    assert health.timeout_count == 1
    assert health.success_rate == pytest.approx(0.5)


def test_health_success_rate_is_none_with_no_requests() -> None:
    tracker = HealthTracker()
    assert tracker.get("mock", "mock-strong") is None


def test_error_category_counters_are_isolated_per_model() -> None:
    tracker = HealthTracker()
    tracker.record_outcome(
        "mock", "a", succeeded=False, latency_ms=1.0, error_category=FailureCategory.RATE_LIMIT
    )
    tracker.record_outcome(
        "mock",
        "b",
        succeeded=False,
        latency_ms=1.0,
        error_category=FailureCategory.PROVIDER_FAILURE,
    )
    a = tracker.get("mock", "a")
    b = tracker.get("mock", "b")
    assert a is not None and a.rate_limit_count == 1 and a.server_error_count == 0
    assert b is not None and b.server_error_count == 1 and b.rate_limit_count == 0


def test_record_response_delegates_to_record_outcome() -> None:
    tracker = HealthTracker()
    tracker.record_response(_response(succeeded=True, latency_ms=42.0))
    health = tracker.get("mock", "mock-strong")
    assert health is not None
    assert health.success_count == 1


def test_ema_latency_converges_toward_recent_samples() -> None:
    tracker = HealthTracker(ema_alpha=0.5)
    tracker.record_outcome("mock", "m", succeeded=True, latency_ms=100.0)
    tracker.record_outcome("mock", "m", succeeded=True, latency_ms=100.0)
    for _ in range(20):
        tracker.record_outcome("mock", "m", succeeded=True, latency_ms=10.0)
    ema = tracker.ema_latency_ms("mock", "m")
    assert ema is not None
    assert ema < 15.0  # should have converged close to the recent 10ms samples


def test_p95_latency_over_known_samples() -> None:
    tracker = HealthTracker()
    for i in range(1, 101):  # 1..100
        tracker.record_outcome("mock", "m", succeeded=True, latency_ms=float(i))
    p95 = tracker.p95_latency_ms("mock", "m")
    assert p95 == 95.0


def test_p95_latency_is_none_with_no_samples() -> None:
    tracker = HealthTracker()
    assert tracker.p95_latency_ms("mock", "m") is None


def test_snapshot_returns_all_tracked_pairs() -> None:
    tracker = HealthTracker()
    tracker.record_outcome("mock", "a", succeeded=True, latency_ms=1.0)
    tracker.record_outcome("mock", "b", succeeded=True, latency_ms=1.0)
    pairs = {(h.provider, h.model) for h in tracker.snapshot()}
    assert pairs == {("mock", "a"), ("mock", "b")}


def test_invalid_ema_alpha_rejected() -> None:
    with pytest.raises(ValueError):
        HealthTracker(ema_alpha=0.0)
    with pytest.raises(ValueError):
        HealthTracker(ema_alpha=1.5)


def test_invalid_latency_window_rejected() -> None:
    with pytest.raises(ValueError):
        HealthTracker(latency_window=0)


def test_invalid_baseline_ema_alpha_rejected() -> None:
    with pytest.raises(ValueError):
        HealthTracker(baseline_ema_alpha=0.0)


def test_ema_success_rate_reacts_faster_than_cumulative_success_rate() -> None:
    tracker = HealthTracker(ema_alpha=0.3)
    for _ in range(50):
        tracker.record_outcome("mock", "m", succeeded=True, latency_ms=1.0)
    for _ in range(10):
        tracker.record_outcome("mock", "m", succeeded=False, latency_ms=1.0)

    health = tracker.get("mock", "m")
    assert health is not None
    assert health.success_rate is not None and health.ema_success_rate is not None
    # Cumulative average barely moves (50 ok / 60 total); the EMA has already
    # collapsed toward the recent run of failures.
    assert health.success_rate == pytest.approx(50 / 60)
    assert health.ema_success_rate < 0.1


def test_ema_timeout_and_rate_limit_rates_track_their_own_category() -> None:
    tracker = HealthTracker(ema_alpha=0.5)
    tracker.record_outcome("mock", "m", succeeded=True, latency_ms=1.0)
    tracker.record_outcome(
        "mock", "m", succeeded=False, latency_ms=1.0, error_category=FailureCategory.TIMEOUT
    )
    health = tracker.get("mock", "m")
    assert health is not None
    # First call initializes the EMA at 0.0 (no timeout); second call blends
    # in a timeout: 0.5*1.0 + 0.5*0.0 = 0.5.
    assert health.ema_timeout_rate == pytest.approx(0.5)
    assert health.ema_rate_limit_rate == pytest.approx(0.0)


def test_consecutive_streak_counters_reset_on_opposite_outcome() -> None:
    tracker = HealthTracker()
    for _ in range(3):
        tracker.record_outcome("mock", "m", succeeded=True, latency_ms=1.0)
    tracker.record_outcome("mock", "m", succeeded=False, latency_ms=1.0)
    tracker.record_outcome("mock", "m", succeeded=True, latency_ms=1.0)
    tracker.record_outcome("mock", "m", succeeded=True, latency_ms=1.0)

    health = tracker.get("mock", "m")
    assert health is not None
    assert health.consecutive_successes == 2
    assert health.consecutive_failures == 0


def test_latency_drift_ratio_detects_a_recent_spike_above_baseline() -> None:
    tracker = HealthTracker(ema_alpha=0.5, baseline_ema_alpha=0.02)
    for _ in range(100):
        tracker.record_outcome("mock", "m", succeeded=True, latency_ms=100.0)
    for _ in range(5):
        tracker.record_outcome("mock", "m", succeeded=True, latency_ms=1000.0)

    drift = tracker.latency_drift_ratio("mock", "m")
    assert drift is not None
    assert drift > 1.5  # fast EMA has jumped toward 1000ms; slow baseline is still near 100ms


def test_latency_drift_ratio_is_none_with_no_samples() -> None:
    tracker = HealthTracker()
    assert tracker.latency_drift_ratio("mock", "m") is None


async def test_wired_into_runtime_via_trace_events() -> None:
    """End-to-end: HealthTracker.record_trace_event actually understands the
    payload shape Runtime emits for PROVIDER_CALL events."""
    tracker = HealthTracker()
    provider = MockProvider()
    runtime = Runtime(
        provider,
        retry_policy=RetryPolicy(max_attempts=2, base_delay_s=0.001, max_delay_s=0.002),
        on_trace_event=tracker.record_trace_event,
    )

    ok_request = InferenceRequest(
        provider="mock", model="mock-strong", messages=[Message(role=Role.USER, content="hi")]
    )
    await runtime.run(ok_request)

    fail_request = InferenceRequest(
        provider="mock",
        model="mock-strong",
        messages=[Message(role=Role.USER, content="hi")],
        metadata={SCENARIO_METADATA_KEY: MockScenario.RATE_LIMIT.value},
    )
    await runtime.run(fail_request)

    health = tracker.get("mock", "mock-strong")
    assert health is not None
    assert health.success_count == 1
    # 2 attempts were made for the failing request (max_attempts=2), both rate-limited.
    assert health.rate_limit_count == 2
    assert health.request_count == 3
