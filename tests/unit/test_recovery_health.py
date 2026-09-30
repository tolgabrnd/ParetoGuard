"""Unit tests for recovery.health: classify_recovery_health's explicit rule
cascade, and select_fallback's health-tier ordering (Phase E.5)."""

from paretoguard.recovery.health import (
    RecoveryHealthPolicy,
    RecoveryHealthStatus,
    classify_recovery_health,
)
from paretoguard.telemetry.health import ModelHealth


def _health(**overrides) -> ModelHealth:
    defaults = {
        "provider": "mock",
        "model": "m",
        "request_count": 10,
        "success_count": 9,
        "ema_success_rate": 0.9,
        "ema_timeout_rate": 0.0,
        "consecutive_failures": 0,
    }
    defaults.update(overrides)
    return ModelHealth(**defaults)


def test_no_health_data_is_unknown() -> None:
    assert (
        classify_recovery_health(None, None, RecoveryHealthPolicy()) == RecoveryHealthStatus.UNKNOWN
    )


def test_below_min_requests_is_unknown_regardless_of_rate() -> None:
    policy = RecoveryHealthPolicy(min_requests_for_health=5)
    health = _health(request_count=2, success_count=0, ema_success_rate=0.0)
    assert classify_recovery_health(health, None, policy) == RecoveryHealthStatus.UNKNOWN


def test_high_success_rate_with_no_other_flags_is_healthy() -> None:
    health = _health(ema_success_rate=0.95)
    assert (
        classify_recovery_health(health, None, RecoveryHealthPolicy())
        == RecoveryHealthStatus.HEALTHY
    )


def test_low_success_rate_is_unhealthy() -> None:
    health = _health(ema_success_rate=0.3)
    assert (
        classify_recovery_health(health, None, RecoveryHealthPolicy())
        == RecoveryHealthStatus.UNHEALTHY
    )


def test_consecutive_failure_streak_forces_unhealthy_even_with_high_ema() -> None:
    """The EMA-independent, immediate signal: a fresh streak of failures
    hasn't had time to move a slow-decaying EMA yet."""
    health = _health(ema_success_rate=0.95, consecutive_failures=3)
    policy = RecoveryHealthPolicy(unhealthy_consecutive_failures_at_or_above=3)
    assert classify_recovery_health(health, None, policy) == RecoveryHealthStatus.UNHEALTHY


def test_mid_range_success_rate_is_degraded() -> None:
    health = _health(ema_success_rate=0.65)
    assert (
        classify_recovery_health(health, None, RecoveryHealthPolicy())
        == RecoveryHealthStatus.DEGRADED
    )


def test_high_timeout_rate_demotes_an_otherwise_healthy_candidate_to_degraded() -> None:
    health = _health(ema_success_rate=0.95, ema_timeout_rate=0.5)
    assert (
        classify_recovery_health(health, None, RecoveryHealthPolicy())
        == RecoveryHealthStatus.DEGRADED
    )


def test_latency_drift_demotes_an_otherwise_healthy_candidate_to_degraded() -> None:
    health = _health(ema_success_rate=0.95)
    assert (
        classify_recovery_health(health, 3.5, RecoveryHealthPolicy())
        == RecoveryHealthStatus.DEGRADED
    )


def test_timeout_flag_alone_never_escalates_to_unhealthy() -> None:
    """A timeout-prone but otherwise-succeeding candidate is a worse
    choice, never an unusable one — only the dedicated unhealthy rules do
    that."""
    health = _health(ema_success_rate=0.95, ema_timeout_rate=1.0)
    assert (
        classify_recovery_health(health, None, RecoveryHealthPolicy())
        == RecoveryHealthStatus.DEGRADED
    )


def test_policy_rejects_invalid_thresholds() -> None:
    import pytest

    with pytest.raises(ValueError, match="min_requests_for_health"):
        RecoveryHealthPolicy(min_requests_for_health=0)
    with pytest.raises(ValueError, match="unhealthy_ema_success_below"):
        RecoveryHealthPolicy(unhealthy_ema_success_below=0.9, healthy_ema_success_at_or_above=0.5)
    with pytest.raises(ValueError, match="unhealthy_consecutive_failures_at_or_above"):
        RecoveryHealthPolicy(unhealthy_consecutive_failures_at_or_above=0)
    with pytest.raises(ValueError, match="degraded_ema_timeout_rate_above"):
        RecoveryHealthPolicy(degraded_ema_timeout_rate_above=1.5)
    with pytest.raises(ValueError, match="degraded_latency_drift_above"):
        RecoveryHealthPolicy(degraded_latency_drift_above=1.0)
