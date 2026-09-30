"""Unit tests for CircuitBreaker: state-machine transitions with an
injected clock — no sleep() anywhere."""

import pytest

from paretoguard.recovery.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_starts_closed() -> None:
    breaker = CircuitBreaker()
    assert breaker.state_for("k") == CircuitState.CLOSED
    assert breaker.allow_request("k")


def test_closed_to_open_after_failure_threshold() -> None:
    config = CircuitBreakerConfig(failure_threshold=3)
    breaker = CircuitBreaker(config)
    breaker.record_failure("k")
    breaker.record_failure("k")
    assert breaker.state_for("k") == CircuitState.CLOSED
    breaker.record_failure("k")
    assert breaker.state_for("k") == CircuitState.OPEN


def test_open_blocks_requests() -> None:
    config = CircuitBreakerConfig(failure_threshold=1)
    breaker = CircuitBreaker(config)
    breaker.record_failure("k")
    assert breaker.state_for("k") == CircuitState.OPEN
    assert not breaker.allow_request("k")


def test_open_to_half_open_after_cooldown() -> None:
    clock = _FakeClock()
    config = CircuitBreakerConfig(failure_threshold=1, cooldown_s=10.0)
    breaker = CircuitBreaker(config, clock=clock)
    breaker.record_failure("k")
    assert breaker.state_for("k") == CircuitState.OPEN
    clock.advance(5.0)
    assert breaker.state_for("k") == CircuitState.OPEN
    clock.advance(5.0)
    assert breaker.state_for("k") == CircuitState.HALF_OPEN


def test_half_open_allows_only_configured_probe_budget() -> None:
    clock = _FakeClock()
    config = CircuitBreakerConfig(failure_threshold=1, cooldown_s=1.0, half_open_probe_budget=2)
    breaker = CircuitBreaker(config, clock=clock)
    breaker.record_failure("k")
    clock.advance(1.0)
    assert breaker.state_for("k") == CircuitState.HALF_OPEN
    assert breaker.allow_request("k")
    assert breaker.allow_request("k")
    assert not breaker.allow_request("k")  # probe budget exhausted


def test_half_open_closes_after_sufficient_successful_probes() -> None:
    clock = _FakeClock()
    config = CircuitBreakerConfig(failure_threshold=1, cooldown_s=1.0, success_threshold_to_close=2)
    breaker = CircuitBreaker(config, clock=clock)
    breaker.record_failure("k")
    clock.advance(1.0)
    assert breaker.state_for("k") == CircuitState.HALF_OPEN
    breaker.record_success("k")
    assert breaker.state_for("k") == CircuitState.HALF_OPEN  # only 1 of 2
    breaker.record_success("k")
    assert breaker.state_for("k") == CircuitState.CLOSED


def test_half_open_reopens_on_probe_failure() -> None:
    clock = _FakeClock()
    config = CircuitBreakerConfig(failure_threshold=1, cooldown_s=1.0)
    breaker = CircuitBreaker(config, clock=clock)
    breaker.record_failure("k")
    clock.advance(1.0)
    assert breaker.state_for("k") == CircuitState.HALF_OPEN
    breaker.record_failure("k")
    assert breaker.state_for("k") == CircuitState.OPEN


def test_reopening_resets_the_cooldown_clock() -> None:
    clock = _FakeClock()
    config = CircuitBreakerConfig(failure_threshold=1, cooldown_s=10.0)
    breaker = CircuitBreaker(config, clock=clock)
    breaker.record_failure("k")
    clock.advance(10.0)
    assert breaker.state_for("k") == CircuitState.HALF_OPEN
    breaker.record_failure("k")  # reopens
    assert breaker.state_for("k") == CircuitState.OPEN
    clock.advance(5.0)
    assert breaker.state_for("k") == CircuitState.OPEN  # cooldown restarted, not yet elapsed
    clock.advance(5.0)
    assert breaker.state_for("k") == CircuitState.HALF_OPEN


def test_independent_keys_do_not_affect_each_other() -> None:
    config = CircuitBreakerConfig(failure_threshold=1)
    breaker = CircuitBreaker(config)
    breaker.record_failure("model-a")
    assert breaker.state_for("model-a") == CircuitState.OPEN
    assert breaker.state_for("model-b") == CircuitState.CLOSED


def test_success_while_closed_resets_failure_streak() -> None:
    config = CircuitBreakerConfig(failure_threshold=3)
    breaker = CircuitBreaker(config)
    breaker.record_failure("k")
    breaker.record_failure("k")
    breaker.record_success("k")
    breaker.record_failure("k")
    breaker.record_failure("k")
    assert breaker.state_for("k") == CircuitState.CLOSED  # streak was reset, never hit 3 in a row


def test_snapshot_reflects_current_state() -> None:
    config = CircuitBreakerConfig(failure_threshold=2)
    breaker = CircuitBreaker(config)
    breaker.record_failure("k")
    snapshot = breaker.snapshot("k")
    assert snapshot.key == "k"
    assert snapshot.state == CircuitState.CLOSED
    assert snapshot.consecutive_failures == 1


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="failure_threshold"):
        CircuitBreakerConfig(failure_threshold=0)
    with pytest.raises(ValueError, match="cooldown_s"):
        CircuitBreakerConfig(cooldown_s=-1.0)
    with pytest.raises(ValueError, match="half_open_probe_budget"):
        CircuitBreakerConfig(half_open_probe_budget=0)
    with pytest.raises(ValueError, match="success_threshold_to_close"):
        CircuitBreakerConfig(success_threshold_to_close=0)


def test_only_reachable_states_ever_occur() -> None:
    """Property-style check: across a long randomized sequence of
    successes/failures/time-advances, the state is always one of the three
    valid enum members (trivially true given the type system, but this also
    exercises many transition sequences without ever raising)."""
    import random

    clock = _FakeClock()
    breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2, cooldown_s=5.0), clock=clock)
    rng = random.Random(0)
    for _ in range(500):
        action = rng.choice(["success", "failure", "advance"])
        if action == "success":
            breaker.record_success("k")
        elif action == "failure":
            breaker.record_failure("k")
        else:
            clock.advance(rng.uniform(0, 10))
        assert breaker.state_for("k") in {
            CircuitState.CLOSED,
            CircuitState.OPEN,
            CircuitState.HALF_OPEN,
        }
