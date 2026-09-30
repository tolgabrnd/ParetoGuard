"""A real CLOSED/OPEN/HALF_OPEN circuit-breaker state machine, one instance
tracking many independent keys (typically `candidate_key(provider, model)`).

Takes an injected clock (`Callable[[], float]`, defaulting to
`time.monotonic`) rather than reading wall-clock time internally, so tests
can advance time deterministically instead of calling `sleep()` — see
`tests/unit/test_circuit_breaker.py`.

**Relationship to `HealthTracker`** (`paretoguard.telemetry.health`): these
are deliberately separate abstractions. `HealthTracker` computes rolling/EMA
statistics from raw outcomes — it has no notion of a request being "allowed"
or "blocked". `CircuitBreaker` is a policy state machine that decides
request admission from discrete success/failure signals (typically fed by
`RecoveryPolicy`, which may itself consult `HealthTracker` to decide when to
report a failure — see `recovery.policy`'s module docstring). Do not read
this module as a replacement for `HealthTracker`, or vice versa.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True)
class CircuitBreakerConfig:
    failure_threshold: int = 5
    """Consecutive failures while CLOSED before transitioning to OPEN."""
    cooldown_s: float = 30.0
    """Time OPEN must elapse before allowing a HALF_OPEN probe."""
    half_open_probe_budget: int = 1
    """Requests allowed through while HALF_OPEN before further requests are
    blocked again (until a probe resolves the state)."""
    success_threshold_to_close: int = 1
    """Consecutive successful HALF_OPEN probes required to transition to CLOSED."""

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.cooldown_s < 0:
            raise ValueError("cooldown_s must be >= 0")
        if self.half_open_probe_budget < 1:
            raise ValueError("half_open_probe_budget must be >= 1")
        if self.success_threshold_to_close < 1:
            raise ValueError("success_threshold_to_close must be >= 1")


@dataclass
class _CircuitRecord:
    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    opened_at: float | None = None
    half_open_probes_used: int = 0


@dataclass(frozen=True)
class CircuitSnapshot:
    """Read-only view of one key's current state, for `RecoveryContext`."""

    key: str
    state: CircuitState
    consecutive_failures: int
    consecutive_successes: int


class CircuitBreaker:
    def __init__(
        self,
        config: CircuitBreakerConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config or CircuitBreakerConfig()
        self._clock = clock
        self._records: dict[str, _CircuitRecord] = {}

    def _record(self, key: str) -> _CircuitRecord:
        return self._records.setdefault(key, _CircuitRecord())

    def state_for(self, key: str) -> CircuitState:
        """Resolves an OPEN circuit whose cooldown has elapsed into
        HALF_OPEN as a side effect — reading state is what drives the
        OPEN -> HALF_OPEN transition, matching "after cooldown" semantics
        without a background timer."""
        record = self._record(key)
        cooldown_elapsed = (
            record.state == CircuitState.OPEN
            and record.opened_at is not None
            and self._clock() - record.opened_at >= self._config.cooldown_s
        )
        if cooldown_elapsed:
            record.state = CircuitState.HALF_OPEN
            record.half_open_probes_used = 0
        return record.state

    def allow_request(self, key: str) -> bool:
        """CLOSED always allows; OPEN always blocks; HALF_OPEN allows up to
        `half_open_probe_budget` requests through as probes, then blocks."""
        state = self.state_for(key)
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.OPEN:
            return False
        record = self._record(key)
        if record.half_open_probes_used < self._config.half_open_probe_budget:
            record.half_open_probes_used += 1
            return True
        return False

    def record_success(self, key: str) -> None:
        state = self.state_for(key)
        record = self._record(key)
        record.consecutive_failures = 0
        record.consecutive_successes += 1
        if state == CircuitState.HALF_OPEN and (
            record.consecutive_successes >= self._config.success_threshold_to_close
        ):
            record.state = CircuitState.CLOSED
            record.opened_at = None
            record.consecutive_successes = 0
            record.half_open_probes_used = 0

    def record_failure(self, key: str) -> None:
        state = self.state_for(key)
        record = self._record(key)
        record.consecutive_successes = 0
        record.consecutive_failures += 1
        if state == CircuitState.HALF_OPEN:
            record.state = CircuitState.OPEN
            record.opened_at = self._clock()
            record.half_open_probes_used = 0
        elif (
            state == CircuitState.CLOSED
            and record.consecutive_failures >= self._config.failure_threshold
        ):
            record.state = CircuitState.OPEN
            record.opened_at = self._clock()

    def snapshot(self, key: str) -> CircuitSnapshot:
        state = self.state_for(key)
        record = self._record(key)
        return CircuitSnapshot(
            key=key,
            state=state,
            consecutive_failures=record.consecutive_failures,
            consecutive_successes=record.consecutive_successes,
        )

    def all_snapshots(self) -> dict[str, CircuitSnapshot]:
        return {key: self.snapshot(key) for key in self._records}
