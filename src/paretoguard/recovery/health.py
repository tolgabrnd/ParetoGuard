"""Health-aware fallback ordering: an advisory signal `select_fallback`
consults when choosing *which* eligible candidate to try next, separate from
(and not a replacement for) `CircuitBreaker`'s hard OPEN/HALF_OPEN/CLOSED
gating.

**Not `routing.reliability.ReliabilityAwareRouter`, deliberately.** That
router ranks *every* candidate for the *initial* routing decision, using one
signal (EMA success-rate tiering) with periodic re-probing. This module only
influences *fallback* ordering within an already-failing recovery chain, and
classifies on multiple independent, explicitly-documented signals (EMA
success rate, EMA timeout rate, latency drift, a consecutive-failure streak)
combined through simple, named, inspectable rules — never blended into one
opaque composite score. `ClosedLoopExecutor` already owns one
`HealthTracker` shared by both routing and recovery; this module reads the
same `ModelHealth` snapshots `ReliabilityAwareRouter` does, it just asks a
different, narrower question ("should this specific fallback candidate be
demoted or skipped right now?") with a different, restrained answer (three
tiers, not a full ranking).

**Health is advisory, not authoritative.** A `HEALTHY`/`UNKNOWN` candidate is
never demoted. A `DEGRADED` candidate is demoted (tried after every
healthy/unknown one) but never skipped. An `UNHEALTHY` candidate is skipped
in the primary selection pass — but never permanently: if every eligible
candidate is unhealthy, `select_fallback` still falls back to trying the
least-bad one rather than reporting no candidate at all (matching this
repo's "never silently drop a case" discipline, and mirroring
`ReliabilityAwareRouter`'s own "no case ever silently dropped" behavior for
the same reason).
"""

from dataclasses import dataclass
from enum import StrEnum

from paretoguard.telemetry.health import ModelHealth


class RecoveryHealthStatus(StrEnum):
    """A candidate's health classification for fallback ordering purposes
    only — not the same enum as `routing.reliability.HealthStatus` (kept
    separate deliberately; see this module's docstring)."""

    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True)
class RecoveryHealthPolicy:
    """Explicit, named, independently-tunable thresholds — no single
    "health score". Each threshold maps to exactly one rule in
    `classify_recovery_health`, documented there."""

    min_requests_for_health: int = 5
    """Cold-start guard: a candidate with fewer observed requests than this
    is `UNKNOWN` — neither demoted nor favored — regardless of what its
    (statistically unreliable) rolling numbers happen to say yet."""
    healthy_ema_success_at_or_above: float = 0.8
    """EMA success rate at or above this, with no other rule tripped: `HEALTHY`."""
    unhealthy_ema_success_below: float = 0.5
    """EMA success rate below this: `UNHEALTHY` outright, regardless of other signals."""
    unhealthy_consecutive_failures_at_or_above: int = 3
    """A streak of at least this many consecutive failures: `UNHEALTHY`
    outright — an EMA-independent, immediate "this candidate is failing
    right now" signal that catches a fresh hot streak before enough calls
    have accumulated to move a slower-moving EMA."""
    degraded_ema_timeout_rate_above: float = 0.2
    """EMA timeout rate above this: at least `DEGRADED` (never on its own
    enough for `UNHEALTHY` — a timeout-prone but otherwise-succeeding
    candidate is a worse choice, not an unusable one)."""
    degraded_latency_drift_above: float = 2.0
    """`HealthTracker.latency_drift_ratio` above this (fast EMA running at
    more than this multiple of the slow baseline EMA): at least `DEGRADED`."""

    def __post_init__(self) -> None:
        if self.min_requests_for_health < 1:
            raise ValueError("min_requests_for_health must be >= 1")
        if (
            not 0.0
            <= self.unhealthy_ema_success_below
            <= self.healthy_ema_success_at_or_above
            <= 1.0
        ):
            raise ValueError(
                "require 0 <= unhealthy_ema_success_below <= healthy_ema_success_at_or_above <= 1"
            )
        if self.unhealthy_consecutive_failures_at_or_above < 1:
            raise ValueError("unhealthy_consecutive_failures_at_or_above must be >= 1")
        if not 0.0 <= self.degraded_ema_timeout_rate_above <= 1.0:
            raise ValueError("degraded_ema_timeout_rate_above must be in [0, 1]")
        if self.degraded_latency_drift_above <= 1.0:
            raise ValueError("degraded_latency_drift_above must be > 1.0 (1.0 = no drift)")


def classify_recovery_health(
    health: ModelHealth | None, latency_drift: float | None, policy: RecoveryHealthPolicy
) -> RecoveryHealthStatus:
    """Pure function: no mutation, no I/O. Rule order matters and is
    deliberately a cascade, not a weighted sum — each rule is independently
    readable and testable:

    1. Cold start (too few requests, or no health recorded at all) -> UNKNOWN.
    2. A live consecutive-failure streak at/above threshold -> UNHEALTHY.
    3. EMA success rate below the unhealthy floor -> UNHEALTHY.
    4. EMA success rate at/above the healthy floor AND no timeout/latency
       flag tripped -> HEALTHY.
    5. Otherwise -> DEGRADED (either a mid-range success rate, or a healthy
       success rate undercut by a timeout-rate or latency-drift flag).
    """
    if health is None or health.request_count < policy.min_requests_for_health:
        return RecoveryHealthStatus.UNKNOWN

    if health.consecutive_failures >= policy.unhealthy_consecutive_failures_at_or_above:
        return RecoveryHealthStatus.UNHEALTHY

    rate = health.ema_success_rate
    if rate is not None and rate < policy.unhealthy_ema_success_below:
        return RecoveryHealthStatus.UNHEALTHY

    timeout_flagged = (
        health.ema_timeout_rate is not None
        and health.ema_timeout_rate > policy.degraded_ema_timeout_rate_above
    )
    latency_flagged = (
        latency_drift is not None and latency_drift > policy.degraded_latency_drift_above
    )

    if rate is not None and rate >= policy.healthy_ema_success_at_or_above:
        if not timeout_flagged and not latency_flagged:
            return RecoveryHealthStatus.HEALTHY
        return RecoveryHealthStatus.DEGRADED

    return RecoveryHealthStatus.DEGRADED
