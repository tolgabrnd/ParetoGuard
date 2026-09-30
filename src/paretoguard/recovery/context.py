"""Shared recovery types: the typed boundary `RecoveryPolicy` consumes and
produces. Split out from `recovery.policy` itself purely so
`recovery.fallback`'s `select_fallback` can type its own signature against
`RecoveryContext` without an import cycle (`policy` calls into `fallback`,
so `fallback` can't import the context type back from `policy`).

`RecoveryContext` is an immutable snapshot the caller (Commit 28's control
plane) builds fresh for each recovery decision — `RecoveryPolicy` never
reaches into `HealthTracker`/`CircuitBreaker` itself, keeping those modules
decoupled from the policy that consumes their state (see
`paretoguard.recovery`'s package docstring).
"""

from dataclasses import dataclass, field
from enum import StrEnum

from paretoguard.core.features import TaskFeatures
from paretoguard.core.models import FailureCategory, RoutingConstraints
from paretoguard.recovery.circuit_breaker import CircuitSnapshot
from paretoguard.routing.types import candidate_key
from paretoguard.telemetry.health import ModelHealth


class RecoveryAction(StrEnum):
    """Every decision a `RecoveryPolicy` can return — no magic strings."""

    RETRY_SAME = "retry_same"
    FALLBACK_MODEL = "fallback_model"
    FALLBACK_PROVIDER = "fallback_provider"
    ESCALATE = "escalate"
    PROBE = "probe"
    ABSTAIN = "abstain"
    FAIL = "fail"


@dataclass(frozen=True)
class AttemptRecord:
    """One prior attempt already made in the current recovery chain."""

    provider: str
    model: str
    failure_category: FailureCategory | None
    succeeded: bool


@dataclass(frozen=True)
class RecoveryContext:
    """Everything a `RecoveryPolicy` needs to decide what happens next,
    after the current attempt has already failed (or, for the very first
    decision in a chain, is about to be attempted with `failure_category`
    unset)."""

    current_provider: str | None
    current_model: str | None
    failure_category: FailureCategory | None
    task_features: TaskFeatures
    """The same routing-time-safe features used everywhere else in this
    repo (`paretoguard.core.features.extract_task_features`) — needed so
    fallback selection can check a candidate's real eligibility (e.g.
    context window) against the actual task, not a placeholder."""
    attempted: tuple[AttemptRecord, ...] = ()
    remaining_call_budget: int | None = None
    remaining_usd_budget: float | None = None
    remaining_latency_budget_ms: float | None = None
    health_snapshots: dict[str, ModelHealth] = field(default_factory=dict)
    """Keyed by `candidate_key(provider, model)` — see `routing.types`."""
    latency_drift: dict[str, float] = field(default_factory=dict)
    """Keyed by `candidate_key(provider, model)` — `HealthTracker
    .latency_drift_ratio` per candidate, a separate snapshot from
    `health_snapshots` because it isn't part of `ModelHealth` itself (see
    `recovery.health`'s module docstring for how it's used)."""
    circuit_snapshots: dict[str, CircuitSnapshot] = field(default_factory=dict)
    constraints: RoutingConstraints = field(default_factory=RoutingConstraints)

    def current_key(self) -> str | None:
        if self.current_provider is None or self.current_model is None:
            return None
        return candidate_key(self.current_provider, self.current_model)

    def attempts_on(self, key: str | None) -> int:
        if key is None:
            return 0
        return sum(
            1 for attempt in self.attempted if candidate_key(attempt.provider, attempt.model) == key
        )


@dataclass(frozen=True)
class RecoveryDecision:
    action: RecoveryAction
    target_provider: str | None = None
    target_model: str | None = None
    reason: str = ""
