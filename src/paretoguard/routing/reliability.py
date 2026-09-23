"""ReliabilityAwareRouter: deprioritizes candidates whose rolling health has
degraded, using `HealthTracker`'s decayed (EMA) signals rather than its
all-time cumulative counters — see `paretoguard.telemetry.health.ModelHealth`
for why the distinction matters (a cumulative average dilutes a recent spike
in failures into near-invisibility).
"""

from dataclasses import dataclass

from paretoguard.core.models import ModelSpec, RoutingDecision
from paretoguard.routing.protocol import Router
from paretoguard.routing.types import (
    NoEligibleCandidateError,
    RoutingRequest,
    apply_soft_constraints,
    candidate_key,
    filter_eligible,
)
from paretoguard.telemetry.health import ModelHealth


@dataclass(frozen=True)
class ReliabilityThresholds:
    """Boundaries for classifying a candidate's rolling health.

    `min_requests_for_health` is a cold-start guard: a candidate with fewer
    observed requests than this has its health treated as UNKNOWN (neither
    penalized nor favored) rather than judged on a handful of samples.
    """

    healthy_ema_success_at_or_above: float = 0.8
    degraded_ema_success_at_or_above: float = 0.5
    min_requests_for_health: int = 5
    probe_interval_calls: int = 15
    """A candidate stuck at DEGRADED/UNHEALTHY and therefore never selected
    would never have its health re-observed by `HealthTracker` — a pure
    exploit-only policy can't detect recovery it never looks for. Every
    `probe_interval_calls` routing calls while a candidate sits below
    HEALTHY and isn't otherwise selected, this router forces one probe call
    to it (a half-open circuit-breaker check), so recovery is still
    detectable without permanently sacrificing traffic to a truly bad model."""


class HealthStatus:
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


_STATUS_RANK = {
    HealthStatus.HEALTHY: 0,
    HealthStatus.UNKNOWN: 1,
    HealthStatus.DEGRADED: 2,
    HealthStatus.UNHEALTHY: 3,
}


class ReliabilityAwareRouter(Router):
    """Ranks eligible candidates primarily by rolling health status
    (healthy > unknown > degraded > unhealthy), then within a status tier by
    rolling success rate, falling back to profile-based predicted success as
    a tie-break when health data is absent or tied.

    A degraded/unhealthy candidate is never hard-excluded — it temporarily
    loses preference, but if every eligible candidate is unhealthy this
    router still returns its least-bad option (with that fact stated in the
    explanation) rather than raising. Total exclusion only happens for hard/
    soft constraint violations, same as every other router here.

    Stateful across calls (unlike the other routers here): it counts calls
    since each degraded/unhealthy candidate was last probed, to periodically
    re-test it (see `ReliabilityThresholds.probe_interval_calls`). A fresh
    instance starts this state at zero.
    """

    def __init__(self, thresholds: ReliabilityThresholds | None = None) -> None:
        self.name = "reliability"
        self._thresholds = thresholds or ReliabilityThresholds()
        self._calls_since_probe: dict[str, int] = {}

    def route(self, routing_request: RoutingRequest) -> RoutingDecision:
        eligible, excluded = filter_eligible(routing_request)
        eligible, soft_excluded = apply_soft_constraints(routing_request, eligible)
        excluded.update(soft_excluded)

        if not eligible:
            raise NoEligibleCandidateError(
                f"ReliabilityAwareRouter: no candidate satisfies constraints; excluded={excluded}"
            )

        statuses: dict[str, str] = {}
        scores: dict[str, float] = {}
        for candidate in eligible:
            key = candidate_key(candidate.provider, candidate.name)
            health = routing_request.health.get(key)
            status, score = self._classify(routing_request, key, health)
            statuses[candidate.name] = status
            scores[candidate.name] = score

        ranked = sorted(
            eligible,
            key=lambda c: (_STATUS_RANK[statuses[c.name]], -scores[c.name], c.name),
        )
        selected = ranked[0]
        probed_name = self._advance_probe_counters(eligible, statuses, top_choice=selected.name)
        if probed_name is not None:
            ranked = [c for c in ranked if c.name == probed_name] + [
                c for c in ranked if c.name != probed_name
            ]
            selected = ranked[0]

        selected_status = statuses[selected.name]
        degraded_or_worse = {
            name: status
            for name, status in statuses.items()
            if status in (HealthStatus.DEGRADED, HealthStatus.UNHEALTHY)
        }

        return RoutingDecision(
            selected_model=selected.name,
            candidate_scores=scores,
            predicted_success=(
                scores[selected.name] if selected_status != HealthStatus.UNKNOWN else None
            ),
            explanation=self._explain(
                selected.name, selected_status, statuses, degraded_or_worse, probed=probed_name
            ),
            fallback_order=[c.name for c in ranked[1:]],
            excluded_candidates=excluded,
        )

    def _advance_probe_counters(
        self, eligible: list[ModelSpec], statuses: dict[str, str], *, top_choice: str
    ) -> str | None:
        """Increments the probe counter for every degraded/unhealthy
        candidate not already selected; returns the name of the first one
        to cross the probe interval (reset to 0), or None if none did."""
        probed: str | None = None
        for candidate in eligible:
            name = candidate.name
            if statuses[name] not in (HealthStatus.DEGRADED, HealthStatus.UNHEALTHY):
                self._calls_since_probe.pop(name, None)
                continue
            if name == top_choice:
                self._calls_since_probe[name] = 0
                continue
            count = self._calls_since_probe.get(name, 0) + 1
            if count >= self._thresholds.probe_interval_calls and probed is None:
                self._calls_since_probe[name] = 0
                probed = name
            else:
                self._calls_since_probe[name] = count
        return probed

    def _classify(
        self, routing_request: RoutingRequest, key: str, health: ModelHealth | None
    ) -> tuple[str, float]:
        if health is None or health.request_count < self._thresholds.min_requests_for_health:
            profile = routing_request.profiles.get(key)
            fallback_score = (
                profile.predicted_success
                if profile is not None and profile.predicted_success is not None
                else 0.5
            )
            return HealthStatus.UNKNOWN, fallback_score

        rate = health.ema_success_rate if health.ema_success_rate is not None else 0.5
        if rate >= self._thresholds.healthy_ema_success_at_or_above:
            return HealthStatus.HEALTHY, rate
        if rate >= self._thresholds.degraded_ema_success_at_or_above:
            return HealthStatus.DEGRADED, rate
        return HealthStatus.UNHEALTHY, rate

    def _explain(
        self,
        selected_name: str,
        selected_status: str,
        statuses: dict[str, str],
        degraded_or_worse: dict[str, str],
        *,
        probed: str | None,
    ) -> str:
        degraded_note = (
            f"; temporarily deprioritized (not excluded): {degraded_or_worse}"
            if degraded_or_worse
            else ""
        )
        all_bad_note = (
            " (no healthy or unknown candidate was available — selected the least-bad option)"
            if selected_status == HealthStatus.UNHEALTHY
            else ""
        )
        probe_note = (
            f" (this is a periodic recovery probe of a {selected_status} candidate, not a "
            f"ranked choice — see ReliabilityThresholds.probe_interval_calls)"
            if probed == selected_name
            else ""
        )
        return (
            f"ReliabilityAwareRouter: selected '{selected_name}' "
            f"(rolling health status={selected_status}){all_bad_note}{probe_note}, "
            f"ranked by EMA-decayed rolling success rate among {len(statuses)} "
            f"eligible candidate(s){degraded_note}."
        )
