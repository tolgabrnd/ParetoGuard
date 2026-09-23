"""A deterministic, seeded degradation/recovery simulation for comparing
routers' reliability behavior.

Every number this module produces is SIMULATION: a synthetic per-timestep
failure-probability schedule fed through a seeded RNG, not a measurement from
any real provider. `SimulationSummary` is labeled accordingly and nothing
here should be read as, or reported alongside, real benchmark results without
that label. See CLAUDE.md: "never fabricate benchmark results... or
reliability numbers" — this generates synthetic ones deliberately and says so.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from paretoguard.core.models import FailureCategory, InferenceRequest, Message, ModelSpec, Role
from paretoguard.routing.protocol import Router
from paretoguard.routing.types import (
    CandidateProfile,
    RoutingRequest,
    candidate_key,
    extract_task_features,
)
from paretoguard.telemetry.health import HealthTracker

TimeoutProbabilitySchedule = Callable[[str, int], float]
"""(model_name, timestep) -> probability in [0, 1] that this step times out."""


@dataclass(frozen=True)
class SimulationStep:
    t: int
    selected_model: str
    succeeded: bool
    cost_usd: float
    latency_ms: float


@dataclass(frozen=True)
class SimulationSummary:
    """SIMULATION-labeled results for one router's run through one scenario."""

    router_name: str
    total_steps: int
    task_success_rate: float
    reroute_count: int
    reroute_frequency: float
    time_to_detect_degradation: int | None
    """Steps between `degraded_start` and the first switch away from the
    degrading model, or None if the router never switched away."""
    time_to_recover: int | None
    """Steps between `recovered_start` and the first switch back to the
    (now-recovered) model, or None if it never switched back (e.g. because
    it never switched away in the first place, or stayed away)."""
    unnecessary_reroute_count: int
    """Reroutes away from the primary model while it was outside the
    degraded window (i.e. not justified by the simulated schedule)."""
    unnecessary_reroute_rate: float | None
    mean_cost_usd: float
    mean_latency_ms: float
    steps: list[SimulationStep] = field(repr=False)

    label: str = "SIMULATION"


def two_model_degradation_schedule(
    *,
    primary_model: str,
    stable_model: str,
    degraded_start: int,
    recovered_start: int,
    baseline_timeout_prob: float = 0.01,
    degraded_timeout_prob: float = 0.25,
    stable_timeout_prob: float = 0.02,
) -> TimeoutProbabilitySchedule:
    """The canonical scenario this module was built for: `primary_model`'s
    timeout probability is `baseline_timeout_prob` for t < degraded_start,
    `degraded_timeout_prob` for degraded_start <= t < recovered_start, and
    back to `baseline_timeout_prob` for t >= recovered_start.
    `stable_model` holds a constant, always-healthier `stable_timeout_prob`
    throughout — a legitimate fallback, not a strictly worse option, so a
    reactive router has a genuine reason to prefer it during degradation.
    """

    def schedule(model_name: str, t: int) -> float:
        if model_name == stable_model:
            return stable_timeout_prob
        if model_name != primary_model:
            return baseline_timeout_prob
        if degraded_start <= t < recovered_start:
            return degraded_timeout_prob
        return baseline_timeout_prob

    return schedule


def run_degradation_recovery_simulation(
    router: Router,
    *,
    candidates: list[ModelSpec],
    schedule: TimeoutProbabilitySchedule,
    profiles_by_model: dict[str, CandidateProfile],
    primary_model: str,
    degraded_start: int,
    recovered_start: int,
    total_steps: int = 300,
    seed: int = 0,
    min_requests_for_reactive_profile: int = 5,
) -> SimulationSummary:
    """Runs `router` through `total_steps` timesteps of the given schedule,
    updating a fresh `HealthTracker` online (as a real deployment would) and
    re-deriving each candidate's `CandidateProfile.predicted_success` from
    the tracker's rolling EMA once enough requests have been observed — so
    profile-driven routers (e.g. ParetoRouter) and health-driven routers
    (ReliabilityAwareRouter) both get a fair chance to react to the same
    underlying signal, one through profiles and one through health directly.
    """
    rng = random.Random(seed)
    tracker = HealthTracker(ema_alpha=0.3)
    provider_name = candidates[0].provider
    steps: list[SimulationStep] = []
    previous_selection: str | None = None
    reroute_count = 0
    unnecessary_reroute_count = 0
    time_to_detect: int | None = None
    time_to_recover: int | None = None
    switched_away_from_primary = False

    for t in range(total_steps):
        health_snapshot = {
            candidate_key(c.provider, c.name): tracker.get(c.provider, c.name) for c in candidates
        }
        live_profiles = dict(profiles_by_model)
        for c in candidates:
            key = candidate_key(c.provider, c.name)
            health = health_snapshot[key]
            if (
                health is not None
                and health.request_count >= min_requests_for_reactive_profile
                and health.ema_success_rate is not None
            ):
                base = live_profiles.get(key, CandidateProfile())
                live_profiles[key] = CandidateProfile(
                    predicted_success=health.ema_success_rate,
                    mean_cost_usd=base.mean_cost_usd,
                    mean_latency_ms=base.mean_latency_ms,
                    task_family=base.task_family,
                    simulated=True,
                )

        request = _request()
        routing_request = RoutingRequest(
            request=request,
            features=extract_task_features(request),
            candidates=candidates,
            health={k: v for k, v in health_snapshot.items() if v is not None},
            profiles=live_profiles,
        )
        decision = router.route(routing_request)
        selected = decision.selected_model

        if previous_selection is not None and selected != previous_selection:
            reroute_count += 1
            in_degraded_window = degraded_start <= t < recovered_start
            if selected != primary_model and not in_degraded_window:
                unnecessary_reroute_count += 1
            if selected != primary_model and in_degraded_window and not switched_away_from_primary:
                time_to_detect = t - degraded_start
                switched_away_from_primary = True
            if (
                selected == primary_model
                and switched_away_from_primary
                and t >= recovered_start
                and time_to_recover is None
            ):
                time_to_recover = t - recovered_start
        previous_selection = selected

        timeout_prob = schedule(selected, t)
        succeeded = rng.random() >= timeout_prob
        profile = live_profiles.get(candidate_key(provider_name, selected))
        cost = profile.mean_cost_usd if profile and profile.mean_cost_usd is not None else 0.0
        latency = (
            profile.mean_latency_ms if profile and profile.mean_latency_ms is not None else 0.0
        )

        tracker.record_outcome(
            provider_name,
            selected,
            succeeded=succeeded,
            latency_ms=latency,
            error_category=FailureCategory.TIMEOUT if not succeeded else None,
        )
        steps.append(
            SimulationStep(
                t=t, selected_model=selected, succeeded=succeeded, cost_usd=cost, latency_ms=latency
            )
        )

    success_rate = sum(s.succeeded for s in steps) / total_steps
    unnecessary_rate = unnecessary_reroute_count / reroute_count if reroute_count else None

    return SimulationSummary(
        router_name=router.name,
        total_steps=total_steps,
        task_success_rate=success_rate,
        reroute_count=reroute_count,
        reroute_frequency=reroute_count / total_steps,
        time_to_detect_degradation=time_to_detect,
        time_to_recover=time_to_recover,
        unnecessary_reroute_count=unnecessary_reroute_count,
        unnecessary_reroute_rate=unnecessary_rate,
        mean_cost_usd=sum(s.cost_usd for s in steps) / total_steps,
        mean_latency_ms=sum(s.latency_ms for s in steps) / total_steps,
        steps=steps,
    )


def _request() -> InferenceRequest:
    return InferenceRequest(
        provider="mock",
        model="mock-strong",
        messages=[Message(role=Role.USER, content="simulated reliability probe")],
    )
