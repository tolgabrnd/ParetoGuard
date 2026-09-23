"""Deterministic degradation/recovery simulation comparing StaticRouter,
ParetoRouter, and ReliabilityAwareRouter.

SIMULATION ONLY: `PRIMARY`'s timeout probability follows a hand-authored
schedule (1% baseline, 25% during a synthetic degraded window, back to 1%
after a synthetic recovery point), driven by a seeded RNG. None of these
numbers are measurements from any real provider — see
paretoguard.routing.reliability_simulation's module docstring.
"""

from paretoguard.core.models import ModelSpec
from paretoguard.routing import (
    CandidateProfile,
    ParetoObjective,
    ParetoRouter,
    ReliabilityAwareRouter,
    StaticRouter,
    candidate_key,
    run_degradation_recovery_simulation,
    two_model_degradation_schedule,
)

PRIMARY = "primary-sim"
STABLE = "stable-sim"
DEGRADED_START = 100
RECOVERED_START = 200
TOTAL_STEPS = 300

_CANDIDATES = [
    ModelSpec(name=PRIMARY, provider="mock", context_window=100_000),
    ModelSpec(name=STABLE, provider="mock", context_window=100_000),
]
# SIMULATION: hand-picked initial profiles, equal cost/latency so any router
# preference shift is attributable to the reliability signal alone, not a
# confounded cost/latency difference.
_INITIAL_PROFILES = {
    candidate_key("mock", PRIMARY): CandidateProfile(
        predicted_success=0.99, mean_cost_usd=0.01, mean_latency_ms=100.0, simulated=True
    ),
    candidate_key("mock", STABLE): CandidateProfile(
        predicted_success=0.98, mean_cost_usd=0.01, mean_latency_ms=100.0, simulated=True
    ),
}
_SCHEDULE = two_model_degradation_schedule(
    primary_model=PRIMARY,
    stable_model=STABLE,
    degraded_start=DEGRADED_START,
    recovered_start=RECOVERED_START,
)


def _run(router):
    return run_degradation_recovery_simulation(
        router,
        candidates=_CANDIDATES,
        schedule=_SCHEDULE,
        profiles_by_model=_INITIAL_PROFILES,
        primary_model=PRIMARY,
        degraded_start=DEGRADED_START,
        recovered_start=RECOVERED_START,
        total_steps=TOTAL_STEPS,
        seed=42,
    )


def test_static_router_never_reroutes_and_absorbs_the_full_degradation() -> None:
    summary = _run(StaticRouter(provider="mock", model=PRIMARY))
    assert summary.reroute_count == 0
    assert summary.time_to_detect_degradation is None
    assert summary.label == "SIMULATION"

    degraded_window = [s for s in summary.steps if DEGRADED_START <= s.t < RECOVERED_START]
    healthy_window = [s for s in summary.steps if s.t < DEGRADED_START]
    degraded_success = sum(s.succeeded for s in degraded_window) / len(degraded_window)
    healthy_success = sum(s.succeeded for s in healthy_window) / len(healthy_window)
    assert degraded_success < healthy_success  # no rerouting, so degradation shows up directly


def test_reliability_router_detects_degradation_and_recovers() -> None:
    summary = _run(ReliabilityAwareRouter())
    assert summary.reroute_count > 0
    assert summary.time_to_detect_degradation is not None
    assert summary.time_to_detect_degradation < 30  # detects well before the window ends
    assert summary.time_to_recover is not None
    assert summary.unnecessary_reroute_rate is not None
    assert summary.unnecessary_reroute_rate < 0.5  # mostly reroutes for real cause, not noise


def test_pareto_router_maximize_success_also_reacts_to_degradation() -> None:
    summary = _run(ParetoRouter(ParetoObjective.MAXIMIZE_SUCCESS))
    assert summary.reroute_count > 0
    assert summary.time_to_detect_degradation is not None


def test_reactive_routers_outperform_static_during_degradation_window() -> None:
    static_summary = _run(StaticRouter(provider="mock", model=PRIMARY))
    reliability_summary = _run(ReliabilityAwareRouter())

    def degraded_window_success(summary):
        window = [s for s in summary.steps if DEGRADED_START <= s.t < RECOVERED_START]
        return sum(s.succeeded for s in window) / len(window)

    assert degraded_window_success(reliability_summary) > degraded_window_success(static_summary)


def test_reliability_router_does_not_reroute_excessively_outside_degraded_window() -> None:
    """Both models are healthy for t < 100 and t >= 200 — a well-behaved
    reliability router shouldn't be constantly flip-flopping in those windows."""
    summary = _run(ReliabilityAwareRouter())
    healthy_window_steps = [
        s for s in summary.steps if s.t < DEGRADED_START or s.t >= RECOVERED_START
    ]
    switches_in_healthy_window = sum(
        1
        for i in range(1, len(healthy_window_steps))
        if healthy_window_steps[i].selected_model != healthy_window_steps[i - 1].selected_model
    )
    # A handful of switches from noisy early EMA convergence is expected; it
    # should not be anywhere near constant flip-flopping.
    assert switches_in_healthy_window < len(healthy_window_steps) * 0.2
