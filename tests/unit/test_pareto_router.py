"""Unit tests for ParetoRouter. All CandidateProfile values are hand-picked
SIMULATION fixtures for testing constrained-objective selection logic, never
real provider costs/latencies/success rates.
"""

import pytest

from paretoguard.core.models import InferenceRequest, Message, ModelSpec, Role, RoutingConstraints
from paretoguard.routing import (
    CandidateProfile,
    NoEligibleCandidateError,
    ParetoObjective,
    ParetoRouter,
    RoutingRequest,
    candidate_key,
    extract_task_features,
)


def _request() -> InferenceRequest:
    return InferenceRequest(
        provider="mock", model="mock-strong", messages=[Message(role=Role.USER, content="hi")]
    )


def _model(name: str, *, provider: str = "mock") -> ModelSpec:
    return ModelSpec(name=name, provider=provider, context_window=100_000)


def _routing_request(
    candidates: list[ModelSpec],
    profiles: dict[str, CandidateProfile],
    *,
    constraints: RoutingConstraints | None = None,
) -> RoutingRequest:
    request = _request()
    return RoutingRequest(
        request=request,
        features=extract_task_features(request),
        candidates=candidates,
        constraints=constraints or RoutingConstraints(),
        profiles=profiles,
    )


# Three synthetic (SIMULATION) profiles with genuine trade-offs: no candidate
# dominates every other, so intelligent routing has something to choose from.
_PROFILE_A = CandidateProfile(predicted_success=0.95, mean_cost_usd=0.05, mean_latency_ms=800)
_PROFILE_B = CandidateProfile(predicted_success=0.80, mean_cost_usd=0.01, mean_latency_ms=200)
_PROFILE_C = CandidateProfile(predicted_success=0.60, mean_cost_usd=0.001, mean_latency_ms=100)


def _three_way_setup() -> tuple[list[ModelSpec], dict[str, CandidateProfile]]:
    candidates = [_model("profile-a"), _model("profile-b"), _model("profile-c")]
    profiles = {
        candidate_key("mock", "profile-a"): _PROFILE_A,
        candidate_key("mock", "profile-b"): _PROFILE_B,
        candidate_key("mock", "profile-c"): _PROFILE_C,
    }
    return candidates, profiles


def test_maximize_success_picks_highest_success_on_frontier() -> None:
    router = ParetoRouter(ParetoObjective.MAXIMIZE_SUCCESS)
    candidates, profiles = _three_way_setup()
    decision = router.route(_routing_request(candidates, profiles))
    assert decision.selected_model == "profile-a"
    assert decision.predicted_success == 0.95


def test_maximize_success_subject_to_cost_excludes_expensive_candidate() -> None:
    router = ParetoRouter(ParetoObjective.MAXIMIZE_SUCCESS)
    candidates, profiles = _three_way_setup()
    constraints = RoutingConstraints(max_cost_usd=0.02)
    decision = router.route(_routing_request(candidates, profiles, constraints=constraints))
    # profile-a excluded by the cost constraint, so the best remaining is profile-b.
    assert decision.selected_model == "profile-b"
    assert "profile-a" in decision.excluded_candidates


def test_minimize_cost_picks_cheapest_on_frontier() -> None:
    router = ParetoRouter(ParetoObjective.MINIMIZE_COST)
    candidates, profiles = _three_way_setup()
    decision = router.route(_routing_request(candidates, profiles))
    assert decision.selected_model == "profile-c"
    assert decision.expected_cost_usd == 0.001


def test_minimize_cost_subject_to_min_success_excludes_low_success_candidate() -> None:
    router = ParetoRouter(ParetoObjective.MINIMIZE_COST)
    candidates, profiles = _three_way_setup()
    constraints = RoutingConstraints(min_predicted_success=0.7)
    decision = router.route(_routing_request(candidates, profiles, constraints=constraints))
    # profile-c (success 0.60) excluded; cheapest of the rest is profile-b.
    assert decision.selected_model == "profile-b"
    assert "profile-c" in decision.excluded_candidates


def test_minimize_cost_subject_to_max_latency_excludes_slow_candidate() -> None:
    router = ParetoRouter(ParetoObjective.MINIMIZE_COST)
    candidates, profiles = _three_way_setup()
    constraints = RoutingConstraints(max_latency_ms=500)
    decision = router.route(_routing_request(candidates, profiles, constraints=constraints))
    # profile-a (800ms) excluded; cheapest of the rest is profile-c.
    assert decision.selected_model == "profile-c"
    assert "profile-a" in decision.excluded_candidates


def test_dominated_candidate_is_never_selected_and_is_explained() -> None:
    router = ParetoRouter(ParetoObjective.MINIMIZE_COST)
    candidates = [_model("good"), _model("strictly-worse")]
    profiles = {
        candidate_key("mock", "good"): CandidateProfile(
            predicted_success=0.9, mean_cost_usd=0.01, mean_latency_ms=100
        ),
        candidate_key("mock", "strictly-worse"): CandidateProfile(
            predicted_success=0.5, mean_cost_usd=0.05, mean_latency_ms=500
        ),
    }
    decision = router.route(_routing_request(candidates, profiles))
    assert decision.selected_model == "good"
    assert "strictly-worse" in decision.excluded_candidates
    assert "dominated" in decision.excluded_candidates["strictly-worse"].lower()
    assert "dominated" in decision.explanation.lower()


def test_candidate_scores_report_only_the_primary_objective() -> None:
    """RoutingDecision.candidate_scores must never be a fabricated composite
    score across success/cost/latency — only the router's one named
    objective (see ParetoRouter's docstring)."""
    router = ParetoRouter(ParetoObjective.MAXIMIZE_SUCCESS)
    candidates, profiles = _three_way_setup()
    decision = router.route(_routing_request(candidates, profiles))
    assert decision.candidate_scores["profile-a"] == _PROFILE_A.predicted_success
    assert decision.candidate_scores["profile-b"] == _PROFILE_B.predicted_success
    assert decision.candidate_scores["profile-c"] == _PROFILE_C.predicted_success


def test_candidate_without_complete_profile_is_excluded_not_crashed() -> None:
    router = ParetoRouter(ParetoObjective.MAXIMIZE_SUCCESS)
    candidates = [_model("complete"), _model("partial")]
    profiles = {
        candidate_key("mock", "complete"): CandidateProfile(
            predicted_success=0.8, mean_cost_usd=0.01, mean_latency_ms=100
        ),
        candidate_key("mock", "partial"): CandidateProfile(
            predicted_success=0.9
        ),  # no cost/latency
    }
    decision = router.route(_routing_request(candidates, profiles))
    assert decision.selected_model == "complete"
    assert "partial" in decision.excluded_candidates


def test_raises_when_no_candidate_has_a_complete_profile() -> None:
    router = ParetoRouter(ParetoObjective.MAXIMIZE_SUCCESS)
    candidates = [_model("no-profile")]
    with pytest.raises(NoEligibleCandidateError):
        router.route(_routing_request(candidates, {}))


def test_fallback_order_never_contains_a_dominated_candidate() -> None:
    router = ParetoRouter(ParetoObjective.MAXIMIZE_SUCCESS)
    candidates, profiles = _three_way_setup()
    decision = router.route(_routing_request(candidates, profiles))
    # profile-a is selected; the fallback order should hold the rest of the
    # frontier only. All three profiles here trade off, so nothing is
    # actually dominated in this fixture — fallback_order should list both.
    assert set(decision.fallback_order) == {"profile-b", "profile-c"}


def test_router_is_deterministic() -> None:
    router = ParetoRouter(ParetoObjective.MINIMIZE_COST)
    candidates, profiles = _three_way_setup()
    routing_request = _routing_request(candidates, profiles)
    decisions = {router.route(routing_request).selected_model for _ in range(5)}
    assert decisions == {"profile-c"}
