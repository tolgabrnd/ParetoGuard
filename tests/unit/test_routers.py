"""Unit tests for StaticRouter, RoundRobinRouter, and RuleRouter."""

import pytest

from paretoguard.core.models import InferenceRequest, Message, ModelSpec, Role, RoutingConstraints
from paretoguard.routing import (
    CandidateProfile,
    NoEligibleCandidateError,
    RoundRobinRouter,
    RoutingRequest,
    RuleRouter,
    StaticRouter,
    candidate_key,
    extract_task_features,
)


def _request(**kwargs: object) -> InferenceRequest:
    defaults: dict[str, object] = {
        "provider": "mock",
        "model": "mock-strong",
        "messages": [Message(role=Role.USER, content="hi there friend")],
    }
    defaults.update(kwargs)
    return InferenceRequest(**defaults)  # type: ignore[arg-type]


def _model(name: str, *, provider: str = "mock", context_window: int = 1000) -> ModelSpec:
    return ModelSpec(name=name, provider=provider, context_window=context_window)


def _routing_request(
    candidates: list[ModelSpec],
    *,
    constraints: RoutingConstraints | None = None,
    profiles: dict[str, CandidateProfile] | None = None,
    task_family: str | None = None,
) -> RoutingRequest:
    request = _request(metadata={"task_family": task_family} if task_family else {})
    return RoutingRequest(
        request=request,
        features=extract_task_features(request),
        candidates=candidates,
        constraints=constraints or RoutingConstraints(),
        profiles=profiles or {},
    )


# -- StaticRouter -----------------------------------------------------------


def test_static_router_always_selects_configured_model() -> None:
    router = StaticRouter(provider="mock", model="mock-strong")
    decision = router.route(_routing_request([_model("mock-cheap")]))
    assert decision.selected_model == "mock-strong"


def test_static_router_is_deterministic_across_calls() -> None:
    router = StaticRouter(provider="mock", model="mock-strong")
    routing_request = _routing_request([_model("mock-strong"), _model("mock-cheap")])
    decisions = [router.route(routing_request).selected_model for _ in range(5)]
    assert decisions == ["mock-strong"] * 5


def test_static_router_warns_when_model_not_among_candidates() -> None:
    router = StaticRouter(provider="mock", model="missing-model")
    decision = router.route(_routing_request([_model("mock-strong")]))
    assert decision.selected_model == "missing-model"
    assert "warning" in decision.explanation.lower()


# -- RoundRobinRouter ---------------------------------------------------------


def test_round_robin_cycles_through_eligible_candidates_in_order() -> None:
    router = RoundRobinRouter()
    candidates = [_model("a"), _model("b"), _model("c")]
    routing_request = _routing_request(candidates)
    selected = [router.route(routing_request).selected_model for _ in range(6)]
    assert selected == ["a", "b", "c", "a", "b", "c"]


def test_round_robin_skips_ineligible_candidates() -> None:
    router = RoundRobinRouter()
    candidates = [_model("small", context_window=1), _model("big", context_window=1000)]
    routing_request = _routing_request(candidates)
    for _ in range(3):
        decision = router.route(routing_request)
        assert decision.selected_model == "big"
        assert "small" in decision.excluded_candidates


def test_round_robin_raises_when_no_candidate_is_eligible() -> None:
    router = RoundRobinRouter()
    routing_request = _routing_request([_model("small", context_window=1)])
    with pytest.raises(NoEligibleCandidateError):
        router.route(routing_request)


# -- RuleRouter ---------------------------------------------------------------


def test_rule_router_prefers_task_family_scoped_success() -> None:
    router = RuleRouter()
    candidates = [_model("a"), _model("b")]
    profiles = {
        candidate_key("mock", "a"): CandidateProfile(predicted_success=0.4, task_family="numeric"),
        candidate_key("mock", "b"): CandidateProfile(predicted_success=0.9, task_family="numeric"),
    }
    routing_request = _routing_request(candidates, profiles=profiles, task_family="numeric")
    decision = router.route(routing_request)
    assert decision.selected_model == "b"
    assert decision.predicted_success == 0.9


def test_rule_router_ignores_profile_for_a_different_task_family() -> None:
    router = RuleRouter()
    candidates = [_model("a"), _model("b")]
    profiles = {
        candidate_key("mock", "a"): CandidateProfile(predicted_success=0.9, task_family="tool_use"),
        candidate_key("mock", "b"): CandidateProfile(predicted_success=0.4, task_family="tool_use"),
    }
    # Neither profile matches "numeric", so both candidates fall back to the
    # deterministic default-order rule and the first configured wins.
    routing_request = _routing_request(candidates, profiles=profiles, task_family="numeric")
    decision = router.route(routing_request)
    assert decision.selected_model == "a"


def test_rule_router_excludes_candidate_over_max_cost() -> None:
    router = RuleRouter()
    candidates = [_model("expensive"), _model("cheap")]
    profiles = {
        candidate_key("mock", "expensive"): CandidateProfile(
            predicted_success=0.99, mean_cost_usd=5.0
        ),
        candidate_key("mock", "cheap"): CandidateProfile(predicted_success=0.5, mean_cost_usd=0.1),
    }
    constraints = RoutingConstraints(max_cost_usd=1.0)
    routing_request = _routing_request(candidates, constraints=constraints, profiles=profiles)
    decision = router.route(routing_request)
    assert decision.selected_model == "cheap"
    assert "expensive" in decision.excluded_candidates


def test_rule_router_excludes_candidate_over_max_latency() -> None:
    router = RuleRouter()
    candidates = [_model("slow"), _model("fast")]
    profiles = {
        candidate_key("mock", "slow"): CandidateProfile(mean_latency_ms=5000.0),
        candidate_key("mock", "fast"): CandidateProfile(mean_latency_ms=100.0),
    }
    constraints = RoutingConstraints(max_latency_ms=1000.0)
    routing_request = _routing_request(candidates, constraints=constraints, profiles=profiles)
    decision = router.route(routing_request)
    assert decision.selected_model == "fast"
    assert "slow" in decision.excluded_candidates


def test_rule_router_excludes_candidate_below_min_predicted_success() -> None:
    router = RuleRouter()
    candidates = [_model("weak"), _model("strong")]
    profiles = {
        candidate_key("mock", "weak"): CandidateProfile(predicted_success=0.2),
        candidate_key("mock", "strong"): CandidateProfile(predicted_success=0.9),
    }
    constraints = RoutingConstraints(min_predicted_success=0.5)
    routing_request = _routing_request(candidates, constraints=constraints, profiles=profiles)
    decision = router.route(routing_request)
    assert decision.selected_model == "strong"
    assert "weak" in decision.excluded_candidates


def test_rule_router_raises_when_every_candidate_is_excluded() -> None:
    router = RuleRouter()
    candidates = [_model("only", context_window=1)]
    routing_request = _routing_request(candidates)
    with pytest.raises(NoEligibleCandidateError):
        router.route(routing_request)


def test_rule_router_falls_back_to_configured_order_without_profiles() -> None:
    router = RuleRouter()
    candidates = [_model("first"), _model("second")]
    routing_request = _routing_request(candidates)
    decision = router.route(routing_request)
    assert decision.selected_model == "first"
    assert "no measured signal" in decision.explanation


def test_rule_router_is_deterministic() -> None:
    router = RuleRouter()
    candidates = [_model("a"), _model("b"), _model("c")]
    routing_request = _routing_request(candidates)
    decisions = {router.route(routing_request).selected_model for _ in range(5)}
    assert decisions == {"a"}
