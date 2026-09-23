"""Unit tests for EscalationRouter: confidence-aware escalation and
explicit abstention."""

import pytest

from paretoguard.core.models import InferenceRequest, Message, ModelSpec, Role, RoutingDecision
from paretoguard.routing.escalation import (
    EscalationPolicy,
    EscalationRouter,
    NoConfidentRouteError,
)
from paretoguard.routing.protocol import Router
from paretoguard.routing.static import StaticRouter
from paretoguard.routing.types import (
    NoEligibleCandidateError,
    RoutingRequest,
    extract_task_features,
)


class _ScriptedRouter(Router):
    """Returns pre-scripted decisions in order, one per `route()` call,
    optionally raising NoEligibleCandidateError instead."""

    def __init__(self, name: str, decisions: list[RoutingDecision | Exception]) -> None:
        self.name = name
        self._decisions = list(decisions)
        self._index = 0

    def route(self, routing_request: RoutingRequest) -> RoutingDecision:
        outcome = self._decisions[min(self._index, len(self._decisions) - 1)]
        self._index += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _request() -> InferenceRequest:
    return InferenceRequest(
        provider="mock", model="mock-strong", messages=[Message(role=Role.USER, content="hi")]
    )


def _routing_request() -> RoutingRequest:
    request = _request()
    candidates = [ModelSpec(name="a", provider="mock", context_window=1000)]
    return RoutingRequest(
        request=request, features=extract_task_features(request), candidates=candidates
    )


def _decision(model: str, predicted_success: float | None) -> RoutingDecision:
    return RoutingDecision(
        selected_model=model, predicted_success=predicted_success, explanation=f"picked {model}"
    )


def test_routes_directly_when_confidence_meets_threshold() -> None:
    inner = _ScriptedRouter("inner", [_decision("model-a", 0.9)])
    router = EscalationRouter(inner, policy=EscalationPolicy(confidence_threshold=0.7))
    decision = router.route(_routing_request())
    assert decision.selected_model == "model-a"


def test_escalates_when_confidence_below_threshold() -> None:
    inner = _ScriptedRouter(
        "inner",
        [
            _decision("weak-model", 0.4),  # initial pick, low confidence
            _decision("strong-model", 0.9),  # escalation attempt
        ],
    )
    router = EscalationRouter(
        inner, policy=EscalationPolicy(confidence_threshold=0.7, max_escalations=1)
    )
    decision = router.route(_routing_request())
    assert decision.selected_model == "strong-model"
    assert decision.confidence == pytest.approx(0.9)
    assert "EscalationRouter" in decision.explanation


def test_stays_with_best_escalation_attempt_if_still_under_threshold() -> None:
    inner = _ScriptedRouter(
        "inner",
        [
            _decision("weak", 0.2),
            _decision("medium", 0.5),  # improves but still below 0.7; no fallback configured
        ],
    )
    router = EscalationRouter(
        inner, policy=EscalationPolicy(confidence_threshold=0.7, max_escalations=1)
    )
    with pytest.raises(NoConfidentRouteError):
        router.route(_routing_request())


def test_falls_back_to_safe_router_when_escalation_insufficient() -> None:
    inner = _ScriptedRouter("inner", [_decision("weak", 0.2), _decision("still-weak", 0.3)])
    fallback = StaticRouter(provider="mock", model="safe-model")
    router = EscalationRouter(
        inner,
        policy=EscalationPolicy(confidence_threshold=0.9, max_escalations=1),
        fallback_router=fallback,
    )
    decision = router.route(_routing_request())
    assert decision.selected_model == "safe-model"
    assert "fallback" in decision.explanation.lower()


def test_abstains_when_no_fallback_and_escalation_insufficient() -> None:
    inner = _ScriptedRouter("inner", [_decision("weak", 0.1), _decision("still-weak", 0.2)])
    router = EscalationRouter(
        inner, policy=EscalationPolicy(confidence_threshold=0.9, max_escalations=1)
    )
    with pytest.raises(NoConfidentRouteError):
        router.route(_routing_request())


def test_stops_escalating_when_inner_router_raises_no_eligible_candidate() -> None:
    inner = _ScriptedRouter(
        "inner", [_decision("weak", 0.1), NoEligibleCandidateError("nothing left")]
    )
    router = EscalationRouter(
        inner, policy=EscalationPolicy(confidence_threshold=0.9, max_escalations=3)
    )
    with pytest.raises(NoConfidentRouteError):
        router.route(_routing_request())


def test_records_escalation_history() -> None:
    inner = _ScriptedRouter("inner", [_decision("weak", 0.3), _decision("strong", 0.85)])
    router = EscalationRouter(
        inner, policy=EscalationPolicy(confidence_threshold=0.7, max_escalations=1)
    )
    routing_request = _routing_request()
    router.route(routing_request)

    record = router.get_record(routing_request.request.request_id)
    assert record is not None
    assert record.initial_selection == "weak"
    assert record.initial_confidence == pytest.approx(0.3)
    assert record.final_selection == "strong"
    assert record.escalation_reason is not None
    assert record.used_fallback is False
    assert record.succeeded is None  # not yet known


def test_record_outcome_attaches_actual_results() -> None:
    inner = _ScriptedRouter("inner", [_decision("model-a", 0.9)])
    router = EscalationRouter(inner, policy=EscalationPolicy(confidence_threshold=0.7))
    routing_request = _routing_request()
    router.route(routing_request)

    request_id = routing_request.request.request_id
    router.record_outcome(request_id, succeeded=True, actual_cost_usd=0.02)
    record = router.get_record(request_id)
    assert record is not None
    assert record.succeeded is True
    assert record.actual_cost_usd == pytest.approx(0.02)


def test_record_outcome_rejects_unknown_request_id() -> None:
    inner = _ScriptedRouter("inner", [_decision("model-a", 0.9)])
    router = EscalationRouter(inner)
    with pytest.raises(KeyError):
        router.record_outcome(_request().request_id, succeeded=True)


def test_no_escalation_needed_records_none_reason() -> None:
    inner = _ScriptedRouter("inner", [_decision("model-a", 0.9)])
    router = EscalationRouter(inner, policy=EscalationPolicy(confidence_threshold=0.7))
    routing_request = _routing_request()
    router.route(routing_request)
    record = router.get_record(routing_request.request.request_id)
    assert record is not None
    assert record.escalation_reason is None
    assert record.used_fallback is False


def test_never_returns_a_decision_below_threshold_without_fallback_or_abstention() -> None:
    """The core constraint-preservation guarantee: EscalationRouter must
    never silently hand back a low-confidence decision."""
    inner = _ScriptedRouter("inner", [_decision("weak", 0.1), _decision("still-weak", 0.15)])
    router = EscalationRouter(
        inner, policy=EscalationPolicy(confidence_threshold=0.99, max_escalations=2)
    )
    with pytest.raises(NoConfidentRouteError):
        router.route(_routing_request())
