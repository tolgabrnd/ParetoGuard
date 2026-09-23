"""StaticRouter: always the same configured model."""

from paretoguard.core.models import RoutingDecision
from paretoguard.routing.protocol import Router
from paretoguard.routing.types import RoutingRequest


class StaticRouter(Router):
    """Always selects one fixed, configured (provider, model) pair.

    Deliberately ignores candidates, constraints, health, and profiles — it
    is the "no routing at all" baseline every other router is compared
    against, not an adaptive strategy. If the fixed model would fail a
    constraint, that's surfaced in the explanation, not silently hidden, but
    it is still selected: a caller that wants constraint enforcement should
    use `RuleRouter` or later, `paretoguard.routing.escalation` (Commit 24).
    """

    def __init__(self, *, provider: str, model: str) -> None:
        self.name = "static"
        self._provider = provider
        self._model = model

    def route(self, routing_request: RoutingRequest) -> RoutingDecision:
        candidate_names = {c.name for c in routing_request.candidates}
        warning = ""
        if candidate_names and self._model not in candidate_names:
            warning = f" (warning: '{self._model}' is not among the given candidates)"

        return RoutingDecision(
            selected_model=self._model,
            explanation=(
                f"StaticRouter: fixed configuration always selects "
                f"provider={self._provider!r} model={self._model!r}, "
                f"regardless of candidates/constraints/health{warning}."
            ),
        )
