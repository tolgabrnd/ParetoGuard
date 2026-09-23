"""RoundRobinRouter: cycles through eligible candidates in a fixed order.

Infrastructure baseline only — it exists to give load-spreading/AB-testing
benchmarks something to compare against, not because rotating candidates is
an intelligent strategy. Do not describe this router as adaptive or as
knowing anything about task success, cost, or reliability: it doesn't.
"""

from itertools import count

from paretoguard.core.models import RoutingDecision
from paretoguard.routing.protocol import Router
from paretoguard.routing.types import NoEligibleCandidateError, RoutingRequest, filter_eligible


class RoundRobinRouter(Router):
    """Rotates through the eligible candidate set on every call, in the
    order `RoutingRequest.candidates` were given. Eligibility (context
    window, structured-output/tool support, explicit exclusions) is still
    enforced — round-robin never selects a candidate that can't serve the
    request — but candidates are not otherwise scored or preferred.
    """

    def __init__(self) -> None:
        self.name = "round_robin"
        self._counter = count()

    def route(self, routing_request: RoutingRequest) -> RoutingDecision:
        eligible, excluded = filter_eligible(routing_request)
        if not eligible:
            raise NoEligibleCandidateError(
                "RoundRobinRouter: no candidate satisfies hard eligibility "
                f"constraints; excluded={excluded}"
            )

        index = next(self._counter) % len(eligible)
        selected = eligible[index]

        return RoutingDecision(
            selected_model=selected.name,
            candidate_scores={c.name: 0.0 for c in eligible},
            explanation=(
                f"RoundRobinRouter (infrastructure baseline, not intelligent "
                f"routing): rotated to '{selected.name}' (position {index} of "
                f"{len(eligible)} eligible candidates)."
            ),
            fallback_order=[c.name for c in eligible if c.name != selected.name],
            excluded_candidates=excluded,
        )
