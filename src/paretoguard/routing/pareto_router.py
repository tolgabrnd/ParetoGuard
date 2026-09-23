"""ParetoRouter: constrained multi-objective routing over the Pareto frontier.

Two constrained-objective forms, chosen at construction via `ParetoObjective`:

  MINIMIZE_COST:    minimize expected cost
                     subject to predicted_success >= constraints.min_predicted_success (Q)
                                 expected_latency_ms <= constraints.max_latency_ms (L)

  MAXIMIZE_SUCCESS:  maximize predicted success
                     subject to expected_cost_usd <= constraints.max_cost_usd (C)

`RoutingConstraints` (already the shared "subject to" type used by every
router) carries Q/L/C; the objective enum only says which axis is primary.
Selection always happens among the non-dominated Pareto frontier — a
dominated candidate is never selected, regardless of the primary objective.
"""

from enum import StrEnum

from paretoguard.core.models import RoutingDecision
from paretoguard.routing.pareto import ObjectiveVector, pareto_frontier
from paretoguard.routing.protocol import Router
from paretoguard.routing.types import (
    NoEligibleCandidateError,
    RoutingRequest,
    apply_soft_constraints,
    candidate_key,
    filter_eligible,
)


class ParetoObjective(StrEnum):
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_SUCCESS = "maximize_success"


class ParetoRouter(Router):
    """Selects the Pareto-optimal candidate for one named, constrained
    objective. Never reduces (success, cost, latency) to a single invented
    score: `RoutingDecision.candidate_scores` reports only the primary
    objective's own value (cost for MINIMIZE_COST, success for
    MAXIMIZE_SUCCESS) — never a fabricated cross-objective blend.
    """

    def __init__(self, objective: ParetoObjective) -> None:
        self.name = "pareto"
        self._objective = objective

    def route(self, routing_request: RoutingRequest) -> RoutingDecision:
        eligible, excluded = filter_eligible(routing_request)
        eligible, soft_excluded = apply_soft_constraints(routing_request, eligible)
        excluded.update(soft_excluded)

        objectives: dict[str, ObjectiveVector] = {}
        for candidate in eligible:
            profile = routing_request.profiles.get(
                candidate_key(candidate.provider, candidate.name)
            )
            if (
                profile is None
                or profile.predicted_success is None
                or profile.mean_cost_usd is None
                or profile.mean_latency_ms is None
            ):
                excluded[candidate.name] = (
                    "insufficient profile data for Pareto optimization "
                    "(need predicted_success, mean_cost_usd, and mean_latency_ms)"
                )
                continue
            objectives[candidate.name] = ObjectiveVector(
                predicted_success=profile.predicted_success,
                expected_cost_usd=profile.mean_cost_usd,
                expected_latency_ms=profile.mean_latency_ms,
            )

        if not objectives:
            raise NoEligibleCandidateError(
                f"ParetoRouter: no candidate has a complete profile to optimize "
                f"over; excluded={excluded}"
            )

        frontier, dominated = pareto_frontier(objectives)
        for name, dominators in dominated.items():
            excluded[name] = (
                f"Pareto-dominated by {sorted(dominators)} (worse or equal on "
                f"every objective, strictly worse on at least one)"
            )

        ranked = self._rank_frontier(frontier, objectives)
        selected_name = ranked[0]
        selected_objective = objectives[selected_name]

        return RoutingDecision(
            selected_model=selected_name,
            candidate_scores=self._primary_objective_scores(objectives),
            predicted_success=selected_objective.predicted_success,
            expected_cost_usd=selected_objective.expected_cost_usd,
            expected_latency_ms=selected_objective.expected_latency_ms,
            explanation=self._explain(
                selected_name,
                eligible_count=len(objectives),
                frontier=frontier,
                dominated=dominated,
            ),
            fallback_order=ranked[1:],
            excluded_candidates=excluded,
        )

    def _rank_frontier(
        self, frontier: set[str], objectives: dict[str, ObjectiveVector]
    ) -> list[str]:
        """Deterministic best-to-worst ordering among frontier candidates
        only — a dominated candidate never appears here, so a caller falling
        back through `fallback_order` never lands on one."""
        if self._objective is ParetoObjective.MINIMIZE_COST:
            return sorted(
                frontier,
                key=lambda name: (
                    objectives[name].expected_cost_usd,
                    -objectives[name].predicted_success,
                    name,
                ),
            )
        return sorted(
            frontier,
            key=lambda name: (
                -objectives[name].predicted_success,
                objectives[name].expected_cost_usd,
                name,
            ),
        )

    def _primary_objective_scores(self, objectives: dict[str, ObjectiveVector]) -> dict[str, float]:
        if self._objective is ParetoObjective.MINIMIZE_COST:
            return {name: obj.expected_cost_usd for name, obj in objectives.items()}
        return {name: obj.predicted_success for name, obj in objectives.items()}

    def _explain(
        self,
        selected_name: str,
        *,
        eligible_count: int,
        frontier: set[str],
        dominated: dict[str, list[str]],
    ) -> str:
        objective_desc = (
            "minimize expected cost subject to predicted_success/latency constraints"
            if self._objective is ParetoObjective.MINIMIZE_COST
            else "maximize predicted success subject to expected_cost constraint"
        )
        dominated_desc = (
            f"{len(dominated)} Pareto-dominated candidate(s) excluded: {sorted(dominated)}"
            if dominated
            else "no candidates were Pareto-dominated"
        )
        return (
            f"ParetoRouter ({objective_desc}): selected '{selected_name}' from the "
            f"Pareto frontier ({len(frontier)} of {eligible_count} candidate(s) with "
            f"complete profiles); {dominated_desc}."
        )
