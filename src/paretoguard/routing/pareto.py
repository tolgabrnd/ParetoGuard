"""Reusable multi-objective Pareto utilities.

Deliberately does not collapse (success, cost, latency) into one invented
scalar "best model" score — see `paretoguard.routing.pareto_router` for how
a router picks *among* the Pareto frontier this module computes, under an
explicit, named objective and constraints, rather than a fabricated weighting.
"""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectiveVector:
    """One candidate's position in objective space. `predicted_success` may
    be measured (real benchmark history) or simulated (synthetic model
    profile) — callers are responsible for not mixing the two without
    labeling, per `CandidateProfile.simulated`."""

    predicted_success: float
    expected_cost_usd: float
    expected_latency_ms: float


def dominates(a: ObjectiveVector, b: ObjectiveVector) -> bool:
    """True if `a` Pareto-dominates `b`: at least as good as `b` on every
    objective (higher success, lower cost, lower latency are "better") and
    strictly better on at least one."""
    at_least_as_good = (
        a.predicted_success >= b.predicted_success
        and a.expected_cost_usd <= b.expected_cost_usd
        and a.expected_latency_ms <= b.expected_latency_ms
    )
    strictly_better = (
        a.predicted_success > b.predicted_success
        or a.expected_cost_usd < b.expected_cost_usd
        or a.expected_latency_ms < b.expected_latency_ms
    )
    return at_least_as_good and strictly_better


def pareto_frontier(
    objectives: Mapping[str, ObjectiveVector],
) -> tuple[set[str], dict[str, list[str]]]:
    """Partitions candidates into the non-dominated frontier and the
    dominated set. Returns `(frontier_names, {dominated_name: [names of
    every candidate that dominates it]})` — every dominated candidate is
    attributed to its dominator(s) so a routing explanation can say exactly
    why it was passed over, not just that it was.

    O(n^2) pairwise comparison: fine for the small (single-digit to low
    dozens) candidate sets a routing decision considers; not intended for
    bulk offline analysis over thousands of points.
    """
    names = list(objectives)
    dominated_by: dict[str, list[str]] = {name: [] for name in names}
    for a_name in names:
        for b_name in names:
            if a_name == b_name:
                continue
            if dominates(objectives[a_name], objectives[b_name]):
                dominated_by[b_name].append(a_name)

    frontier = {name for name, dominators in dominated_by.items() if not dominators}
    dominated = {name: dominators for name, dominators in dominated_by.items() if dominators}
    return frontier, dominated
