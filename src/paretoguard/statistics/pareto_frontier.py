"""Reporting-facing Pareto frontier analysis.

Reuses `routing.pareto`'s dominance/frontier algorithm rather than
reimplementing it — a candidate that Pareto-dominates another for routing
purposes is dominated the same way when reporting on measured results after
the fact; the only difference is where the objective vectors come from
(a router's predictions vs. a benchmark's measured outcomes).

Never declares one configuration universally "best" (see `routing.pareto`'s
own module docstring and the Phase F spec: "do not declare one universally
best; if a point is recommended, state the explicit constraint/objective
that selects it").
"""

from dataclasses import dataclass

from paretoguard.routing.pareto import ObjectiveVector, pareto_frontier


@dataclass(frozen=True)
class ParetoAnalysis:
    frontier: list[str]
    """Names on the non-dominated frontier, in the order they were given."""
    dominated: dict[str, list[str]]
    """`{name: [names of every point that dominates it]}` for every
    dominated point — an explanation, not just a verdict."""
    objectives: dict[str, ObjectiveVector]


def analyze_pareto_frontier(
    success_rates: dict[str, float],
    mean_costs_usd: dict[str, float],
    mean_latencies_ms: dict[str, float] | None = None,
) -> ParetoAnalysis:
    """Builds a `ParetoAnalysis` over success rate (higher better), mean
    cost (lower better), and optionally mean latency (lower better, `0.0`
    for every point — i.e. latency excluded from dominance — when not
    supplied, so a caller reporting only cost/success doesn't need to
    fabricate latency data to use this function)."""
    names = list(success_rates)
    if set(mean_costs_usd) != set(names):
        raise ValueError("success_rates and mean_costs_usd must cover the same names")
    if mean_latencies_ms is not None and set(mean_latencies_ms) != set(names):
        raise ValueError("mean_latencies_ms must cover the same names as success_rates")

    objectives = {
        name: ObjectiveVector(
            predicted_success=success_rates[name],
            expected_cost_usd=mean_costs_usd[name],
            expected_latency_ms=(mean_latencies_ms or {}).get(name, 0.0),
        )
        for name in names
    }
    frontier_set, dominated = pareto_frontier(objectives)
    return ParetoAnalysis(
        frontier=[n for n in names if n in frontier_set],
        dominated=dominated,
        objectives=objectives,
    )
