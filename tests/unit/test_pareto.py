"""Unit tests for the reusable Pareto utilities (SIMULATION-only fixtures).

Every ObjectiveVector below is a hand-picked synthetic value for testing
domination logic, not a claim about any real provider or model.
"""

from paretoguard.routing.pareto import ObjectiveVector, dominates, pareto_frontier


def test_dominates_requires_at_least_as_good_on_every_objective() -> None:
    better = ObjectiveVector(predicted_success=0.9, expected_cost_usd=0.01, expected_latency_ms=100)
    worse = ObjectiveVector(predicted_success=0.5, expected_cost_usd=0.05, expected_latency_ms=500)
    assert dominates(better, worse)
    assert not dominates(worse, better)


def test_dominates_is_false_for_identical_vectors() -> None:
    a = ObjectiveVector(predicted_success=0.8, expected_cost_usd=0.02, expected_latency_ms=200)
    b = ObjectiveVector(predicted_success=0.8, expected_cost_usd=0.02, expected_latency_ms=200)
    assert not dominates(a, b)
    assert not dominates(b, a)


def test_dominates_is_false_when_mixed_tradeoff() -> None:
    cheap_slow = ObjectiveVector(
        predicted_success=0.7, expected_cost_usd=0.01, expected_latency_ms=1000
    )
    expensive_fast = ObjectiveVector(
        predicted_success=0.7, expected_cost_usd=0.05, expected_latency_ms=100
    )
    assert not dominates(cheap_slow, expensive_fast)
    assert not dominates(expensive_fast, cheap_slow)


def test_pareto_frontier_separates_dominated_from_non_dominated() -> None:
    objectives = {
        # SIMULATION: synthetic profiles, not real provider numbers.
        "strictly-worse": ObjectiveVector(0.5, 0.05, 500),
        "dominator": ObjectiveVector(0.9, 0.01, 100),
        "tradeoff-cheap": ObjectiveVector(0.6, 0.005, 800),
    }
    frontier, dominated = pareto_frontier(objectives)
    assert frontier == {"dominator", "tradeoff-cheap"}
    assert dominated == {"strictly-worse": ["dominator"]}


def test_pareto_frontier_with_no_domination_returns_everyone() -> None:
    objectives = {
        "a": ObjectiveVector(0.9, 0.05, 100),
        "b": ObjectiveVector(0.5, 0.01, 100),
        "c": ObjectiveVector(0.5, 0.05, 10),
    }
    frontier, dominated = pareto_frontier(objectives)
    assert frontier == {"a", "b", "c"}
    assert dominated == {}


def test_pareto_frontier_attributes_multiple_dominators() -> None:
    objectives = {
        "worst": ObjectiveVector(0.1, 0.10, 1000),
        "good-a": ObjectiveVector(0.8, 0.02, 200),
        "good-b": ObjectiveVector(0.9, 0.03, 150),
    }
    frontier, dominated = pareto_frontier(objectives)
    assert frontier == {"good-a", "good-b"}
    assert sorted(dominated["worst"]) == ["good-a", "good-b"]
