"""Unit tests for statistics.pareto_frontier and statistics.effect_size."""

import pytest

from paretoguard.statistics.effect_size import (
    continuous_delta,
    odds_ratio,
    risk_difference,
)
from paretoguard.statistics.pareto_frontier import analyze_pareto_frontier


def test_risk_difference_basic() -> None:
    d = risk_difference(0.5, 0.8)
    assert d.absolute_difference == pytest.approx(0.3)
    assert d.relative_difference == pytest.approx(0.6)


def test_risk_difference_undefined_relative_when_baseline_zero() -> None:
    d = risk_difference(0.0, 0.5)
    assert d.relative_difference is None


def test_odds_ratio_basic() -> None:
    r = odds_ratio(baseline_successes=10, baseline_n=20, candidate_successes=18, candidate_n=20)
    assert r.value is not None
    assert r.value > 1.0  # candidate has better odds


def test_odds_ratio_none_on_zero_cell() -> None:
    r = odds_ratio(baseline_successes=0, baseline_n=20, candidate_successes=10, candidate_n=20)
    assert r.value is None


def test_continuous_delta_basic() -> None:
    d = continuous_delta([10.0, 20.0], [5.0, 15.0])
    assert d.baseline_mean == 15.0
    assert d.candidate_mean == 10.0
    assert d.absolute_delta == -5.0
    assert d.relative_delta == pytest.approx(-1 / 3)


def test_continuous_delta_rejects_empty_sample() -> None:
    with pytest.raises(ValueError, match="empty"):
        continuous_delta([], [1.0])


def test_pareto_frontier_identifies_non_dominated_points() -> None:
    analysis = analyze_pareto_frontier(
        success_rates={"cheap": 0.8, "balanced": 0.9, "expensive": 0.95},
        mean_costs_usd={"cheap": 0.01, "balanced": 0.03, "expensive": 0.10},
    )
    assert set(analysis.frontier) == {"cheap", "balanced", "expensive"}
    assert analysis.dominated == {}


def test_pareto_frontier_excludes_a_strictly_dominated_point() -> None:
    analysis = analyze_pareto_frontier(
        success_rates={"good": 0.9, "strictly_worse": 0.8, "cheap": 0.7},
        mean_costs_usd={"good": 0.02, "strictly_worse": 0.03, "cheap": 0.01},
    )
    # strictly_worse: lower success AND higher cost than "good" -> dominated
    assert "strictly_worse" not in analysis.frontier
    assert "good" in analysis.dominated["strictly_worse"]


def test_pareto_frontier_never_declares_a_single_universal_best() -> None:
    """With genuine tradeoffs (no point dominates every other), the
    frontier must contain more than one member — this module must not
    collapse multiple valid tradeoffs into one "best" pick."""
    analysis = analyze_pareto_frontier(
        success_rates={"a": 0.99, "b": 0.5},
        mean_costs_usd={"a": 10.0, "b": 0.01},
    )
    assert len(analysis.frontier) == 2


def test_pareto_frontier_requires_matching_names() -> None:
    with pytest.raises(ValueError, match="same names"):
        analyze_pareto_frontier(success_rates={"a": 0.9}, mean_costs_usd={"b": 0.1})


def test_pareto_frontier_with_latency_included() -> None:
    analysis = analyze_pareto_frontier(
        success_rates={"fast": 0.9, "slow": 0.9},
        mean_costs_usd={"fast": 0.01, "slow": 0.01},
        mean_latencies_ms={"fast": 100.0, "slow": 500.0},
    )
    # identical success/cost, "slow" strictly worse on latency -> dominated
    assert "slow" not in analysis.frontier
    assert "fast" in analysis.frontier
