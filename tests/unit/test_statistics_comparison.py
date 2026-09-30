"""Unit tests for statistics.comparison: auto-selected paired/independent
tests, effect sizes, and descriptive conclusions."""

import pytest

from paretoguard.statistics.comparison import (
    ComparisonConclusion,
    compare_binary_outcomes,
    compare_continuous_outcomes,
)


def _ids(n: int, prefix: str = "t") -> list[str]:
    return [f"{prefix}{i}" for i in range(n)]


def test_same_task_ids_in_same_order_are_treated_as_paired() -> None:
    ids = _ids(20)
    baseline = [False] * 10 + [True] * 10
    candidate = [True] * 20
    result = compare_binary_outcomes(ids, baseline, ids, candidate)
    assert result.test.paired
    assert result.test.test_name == "mcnemar_exact"


def test_different_task_ids_are_treated_as_independent() -> None:
    ids_a = _ids(20, "a")
    ids_b = _ids(20, "b")
    baseline = [False] * 10 + [True] * 10
    candidate = [True] * 20
    result = compare_binary_outcomes(ids_a, baseline, ids_b, candidate)
    assert not result.test.paired
    assert result.test.test_name == "fisher_exact"


def test_same_ids_different_order_are_not_treated_as_paired() -> None:
    """Pairing requires identical order, not just identical set — reordered
    task ids can't be assumed to still correspond position-by-position."""
    ids_a = _ids(20)
    ids_b = list(reversed(ids_a))
    baseline = [False] * 10 + [True] * 10
    candidate = [True] * 20
    result = compare_binary_outcomes(ids_a, baseline, ids_b, candidate)
    assert not result.test.paired


def test_below_min_sample_size_is_insufficient_evidence() -> None:
    ids = _ids(5)
    result = compare_binary_outcomes(ids, [True] * 5, ids, [False] * 5)
    assert result.conclusion == ComparisonConclusion.INSUFFICIENT_EVIDENCE


def test_large_clear_improvement_is_detected() -> None:
    ids = _ids(40)
    baseline = [False] * 20 + [True] * 20
    candidate = [True] * 40
    result = compare_binary_outcomes(ids, baseline, ids, candidate)
    assert result.conclusion == ComparisonConclusion.IMPROVEMENT_DETECTED
    assert result.absolute_delta == pytest.approx(0.5)


def test_large_clear_regression_is_detected() -> None:
    ids = _ids(40)
    baseline = [True] * 40
    candidate = [False] * 20 + [True] * 20
    result = compare_binary_outcomes(ids, baseline, ids, candidate)
    assert result.conclusion == ComparisonConclusion.REGRESSION_DETECTED


def test_identical_outcomes_give_no_clear_difference() -> None:
    ids = _ids(40)
    outcomes = [True] * 30 + [False] * 10
    result = compare_binary_outcomes(ids, outcomes, ids, outcomes)
    assert result.conclusion == ComparisonConclusion.NO_CLEAR_DIFFERENCE


def test_adjusted_p_value_can_flip_the_conclusion_to_no_clear_difference() -> None:
    ids = _ids(40)
    baseline = [False] * 8 + [True] * 32
    candidate = [True] * 40
    # A modest effect that would be "significant" raw but not after a harsh
    # multiple-comparisons correction.
    result_raw = compare_binary_outcomes(ids, baseline, ids, candidate)
    result_adjusted = compare_binary_outcomes(ids, baseline, ids, candidate, adjusted_p_value=0.99)
    assert result_raw.conclusion == ComparisonConclusion.IMPROVEMENT_DETECTED
    assert result_adjusted.conclusion == ComparisonConclusion.NO_CLEAR_DIFFERENCE


def test_continuous_comparison_defaults_to_lower_is_better() -> None:
    ids = _ids(20)
    baseline_latency = [100.0] * 20
    candidate_latency = [50.0] * 20
    result = compare_continuous_outcomes(ids, baseline_latency, ids, candidate_latency)
    assert result.conclusion == ComparisonConclusion.IMPROVEMENT_DETECTED
    assert result.absolute_delta < 0


def test_continuous_comparison_higher_is_better_flips_conclusion() -> None:
    ids = _ids(20)
    baseline_quality = [0.5] * 20
    candidate_quality = [0.9] * 20
    result = compare_continuous_outcomes(
        ids,
        baseline_quality,
        ids,
        candidate_quality,
        metric_name="quality_score",
        higher_is_better=True,
    )
    assert result.conclusion == ComparisonConclusion.IMPROVEMENT_DETECTED


def test_continuous_comparison_selects_paired_or_independent_test() -> None:
    ids = _ids(20)
    other_ids = _ids(20, "z")
    values_a = [10.0] * 20
    values_b = [5.0] * 20
    paired = compare_continuous_outcomes(ids, values_a, ids, values_b)
    independent = compare_continuous_outcomes(ids, values_a, other_ids, values_b)
    assert paired.test.test_name == "wilcoxon_signed_rank"
    assert independent.test.test_name == "mann_whitney_u"


def test_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="empty"):
        compare_binary_outcomes([], [], [], [True])
    with pytest.raises(ValueError, match="empty"):
        compare_continuous_outcomes([], [], [], [1.0])


def test_confidence_interval_is_computed_and_deterministic() -> None:
    ids = _ids(30)
    baseline = [False] * 15 + [True] * 15
    candidate = [True] * 30
    a = compare_binary_outcomes(ids, baseline, ids, candidate, seed=7)
    b = compare_binary_outcomes(ids, baseline, ids, candidate, seed=7)
    assert a.confidence_interval is not None
    assert a.confidence_interval == b.confidence_interval


def test_relative_delta_is_none_when_baseline_rate_is_zero() -> None:
    ids = _ids(20)
    baseline = [False] * 20
    candidate = [True] * 10 + [False] * 10
    result = compare_binary_outcomes(ids, baseline, ids, candidate)
    assert result.baseline_estimate == 0.0
    assert result.relative_delta is None
