"""Unit tests for statistics.significance: McNemar, Fisher, Wilcoxon,
Mann-Whitney, and Holm-Bonferroni correction."""

import pytest

from paretoguard.statistics.significance import (
    fisher_exact_test,
    holm_bonferroni,
    mann_whitney_u_test,
    mcnemar_exact_test,
    wilcoxon_signed_rank_test,
)


def test_mcnemar_requires_equal_length_paired_sequences() -> None:
    with pytest.raises(ValueError, match="same length"):
        mcnemar_exact_test([True, False], [True])


def test_mcnemar_no_discordant_pairs_gives_p_one() -> None:
    result = mcnemar_exact_test([True, False, True], [True, False, True])
    assert result.p_value == 1.0
    assert result.paired


def test_mcnemar_detects_a_strong_paired_improvement() -> None:
    baseline = [False] * 18 + [True] * 2
    candidate = [True] * 18 + [True] * 2  # every baseline failure now succeeds
    result = mcnemar_exact_test(baseline, candidate)
    assert result.p_value < 0.05


def test_mcnemar_ignores_concordant_pairs() -> None:
    """Two configs that agree on every task (all success or all failure)
    give no discordant pairs regardless of how many tasks there are."""
    baseline = [True] * 5 + [False] * 5
    candidate = [True] * 5 + [False] * 5
    result = mcnemar_exact_test(baseline, candidate)
    assert result.p_value == 1.0


def test_fisher_exact_detects_a_strong_independent_difference() -> None:
    baseline = [True] * 2 + [False] * 18
    candidate = [True] * 18 + [False] * 2
    result = fisher_exact_test(baseline, candidate)
    assert result.p_value < 0.01
    assert not result.paired


def test_wilcoxon_requires_equal_length_paired_sequences() -> None:
    with pytest.raises(ValueError, match="same length"):
        wilcoxon_signed_rank_test([1.0, 2.0], [1.0])


def test_wilcoxon_all_tied_gives_p_one() -> None:
    result = wilcoxon_signed_rank_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert result.p_value == 1.0


def test_wilcoxon_detects_a_consistent_paired_shift() -> None:
    baseline = [10.0, 12.0, 11.0, 13.0, 9.0, 14.0, 10.5, 11.5, 12.5, 13.5]
    candidate = [b - 2.0 for b in baseline]  # every task strictly improves
    result = wilcoxon_signed_rank_test(baseline, candidate)
    assert result.p_value < 0.05


def test_mann_whitney_detects_an_independent_shift() -> None:
    baseline = [10.0] * 15
    candidate = [5.0] * 15
    result = mann_whitney_u_test(baseline, candidate)
    assert result.p_value < 0.05
    assert not result.paired


def test_holm_bonferroni_matches_expected_step_down_values() -> None:
    # Standard textbook example: p = [0.01, 0.02, 0.03, 0.04], n=4
    # sorted ascending, multipliers 4,3,2,1, then running max.
    adjusted = holm_bonferroni([0.01, 0.02, 0.03, 0.04])
    assert adjusted == pytest.approx([0.04, 0.06, 0.06, 0.06])


def test_holm_bonferroni_never_exceeds_one() -> None:
    adjusted = holm_bonferroni([0.5, 0.6, 0.9])
    assert all(p <= 1.0 for p in adjusted)


def test_holm_bonferroni_preserves_input_order() -> None:
    # Largest p-value first in the input; output must stay aligned to input
    # order (not sorted): p=0.01 (index 1) gets multiplier 2 -> 0.02;
    # p=0.04 (index 0) gets multiplier 1 -> 0.04, and the step-down running
    # max leaves it at 0.04 since 0.04 > 0.02.
    adjusted = holm_bonferroni([0.04, 0.01])
    assert adjusted == pytest.approx([0.04, 0.02])


def test_holm_bonferroni_empty_input() -> None:
    assert holm_bonferroni([]) == []
