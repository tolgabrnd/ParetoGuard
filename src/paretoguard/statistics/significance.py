"""Statistical significance tests, chosen by data structure, not applied by
default.

**Binary paired vs. independent**: when baseline and candidate were
evaluated on the *same* task IDs (the normal case for two recovery configs
run over the identical suite), outcomes are paired, not independent — the
correct test is an exact McNemar test (a binomial test on the *discordant*
pairs only), not Fisher's exact test on a naive 2x2 contingency table.
Fisher's exact test assumes independent samples and discards the pairing
information entirely, which typically under-powers a comparison that is
actually about "did this specific task's outcome change" (see the Phase F
spec: "do not automatically use an independent-sample test if results are
naturally paired by task"). Fisher's exact test is provided here for the
genuinely-independent case (e.g. two different task populations) — this
module never silently substitutes one for the other; `statistics.comparison`
inspects the task IDs and picks the correct one, recording the choice in
the returned result.

**Continuous paired vs. independent**: Wilcoxon signed-rank for paired
continuous measurements (same tasks, e.g. cost or latency per task under
two configs); Mann-Whitney U only for genuinely independent samples. Both
are rank-based/non-parametric — this repo's cost and latency distributions
are typically right-skewed (a handful of slow/expensive outliers), so a
paired/independent t-test's normality assumption is not assumed without
checking, per the spec's "choose based on the data structure" instruction.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from scipy import stats


@dataclass(frozen=True)
class TestResult:
    test_name: str
    statistic: float | None
    p_value: float
    paired: bool
    n: int


def mcnemar_exact_test(baseline: Sequence[bool], candidate: Sequence[bool]) -> TestResult:
    """Exact McNemar test on paired binary outcomes — `baseline[i]` and
    `candidate[i]` must be the same task's outcome under each config.
    Concordant pairs (both succeeded or both failed) carry no information
    about whether the treatment changed the outcome and are correctly
    excluded, not double-counted; only the discordant pairs (`n10`: baseline
    succeeded, candidate failed; `n01`: the reverse) are tested."""
    if len(baseline) != len(candidate):
        raise ValueError("baseline and candidate must be the same length (paired by task)")
    n10 = sum(1 for b, c in zip(baseline, candidate, strict=True) if b and not c)
    n01 = sum(1 for b, c in zip(baseline, candidate, strict=True) if not b and c)
    discordant = n10 + n01
    if discordant == 0:
        # No pair disagreed — nothing to test; the two configs never
        # differed on any task. Not a "significant" result, not "no
        # result" either: a plain p=1.0 (no evidence of any difference).
        return TestResult("mcnemar_exact", statistic=0.0, p_value=1.0, paired=True, n=len(baseline))
    result = stats.binomtest(min(n10, n01), discordant, p=0.5, alternative="two-sided")
    return TestResult(
        "mcnemar_exact",
        statistic=float(discordant),
        p_value=float(result.pvalue),
        paired=True,
        n=len(baseline),
    )


def fisher_exact_test(baseline: Sequence[bool], candidate: Sequence[bool]) -> TestResult:
    """Fisher's exact test on an unpaired 2x2 contingency table. Use only
    when baseline and candidate are genuinely independent samples (not the
    same task IDs evaluated twice) — see this module's docstring."""
    a = sum(baseline)
    b = len(baseline) - a
    c = sum(candidate)
    d = len(candidate) - c
    odds_ratio_value, p_value = stats.fisher_exact([[a, b], [c, d]])
    return TestResult(
        "fisher_exact",
        statistic=float(odds_ratio_value),
        p_value=float(p_value),
        paired=False,
        n=len(baseline) + len(candidate),
    )


def wilcoxon_signed_rank_test(baseline: Sequence[float], candidate: Sequence[float]) -> TestResult:
    """Paired, non-parametric test for continuous measurements on the same
    tasks. `scipy.stats.wilcoxon`'s default `zero_method="wilcox"` drops
    zero-difference pairs before ranking; if every pair is tied, there is
    nothing to rank and this returns p=1.0 directly rather than letting
    scipy raise."""
    if len(baseline) != len(candidate):
        raise ValueError("baseline and candidate must be the same length (paired by task)")
    if all(c == b for b, c in zip(baseline, candidate, strict=True)):
        return TestResult(
            "wilcoxon_signed_rank", statistic=0.0, p_value=1.0, paired=True, n=len(baseline)
        )
    statistic, p_value = stats.wilcoxon(baseline, candidate)
    return TestResult(
        "wilcoxon_signed_rank",
        statistic=float(statistic),
        p_value=float(p_value),
        paired=True,
        n=len(baseline),
    )


def mann_whitney_u_test(baseline: Sequence[float], candidate: Sequence[float]) -> TestResult:
    """Independent-samples, non-parametric test for continuous
    measurements. Use only when baseline and candidate are genuinely
    independent (not paired by task)."""
    statistic, p_value = stats.mannwhitneyu(baseline, candidate, alternative="two-sided")
    return TestResult(
        "mann_whitney_u",
        statistic=float(statistic),
        p_value=float(p_value),
        paired=False,
        n=len(baseline) + len(candidate),
    )


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down multiple-comparisons correction. Returns
    adjusted p-values in the *same order* as `p_values` (not sorted) — the
    family of hypotheses being corrected is exactly the sequence passed in;
    callers must ensure that family is the intended one (e.g. every
    pairwise comparison within one flagship benchmark, not an unrelated
    mix of analyses — see the Phase F spec: "do not apply correction
    mechanically to unrelated analyses")."""
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        adjusted_p = min(1.0, (n - rank) * p_values[idx])
        running_max = max(running_max, adjusted_p)
        adjusted[idx] = running_max
    return adjusted
