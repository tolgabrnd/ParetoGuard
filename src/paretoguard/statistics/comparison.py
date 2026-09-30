"""The typed experiment-comparison result, and the orchestration functions
that build one from raw paired/independent observations — auto-selecting
the statistically correct test based on data structure (see
`significance`'s module docstring). Conclusions are always descriptive
("improvement detected"), never causal — this module never claims a
recovery config *caused* a success-rate change, only that one was observed
(see the Phase F spec: "do not claim causality").
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from paretoguard.statistics.bootstrap import (
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_N_RESAMPLES,
    MIN_RELIABLE_N,
    BootstrapResult,
)
from paretoguard.statistics.effect_size import (
    ContinuousDelta,
    RiskDifference,
    continuous_delta,
    risk_difference,
)
from paretoguard.statistics.significance import (
    TestResult,
    fisher_exact_test,
    mann_whitney_u_test,
    mcnemar_exact_test,
    wilcoxon_signed_rank_test,
)

MIN_SAMPLE_SIZE_FOR_EVIDENCE = 10
"""Below this per-arm sample size, a comparison's conclusion is always
`INSUFFICIENT_EVIDENCE` regardless of the p-value — a "significant" result
from a handful of observations is not evidence this repo reports as such
(see CLAUDE.md: never overclaim)."""


class ComparisonConclusion(StrEnum):
    """Deliberately descriptive, never causal (see module docstring)."""

    IMPROVEMENT_DETECTED = "improvement_detected"
    REGRESSION_DETECTED = "regression_detected"
    NO_CLEAR_DIFFERENCE = "no_clear_difference"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class ExperimentComparison:
    """Baseline vs. candidate on one metric — the complete record the
    Phase F spec asks for: both estimates, both kinds of delta, a
    confidence interval, the test actually used (and why), raw and
    (optionally) multiple-comparison-adjusted p-values, an effect size, the
    sample size, and a descriptive conclusion."""

    metric_name: str
    baseline_run_id: str | None
    candidate_run_id: str | None
    baseline_estimate: float
    candidate_estimate: float
    absolute_delta: float
    relative_delta: float | None
    confidence_interval: BootstrapResult | None
    test: TestResult
    raw_p_value: float
    adjusted_p_value: float | None
    effect_size: RiskDifference | ContinuousDelta
    sample_size: int
    conclusion: ComparisonConclusion
    alpha: float
    """Significance threshold used for `conclusion`'s statistical check."""
    higher_is_better: bool


def _paired(baseline_task_ids: Sequence[str], candidate_task_ids: Sequence[str]) -> bool:
    """The only case this module treats as "paired": the identical set of
    task ids, in the identical order, evaluated under both configs."""
    return len(baseline_task_ids) == len(candidate_task_ids) and list(baseline_task_ids) == list(
        candidate_task_ids
    )


def _classify(
    *,
    n_baseline: int,
    n_candidate: int,
    p_value: float,
    adjusted_p_value: float | None,
    alpha: float,
    absolute_delta: float,
    higher_is_better: bool,
) -> ComparisonConclusion:
    if n_baseline < MIN_SAMPLE_SIZE_FOR_EVIDENCE or n_candidate < MIN_SAMPLE_SIZE_FOR_EVIDENCE:
        return ComparisonConclusion.INSUFFICIENT_EVIDENCE
    effective_p = adjusted_p_value if adjusted_p_value is not None else p_value
    if effective_p >= alpha or absolute_delta == 0:
        return ComparisonConclusion.NO_CLEAR_DIFFERENCE
    improved = (absolute_delta > 0) == higher_is_better
    return (
        ComparisonConclusion.IMPROVEMENT_DETECTED
        if improved
        else ComparisonConclusion.REGRESSION_DETECTED
    )


def _bootstrap_binary_delta_ci(
    baseline: Sequence[bool],
    candidate: Sequence[bool],
    *,
    paired: bool,
    seed: int,
    n_resamples: int,
    confidence_level: float,
) -> BootstrapResult:
    b = np.asarray([1.0 if x else 0.0 for x in baseline])
    c = np.asarray([1.0 if x else 0.0 for x in candidate])
    return _bootstrap_delta_ci(
        b, c, paired=paired, seed=seed, n_resamples=n_resamples, confidence_level=confidence_level
    )


def _bootstrap_delta_ci(
    b: "np.ndarray",
    c: "np.ndarray",
    *,
    paired: bool,
    seed: int,
    n_resamples: int,
    confidence_level: float,
) -> BootstrapResult:
    rng = np.random.default_rng(seed)
    n_b, n_c = len(b), len(c)
    deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        if paired:
            idx = rng.integers(0, n_b, size=n_b)
            deltas[i] = c[idx].mean() - b[idx].mean()
        else:
            idx_b = rng.integers(0, n_b, size=n_b)
            idx_c = rng.integers(0, n_c, size=n_c)
            deltas[i] = c[idx_c].mean() - b[idx_b].mean()
    point = float(c.mean() - b.mean())
    alpha = 1.0 - confidence_level
    lower = float(np.percentile(deltas, 100 * (alpha / 2)))
    upper = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
    return BootstrapResult(
        point_estimate=point,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        n=n_b + n_c,
        seed=seed,
        reliable=n_b >= MIN_RELIABLE_N and n_c >= MIN_RELIABLE_N,
    )


def compare_binary_outcomes(
    baseline_task_ids: Sequence[str],
    baseline_outcomes: Sequence[bool],
    candidate_task_ids: Sequence[str],
    candidate_outcomes: Sequence[bool],
    *,
    metric_name: str = "success_rate",
    baseline_run_id: str | None = None,
    candidate_run_id: str | None = None,
    alpha: float = 0.05,
    adjusted_p_value: float | None = None,
    seed: int = 0,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> ExperimentComparison:
    """Compares two binary-outcome (success/fail) samples, auto-selecting
    McNemar (paired, same task ids) or Fisher's exact (independent)."""
    if not baseline_outcomes or not candidate_outcomes:
        raise ValueError("cannot compare an empty outcome set")
    paired = _paired(baseline_task_ids, candidate_task_ids)

    if (
        len(baseline_outcomes) < MIN_SAMPLE_SIZE_FOR_EVIDENCE
        or len(candidate_outcomes) < MIN_SAMPLE_SIZE_FOR_EVIDENCE
    ):
        test = TestResult("insufficient_data", None, 1.0, paired, len(baseline_outcomes))
    elif paired:
        test = mcnemar_exact_test(baseline_outcomes, candidate_outcomes)
    else:
        test = fisher_exact_test(baseline_outcomes, candidate_outcomes)

    baseline_rate = sum(baseline_outcomes) / len(baseline_outcomes)
    candidate_rate = sum(candidate_outcomes) / len(candidate_outcomes)
    effect = risk_difference(baseline_rate, candidate_rate)

    ci = _bootstrap_binary_delta_ci(
        baseline_outcomes,
        candidate_outcomes,
        paired=paired,
        seed=seed,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
    )

    conclusion = _classify(
        n_baseline=len(baseline_outcomes),
        n_candidate=len(candidate_outcomes),
        p_value=test.p_value,
        adjusted_p_value=adjusted_p_value,
        alpha=alpha,
        absolute_delta=effect.absolute_difference,
        higher_is_better=True,
    )

    return ExperimentComparison(
        metric_name=metric_name,
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        baseline_estimate=baseline_rate,
        candidate_estimate=candidate_rate,
        absolute_delta=effect.absolute_difference,
        relative_delta=effect.relative_difference,
        confidence_interval=ci,
        test=test,
        raw_p_value=test.p_value,
        adjusted_p_value=adjusted_p_value,
        effect_size=effect,
        sample_size=len(baseline_outcomes) + len(candidate_outcomes),
        conclusion=conclusion,
        alpha=alpha,
        higher_is_better=True,
    )


def compare_continuous_outcomes(
    baseline_task_ids: Sequence[str],
    baseline_values: Sequence[float],
    candidate_task_ids: Sequence[str],
    candidate_values: Sequence[float],
    *,
    metric_name: str = "cost_usd",
    higher_is_better: bool = False,
    baseline_run_id: str | None = None,
    candidate_run_id: str | None = None,
    alpha: float = 0.05,
    adjusted_p_value: float | None = None,
    seed: int = 0,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> ExperimentComparison:
    """Compares two continuous samples (cost, latency), auto-selecting
    Wilcoxon signed-rank (paired) or Mann-Whitney U (independent).
    `higher_is_better=False` by default — lower cost/latency is normally
    the improvement; pass `True` for a metric where higher is better."""
    if not baseline_values or not candidate_values:
        raise ValueError("cannot compare an empty value set")
    paired = _paired(baseline_task_ids, candidate_task_ids)

    if (
        len(baseline_values) < MIN_SAMPLE_SIZE_FOR_EVIDENCE
        or len(candidate_values) < MIN_SAMPLE_SIZE_FOR_EVIDENCE
    ):
        test = TestResult("insufficient_data", None, 1.0, paired, len(baseline_values))
    elif paired:
        test = wilcoxon_signed_rank_test(baseline_values, candidate_values)
    else:
        test = mann_whitney_u_test(baseline_values, candidate_values)

    effect = continuous_delta(baseline_values, candidate_values)
    b_arr = np.asarray(baseline_values, dtype=float)
    c_arr = np.asarray(candidate_values, dtype=float)
    ci = _bootstrap_delta_ci(
        b_arr,
        c_arr,
        paired=paired,
        seed=seed,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
    )

    conclusion = _classify(
        n_baseline=len(baseline_values),
        n_candidate=len(candidate_values),
        p_value=test.p_value,
        adjusted_p_value=adjusted_p_value,
        alpha=alpha,
        absolute_delta=effect.absolute_delta,
        higher_is_better=higher_is_better,
    )

    return ExperimentComparison(
        metric_name=metric_name,
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        baseline_estimate=effect.baseline_mean,
        candidate_estimate=effect.candidate_mean,
        absolute_delta=effect.absolute_delta,
        relative_delta=effect.relative_delta,
        confidence_interval=ci,
        test=test,
        raw_p_value=test.p_value,
        adjusted_p_value=adjusted_p_value,
        effect_size=effect,
        sample_size=len(baseline_values) + len(candidate_values),
        conclusion=conclusion,
        alpha=alpha,
        higher_is_better=higher_is_better,
    )
