"""Regression detection: compares two persisted `ExperimentStore` runs and
decides whether a configured `RegressionPolicy` is triggered.

**Operates on actual stored data, never manually entered summary values**
(per the Phase F spec) — every metric is extracted fresh from
`store.get_eval_results`/`store.get_outcome_events` for the two run ids
given, through `statistics.comparison`'s auto-selected paired/independent
tests.

**Practical vs. statistical significance**: a `MetricRegressionPolicy` has
*two* independent gates — `max_practical_change` (a real-world-meaningful
threshold in the metric's own units) and, optionally,
`require_statistical_significance` (the change must also be distinguishable
from noise). A tiny but statistically-significant change at huge `n`
(e.g. success rate down 0.2 percentage points with thousands of tasks) does
not trigger a regression on its own unless it also clears the practical
threshold — see the spec's own example. Both gates must pass (when both are
configured) for a metric to be `triggered`.

**Paired regression**: when baseline and candidate share the same task
IDs, `statistics.comparison` already detects and uses the paired test.
This module additionally treats a *partial* task-ID mismatch under a
matching suite name/version as a run-compatibility failure rather than
silently degrading to an independent-sample comparison — a candidate run
that accidentally covers a different subset of the same suite is a data
problem to surface loudly, not paper over (see the spec: "fail loudly if
the comparison expects paired data but runs are not compatible").
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import IntEnum, StrEnum

from paretoguard.core.models import RunManifest
from paretoguard.evals.models import EvalResult, GraderKind
from paretoguard.statistics.comparison import (
    MIN_SAMPLE_SIZE_FOR_EVIDENCE,
    ExperimentComparison,
    compare_binary_outcomes,
    compare_continuous_outcomes,
)
from paretoguard.statistics.significance import holm_bonferroni
from paretoguard.storage import ExperimentStore

_BINARY_METRICS = frozenset(
    {
        "success_rate",
        "recovery_rate",
        "structured_output_validity_rate",
        "tool_use_correctness_rate",
    }
)
_CONTINUOUS_METRICS = frozenset({"cost_usd", "latency_ms", "p95_latency_ms"})
KNOWN_METRICS = _BINARY_METRICS | _CONTINUOUS_METRICS


class RegressionOutcome(StrEnum):
    OK = "ok"
    REGRESSION_DETECTED = "regression_detected"
    INCOMPATIBLE_RUNS = "incompatible_runs"
    INSUFFICIENT_DATA = "insufficient_data"
    CONFIG_ERROR = "config_error"


class RegressionExitCode(IntEnum):
    """Mirrors `RegressionOutcome` for a future CLI (`paretoguard regress`,
    Phase G) — tested at this service layer per the spec, since CLI
    exposure itself is out of Phase F's scope."""

    OK = 0
    REGRESSION_DETECTED = 1
    INCOMPATIBLE_RUNS = 2
    INSUFFICIENT_DATA = 3
    CONFIG_ERROR = 4


_EXIT_CODE_BY_OUTCOME: dict[RegressionOutcome, RegressionExitCode] = {
    RegressionOutcome.OK: RegressionExitCode.OK,
    RegressionOutcome.REGRESSION_DETECTED: RegressionExitCode.REGRESSION_DETECTED,
    RegressionOutcome.INCOMPATIBLE_RUNS: RegressionExitCode.INCOMPATIBLE_RUNS,
    RegressionOutcome.INSUFFICIENT_DATA: RegressionExitCode.INSUFFICIENT_DATA,
    RegressionOutcome.CONFIG_ERROR: RegressionExitCode.CONFIG_ERROR,
}


@dataclass(frozen=True)
class MetricRegressionPolicy:
    """One metric's regression rule.

    `max_practical_change`: the maximum tolerable change, in the *bad*
    direction, before this metric counts as regressed — in the metric's own
    units (percentage points for a rate, USD for cost, milliseconds for
    latency). E.g. `higher_is_better=True, max_practical_change=0.02` means
    "a decrease of more than 2 percentage points is a regression";
    `higher_is_better=False, max_practical_change=0.01` means "an increase
    of more than $0.01 is a regression". `0.0` means "any move in the bad
    direction counts" (still gated by `require_statistical_significance`
    unless that's also disabled).
    """

    metric_name: str
    higher_is_better: bool = True
    max_practical_change: float = 0.0
    require_statistical_significance: bool = True
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if self.metric_name not in KNOWN_METRICS:
            raise ValueError(
                f"unknown metric_name {self.metric_name!r}; must be one of {sorted(KNOWN_METRICS)}"
            )
        if self.metric_name in _BINARY_METRICS and not self.higher_is_better:
            raise ValueError(
                f"{self.metric_name!r} is a binary rate metric: higher_is_better must be True"
            )
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")


@dataclass(frozen=True)
class RegressionPolicy:
    metrics: tuple[MetricRegressionPolicy, ...]
    apply_multiple_comparison_correction: bool = True
    """Holm-Bonferroni across every configured metric's raw p-value — the
    family of hypotheses is exactly `metrics`, per `holm_bonferroni`'s own
    docstring on scoping the family correctly."""
    allow_simulation_vs_live: bool = False
    """Comparing a SIMULATION-labeled run against a LIVE-labeled run (or
    either against an unlabeled one) is rejected as incompatible unless
    this is explicitly set — see `check_run_compatibility`."""


@dataclass(frozen=True)
class MetricRegressionResult:
    metric_name: str
    available: bool
    """False if this metric's data was unavailable or insufficient
    (`< MIN_SAMPLE_SIZE_FOR_EVIDENCE` per arm) in either run — never
    fabricated as a zero change."""
    comparison: ExperimentComparison | None
    practical_change: float | None
    """Positive means "moved in the bad direction" regardless of whether
    the metric is higher- or lower-is-better — always directly comparable
    to `max_practical_change`. `None` when `available` is `False`."""
    exceeds_practical_threshold: bool
    statistically_significant: bool
    """Uses the Holm-Bonferroni-adjusted p-value when the policy applies
    correction, the raw p-value otherwise."""
    triggered: bool
    reason: str


@dataclass(frozen=True)
class RegressionCheckResult:
    outcome: RegressionOutcome
    exit_code: int
    metric_results: tuple[MetricRegressionResult, ...]
    incompatibility_reasons: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.outcome == RegressionOutcome.OK


def check_run_compatibility(
    baseline: RunManifest, candidate: RunManifest, *, allow_simulation_vs_live: bool = False
) -> list[str]:
    """Validates suite identity and simulation/live labeling before any
    metric is compared. Returns a list of human-readable issues (empty if
    compatible) — never raises, so a caller can report every issue at once
    rather than stopping at the first."""
    issues: list[str] = []
    if (
        baseline.suite_name is not None
        and candidate.suite_name is not None
        and baseline.suite_name != candidate.suite_name
    ):
        issues.append(f"suite_name mismatch: {baseline.suite_name!r} vs {candidate.suite_name!r}")
    if (
        baseline.suite_version is not None
        and candidate.suite_version is not None
        and baseline.suite_version != candidate.suite_version
    ):
        issues.append(
            f"suite_version mismatch: {baseline.suite_version!r} vs {candidate.suite_version!r}"
        )
    if not allow_simulation_vs_live and baseline.label != candidate.label:
        issues.append(
            f"label mismatch: {baseline.label!r} vs {candidate.label!r} — comparing a "
            "simulation run against a live run (or either against an unlabeled run) "
            "requires RegressionPolicy(allow_simulation_vs_live=True)"
        )
    return issues


def _percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile — same method as `evals.metrics._percentile`
    (not imported directly since that one is private to its module); kept
    consistent across the codebase deliberately, not reimplemented
    differently."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0, min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1))
    return ordered[rank]


def _extract_binary(
    metric_name: str, results: Sequence[EvalResult]
) -> tuple[list[str], list[bool]]:
    if metric_name == "success_rate":
        filtered = list(results)
    elif metric_name == "structured_output_validity_rate":
        filtered = [r for r in results if r.grader_kind == GraderKind.JSON_SCHEMA]
    elif metric_name == "tool_use_correctness_rate":
        filtered = [r for r in results if r.grader_kind == GraderKind.TOOL_TRAJECTORY]
    else:
        raise AssertionError(f"unhandled binary metric: {metric_name}")  # pragma: no cover
    return [r.case_id for r in filtered], [r.succeeded for r in filtered]


def _extract_continuous(
    metric_name: str, results: Sequence[EvalResult]
) -> tuple[list[str], list[float]]:
    if metric_name == "cost_usd":
        filtered = [r for r in results if r.cost_usd is not None]
        return [r.case_id for r in filtered], [
            r.cost_usd for r in filtered if r.cost_usd is not None
        ]
    # "latency_ms" / "p95_latency_ms" both compare the raw per-task latency
    # distribution — see this module's docstring on why p95 uses the raw
    # values for its statistical test even though the *practical* threshold
    # is later applied to the aggregate p95 figure specifically.
    return [r.case_id for r in results], [r.latency_ms for r in results]


def _extract_recovery_rate(
    store: ExperimentStore, run_id: str
) -> tuple[list[str], list[bool]] | None:
    events = store.get_outcome_events(run_id)
    raw_failures = [e for e in events if e.attempt_count > 1 or not e.succeeded]
    if not raw_failures:
        return None
    return [e.task_id or str(e.outcome_id) for e in raw_failures], [
        e.succeeded for e in raw_failures
    ]


def check_regression(
    store: ExperimentStore,
    baseline_run_id: str,
    candidate_run_id: str,
    policy: RegressionPolicy,
) -> RegressionCheckResult:
    """The regression engine's entry point. Always returns a
    `RegressionCheckResult` — never raises for a data problem (missing run,
    incompatible runs, insufficient data); those become the corresponding
    `RegressionOutcome`, distinguishable by `exit_code`."""
    if not policy.metrics:
        return RegressionCheckResult(
            RegressionOutcome.CONFIG_ERROR,
            _EXIT_CODE_BY_OUTCOME[RegressionOutcome.CONFIG_ERROR],
            (),
            ("RegressionPolicy.metrics must not be empty",),
        )

    baseline_manifest = store.get_run(baseline_run_id)
    candidate_manifest = store.get_run(candidate_run_id)
    if baseline_manifest is None or candidate_manifest is None:
        missing = [
            run_id
            for run_id, manifest in (
                (baseline_run_id, baseline_manifest),
                (candidate_run_id, candidate_manifest),
            )
            if manifest is None
        ]
        return RegressionCheckResult(
            RegressionOutcome.CONFIG_ERROR,
            _EXIT_CODE_BY_OUTCOME[RegressionOutcome.CONFIG_ERROR],
            (),
            tuple(f"run not found: {run_id!r}" for run_id in missing),
        )

    issues = check_run_compatibility(
        baseline_manifest,
        candidate_manifest,
        allow_simulation_vs_live=policy.allow_simulation_vs_live,
    )
    baseline_results = store.get_eval_results(baseline_run_id)
    candidate_results = store.get_eval_results(candidate_run_id)
    baseline_task_ids = {r.case_id for r in baseline_results}
    candidate_task_ids = {r.case_id for r in candidate_results}
    if (
        baseline_manifest.suite_name is not None
        and baseline_manifest.suite_name == candidate_manifest.suite_name
        and baseline_task_ids
        and candidate_task_ids
        and baseline_task_ids != candidate_task_ids
    ):
        issues.append(
            "task id sets differ under the same suite_name/suite_version — "
            "candidate does not cover the same tasks as baseline; refusing to "
            "silently fall back to an independent-sample comparison"
        )
    if issues:
        return RegressionCheckResult(
            RegressionOutcome.INCOMPATIBLE_RUNS,
            _EXIT_CODE_BY_OUTCOME[RegressionOutcome.INCOMPATIBLE_RUNS],
            (),
            tuple(issues),
        )

    raw: list[tuple[MetricRegressionPolicy, ExperimentComparison | None]] = []
    for metric_policy in policy.metrics:
        comparison = _compare_one_metric(
            store,
            baseline_run_id,
            candidate_run_id,
            baseline_results,
            candidate_results,
            metric_policy,
        )
        raw.append((metric_policy, comparison))

    adjusted_by_metric: dict[str, float] = {}
    if policy.apply_multiple_comparison_correction:
        available_raw_p = [c.raw_p_value for _, c in raw if c is not None]
        adjusted_values = holm_bonferroni(available_raw_p)
        adjusted_iter = iter(adjusted_values)
        for metric_policy, comparison in raw:
            if comparison is not None:
                adjusted_by_metric[metric_policy.metric_name] = next(adjusted_iter)
        # Write the adjusted p-value back onto each ExperimentComparison
        # itself — otherwise a caller inspecting MetricRegressionResult
        # .comparison.adjusted_p_value would see None and wrongly conclude
        # no correction was applied, even though `triggered` below already
        # used the corrected value.
        raw = [
            (
                metric_policy,
                replace(comparison, adjusted_p_value=adjusted_by_metric[metric_policy.metric_name])
                if comparison is not None
                else None,
            )
            for metric_policy, comparison in raw
        ]

    metric_results: list[MetricRegressionResult] = []
    any_triggered = False
    any_available = False
    for metric_policy, comparison in raw:
        if comparison is None:
            metric_results.append(
                MetricRegressionResult(
                    metric_name=metric_policy.metric_name,
                    available=False,
                    comparison=None,
                    practical_change=None,
                    exceeds_practical_threshold=False,
                    statistically_significant=False,
                    triggered=False,
                    reason="metric data unavailable or below minimum sample size in one or both runs",
                )
            )
            continue
        any_available = True
        effective_p = adjusted_by_metric.get(metric_policy.metric_name, comparison.raw_p_value)
        significant = effective_p < metric_policy.alpha
        baseline_estimate, candidate_estimate = (
            comparison.baseline_estimate,
            comparison.candidate_estimate,
        )
        if metric_policy.metric_name == "p95_latency_ms":
            # The statistical test above ran on raw per-task latencies
            # (appropriate for "did the distribution shift at all"), but the
            # *practical* threshold is about the aggregate p95 figure
            # specifically, not the mean compare_continuous_outcomes reports
            # — see this module's docstring.
            _, baseline_latencies = _extract_continuous("latency_ms", baseline_results)
            _, candidate_latencies = _extract_continuous("latency_ms", candidate_results)
            baseline_estimate = _percentile(baseline_latencies, 0.95)
            candidate_estimate = _percentile(candidate_latencies, 0.95)
        if metric_policy.higher_is_better:
            practical_change = baseline_estimate - candidate_estimate
        else:
            practical_change = candidate_estimate - baseline_estimate
        exceeds = practical_change > metric_policy.max_practical_change
        triggered = exceeds and (
            significant if metric_policy.require_statistical_significance else True
        )
        any_triggered = any_triggered or triggered
        reason = (
            f"{'regressed' if triggered else 'within policy'}: practical_change="
            f"{practical_change:.6g} (threshold={metric_policy.max_practical_change:.6g}), "
            f"p={effective_p:.4g} (alpha={metric_policy.alpha:.4g})"
        )
        metric_results.append(
            MetricRegressionResult(
                metric_name=metric_policy.metric_name,
                available=True,
                comparison=comparison,
                practical_change=practical_change,
                exceeds_practical_threshold=exceeds,
                statistically_significant=significant,
                triggered=triggered,
                reason=reason,
            )
        )

    if any_triggered:
        outcome = RegressionOutcome.REGRESSION_DETECTED
    elif not any_available:
        outcome = RegressionOutcome.INSUFFICIENT_DATA
    else:
        outcome = RegressionOutcome.OK

    return RegressionCheckResult(outcome, _EXIT_CODE_BY_OUTCOME[outcome], tuple(metric_results))


def _compare_one_metric(
    store: ExperimentStore,
    baseline_run_id: str,
    candidate_run_id: str,
    baseline_results: Sequence[EvalResult],
    candidate_results: Sequence[EvalResult],
    metric_policy: MetricRegressionPolicy,
) -> ExperimentComparison | None:
    name = metric_policy.metric_name
    if name == "recovery_rate":
        baseline_extraction = _extract_recovery_rate(store, baseline_run_id)
        candidate_extraction = _extract_recovery_rate(store, candidate_run_id)
        if baseline_extraction is None or candidate_extraction is None:
            return None
        baseline_ids, baseline_values = baseline_extraction
        candidate_ids, candidate_values = candidate_extraction
        if (
            len(baseline_values) < MIN_SAMPLE_SIZE_FOR_EVIDENCE
            or len(candidate_values) < MIN_SAMPLE_SIZE_FOR_EVIDENCE
        ):
            return None
        return compare_binary_outcomes(
            baseline_ids,
            baseline_values,
            candidate_ids,
            candidate_values,
            metric_name=name,
            baseline_run_id=baseline_run_id,
            candidate_run_id=candidate_run_id,
            alpha=metric_policy.alpha,
        )

    if name in _BINARY_METRICS:
        baseline_ids, baseline_values = _extract_binary(name, baseline_results)
        candidate_ids, candidate_values = _extract_binary(name, candidate_results)
        if (
            len(baseline_values) < MIN_SAMPLE_SIZE_FOR_EVIDENCE
            or len(candidate_values) < MIN_SAMPLE_SIZE_FOR_EVIDENCE
        ):
            return None
        return compare_binary_outcomes(
            baseline_ids,
            baseline_values,
            candidate_ids,
            candidate_values,
            metric_name=name,
            baseline_run_id=baseline_run_id,
            candidate_run_id=candidate_run_id,
            alpha=metric_policy.alpha,
        )

    baseline_ids, baseline_floats = _extract_continuous(name, baseline_results)
    candidate_ids, candidate_floats = _extract_continuous(name, candidate_results)
    if (
        len(baseline_floats) < MIN_SAMPLE_SIZE_FOR_EVIDENCE
        or len(candidate_floats) < MIN_SAMPLE_SIZE_FOR_EVIDENCE
    ):
        return None
    return compare_continuous_outcomes(
        baseline_ids,
        baseline_floats,
        candidate_ids,
        candidate_floats,
        metric_name=name,
        higher_is_better=metric_policy.higher_is_better,
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        alpha=metric_policy.alpha,
    )
