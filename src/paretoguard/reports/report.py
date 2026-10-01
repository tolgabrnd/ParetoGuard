"""Reproducible benchmark reports — generated entirely from persisted
`ExperimentStore` data, never hand-entered values. Every field that can be
unavailable (no cost data, no baseline given, no `OutcomeEvent`s for this
run) is `None`/omitted with an explicit "not available" note in
`limitations`, never a fabricated zero (see CLAUDE.md and the Phase F spec).
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from paretoguard.core.models import RunManifest
from paretoguard.evals.metrics import (
    ResilienceMetricsSummary,
    compute_metrics,
    compute_resilience_metrics,
)
from paretoguard.statistics.bootstrap import (
    DEFAULT_CONFIDENCE_LEVEL,
    BootstrapResult,
    bootstrap_proportion_ci,
)
from paretoguard.statistics.comparison import ExperimentComparison, compare_binary_outcomes
from paretoguard.statistics.pareto_frontier import ParetoAnalysis
from paretoguard.statistics.regression import check_run_compatibility
from paretoguard.storage import ExperimentStore

SIMULATION_BANNER = "SIMULATION — NOT LIVE PROVIDER PERFORMANCE"


@dataclass(frozen=True)
class FailureTaxonomyEntry:
    category: str
    count: int
    rate: float
    """`count / sample_size` — the fraction of *all* tasks, not just failures."""


@dataclass(frozen=True)
class BenchmarkReport:
    run_id: str
    manifest: RunManifest
    sample_size: int
    success_rate: float | None
    success_rate_ci: BootstrapResult | None
    cost_per_success_usd: float | None
    mean_latency_ms: float | None
    failure_taxonomy: list[FailureTaxonomyEntry]
    recovery_summary: ResilienceMetricsSummary | None
    """From this run's `OutcomeEvent`s, if any were persisted — `None` for
    a plain `BenchmarkRunner`/`RoutedBenchmarkRunner` run with no closed-loop
    recovery to summarize."""
    comparison: ExperimentComparison | None
    """Vs. a baseline run, if one was given to `generate_report`."""
    pareto_analysis: ParetoAnalysis | None
    """Only populated if the caller supplies one — a single run's
    `EvalResult`s alone don't carry the multi-candidate cost/success data a
    Pareto frontier needs (that's `evals.matrix.MatrixRunner`'s shape, an
    orthogonal offline path — see `docs/ARCHITECTURE.md`)."""
    limitations: list[str] = field(default_factory=list)


def generate_report(
    store: ExperimentStore,
    run_id: str,
    *,
    baseline_run_id: str | None = None,
    comparison_metric: str = "success_rate",
    pareto_analysis: ParetoAnalysis | None = None,
    seed: int = 0,
) -> BenchmarkReport:
    """Builds a `BenchmarkReport` entirely from `store`'s persisted rows for
    `run_id` (and, optionally, `baseline_run_id` for a comparison). Raises
    `ValueError` if `run_id` itself doesn't exist — there is no meaningful
    report over a run that was never recorded."""
    manifest = store.get_run(run_id)
    if manifest is None:
        raise ValueError(f"no run found for run_id={run_id!r}")

    limitations: list[str] = []
    if manifest.label is None:
        limitations.append(
            "This run has no SIMULATION/LIVE label — treat every number here cautiously."
        )

    results = store.get_eval_results(run_id)
    if not results:
        return BenchmarkReport(
            run_id=run_id,
            manifest=manifest,
            sample_size=0,
            success_rate=None,
            success_rate_ci=None,
            cost_per_success_usd=None,
            mean_latency_ms=None,
            failure_taxonomy=[],
            recovery_summary=None,
            comparison=None,
            pareto_analysis=pareto_analysis,
            limitations=[*limitations, "No EvalResult rows found for this run."],
        )

    metrics = compute_metrics(results)
    success_ci = bootstrap_proportion_ci(
        [r.succeeded for r in results], seed=seed, confidence_level=DEFAULT_CONFIDENCE_LEVEL
    )

    unrecovered = [r for r in results if not r.succeeded]
    category_counts = Counter(
        (r.response_error_category.value if r.response_error_category else "unknown")
        for r in unrecovered
    )
    taxonomy = [
        FailureTaxonomyEntry(category=cat, count=count, rate=count / len(results))
        for cat, count in sorted(category_counts.items(), key=lambda kv: -kv[1])
    ]

    outcome_events = store.get_outcome_events(run_id)
    recovery_summary = compute_resilience_metrics(outcome_events) if outcome_events else None

    comparison = None
    if baseline_run_id is not None:
        baseline_manifest = store.get_run(baseline_run_id)
        if baseline_manifest is None:
            limitations.append(f"baseline run {baseline_run_id!r} not found; comparison omitted.")
        else:
            issues = check_run_compatibility(baseline_manifest, manifest)
            if issues:
                limitations.append(
                    "Baseline comparison omitted (incompatible runs): " + "; ".join(issues)
                )
            else:
                baseline_results = store.get_eval_results(baseline_run_id)
                if not baseline_results:
                    limitations.append(
                        f"baseline run {baseline_run_id!r} has no EvalResult rows; comparison omitted."
                    )
                else:
                    comparison = compare_binary_outcomes(
                        [r.case_id for r in baseline_results],
                        [r.succeeded for r in baseline_results],
                        [r.case_id for r in results],
                        [r.succeeded for r in results],
                        metric_name=comparison_metric,
                        baseline_run_id=baseline_run_id,
                        candidate_run_id=run_id,
                        seed=seed,
                    )

    if metrics.cost_per_success_usd is None:
        limitations.append(
            "No cost data available for this run (no PricingTable configured, or no successes)."
        )
    if pareto_analysis is None:
        limitations.append(
            "No Pareto analysis available: a single run's EvalResults don't carry the "
            "multi-candidate data a frontier needs (see evals.matrix.MatrixRunner)."
        )

    return BenchmarkReport(
        run_id=run_id,
        manifest=manifest,
        sample_size=len(results),
        success_rate=metrics.success_rate,
        success_rate_ci=success_ci,
        cost_per_success_usd=metrics.cost_per_success_usd,
        mean_latency_ms=metrics.mean_latency_ms_per_success,
        failure_taxonomy=taxonomy,
        recovery_summary=recovery_summary,
        comparison=comparison,
        pareto_analysis=pareto_analysis,
        limitations=limitations,
    )


def _na(value: object, fmt: str = "{}") -> str:
    return "not available" if value is None else fmt.format(value)


def render_markdown(report: BenchmarkReport) -> str:
    lines: list[str] = []
    if report.manifest.label == "SIMULATION":
        lines.append(f"**{SIMULATION_BANNER}**")
        lines.append("")
    lines.append(f"# Benchmark report: {report.run_id}")
    lines.append("")
    lines.append("## Run identity (reproducibility)")
    lines.append(f"- run_id: `{report.run_id}`")
    lines.append(f"- label: {report.manifest.label or 'unlabeled'}")
    lines.append(f"- git_sha: {_na(report.manifest.git_sha)}")
    lines.append(f"- paretoguard_version: {report.manifest.paretoguard_version}")
    lines.append(
        f"- suite: {_na(report.manifest.suite_name)} v{_na(report.manifest.suite_version)}"
    )
    lines.append(f"- seed: {_na(report.manifest.seed)}")
    lines.append(f"- router: {_na(report.manifest.router_name)}")
    lines.append(f"- pricing_config_version: {_na(report.manifest.pricing_config_version)}")
    lines.append(f"- os / python: {report.manifest.os} / {report.manifest.python_version}")
    lines.append(f"- sample size: {report.sample_size}")
    lines.append("")

    lines.append("## Metrics")
    if report.success_rate is None:
        lines.append("- success rate: not available")
    else:
        ci = report.success_rate_ci
        assert ci is not None
        reliability_note = "" if ci.reliable else " — LOW CONFIDENCE (n < 30)"
        lines.append(
            f"- success rate: {report.success_rate:.3f} "
            f"(95% CI [{ci.lower:.3f}, {ci.upper:.3f}]{reliability_note})"
        )
    lines.append(f"- cost per successful task: {_na(report.cost_per_success_usd, '${:.4f}')}")
    lines.append(f"- mean latency (successful tasks): {_na(report.mean_latency_ms, '{:.2f} ms')}")
    lines.append("")

    lines.append("## Failure taxonomy")
    if not report.failure_taxonomy:
        lines.append("- no failures recorded")
    else:
        for entry in report.failure_taxonomy:
            lines.append(f"- {entry.category}: {entry.count} ({entry.rate:.1%} of all tasks)")
    lines.append("")

    if report.recovery_summary is not None:
        r = report.recovery_summary
        lines.append("## Recovery / circuit-breaker outcomes")
        lines.append(f"- raw failure rate: {r.raw_failure_rate:.3f}")
        lines.append(f"- recovery rate: {_na(r.recovery_rate, '{:.3f}')}")
        lines.append(f"- retry rate: {r.retry_rate:.3f}")
        lines.append(f"- fallback rate: {r.fallback_rate:.3f}")
        lines.append(f"- escalation rate: {r.escalation_rate:.3f}")
        lines.append(f"- probe rate: {r.probe_rate:.3f}")
        lines.append(f"- average attempts: {r.average_attempts:.2f}")
        lines.append("")

    lines.append("## Pareto analysis")
    if report.pareto_analysis is None:
        lines.append("- not available")
    else:
        lines.append(f"- frontier: {', '.join(report.pareto_analysis.frontier)}")
        for name, dominators in report.pareto_analysis.dominated.items():
            lines.append(f"- {name}: dominated by {', '.join(dominators)}")
    lines.append("")

    if report.comparison is not None:
        c = report.comparison
        lines.append("## Comparison vs. baseline")
        lines.append(f"- baseline_run_id: `{c.baseline_run_id}`")
        lines.append(f"- metric: {c.metric_name}")
        lines.append(
            f"- baseline: {c.baseline_estimate:.4f}, candidate: {c.candidate_estimate:.4f}"
        )
        lines.append(f"- absolute delta: {c.absolute_delta:+.4f}")
        lines.append(f"- test: {c.test.test_name} (paired={c.test.paired}, p={c.raw_p_value:.4g})")
        lines.append(f"- conclusion: {c.conclusion.value}")
        lines.append("(descriptive only — not a causal claim)")
        lines.append("")

    if report.limitations:
        lines.append("## Limitations")
        for lim in report.limitations:
            lines.append(f"- {lim}")

    return "\n".join(lines)


def render_json(report: BenchmarkReport) -> dict[str, Any]:
    return {
        "run_id": report.run_id,
        "manifest": report.manifest.model_dump(mode="json"),
        "sample_size": report.sample_size,
        "success_rate": report.success_rate,
        "success_rate_ci": (
            None
            if report.success_rate_ci is None
            else {
                "lower": report.success_rate_ci.lower,
                "upper": report.success_rate_ci.upper,
                "confidence_level": report.success_rate_ci.confidence_level,
                "reliable": report.success_rate_ci.reliable,
            }
        ),
        "cost_per_success_usd": report.cost_per_success_usd,
        "mean_latency_ms": report.mean_latency_ms,
        "failure_taxonomy": [
            {"category": e.category, "count": e.count, "rate": e.rate}
            for e in report.failure_taxonomy
        ],
        "recovery_summary": (
            None if report.recovery_summary is None else report.recovery_summary.model_dump()
        ),
        "pareto_analysis": (
            None
            if report.pareto_analysis is None
            else {
                "frontier": report.pareto_analysis.frontier,
                "dominated": report.pareto_analysis.dominated,
            }
        ),
        "comparison": (
            None
            if report.comparison is None
            else {
                "baseline_run_id": report.comparison.baseline_run_id,
                "candidate_run_id": report.comparison.candidate_run_id,
                "metric_name": report.comparison.metric_name,
                "baseline_estimate": report.comparison.baseline_estimate,
                "candidate_estimate": report.comparison.candidate_estimate,
                "absolute_delta": report.comparison.absolute_delta,
                "raw_p_value": report.comparison.raw_p_value,
                "conclusion": report.comparison.conclusion.value,
            }
        ),
        "limitations": report.limitations,
    }
