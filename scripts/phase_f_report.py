"""Phase F flagship analysis: statistics, regression, and reporting tied
together end-to-end over real persisted data.

**SIMULATION — NOT LIVE PROVIDER PERFORMANCE.** Every number here comes
from `MockProvider` and injected/scripted synthetic faults — see
`docs/LIMITATIONS.md`'s Phase E/E.5 sections for the full experimental
design and honest findings each scenario below reuses.

This script deliberately does two different things with two different
scales, for a reason documented in `docs/LIMITATIONS.md`
("`ExperimentStore` transaction batching"): `ExperimentStore` writes have
real per-statement cost on this development machine, so:

1. The **curve/summary data** for all three flagship comparisons (i.i.d.
   fault levels, sustained outage, escalation) reuses the existing
   `run_resilience_benchmark`/`run_sustained_outage_scenario`/
   `run_escalation_scenario` functions *without* a store — the same
   fast, already-verified-deterministic path `scripts/phase_e5_experiment.py`
   uses for its own larger-scale run.
2. A **small, store-backed slice** (one fault level, all five recovery
   configs, `RESILIENCE_STORE_NUM_CASES` tasks/cell) is *additionally* run
   through a real `ExperimentStore`, specifically to demonstrate and verify
   the full `statistics.regression.check_regression` ->
   `reports.generate_report` -> `reports.charts` pipeline against genuinely
   persisted rows, not just in-memory objects — this is what Commit 32's
   "reports must be generated entirely from persisted/raw experiment
   artifacts" is checked against.

Routing-only degradation (no recovery layer) reuses Phase D's
`routing.reliability_simulation` directly, unchanged — see that module for
its own SIMULATION labeling.
"""

import asyncio
import json
from pathlib import Path

from paretoguard.core.models import ModelSpec
from paretoguard.evals.suites import resilience_v1
from paretoguard.reports import (
    generate_report,
    plot_baseline_vs_candidate,
    plot_failure_taxonomy,
    plot_recovery_rate_by_level,
    plot_resilience_curve,
    plot_selection_distribution,
    render_json,
    render_markdown,
)
from paretoguard.reports.charts import ResilienceCurvePoint
from paretoguard.routing.escalation_benchmark import run_escalation_scenario
from paretoguard.routing.pareto_router import ParetoObjective, ParetoRouter
from paretoguard.routing.reliability import ReliabilityAwareRouter
from paretoguard.routing.reliability_simulation import (
    run_degradation_recovery_simulation,
    two_model_degradation_schedule,
)
from paretoguard.routing.resilience_benchmark import (
    ResilienceBenchmarkResult,
    run_resilience_benchmark,
)
from paretoguard.routing.static import StaticRouter
from paretoguard.routing.sustained_outage_benchmark import run_sustained_outage_scenario
from paretoguard.routing.types import CandidateProfile, candidate_key
from paretoguard.statistics.regression import (
    MetricRegressionPolicy,
    RegressionPolicy,
    check_regression,
)
from paretoguard.storage import ExperimentStore

RESULTS_DIR = Path(__file__).resolve().parent / "phase_f_results"
RESILIENCE_STORE_NUM_CASES = 30
RESILIENCE_STORE_FAULT_LEVEL = 0.30


async def run_iid_fault_curves() -> ResilienceBenchmarkResult:
    print("\n=== 1/4: recovery configs under i.i.d. faults (resilience_v1) ===")
    suite = resilience_v1.build_suite(seed=0, num_cases=24)
    result = await run_resilience_benchmark(suite, seed=0)
    for row in result.rows:
        print(
            f"  level={row.fault_level:<5.2f} {row.recovery_config:<35s} "
            f"success={row.metrics.task_success_rate:.3f} n={row.metrics.n}"
        )
    return result


async def run_sustained_outage_summary() -> dict:
    print("\n=== 2/4: recovery configs under sustained/correlated outage ===")
    results = await run_sustained_outage_scenario()
    summary = {}
    for r in results:
        m = r.metrics
        summary[r.recovery_config] = {
            "task_success_rate": m.task_success_rate,
            "task_success_rate_during_outage": m.task_success_rate_during_outage,
            "average_attempts": m.average_attempts,
            "fallback_count": m.fallback_count,
            "circuit_open_transitions": m.circuit_open_transitions,
        }
        print(
            f"  {r.recovery_config:<35s} outage_success={m.task_success_rate_during_outage:.3f} "
            f"fallback={m.fallback_count} circuit_opens={m.circuit_open_transitions}"
        )
    return summary


async def run_escalation_summary() -> dict:
    print("\n=== 3/4: escalation vs. no validation under quality failures ===")
    result = await run_escalation_scenario(num_cases=20, quality_fault_rate=0.6, seed=0)
    print(
        f"  no_validation: success={result.no_validation.task_success_rate:.3f} "
        f"escalate={result.no_validation.escalation_rate:.3f}"
    )
    print(
        f"  validation_and_escalation: success={result.validation_and_escalation.task_success_rate:.3f} "
        f"escalate={result.validation_and_escalation.escalation_rate:.3f}"
    )
    return {
        "no_validation_success_rate": result.no_validation.task_success_rate,
        "validation_success_rate": result.validation_and_escalation.task_success_rate,
        "success_rate_delta": result.success_rate_delta,
        "added_average_attempts": result.added_average_attempts,
    }


def run_routing_only_degradation() -> dict:
    print("\n=== 4/4: routing-only degradation (no recovery layer, Phase D) ===")
    candidates = [
        ModelSpec(name="primary-sim", provider="mock", context_window=100_000),
        ModelSpec(name="stable-sim", provider="mock", context_window=100_000),
    ]
    profiles = {
        candidate_key("mock", "primary-sim"): CandidateProfile(
            predicted_success=0.99, mean_cost_usd=0.01, mean_latency_ms=100.0, simulated=True
        ),
        candidate_key("mock", "stable-sim"): CandidateProfile(
            predicted_success=0.98, mean_cost_usd=0.01, mean_latency_ms=100.0, simulated=True
        ),
    }
    schedule = two_model_degradation_schedule(
        primary_model="primary-sim",
        stable_model="stable-sim",
        degraded_start=100,
        recovered_start=200,
    )
    summary = {}
    for router in [
        StaticRouter(provider="mock", model="primary-sim"),
        ParetoRouter(ParetoObjective.MAXIMIZE_SUCCESS),
        ReliabilityAwareRouter(),
    ]:
        result = run_degradation_recovery_simulation(
            router,
            candidates=candidates,
            schedule=schedule,
            profiles_by_model=profiles,
            primary_model="primary-sim",
            degraded_start=100,
            recovered_start=200,
            total_steps=300,
            seed=42,
        )
        summary[result.router_name] = {
            "task_success_rate": result.task_success_rate,
            "reroute_count": result.reroute_count,
            "time_to_detect_degradation": result.time_to_detect_degradation,
            "time_to_recover": result.time_to_recover,
        }
        print(
            f"  {result.router_name:<15s} success={result.task_success_rate:.3f} "
            f"reroutes={result.reroute_count} detect={result.time_to_detect_degradation} "
            f"recover={result.time_to_recover} [{result.label}]"
        )
    return summary


async def run_store_backed_slice(store: ExperimentStore) -> tuple[str, str]:
    """The small, store-backed demonstration slice: one fault level, all
    five recovery configs, persisted for real. Returns
    (baseline_run_id, candidate_run_id) for A-no_recovery vs E-full_policy."""
    print(
        f"\n=== Store-backed slice: fault_level={RESILIENCE_STORE_FAULT_LEVEL}, "
        f"{RESILIENCE_STORE_NUM_CASES} tasks/cell, persisted ==="
    )
    suite = resilience_v1.build_suite(seed=0, num_cases=RESILIENCE_STORE_NUM_CASES)
    await run_resilience_benchmark(
        suite, fault_levels=[RESILIENCE_STORE_FAULT_LEVEL], seed=0, store=store
    )
    baseline_run_id = f"resilience-{suite.name}-{RESILIENCE_STORE_FAULT_LEVEL}-A-no_recovery"
    candidate_run_id = f"resilience-{suite.name}-{RESILIENCE_STORE_FAULT_LEVEL}-E-full_policy"
    for run_id in (baseline_run_id, candidate_run_id):
        manifest = store.get_run(run_id)
        if manifest is None:
            raise RuntimeError(f"expected run_resilience_benchmark to have recorded {run_id!r}")
        store.record_run(manifest.model_copy(update={"label": "SIMULATION"}))
    return baseline_run_id, candidate_run_id


async def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    print("SIMULATION — NOT LIVE PROVIDER PERFORMANCE")

    iid_result = await run_iid_fault_curves()
    iid_points = [
        ResilienceCurvePoint(
            row.fault_level, row.recovery_config, row.metrics.task_success_rate, row.metrics.n
        )
        for row in iid_result.rows
    ]
    recovery_rate_map = {
        (row.recovery_config, row.fault_level): row.metrics.recovery_rate for row in iid_result.rows
    }
    outage_summary = await run_sustained_outage_summary()
    escalation_summary = await run_escalation_summary()
    routing_only_summary = run_routing_only_degradation()

    db_path = RESULTS_DIR / "phase_f_experiment.duckdb"
    db_path.unlink(missing_ok=True)
    with ExperimentStore(db_path) as store:
        baseline_run_id, candidate_run_id = await run_store_backed_slice(store)

        print("\n=== Regression check: A-no_recovery (baseline) vs E-full_policy (candidate) ===")
        policy = RegressionPolicy(
            metrics=(MetricRegressionPolicy("success_rate", max_practical_change=0.0),)
        )
        regression_result = check_regression(store, baseline_run_id, candidate_run_id, policy)
        print(
            f"  outcome={regression_result.outcome.value} exit_code={regression_result.exit_code}"
        )
        for m in regression_result.metric_results:
            print(f"  {m.metric_name}: {m.reason}")

        print("\n=== Generating report for E-full_policy vs. A-no_recovery baseline ===")
        report = generate_report(store, candidate_run_id, baseline_run_id=baseline_run_id)
        # Explicit encoding="utf-8": Path.write_text defaults to
        # locale.getpreferredencoding() (cp1252 on this Windows machine),
        # which cannot represent the em dashes render_markdown/render_json
        # embed (e.g. the "descriptive only -- not a causal claim" note) and
        # silently produces an invalid-UTF-8 file instead of raising.
        (RESULTS_DIR / "report.md").write_text(render_markdown(report), encoding="utf-8")
        (RESULTS_DIR / "report.json").write_text(
            json.dumps(render_json(report), indent=2, default=str), encoding="utf-8"
        )
        print(f"  wrote {RESULTS_DIR / 'report.md'} and report.json")

        print("\n=== Generating charts ===")
        chart_results = {}
        r1 = plot_resilience_curve(
            iid_points, output_path=RESULTS_DIR / "charts" / "resilience_curve.png"
        )
        if r1:
            chart_results["resilience_curve"] = r1.path.name

        r2 = plot_recovery_rate_by_level(
            iid_points, recovery_rate_map, output_path=RESULTS_DIR / "charts" / "recovery_rate.png"
        )
        if r2:
            chart_results["recovery_rate_by_level"] = r2.path.name

        if report.failure_taxonomy:
            r3 = plot_failure_taxonomy(
                {e.category: e.count for e in report.failure_taxonomy},
                output_path=RESULTS_DIR / "charts" / "failure_taxonomy.png",
                run_id=candidate_run_id,
            )
            if r3:
                chart_results["failure_taxonomy"] = r3.path.name

        if report.recovery_summary is not None:
            action_counts: dict[str, int] = {}
            for event in store.get_outcome_events(candidate_run_id):
                for action in event.recovery_actions:
                    action_counts[action] = action_counts.get(action, 0) + 1
            r4 = plot_selection_distribution(
                action_counts,
                output_path=RESULTS_DIR / "charts" / "recovery_actions.png",
                title="Recovery action distribution (E-full_policy)",
            )
            if r4:
                chart_results["recovery_action_distribution"] = r4.path.name

        if report.comparison is not None:
            r5 = plot_baseline_vs_candidate(
                report.comparison, output_path=RESULTS_DIR / "charts" / "baseline_vs_candidate.png"
            )
            chart_results["baseline_vs_candidate"] = r5.path.name

        print(f"  wrote {len(chart_results)} charts to {RESULTS_DIR / 'charts'}")

    summary = {
        "label": "SIMULATION",
        "iid_fault_curve": [
            {
                "fault_level": p.fault_level,
                "config": p.config_name,
                "success_rate": p.success_rate,
                "n": p.n,
            }
            for p in iid_points
        ],
        "sustained_outage": outage_summary,
        "escalation": escalation_summary,
        "routing_only_degradation": routing_only_summary,
        "regression_check": {
            "outcome": regression_result.outcome.value,
            "exit_code": regression_result.exit_code,
        },
        "charts": chart_results,
    }
    (RESULTS_DIR / "flagship_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nPhase F flagship analysis complete. Raw data in: {RESULTS_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
