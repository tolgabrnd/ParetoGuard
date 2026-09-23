"""Phase D end-to-end offline routing experiment.

**SIMULATION ONLY.** Every candidate model here is a synthetic profile
(`paretoguard.providers.profiled_mock.simulated_profile_suite`) with
invented success/cost/latency — never a real provider. Every number this
script prints or writes is a simulated/offline result and must be read as
such (see CLAUDE.md: "never fabricate benchmark results... or reliability
numbers" — this generates synthetic ones deliberately and labels them).

Two independent experiments, both against the same 3 simulated profiles
(profile-a/b/c — see `simulated_profile_suite`'s docstring for their
deliberately non-dominant trade-offs):

1. **Routing comparison**: builds an exhaustive offline training matrix
   (`MatrixRunner`, every task x every candidate) over all four built-in
   eval suites, splits it group-aware train/val/test, trains the learned
   routers on train, and evaluates every router's counterfactual
   performance on the held-out test split (`evaluate_router_offline`) —
   never on training rows. Compares: RuleRouter, ParetoRouter,
   ReliabilityAwareRouter, LearnedRouter (logistic regression, random
   forest), TorchRouter (if PyTorch is installed), the strongest/cheapest
   fixed profile, and the oracle upper bound (NON-DEPLOYABLE, analysis
   only — see `oracle_upper_bound`'s docstring).

2. **Reliability degradation/recovery**: the deterministic simulation from
   Commit 20 (`run_degradation_recovery_simulation`), comparing
   StaticRouter/ParetoRouter/ReliabilityAwareRouter's reroute behavior.

Raw results are persisted through the normal experiment pipeline
(`ExperimentStore`, via `MatrixRunner(store=...)`) and this script's own
summary tables are written to `scripts/phase_d_results/` as CSV. Chart/
report generation remains Phase F — this script only guarantees the raw
data exists, is reproducible (fixed seeds throughout), and is generated
from actual execution, never hard-coded.
"""

import asyncio
from datetime import date
from pathlib import Path

import polars as pl

from paretoguard.core.config import PricingTable
from paretoguard.core.models import CostBasis, ModelSpec, PricingEntry
from paretoguard.evals.matrix import MatrixConfig, MatrixRunner
from paretoguard.evals.models import EvalSuite
from paretoguard.evals.suites import (
    long_context_retrieval_v1,
    numeric_reasoning_v1,
    structured_extraction_v1,
    tool_use_v1,
)
from paretoguard.providers.profiled_mock import ProfiledMockProvider, simulated_profile_suite
from paretoguard.routing import (
    LearnedRouter,
    ParetoObjective,
    ParetoRouter,
    ReliabilityAwareRouter,
    RuleRouter,
    StaticRouter,
    best_fixed_model,
    build_dataset,
    build_profiles_from_train_split,
    cheapest_fixed_model,
    evaluate_calibration,
    evaluate_router_offline,
    fixed_model_stats,
    oracle_upper_bound,
    run_degradation_recovery_simulation,
    train_learned_router_model,
    two_model_degradation_schedule,
)
from paretoguard.storage import ExperimentStore

RESULTS_DIR = Path(__file__).resolve().parent / "phase_d_results"
PRICING_VERSION = "sim-2026-09-23"


def _tag_task_family(suite: EvalSuite) -> EvalSuite:
    """Built-in suites don't set `task_family` metadata themselves (they
    weren't written with routing in mind) — the task family is exactly the
    suite name, so this tags every case with it, additively."""
    cases = [
        case.model_copy(update={"metadata": {**case.metadata, "task_family": suite.name}})
        for case in suite.cases
    ]
    return suite.model_copy(update={"cases": cases})


def _combined_suite() -> EvalSuite:
    suites = [
        _tag_task_family(structured_extraction_v1.build_suite(seed=1, num_cases=15)),
        _tag_task_family(numeric_reasoning_v1.build_suite(seed=1, num_cases=15)),
        _tag_task_family(long_context_retrieval_v1.build_suite(seed=1, cases_per_difficulty=5)),
        _tag_task_family(tool_use_v1.build_suite(seed=1, num_cases=15)),
    ]
    all_cases = [case for suite in suites for case in suite.cases]
    return EvalSuite(
        name="phase_d_combined",
        version="1.0.0",
        description="SIMULATION: combined Phase D routing-comparison suite.",
        seed=1,
        cases=all_cases,
    )


def _candidates() -> list[ModelSpec]:
    return [
        ModelSpec(
            name=name,
            provider="sim",
            context_window=128_000,
            supports_tools=True,
            supports_structured_output=True,
        )
        for name in simulated_profile_suite()
    ]


def _pricing_table() -> PricingTable:
    """SIMULATED prices, deliberately matching each profile's described
    cost tier (profile-a: expensive, profile-b: cheap, profile-c: medium) —
    invented for this experiment, never real provider prices (see
    CostBasis.SIMULATED and CLAUDE.md's pricing rules)."""
    today = date(2026, 9, 23)
    tiers = {"profile-a": (15.0, 60.0), "profile-b": (0.5, 1.5), "profile-c": (3.0, 12.0)}
    return PricingTable(
        entries=[
            PricingEntry(
                provider="sim",
                model=model,
                input_price_per_million_usd=input_price,
                output_price_per_million_usd=output_price,
                version=PRICING_VERSION,
                effective_date=today,
                basis=CostBasis.SIMULATED,
            )
            for model, (input_price, output_price) in tiers.items()
        ]
    )


async def run_routing_comparison() -> None:
    suite = _combined_suite()
    candidates = _candidates()
    provider = ProfiledMockProvider(simulated_profile_suite(), seed=7)
    pricing = _pricing_table()

    print(f"[SIMULATION] {suite.case_count} tasks x {len(candidates)} candidates x 3 repetitions")
    db_path = RESULTS_DIR / "phase_d_experiment.duckdb"
    db_path.unlink(missing_ok=True)

    with ExperimentStore(db_path) as store:
        matrix_runner = MatrixRunner(
            candidates,
            {"sim": provider},
            config=MatrixConfig(repetitions=3, max_concurrency=8),
            pricing_table=pricing,
            store=store,
        )
        matrix_result = await matrix_runner.run(suite)
        print(f"Matrix run_id={matrix_result.run_id} rows={len(matrix_result.rows)}")

        df = build_dataset(matrix_result.rows, seed=0)
        split_counts = df["split"].value_counts().sort("split")
        print(f"Dataset rows: {df.height}, splits: {split_counts.to_dicts()}")
        df.write_csv(RESULTS_DIR / "training_matrix.csv")

        train_profiles = build_profiles_from_train_split(df)

        logreg_model = train_learned_router_model(df, model_kind="logistic_regression", seed=0)
        rf_model = train_learned_router_model(df, model_kind="random_forest", seed=0)
        calibration_rows = [
            _calibration_row(
                "logistic_regression", evaluate_calibration(logreg_model, df, split="test")
            ),
            _calibration_row("random_forest", evaluate_calibration(rf_model, df, split="test")),
        ]

        routers: list = [
            RuleRouter(),
            ParetoRouter(ParetoObjective.MAXIMIZE_SUCCESS),
            ReliabilityAwareRouter(),
        ]
        learned_logreg = LearnedRouter(logreg_model)
        learned_logreg.name = "learned_logistic_regression"
        routers.append(learned_logreg)
        learned_rf = LearnedRouter(rf_model)
        learned_rf.name = "learned_random_forest"
        routers.append(learned_rf)

        try:
            from paretoguard.routing.torch_router import TorchRouter, train_torch_router_model

            torch_model = train_torch_router_model(df, epochs=150, seed=0)
            calibration_rows.append(
                _calibration_row("torch_mlp", evaluate_calibration(torch_model, df, split="test"))
            )
            torch_router = TorchRouter(torch_model)
            torch_router.name = "learned_torch_mlp"
            routers.append(torch_router)
        except ImportError:
            print("[SKIP] torch not installed; TorchRouter excluded from this run")

        pl.DataFrame(calibration_rows).write_csv(RESULTS_DIR / "calibration_results.csv")
        print("\n=== Calibration (SIMULATION, test split, NEVER training rows) ===")
        for row in calibration_rows:
            print(row)

        summaries = []
        selection_rows = []
        for router in routers:
            try:
                summary = evaluate_router_offline(
                    router, df, candidates, split="test", profiles=train_profiles
                )
            except Exception as exc:
                print(f"[WARN] router {router.name!r} failed to evaluate: {exc}")
                continue
            summaries.append(summary)
            for model_name, count in summary.selection_distribution.items():
                selection_rows.append({"router": router.name, "model": model_name, "count": count})

        best_model = best_fixed_model(df, split="test")
        cheapest_model = cheapest_fixed_model(df, split="test")
        stats = fixed_model_stats(df, split="test")
        oracle = oracle_upper_bound(df, candidates, split="test")
        stats.write_csv(RESULTS_DIR / "fixed_model_stats.csv")
        pl.DataFrame(selection_rows).write_csv(RESULTS_DIR / "selection_distribution.csv")

        print("\n=== Routing summary (SIMULATION, held-out test split) ===")
        summary_rows = []
        for s in summaries:
            print(
                f"{s.router_name:30s} success={s.success_rate:.3f} "
                f"cost={s.mean_cost_usd} latency_ms={s.mean_latency_ms:.1f} "
                f"regret_vs_oracle={s.regret_vs_oracle:.3f} n={s.n_tasks} "
                f"selections={s.selection_distribution}"
            )
            summary_rows.append(
                {
                    "router": s.router_name,
                    "success_rate": s.success_rate,
                    "mean_cost_usd": s.mean_cost_usd,
                    "mean_latency_ms": s.mean_latency_ms,
                    "regret_vs_oracle": s.regret_vs_oracle,
                    "n_tasks": s.n_tasks,
                    "label": "SIMULATION",
                }
            )
        print(f"best_fixed_profile (by test success)   = {best_model!r}")
        print(f"cheapest_fixed_profile (by test cost)  = {cheapest_model!r}")
        print(f"oracle_upper_bound (NON-DEPLOYABLE)    = {oracle:.3f}")
        summary_rows.append(
            {
                "router": "oracle_NON_DEPLOYABLE",
                "success_rate": oracle,
                "mean_cost_usd": None,
                "mean_latency_ms": None,
                "regret_vs_oracle": 0.0,
                "n_tasks": None,
                "label": "SIMULATION-ANALYSIS-ONLY-NEVER-DEPLOY",
            }
        )
        pl.DataFrame(summary_rows).write_csv(RESULTS_DIR / "routing_summary.csv")


def _calibration_row(model_kind: str, metrics: object) -> dict:
    return {
        "model_kind": model_kind,
        "brier_score": metrics.brier_score,  # type: ignore[attr-defined]
        "n": metrics.n,  # type: ignore[attr-defined]
        "mean_predicted_success": metrics.mean_predicted_success,  # type: ignore[attr-defined]
        "mean_actual_success": metrics.mean_actual_success,  # type: ignore[attr-defined]
        "label": "SIMULATION",
    }


def run_reliability_experiment() -> None:
    print("\n=== Reliability degradation/recovery experiment (SIMULATION) ===")
    candidates = [
        ModelSpec(name="primary-sim", provider="mock", context_window=100_000),
        ModelSpec(name="stable-sim", provider="mock", context_window=100_000),
    ]
    from paretoguard.routing.types import CandidateProfile, candidate_key

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

    rows = []
    for router in [
        StaticRouter(provider="mock", model="primary-sim"),
        ParetoRouter(ParetoObjective.MAXIMIZE_SUCCESS),
        ReliabilityAwareRouter(),
    ]:
        summary = run_degradation_recovery_simulation(
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
        print(
            f"{summary.router_name:15s} task_success={summary.task_success_rate:.3f} "
            f"reroutes={summary.reroute_count} time_to_detect={summary.time_to_detect_degradation} "
            f"time_to_recover={summary.time_to_recover} "
            f"unnecessary_reroute_rate={summary.unnecessary_reroute_rate} "
            f"[{summary.label}]"
        )
        rows.append(
            {
                "router": summary.router_name,
                "task_success_rate": summary.task_success_rate,
                "reroute_count": summary.reroute_count,
                "reroute_frequency": summary.reroute_frequency,
                "time_to_detect_degradation": summary.time_to_detect_degradation,
                "time_to_recover": summary.time_to_recover,
                "unnecessary_reroute_count": summary.unnecessary_reroute_count,
                "unnecessary_reroute_rate": summary.unnecessary_reroute_rate,
                "mean_cost_usd": summary.mean_cost_usd,
                "mean_latency_ms": summary.mean_latency_ms,
                "label": summary.label,
            }
        )
    pl.DataFrame(rows).write_csv(RESULTS_DIR / "reliability_experiment_summary.csv")


async def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    await run_routing_comparison()
    run_reliability_experiment()
    print(f"\nPhase D experiment complete. Raw data in: {RESULTS_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
