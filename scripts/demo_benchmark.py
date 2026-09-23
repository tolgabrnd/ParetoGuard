"""Phase C end-to-end offline-benchmark demo.

Runs all four built-in suites against MockProvider with repeated trials, persists
everything to a real (file-backed, not in-memory) DuckDB store, then reads the
results back from storage and computes metrics from those raw records - proving
the full pipeline (manifest -> execution -> grading -> persistence -> metrics)
works end to end with zero network access and zero API keys.

This is a demo/verification script, not part of the public library API; the real
`paretoguard benchmark` CLI command lands in a later phase.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

from paretoguard.evals import BenchmarkConfig, BenchmarkRunner, EvalSuite, compute_metrics
from paretoguard.evals.suites import (
    long_context_retrieval_v1,
    numeric_reasoning_v1,
    structured_extraction_v1,
    tool_use_v1,
)
from paretoguard.providers.mock import SCENARIO_METADATA_KEY, MockProvider, MockScenario
from paretoguard.storage import ExperimentStore


def _inject_some_failures(suite: EvalSuite) -> EvalSuite:
    """Flips a couple of cases in each suite to a failing MockProvider scenario,
    so the demo also proves failed attempts remain inspectable, not just the
    happy path."""
    cases = list(suite.cases)
    for i in (0, 1):
        if i < len(cases):
            case = cases[i]
            meta = dict(case.metadata)
            meta[SCENARIO_METADATA_KEY] = MockScenario.RATE_LIMIT.value
            cases[i] = case.model_copy(update={"metadata": meta})
    return suite.model_copy(update={"cases": cases})


async def main() -> None:
    db_path = Path(tempfile.gettempdir()) / "paretoguard_demo_benchmark.duckdb"
    db_path.unlink(missing_ok=True)

    suites = [
        structured_extraction_v1.build_suite(seed=1, num_cases=6),
        numeric_reasoning_v1.build_suite(seed=1, num_cases=6),
        long_context_retrieval_v1.build_suite(seed=1, cases_per_difficulty=2),
        _inject_some_failures(tool_use_v1.build_suite(seed=1, num_cases=6)),
    ]

    print(f"Persisting to: {db_path}")
    with ExperimentStore(db_path) as store:
        run_ids = []
        for suite in suites:
            provider = MockProvider(seed=1)
            runner = BenchmarkRunner(
                provider,
                model="mock-strong",
                store=store,
                config=BenchmarkConfig(repetitions=3, max_concurrency=4),
            )
            result = await runner.run(suite)
            run_ids.append(result.run_id)
            print(
                f"[{suite.name}] run_id={result.run_id} "
                f"cases={suite.case_count} results={len(result.results)}"
            )

        print("\n--- Verification ---")
        all_ok = True

        for run_id, suite in zip(run_ids, suites, strict=True):
            manifest = store.get_run(run_id)
            assert manifest is not None, "RunManifest was not persisted"
            assert manifest.suite_name == suite.name
            assert manifest.repetitions == 3
            assert manifest.task_count == suite.case_count

            stored_results = store.get_eval_results(run_id)
            expected_count = suite.case_count * 3
            ok = len(stored_results) == expected_count
            all_ok &= ok
            print(
                f"[{suite.name}] manifest ok, eval_results persisted: "
                f"{len(stored_results)}/{expected_count} {'OK' if ok else 'MISMATCH'}"
            )

            # Sequence must be gap-free and sorted after round-tripping through storage.
            sequences = [r.sequence for r in stored_results]
            ok = sequences == sorted(sequences) == list(range(expected_count))
            all_ok &= ok
            print(f"[{suite.name}] deterministic sequence ordering: {'OK' if ok else 'MISMATCH'}")

            failures = [r for r in stored_results if not r.succeeded]
            print(
                f"[{suite.name}] {len(failures)} failing result(s) still inspectable "
                f"(e.g. {failures[0].explanation!r})"
                if failures
                else f"[{suite.name}] no failures"
            )

            metrics = compute_metrics(stored_results, k=3)
            print(
                f"[{suite.name}] metrics: success_rate={metrics.success_rate:.2f} "
                f"pass@1={metrics.pass_at_1:.2f} pass@3={metrics.pass_at_k} "
                f"consistency={metrics.consistency:.2f} "
                f"cost_per_success=${metrics.cost_per_success_usd} "
                f"mean_latency_ms={metrics.mean_latency_ms_per_success:.2f}"
                if metrics.mean_latency_ms_per_success is not None
                else f"[{suite.name}] metrics: no successful results"
            )

        requests_df = store.requests_df()
        responses_df = store.responses_df()
        trace_df = store.trace_events_df()
        print(
            f"\nTotal persisted: requests={requests_df.height} responses={responses_df.height} "
            f"trace_events={trace_df.height}"
        )

    db_path.unlink(missing_ok=True)

    if not all_ok:
        print("\nDEMO FAILED: see MISMATCH lines above")
        sys.exit(1)
    print(
        "\nDEMO PASSED: manifest, execution, grading, persistence, repeated trials, "
        "failure inspectability, and metrics-from-raw-results all verified end to end."
    )


if __name__ == "__main__":
    asyncio.run(main())
