"""Phase E end-to-end resilience/recovery experiment.

**SIMULATION ONLY.** Every failure this script observes is a synthetic,
injected `chaos.FaultInjector` fault against `MockProvider` — never a real
provider's actual reliability (see CLAUDE.md: never fabricate reliability
numbers). Every number this script prints or writes is a simulated/offline
result and must be read as such.

Two independent experiments:

1. **Flagship recovery comparison** (`routing.resilience_benchmark`): the
   five recovery configs (A: no recovery, B: retry-only, C: retry+fallback,
   D: +circuit-breaker, E: +escalation) against `resilience_v1` at fault
   levels 0%/5%/15%/30% — see that module's docstring for the full
   experimental design and its stated limitations (i.i.d. fault model,
   zero expected escalation rate for this particular suite).

2. **structured_agent_v1**: the multi-step tool-use agent suite run once
   through `AgentSimulator`, reporting the required "average tool calls"
   metric this repo has nowhere else to source from (`OutcomeEvent` has no
   tool-call concept — see `evals.metrics.mean_tool_call_count`).

Raw results are persisted through the normal experiment pipeline
(`ExperimentStore`, via `run_resilience_benchmark(..., store=...)`) and this
script's own summary tables are written to `scripts/phase_e_results/` as
CSV. This script is also the determinism-verification harness the Phase E
spec requires: `--verify-determinism` runs the flagship comparison twice
from a fresh process each time and diffs every field except latency
(real wall-clock time — see `test_resilience_benchmark.py`'s
`test_determinism_two_runs_produce_identical_metrics` for why that one
field is excluded).
"""

import asyncio
import json
import sys
from pathlib import Path

import polars as pl

from paretoguard.agents.executor import AgentExecutor, AgentLimits
from paretoguard.agents.simulator import AgentSimulator
from paretoguard.agents.tools import default_tools
from paretoguard.evals.metrics import mean_tool_call_count
from paretoguard.evals.suites import resilience_v1, structured_agent_v1
from paretoguard.routing.resilience_benchmark import run_resilience_benchmark
from paretoguard.runtime import RetryPolicy, Runtime
from paretoguard.storage import ExperimentStore

RESULTS_DIR = Path(__file__).resolve().parent / "phase_e_results"
_LATENCY_FIELDS = {"median_latency_ms", "p95_latency_ms"}


def _row_dict(row) -> dict:  # type: ignore[no-untyped-def]
    d = {"fault_level": row.fault_level, "recovery_config": row.recovery_config}
    d.update(row.metrics.model_dump())
    # CSV has no nested-value support; the distribution is still full-fidelity
    # in the DuckDB store this script also writes (outcome_events table).
    d["unrecovered_failure_distribution"] = json.dumps(d["unrecovered_failure_distribution"])
    d["label"] = "SIMULATION"
    return d


async def run_flagship_comparison() -> list[dict]:
    print("\n=== Flagship recovery comparison (SIMULATION) ===")
    suite = resilience_v1.build_suite(seed=0, num_cases=24)
    print(f"[SIMULATION] {suite.case_count} tasks x 5 recovery configs x 4 fault levels")

    db_path = RESULTS_DIR / "phase_e_experiment.duckdb"
    db_path.unlink(missing_ok=True)

    with ExperimentStore(db_path) as store:
        result = await run_resilience_benchmark(suite, seed=0, store=store)

    rows = [_row_dict(row) for row in result.rows]
    pl.DataFrame(rows).write_csv(RESULTS_DIR / "resilience_comparison.csv")

    for row in result.rows:
        m = row.metrics
        print(
            f"level={row.fault_level:<5.2f} {row.recovery_config:<35s} "
            f"success={m.task_success_rate:.3f} raw_fail={m.raw_failure_rate:.3f} "
            f"recovery={m.recovery_rate} retry={m.retry_rate:.2f} "
            f"fallback={m.fallback_rate:.2f} escalate={m.escalation_rate:.2f} "
            f"avg_attempts={m.average_attempts:.2f} n={m.n} [SIMULATION]"
        )
    return rows


async def run_structured_agent_suite() -> dict:
    print("\n=== structured_agent_v1 (SIMULATION) ===")
    tasks = structured_agent_v1.build_tasks()
    provider = structured_agent_v1.ScriptedAgentProvider()
    runtime = Runtime(provider, retry_policy=RetryPolicy(max_attempts=1))
    executor = AgentExecutor(runtime, default_tools(), limits=AgentLimits())
    simulator = AgentSimulator(executor, provider="mock", model="m")

    results = await simulator.run(tasks)
    trajectories = [r.trajectory for r in results]
    n_succeeded = sum(r.outcome.succeeded for r in results)
    avg_tool_calls = mean_tool_call_count(trajectories)

    summary = {
        "n_tasks": len(results),
        "n_succeeded": n_succeeded,
        "success_rate": n_succeeded / len(results),
        "average_tool_calls": avg_tool_calls,
        "label": "SIMULATION",
    }
    print(
        f"n={summary['n_tasks']} succeeded={n_succeeded} "
        f"success_rate={summary['success_rate']:.3f} "
        f"avg_tool_calls={avg_tool_calls:.2f} [SIMULATION]"
    )
    for r in results:
        status = "OK" if r.outcome.succeeded else "FAIL"
        print(f"  [{status}] {r.task.task_id}: {r.outcome.termination_reason.value}")

    pl.DataFrame([summary]).write_csv(RESULTS_DIR / "structured_agent_summary.csv")
    return summary


def _stable_comparison_rows(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in row.items() if k not in _LATENCY_FIELDS} for row in rows]


async def verify_determinism() -> bool:
    """Runs the flagship comparison twice, independently, and diffs every
    field except latency. Explicitly VERIFIES (never assumes) determinism —
    the Phase D audit's own finding (a random-UUID4 default that silently
    broke reproducibility) is exactly why this script does not just trust
    the seeding discipline and print a result."""
    print("\n=== Determinism verification (run twice, diff) ===")
    suite = resilience_v1.build_suite(seed=0, num_cases=24)
    result_a = await run_resilience_benchmark(suite, seed=0)
    result_b = await run_resilience_benchmark(suite, seed=0)

    stable_a = _stable_comparison_rows([_row_dict(r) for r in result_a.rows])
    stable_b = _stable_comparison_rows([_row_dict(r) for r in result_b.rows])

    if stable_a == stable_b:
        print(
            f"DETERMINISTIC: {len(stable_a)} rows identical across two independent runs "
            "(every field except wall-clock latency)."
        )
        return True

    print("NON-DETERMINISTIC: mismatch found between the two runs:")
    for i, (a, b) in enumerate(zip(stable_a, stable_b, strict=True)):
        if a != b:
            print(f"  row {i}: {a}")
            print(f"       vs {b}")
    return False


async def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    verify_only = "--verify-determinism" in sys.argv

    if verify_only:
        ok = await verify_determinism()
        (RESULTS_DIR / "determinism_check.json").write_text(
            json.dumps({"deterministic": ok}, indent=2)
        )
        sys.exit(0 if ok else 1)

    await run_flagship_comparison()
    await run_structured_agent_suite()
    ok = await verify_determinism()
    (RESULTS_DIR / "determinism_check.json").write_text(json.dumps({"deterministic": ok}, indent=2))
    print(f"\nPhase E experiment complete. Raw data in: {RESULTS_DIR}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
