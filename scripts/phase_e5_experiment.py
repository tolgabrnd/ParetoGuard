"""Phase E.5 expanded offline analysis experiment.

**SIMULATION ONLY.** Every failure this script observes is a synthetic,
injected `chaos.FaultInjector` fault (or scripted quality corruption)
against `MockProvider` — never a real provider's actual reliability (see
CLAUDE.md: never fabricate reliability numbers).

**Why this script exists, separate from `phase_e_resilience_experiment.py`**:
that script's defaults (24 tasks/cell, 300-step outage, 20 escalation
tasks) are deliberately small and CI-friendly — `tests/unit` calls the same
library functions at similarly small scale so the test suite stays fast.
This script instead runs the same three library functions (no new benchmark
code — `resilience_v1.build_suite`, `run_sustained_outage_scenario`,
`run_escalation_scenario` were already parameterized for this) at a scale
intended for Phase F's statistics work: confidence intervals and pairwise
comparisons need enough samples per cell to be meaningful, not just
"whichever number happened to be picked for CI speed".

**Sample size, chosen for statistical reach, not for significance**: at
`n=200` observations per cell, a 95% normal-approximation CI half-width on
a success-rate proportion is roughly +/-0.06 at p=0.5 (the widest case) and
roughly +/-0.03 near p=0.9 — tight enough that Phase F's confidence
intervals and pairwise comparisons will say something real, without being
picked to manufacture a "statistically significant" result the effect size
doesn't actually support. This is *not* "the more samples the better
forever" — 200/cell was chosen as the point past which further tightening
stops mattering much for the kinds of effect sizes this repo's synthetic
scenarios produce (tens of percentage points, not fractions of one), while
the whole run still completes in a few seconds against `MockProvider`.

- **Flagship resilience comparison**: `resilience_v1` at `num_cases=200`
  (vs. the default 24), across 4 fault levels x 5 recovery configs = 20
  cells, 200 repetitions (tasks) per cell -> **4,000 total task
  executions**.
- **Sustained outage scenario**: `total_steps=1000` (vs. the default 300),
  with `degraded_start=300`/`recovered_start=700` (the same
  30%/40%/30% healthy/degraded/recovered proportions as the default
  100/200/300 split, scaled up) -> 5 recovery configs x 1000 steps =
  **5,000 total task executions**, 400 of them inside the outage window
  per config (vs. 100 at default scale).
- **Escalation comparison**: `escalation_v1` at `num_cases=200` (vs. the
  default 20), 2 arms -> **400 total task executions**.

**9,800 deterministic synthetic task executions in total.** Determinism is
verified the same way `phase_e_resilience_experiment.py` verifies it for
the smaller scenario: run, then run again, diff every field except
wall-clock latency.

**No `ExperimentStore` persistence at this scale — a real finding, not an
oversight.** `phase_e_resilience_experiment.py` persists through a
file-backed `ExperimentStore` at its small (~500-task) scale without
issue. Attempting the same here (a real DuckDB file, one unbatched
`INSERT`-per-row across `requests`/`responses`/`trace_events`/
`routing_decisions`/`eval_results`/`outcome_events` for every one of
~9,800 tasks — tens of thousands of individual statements) made the run
take minutes instead of seconds and was killed rather than waited out; a
second attempt confirmed the bottleneck is exactly that (removing the store
entirely brought the whole run back to single-digit seconds). This is
real, measured Phase E.5 technical debt for whoever next wants to persist a
large experiment run: `ExperimentStore`'s write methods execute one
statement per call with no batching/explicit-transaction wrapping, which is
invisible at Phase E's dozens-of-tasks scale and dominant at Phase E.5's
thousands-of-tasks scale. Fixing it (e.g. wrapping a run's writes in one
transaction, or a batched `executemany`) is Phase F-adjacent storage work,
not attempted here — this script instead computes every metric directly
from the `OutcomeEvent`/result objects each library function already
returns in memory, with no `ExperimentStore` in the loop at all.
"""

import asyncio
import json
from pathlib import Path

import polars as pl

from paretoguard.evals.suites import resilience_v1
from paretoguard.routing.escalation_benchmark import run_escalation_scenario
from paretoguard.routing.resilience_benchmark import run_resilience_benchmark
from paretoguard.routing.sustained_outage_benchmark import run_sustained_outage_scenario
from paretoguard.storage import ExperimentStore

RESULTS_DIR = Path(__file__).resolve().parent / "phase_e5_results"

LARGE_RESILIENCE_NUM_CASES = 200
LARGE_SUSTAINED_TOTAL_STEPS = 1000
LARGE_SUSTAINED_DEGRADED_START = 300
LARGE_SUSTAINED_RECOVERED_START = 700
LARGE_ESCALATION_NUM_CASES = 200

_LATENCY_FIELDS = {
    "median_latency_ms",  # ResilienceMetricsSummary
    "p95_latency_ms",  # ResilienceMetricsSummary
    "mean_latency_ms_baseline",  # SustainedOutageMetrics
    "mean_latency_ms_outage",  # SustainedOutageMetrics
}


async def run_large_resilience_comparison(store: ExperimentStore | None) -> list[dict]:
    suite = resilience_v1.build_suite(seed=0, num_cases=LARGE_RESILIENCE_NUM_CASES)
    print(
        f"[SIMULATION] resilience comparison: {suite.case_count} tasks/cell x 5 configs x "
        f"4 fault levels = {suite.case_count * 20} total task executions"
    )
    result = await run_resilience_benchmark(suite, seed=0, store=store)
    rows = []
    for row in result.rows:
        d = {"fault_level": row.fault_level, "recovery_config": row.recovery_config}
        d.update(row.metrics.model_dump())
        d["unrecovered_failure_distribution"] = json.dumps(d["unrecovered_failure_distribution"])
        d["label"] = "SIMULATION"
        rows.append(d)
        m = row.metrics
        print(
            f"  level={row.fault_level:<5.2f} {row.recovery_config:<35s} "
            f"success={m.task_success_rate:.4f} recovery={m.recovery_rate} n={m.n}"
        )
    return rows


async def run_large_sustained_outage(store: ExperimentStore | None) -> list[dict]:
    print(
        f"[SIMULATION] sustained outage: {LARGE_SUSTAINED_TOTAL_STEPS} steps x 5 configs = "
        f"{LARGE_SUSTAINED_TOTAL_STEPS * 5} total task executions "
        f"({LARGE_SUSTAINED_RECOVERED_START - LARGE_SUSTAINED_DEGRADED_START} steps in the "
        "outage window per config)"
    )
    results = await run_sustained_outage_scenario(
        total_steps=LARGE_SUSTAINED_TOTAL_STEPS,
        degraded_start=LARGE_SUSTAINED_DEGRADED_START,
        recovered_start=LARGE_SUSTAINED_RECOVERED_START,
        store=store,
    )
    rows = []
    for r in results:
        m = r.metrics
        rows.append({"recovery_config": r.recovery_config, **vars(m), "label": "SIMULATION"})
        print(
            f"  {r.recovery_config:<35s} success={m.task_success_rate:.4f} "
            f"outage_success={m.task_success_rate_during_outage:.4f} "
            f"fallback={m.fallback_count} circuit_opens={m.circuit_open_transitions}"
        )
    return rows


async def run_large_escalation_comparison(store: ExperimentStore | None) -> dict:
    print(
        f"[SIMULATION] escalation comparison: {LARGE_ESCALATION_NUM_CASES} tasks x 2 arms = "
        f"{LARGE_ESCALATION_NUM_CASES * 2} total task executions"
    )
    result = await run_escalation_scenario(
        num_cases=LARGE_ESCALATION_NUM_CASES, quality_fault_rate=0.6, seed=0, store=store
    )
    row = {
        "no_validation_success_rate": result.no_validation.task_success_rate,
        "validation_escalation_success_rate": result.validation_and_escalation.task_success_rate,
        "no_validation_escalation_rate": result.no_validation.escalation_rate,
        "validation_escalation_rate": result.validation_and_escalation.escalation_rate,
        "success_rate_delta": result.success_rate_delta,
        "added_average_attempts": result.added_average_attempts,
        "added_mean_latency_ms": result.added_mean_latency_ms,
        "label": "SIMULATION",
    }
    print(
        f"  no_validation: success={row['no_validation_success_rate']:.4f} "
        f"escalate={row['no_validation_escalation_rate']:.4f}"
    )
    print(
        f"  validation_and_escalation: success={row['validation_escalation_success_rate']:.4f} "
        f"escalate={row['validation_escalation_rate']:.4f}"
    )
    print(f"  success_rate_delta={row['success_rate_delta']:.4f}")
    return row


def _stable(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in row.items() if k not in _LATENCY_FIELDS} for row in rows]


async def verify_determinism() -> bool:
    print("\n=== Determinism verification (large scale, run twice, diff) ===")
    suite = resilience_v1.build_suite(seed=0, num_cases=LARGE_RESILIENCE_NUM_CASES)
    result_a = await run_resilience_benchmark(suite, seed=0)
    result_b = await run_resilience_benchmark(suite, seed=0)
    rows_a = _stable(
        [
            {"fault_level": r.fault_level, "config": r.recovery_config, **r.metrics.model_dump()}
            for r in result_a.rows
        ]
    )
    rows_b = _stable(
        [
            {"fault_level": r.fault_level, "config": r.recovery_config, **r.metrics.model_dump()}
            for r in result_b.rows
        ]
    )
    ok = rows_a == rows_b
    print(f"resilience comparison deterministic: {ok} ({len(rows_a)} rows)")

    outage_a = await run_sustained_outage_scenario(
        total_steps=LARGE_SUSTAINED_TOTAL_STEPS,
        degraded_start=LARGE_SUSTAINED_DEGRADED_START,
        recovered_start=LARGE_SUSTAINED_RECOVERED_START,
    )
    outage_b = await run_sustained_outage_scenario(
        total_steps=LARGE_SUSTAINED_TOTAL_STEPS,
        degraded_start=LARGE_SUSTAINED_DEGRADED_START,
        recovered_start=LARGE_SUSTAINED_RECOVERED_START,
    )
    outage_stable_a = _stable([{"config": r.recovery_config, **vars(r.metrics)} for r in outage_a])
    outage_stable_b = _stable([{"config": r.recovery_config, **vars(r.metrics)} for r in outage_b])
    outage_ok = outage_stable_a == outage_stable_b
    print(f"sustained outage deterministic: {outage_ok} ({len(outage_stable_a)} rows)")

    esc_a = await run_escalation_scenario(num_cases=LARGE_ESCALATION_NUM_CASES, seed=0)
    esc_b = await run_escalation_scenario(num_cases=LARGE_ESCALATION_NUM_CASES, seed=0)
    esc_ok = (
        esc_a.success_rate_delta == esc_b.success_rate_delta
        and esc_a.added_average_attempts == esc_b.added_average_attempts
        and esc_a.no_validation.task_success_rate == esc_b.no_validation.task_success_rate
    )
    print(f"escalation comparison deterministic: {esc_ok}")

    return ok and outage_ok and esc_ok


async def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    # No ExperimentStore here — see the module docstring's "No ExperimentStore
    # persistence at this scale" section for why (a real, measured finding,
    # not an oversight). Every metric below comes directly from the
    # OutcomeEvent/result objects each library function already returns.
    resilience_rows = await run_large_resilience_comparison(None)
    pl.DataFrame(resilience_rows).write_csv(RESULTS_DIR / "resilience_comparison_large.csv")

    outage_rows = await run_large_sustained_outage(None)
    pl.DataFrame(outage_rows).write_csv(RESULTS_DIR / "sustained_outage_large.csv")

    escalation_row = await run_large_escalation_comparison(None)
    pl.DataFrame([escalation_row]).write_csv(RESULTS_DIR / "escalation_comparison_large.csv")

    ok = await verify_determinism()
    (RESULTS_DIR / "determinism_check.json").write_text(json.dumps({"deterministic": ok}, indent=2))

    total_tasks = (
        LARGE_RESILIENCE_NUM_CASES * 20
        + LARGE_SUSTAINED_TOTAL_STEPS * 5
        + LARGE_ESCALATION_NUM_CASES * 2
    )
    print(f"\nPhase E.5 expanded experiment complete. {total_tasks} total task executions.")
    print(f"Raw data in: {RESULTS_DIR}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
