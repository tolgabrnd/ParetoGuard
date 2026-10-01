# Benchmarks

This document is the index of every benchmark/experiment this repo can
run, what each one measures, and where its results live. See
`docs/EVALUATION.md` for suite/grader/metric details,
`docs/CHAOS_ENGINEERING.md` for the fault-injection model,
`docs/STATISTICS.md` for how results are compared, and
`docs/LIMITATIONS.md` for what each benchmark's numbers do and don't show.

**Every result in this repo is `SIMULATION` unless explicitly labeled
`LIVE`.** No benchmark here has ever been run against a real provider.

## Single-model / routing benchmarks (Phase C/D)

| Suite/experiment | Module | What it measures |
|---|---|---|
| `structured_extraction_v1`, `numeric_reasoning_v1`, `long_context_retrieval_v1`, `tool_use_v1` | `evals.suites` | Single-turn task correctness via `evals.runner.BenchmarkRunner` |
| Offline training matrix | `evals.matrix.MatrixRunner` | Every candidate x every task, for learned-router training data |
| Routing comparison | `routing.evaluation.evaluate_router_offline` | A router's counterfactual performance on held-out data, vs. an oracle upper bound (non-deployable) and fixed-model baselines |
| Degradation/recovery (routing-only) | `routing.reliability_simulation` | `StaticRouter`/`ParetoRouter`/`ReliabilityAwareRouter` reroute behavior under a synthetic degradation schedule — no recovery layer involved, routing decisions only |
| `scripts/phase_d_experiment.py` | — | The Phase D end-to-end run tying the above together; results in `scripts/phase_d_results/` |

## Resilience / recovery benchmarks (Phase E/E.5)

| Suite/experiment | Module | What it measures |
|---|---|---|
| `resilience_v1` flagship comparison | `routing.resilience_benchmark` | Five recovery configs (A: none, B: retry, C: +fallback, D: +circuit-breaker, E: +escalation) at 0/5/15/30% i.i.d. fault levels |
| Sustained outage | `routing.sustained_outage_benchmark` | The same configs against a correlated healthy->degraded->recovered timeline (one provider), where fallback and circuit-breaker have a real, non-i.i.d. story to tell |
| Escalation | `routing.escalation_benchmark` | `validate_quality=False` (pre-Phase-E.5 behavior) vs. `validate_quality=True` + escalation, against an injected quality (malformed-structured-output) fault |
| `structured_agent_v1` | `evals.suites.structured_agent_v1` via `agents.simulator.AgentSimulator` | Multi-step, multi-hop tool-use correctness and graceful failure handling |
| `scripts/phase_e_resilience_experiment.py` | — | CI-friendly scale (24 tasks/cell, 300-step outage, 20 escalation tasks); results in `scripts/phase_e_results/` |
| `scripts/phase_e5_experiment.py` | — | Larger analysis scale (200 tasks/cell, 1,000-step outage, 200 escalation tasks — 9,400 total task executions); results in `scripts/phase_e5_results/`. No `ExperimentStore` persistence at this scale (see `docs/LIMITATIONS.md`) — computed directly from in-memory results |

## What each flagship comparison actually found

The honest, actual numbers (not a projection) are in `docs/LIMITATIONS.md`'s
Phase E/E.5 sections, including the negative/surprising results: retry-only
ties fallback under i.i.d. faults (as expected for that fault model);
fallback meaningfully beats retry-only under a *correlated* outage; circuit
breaker reduces attempt overhead without improving success rate under
i.i.d. faults, but at 30% fault with a larger sample it can measurably
*hurt* success rate due to a real-clock cooldown interacting badly with a
fast synthetic benchmark; validating and escalating on quality failures
took success from 0.35 to 1.00 on an injected-quality-fault suite, at a
real, measured attempt cost.

## Running a benchmark and generating a report

```python
from paretoguard.storage import ExperimentStore
from paretoguard.reports import generate_report, render_markdown

with ExperimentStore("my_experiment.duckdb") as store:
    # ... run a BenchmarkRunner/RoutedBenchmarkRunner/ClosedLoopExecutor
    # pass, or run one of scripts/phase_e*.py, with store=store ...
    report = generate_report(store, run_id="my-run-id")
    print(render_markdown(report))
```

Pass `baseline_run_id=...` to `generate_report` for a statistical
comparison against a prior run (see `docs/STATISTICS.md`), or
`statistics.regression.check_regression` directly for a pass/fail
regression check against a configured policy.
