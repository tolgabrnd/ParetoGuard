# Evaluation

This document describes `paretoguard.evals`: task schemas, deterministic
grading, benchmark orchestration, and the metrics computed over results. It
covers both the Phase C single-shot benchmark path and the Phase E
recovery/agent extensions (Commit 29).

## Task schema

An `EvalSuite` is a versioned, seeded, reproducible collection of `EvalCase`s
— generated deterministically by each suite module's `build_suite(seed, ...)`
function, never loaded from a static fixture file, so a suite is fully
reproducible from (module version, seed) alone.

An `EvalCase` carries its messages/tools/structured-output-schema, a
`GraderConfig` (which deterministic grader to use), a `GroundTruth` (the
grader-specific expected shape — text, number, label, JSON schema, expected
fields, expected substrings, or expected tool calls), and `metadata` passed
through to the `InferenceRequest` (including `MockProvider`-recognized keys
so a case can demonstrate pass/fail offline).

## Built-in suites

| Suite | What it tests | Grader |
|---|---|---|
| `structured_extraction_v1` | JSON-schema-conformant extraction | `JSON_SCHEMA`/`FIELD_SCORING` |
| `numeric_reasoning_v1` | Multi-step arithmetic with a distractor sentence | `NUMERIC` |
| `long_context_retrieval_v1` | Needle-in-haystack retrieval at varying context lengths | `SUBSTRING_PRESENCE` |
| `tool_use_v1` | Single-turn tool selection/argument validity | `TOOL_TRAJECTORY` |
| `resilience_v1` (Commit 29) | Fixed-difficulty single-turn Q&A — a task *load* for the recovery-policy comparison, not a difficulty axis | `NORMALIZED_TEXT_MATCH` |
| `structured_agent_v1` (Commit 29) | Multi-step, multi-hop tool-use agent tasks over `agents.tools`'s fixture world | `agents.simulator.grade_trajectory` (not an `evals.graders` grader — see below) |

`resilience_v1` is deliberately trivial: every case succeeds with certainty
under `MockProvider`'s default behavior absent injected chaos. The point of
that suite is to hold task difficulty constant so the only source of failure
in a chaos-driven benchmark is the injected fault, never the task itself —
see `docs/CHAOS_ENGINEERING.md`.

## Two grading paths

`evals.graders.grade_case(case, response) -> GradeOutcome` grades a single
`InferenceResponse` against an `EvalCase`'s `GroundTruth` — this is the path
every suite above except `structured_agent_v1` uses, via `BenchmarkRunner`,
`routing.execution.RoutedBenchmarkRunner`, or `routing.execution
.ClosedLoopExecutor`.

`agents.simulator.grade_trajectory(trajectory, expected) -> AgentGradeOutcome`
grades a whole multi-step `AgentTrajectory` against an `ExpectedTrajectory`
(expected tool call sequence, expected final field/value or answer
substring) — this is a *different* shape of grading problem (a sequence of
steps, not one response), so it is not folded into `evals.graders`; both
paths are equally deterministic, state-based, and judge-model-free (see
CLAUDE.md: "Prefer deterministic evaluation").

LLM-as-judge, where it lands, will be a third, clearly-separate,
optional path — never silently mixed into either of the above.

## Benchmark orchestrators

| Orchestrator | Candidate selection | Recovery | Use case |
|---|---|---|---|
| `evals.runner.BenchmarkRunner` | Fixed (provider, model) | None | A single model's raw performance on a suite |
| `evals.matrix.MatrixRunner` | Every candidate, every task | None | Offline training-matrix generation for learned routers |
| `routing.execution.RoutedBenchmarkRunner` | One `Router` decision per task | None | A router's live selection behavior on a suite |
| `routing.execution.ClosedLoopExecutor` | One `Router` decision, then `RecoveryPolicy`-driven | Automatic retry/fallback/escalate/probe/abstain | Phase E: the resilience/recovery comparison |
| `agents.executor.AgentExecutor` via `agents.simulator.AgentSimulator` | N/A (agent tasks are provider/model-fixed per batch) | Tool-level chaos only (Phase E); no mid-trajectory recovery yet (see `docs/LIMITATIONS.md`) | `structured_agent_v1` |

## Metrics

`evals.metrics.compute_metrics(results: Sequence[EvalResult]) -> MetricsSummary`
(Phase C): success rate, pass@1/pass@k (Chen et al. 2021's unbiased
estimator), consistency, cross-repetition variance, cost-per-success,
mean-latency-per-success, and JSON-Schema validity rate specifically. Every
metric that can be undefined for the given results (no cost data, no
JSON_SCHEMA results, ...) is `None`, never a misleading `0.0`.

`evals.metrics.compute_resilience_metrics(outcomes: Sequence[OutcomeEvent], *,
average_tool_calls=None) -> ResilienceMetricsSummary` (Commit 29) — one level
up: an `OutcomeEvent` already summarizes a whole recovery chain (possibly
many attempts), where `EvalResult` summarizes one attempt. Every field is
defined precisely in the class's own docstring (required by the Phase E
spec's "every metric defined precisely" rule); the ones worth calling out
here:

- **`raw_failure_rate`**: the failure rate a deployment with *no* recovery
  layer at all would have seen on the same task/fault stream, reconstructed
  from `attempt_count > 1 OR not succeeded` (recovery is only ever invoked
  after a failed first attempt).
- **`recovery_rate`**: of those raw failures, the fraction that ended up
  `succeeded=True`. `None` (not `0.0` or `1.0`) when there were no raw
  failures to recover from — a benchmark cell with zero raw failures says
  nothing about recovery effectiveness.
- **`retry_rate`/`fallback_rate`/`escalation_rate`/`probe_rate`**: fraction of
  tasks whose `recovery_actions` include that action at least once.
- **`unrecovered_failure_distribution`**: a `Counter` of `failure_category`
  (or `"unknown"`) among tasks that ended `succeeded=False` — the failures
  recovery did *not* fix, broken down by cause.
- **`average_tool_calls`**: `None` unless explicitly supplied via
  `mean_tool_call_count(trajectories)` — `OutcomeEvent` has no tool-call
  concept, so this is never derived from `outcomes` alone. Used when
  reporting `structured_agent_v1` results alongside a resilience comparison.

`median_latency_ms`/`p95_latency_ms` use a dependency-free nearest-rank
percentile (`evals.metrics._percentile`) — deterministic given the same
latency values, but latency values themselves are real wall-clock
measurements (`time.perf_counter()` inside `MockProvider.complete`), so
these two fields are excluded from every determinism check in this repo
(see `docs/LIMITATIONS.md`).

## Persistence

`BenchmarkRunner`/`RoutedBenchmarkRunner` persist `requests`, `responses`,
`trace_events`, `routing_decisions` (routed only), and `eval_results` through
`storage.ExperimentStore`. `ClosedLoopExecutor` additionally persists one
`outcome_events` row per task (`ExperimentStore.record_outcome_event` /
`.get_outcome_events` / `.outcome_events_df`, schema migration 004,
Commit 29) — the one-row-per-task summary `eval_results` alone cannot
provide once a task's execution spans multiple attempts.

## Reproducibility discipline

Every `InferenceRequest.request_id` used across a multi-attempt or
multi-step chain is derived via `core.ids.deterministic_request_id` (UUID5,
not `uuid4()`) — this is the fix for the exact bug the Phase D integrity
audit found (a random default silently breaking reproducibility). Every
chaos-affected benchmark additionally derives its fault decisions from a
seeded `Random` keyed on stable inputs only (`docs/CHAOS_ENGINEERING.md`).
`scripts/phase_e_resilience_experiment.py --verify-determinism` is the
harness that checks this explicitly, rather than assuming it holds.
