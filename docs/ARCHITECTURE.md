# Architecture

## Overview

ParetoGuard sits between an application and one or more LLM providers. It normalizes
provider behavior, routes each request to a model under explicit constraints, records
what happened, and can inject/recover from faults for evaluation purposes.

```mermaid
flowchart TD
    Req[InferenceRequest] --> Feat[Feature extraction]
    Feat --> Filt[Constraint filter]
    Filt --> Router{Router}
    Router -->|static| RStatic[StaticRouter]
    Router -->|rule| RRule[RuleRouter]
    Router -->|pareto| RPareto[ParetoRouter]
    Router -->|reliability| RRel[ReliabilityAwareRouter]
    Router -->|learned| RLearn[LearnedRouter]
    RStatic --> Runtime[Provider runtime]
    RRule --> Runtime
    RPareto --> Runtime
    RRel --> Runtime
    RLearn --> Runtime
    Runtime --> Provider[LLM provider / MockProvider]
    Provider --> Validate{Validation}
    Validate -->|success| Telemetry[Telemetry + storage]
    Validate -->|failure| Recovery[Recovery policy]
    Recovery --> Retry[Retry]
    Recovery --> Fallback[Fallback]
    Recovery --> Escalate[Escalate]
    Recovery --> FailSafe[Fail safely]
    Retry --> Runtime
    Fallback --> Runtime
    Escalate --> Runtime

    Health[Health monitor] -.-> RRel
    Chaos[Chaos injector] -.-> Runtime
    Bench[Benchmark / eval runner] -.-> Req
    Storage[(DuckDB store)] -.-> Telemetry
    Reports[Report generator] -.-> Storage
```

## Module boundaries

| Module | Responsibility | Must not do |
|---|---|---|
| `core` | Typed domain models, configuration | Contain business logic |
| `providers` | Normalize provider I/O | Know about routing or evaluation |
| `runtime` | Concurrency, timeout, retry, budget | Know about specific providers |
| `routing` | Select a model given constraints/health | Call providers directly |
| `evals` | Task schemas, grading, benchmark orchestration | Assume a specific router |
| `evals.matrix` | Offline: every candidate x every task, for learned-router training data | Assume a specific router (candidates are given, not chosen) |
| `routing.execution` | **The one exception**: dispatches a `Router`'s `RoutingDecision` to the selected provider via `Runtime`. Also home to `ClosedLoopExecutor` (Phase E): the same dispatch responsibility, extended with automatic retry/fallback/escalate/probe/abstain via `recovery`, feeding every outcome back into `telemetry`/`recovery` without a caller having to close the loop manually | — (every other file in `routing` still must not call providers; only this one composes `routing` + `evals` + `runtime` + `recovery` for live router-driven benchmarking and closed-loop execution) |
| `routing.resilience_benchmark` | Phase E flagship recovery-policy comparison: composes `chaos` + `recovery` + `routing.execution.ClosedLoopExecutor` + `evals` over `resilience_v1` (i.i.d. per-attempt faults) | Tune the comparison to guarantee any config wins (see its own module docstring) |
| `routing.sustained_outage_benchmark` | Phase E.5: the same recovery configs against a deterministic healthy->degraded->recovered timeline (one provider, correlated faults, not i.i.d.) — reuses `chaos.scenarios.provider_degradation_schedule` and a virtual clock for `CircuitBreaker` | Tune the scenario to guarantee circuit-breaker or fallback "wins" (see its own module docstring) |
| `agents` | Tool-use loop, deterministic tools. Optionally consults a `chaos.FaultInjector` (constructor param) to inject tool-level faults — the one injection point it uniquely owns | Execute arbitrary code or shell |
| `chaos` | Deterministic fault injection | Run unconditionally outside experiments |
| `recovery` | Retry/fallback/circuit-breaker/escalation policy. `recovery.health` (Phase E.5) advises fallback *ordering* from `HealthTracker` signals (EMA success/timeout rate, latency drift, consecutive-failure streak) — a separate, narrower mechanism from both `routing.reliability.ReliabilityAwareRouter` (initial-routing ranking) and `CircuitBreaker` (hard state gating) | Hide failures silently. Never hard-exclude a candidate on health alone — see `recovery.health`'s module docstring |
| `telemetry` | Trace events, rolling health metrics | Persist directly to disk (delegates to storage) |
| `storage` | DuckDB schema, migrations, export | Contain domain logic |
| `statistics` | Confidence intervals, comparisons, regression | Be called from CLI/API handlers directly for business decisions |
| `reports` | Render stored results into docs/charts | Compute new metrics not already stored |
| `api` | FastAPI HTTP boundary | Contain business logic |
| `cli` | Typer command boundary | Contain business logic |

CLI and API are thin: they parse input, call into the library modules above, and
format output. This keeps every capability usable both as a library and as a service.

**Phase E addition**: `evals.metrics.compute_resilience_metrics` reads `RecoveryAction`
string values (via a function-local import of `recovery.context`, not a module-level
one — `recovery.context` transitively imports `routing.types`, and `routing/__init__.py`
imports `evals.matrix`, so a module-level import here would cycle back to this file) and
accepts an optional `average_tool_calls` computed from `agents.state.AgentTrajectory`s
(`evals.metrics.mean_tool_call_count`) — `evals` metrics now read from `recovery`'s and
`agents`' *typed output shapes* (`OutcomeEvent.recovery_actions`,
`AgentTrajectory.tool_call_count`) without either of those modules depending back on
`evals`.

## Data flow for a benchmark run

1. A `BenchmarkRun` is created from an `EvalSuite` + `RunManifest` (captures git SHA,
   seed, config versions).
2. For each `EvalCase`, the runtime builds an `InferenceRequest`, asks the configured
   `Router` for a `RoutingDecision`, and dispatches to the provider runtime.
3. The chaos injector may intercept the request/response to apply a `FaultSpec`
   (deterministic, seeded).
4. Providers return a normalized `InferenceResponse`; graders in `evals` compute an
   `EvalResult`.
5. On failure, the `recovery` module decides retry/fallback/escalate/fail based on the
   error taxonomy. For a `ClosedLoopExecutor`-driven run (Phase E), this repeats
   automatically until a terminal outcome, and the whole chain is summarized as one
   `OutcomeEvent` (`storage.ExperimentStore.record_outcome_event`) alongside the
   per-attempt `requests`/`responses`/`trace_events` rows.
6. Every step emits `TraceEvent`s to `telemetry`, which updates rolling `ModelHealth`
   and persists to the DuckDB `storage` layer.
7. `reports` reads only from storage to produce Markdown/JSON/chart output — it never
   recomputes metrics from live provider calls.

## Non-goals

No Kubernetes, message queues, or distributed infrastructure. The reliability story is
about the LLM/agent layer (retries, fallbacks, circuit breakers, adaptive routing), not
about distributed systems infrastructure.
