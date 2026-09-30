# Chaos Engineering

This document describes ParetoGuard's deterministic fault-injection subsystem
(`paretoguard.chaos`, Commit 26) and how it composes with the recovery engine
(`paretoguard.recovery`, Commit 27) and the closed-loop control plane
(`paretoguard.routing.execution.ClosedLoopExecutor`, Commit 28) to produce the
Phase E resilience benchmarks (`paretoguard.routing.resilience_benchmark`,
Commit 29).

**Every fault this subsystem injects is synthetic.** Nothing here observes or
replays a real provider outage; every number a chaos-driven benchmark produces
is SIMULATION (see `CLAUDE.md`: never fabricate reliability numbers).

## Why deterministic, not random

Phase D's integrity audit found a real bug: a random `uuid4()` default made an
otherwise "seeded" simulation non-reproducible across runs. Every part of
`chaos` is built to avoid that class of bug from the ground up:

- `FaultInjector` never touches `random` module global state, `time.time()`,
  or an unseeded RNG. Every fault decision is drawn from
  `random.Random(f"{seed}:{task_id}:{step}:{fault_id}")` — a fresh,
  independently-seeded `Random` instance per decision, keyed only on stable
  inputs the caller controls.
- The same `(seed, task_id, step, fault_id)` always yields the same fire/
  no-fire decision, across processes, forever — this is what makes
  `scripts/phase_e_resilience_experiment.py --verify-determinism` a
  meaningful check rather than a formality.
- Nothing in `chaos` monkey-patches global state. `FaultyProvider` wraps a
  `Provider` explicitly; `check_tool_fault`/`apply_tool_fault` are plain
  functions a caller invokes explicitly. A `FaultInjector` instance is inert
  until something calls `faults_for` on it.

## The fault model

Three categories (`FaultCategory`), each with its own typed `*FaultKind` enum
— never an untyed string:

| Category | Kind enum | Applied by |
|---|---|---|
| `PROVIDER` | `ProviderFaultKind` (timeout, rate_limit, server_error, latency_spike, malformed_structured_output, truncated_output) | `chaos.injector.FaultyProvider` wraps a `Provider` |
| `TOOL` | `ToolFaultKind` (exception, timeout, stale_result, missing_field, schema_mismatch, partial_result, temporary_unavailable) | `chaos.injector.check_tool_fault`/`apply_tool_fault`, called by `agents.executor.AgentExecutor` when constructed with a `fault_injector` |
| `CONTEXT` | `ContextFaultKind` (distractor_injection, missing_evidence, stale_record, contradictory_stale_record) | `chaos.scenarios.apply_context_fault`, applied at benchmark **construction** time (perturbing a prompt before it's ever sent), never at dispatch time |

Every `FaultEvent` also carries an author's own `ExpectedRecoverability`
(`RECOVERABLE`/`UNRECOVERABLE`/`UNKNOWN`) — recorded so a benchmark can later
distinguish "the system failed to recover from something that should have
been recoverable" (a real finding) from "the fault was deliberately
unrecoverable" (expected, not a defect).

## Scheduling: probabilistic and explicit-schedule forms

`FaultSchedule` is a `Protocol` with one method, `probability_at(step) -> float`:

- `ConstantProbability(probability)` — the same fire probability at every
  step. Used by `resilience_v1`'s flagship comparison (0%/5%/15%/30%).
- `StepRangeSchedule(((start, probability), ...))` — an explicit,
  non-stationary schedule (e.g. healthy → degraded → recovered). Generalizes
  the shape `routing.reliability_simulation.two_model_degradation_schedule`
  established in Phase D; `chaos.scenarios.provider_degradation_schedule`
  builds one directly from that same (degraded_start, recovered_start) shape.

## `chaos_step`: why every retry needs its own step index

`FaultInjector`'s RNG key includes `step`, deliberately **not** the candidate
being called — two different candidates hit at the same step draw
independent decisions only if they're covered by policies with different
`fault_id`s (see `routing.resilience_benchmark`'s module docstring for why
its three candidate providers each get their own `fault_id`). Within *one*
candidate's own retry chain, though, `step` is what keeps a retry from
replaying the identical decision as the attempt it's retrying.

Two call sites set this convention:

- `routing.execution.ClosedLoopExecutor` sets `chaos_step = attempt_number - 1`
  on every attempt's request metadata — attempt 1 draws step 0, a retry draws
  step 1, and so on, regardless of which candidate is currently selected.
- `agents.executor.AgentExecutor` sets `chaos_step = step_index` on every
  agent-loop step's request metadata, for the same reason at the agent-step
  granularity.

`FaultyProvider` reads `request.metadata.get("chaos_step", 0)` — a provider
never wrapped in chaos, or a request that never sets this key, behaves
exactly as if no chaos were configured (default step 0, and typically no
policy targets it at all).

## Composing chaos with recovery: the resilience benchmark

`routing.resilience_benchmark.run_resilience_benchmark` is the reference
composition: `resilience_v1` (a fixed, trivially-correct task load) run
through `ClosedLoopExecutor` against `FaultyProvider`-wrapped `MockProvider`s,
at each of five recovery-policy configurations (A: no recovery, B: retry-only,
C: retry+fallback, D: +circuit-breaker, E: +escalation), at each of four fault
levels. See that module's own docstring for the full experimental design,
including why the fault stream is held constant across configs at a given
level (so the comparison is fair) and what limitations the i.i.d. fault model
has (see `docs/LIMITATIONS.md`'s Phase E section for the actual numbers this
produced).

## Composing chaos with the agent loop

`agents.executor.AgentExecutor`'s optional `fault_injector` constructor
argument applies **tool**-category faults immediately before each
`tool.run()` call — the one injection point the executor uniquely owns.
Provider-category faults need no executor-side integration at all: wrap the
`Provider` passed to the executor's own `Runtime` in `FaultyProvider`, and
every step's request already carries `chaos_step` for it to key on. See
`structured_agent_v1`'s module docstring for why that suite itself does not
use fault injection (it uses a fixed deterministic script instead, to
isolate "does the executor handle a scripted failure gracefully" from "did
chaos draw a failure this run") and `docs/LIMITATIONS.md` for what is and
is not wired between `agents` and `chaos` in Phase E.
