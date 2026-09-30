# Limitations

This document is updated as features land. It exists so results are never read as
stronger claims than the evidence supports.

## Reproducibility

- Results against MockProvider are fully deterministic given the same seed and config.
- Results against real provider APIs are **not** bit-for-bit reproducible: providers
  change model weights, safety behavior, and latency characteristics over time without
  notice.
- Estimated cost is computed from a versioned local pricing table (`configs/pricing.example.yaml`)
  and may drift from actual provider billing if prices change after that version's
  effective date.
- Provider REST APIs themselves change — see `docs/PROVIDER_COMPATIBILITY.md` for
  which API surface each adapter targets, when it was last verified against
  official docs, and known deprecation risk per provider.

## Benchmarks

- Built-in suites are synthetic and intentionally small; they are designed to isolate
  specific failure modes (structured output, tool use, long context), not to be a
  general-purpose model leaderboard.
- Task distributions do not necessarily reflect any specific production workload.
- Simulated model profiles (used when no live credentials are configured) are labeled
  as simulations everywhere they appear; they must never be presented as live-model
  results.

## Routing

- Learned and bandit routers are only as good as the benchmark data they are trained
  on; they are not guaranteed to generalize to task distributions outside that data.
- Predicted success probabilities are model outputs, not guarantees, and may be
  miscalibrated; calibration metrics are reported alongside predictions so this can be
  checked rather than assumed.
- Online/bandit routing evaluated offline has known selection-bias and counterfactual
  estimation limitations, documented at the point that router is introduced.

### Phase D specifics (2026-09-23)

- **Logistic regression cannot represent non-additive (task, candidate) interactions.**
  `paretoguard.routing.learned`'s test suite includes a deliberately XOR-structured
  fixture (`strong` succeeds on family-a/fails on family-b, `weak` the exact
  opposite) that the logistic-regression baseline cannot separate from additive
  one-hot features; the random-forest baseline can. This is the documented,
  concrete justification for keeping a tree-based baseline (see Commit 22), not a
  hypothetical one — check `evaluate_calibration`'s Brier score per model before
  trusting either baseline on a new dataset with unknown interaction structure.
- **`RuleRouter`/`ParetoRouter` given only overall (non-task-family-scoped) profiles
  converge to a single "generalist" candidate.** In the Phase D end-to-end
  experiment (`scripts/phase_d_experiment.py`), `build_profiles_from_train_split`
  aggregates one success rate per candidate across every task family; routers that
  only see this aggregate picked `profile-a` for all 11 held-out test tasks, never
  the family-specialized `profile-b`/`profile-c`, even though `profile-c` is the
  strongest choice specifically for `structured_extraction_v1`. The learned routers
  (which take `task_family` as an input feature) picked `profile-b`/`profile-c` on
  a handful of tasks instead — see `scripts/phase_d_results/selection_distribution.csv`
  for the actual counts. This is a genuine property of the profile representation,
  not a bug: a rule/Pareto router needs task-family-scoped `CandidateProfile`s (which
  `RuleRouter` does support, see Commit 18) to specialize; Phase D's experiment script
  did not build them, so this is a known gap in the experiment, not the router logic.
- **`ReliabilityAwareRouter` cannot detect recovery without periodic re-probing.** A
  purely exploit-only policy that fully stops calling a degraded candidate can never
  observe it recovering — fixed in Commit 20 via a half-open circuit-breaker-style
  probe (`ReliabilityThresholds.probe_interval_calls`), but the probe interval is a
  tunable tradeoff: too frequent wastes traffic on a genuinely-bad candidate, too
  infrequent slows recovery detection. The Phase D degradation/recovery experiment
  measured a detection delay of 7 steps and a recovery delay of 70 steps at the
  default interval (15 calls) — see `scripts/phase_d_results/reliability_experiment_summary.csv`.
- **Group-aware train/val/test splitting depends on suites tagging `template_id`.**
  Only `numeric_reasoning_v1` and `tool_use_v1` generate cases from a small,
  enumerable set of distinguishable templates and tag them; `structured_extraction_v1`
  and `long_context_retrieval_v1` do not need to (single template, randomized field
  values only — see `paretoguard.routing.splits`'s module docstring for the
  reasoning). A future suite with real near-duplicate structure that doesn't tag
  `template_id` would not be protected by `verify_no_group_leakage` — the guard only
  catches what it's told about.

## LLM-as-judge

- Where used (optional, never the sole grading path for primary metrics), the judge
  model and version are recorded, and judge-based results are kept visually and
  numerically distinct from deterministic grading results.

## Chaos / agent simulation

- Simulated tool environments are simplified stand-ins for production tools; recovery
  behavior observed in simulation is evidence about the recovery *logic*, not a
  guarantee about production tool failure modes.

### Phase E specifics (2026-09-30)

- **`resilience_v1`'s fault model is i.i.d. per attempt, not a sustained/correlated
  outage.** `chaos.policies.ConstantProbability` draws an independent fault decision
  at every attempt (keyed by `(seed, task_id, chaos_step, fault_id)`); it does not
  model a provider that is *reliably* down for a stretch of time the way
  `routing.reliability_simulation`'s `two_model_degradation_schedule` does. Under an
  i.i.d. stream, retry-same and fallback face statistically the same per-attempt
  success probability, so `routing.resilience_benchmark`'s flagship comparison is not
  expected to show fallback systematically beating retry-only — and the actual
  0/5/15/30% run (`scripts/phase_e_results/resilience_comparison.csv`) does not: at
  24 tasks per cell, every recovery-enabled config (B–E) recovers 100% of raw
  failures at every nonzero fault level tested, with only *how* they got there
  (more retries vs. more fallback switches, visible in `average_attempts`) differing.
  A larger suite, a higher fault level, or a correlated-outage fault schedule would be
  needed to observe a recovery-enabled config *failing* to fully recover — this run
  does not exercise that regime, and no claim here should be read as "recovery always
  achieves 100%."
- **`resilience_v1` injects no quality-category faults**, so `RecoveryAction.ESCALATE`
  is exercised by unit tests (`test_recovery_policy.py`) but not by this benchmark —
  `escalation_rate == 0.0` in every row of the flagship comparison is expected, not a
  defect, and remains true after the Phase E.5 fix below (verified unchanged:
  `resilience_v1` still only injects transient/provider-category faults). See
  `routing.escalation_benchmark` (Phase E.5) for the scenario that does exercise it.
- **RESOLVED in Phase E.5, and a deeper bug than originally stated**: the paragraph
  above previously ended with "a future suite would need to inject quality-shaped
  faults to exercise escalation end-to-end" — true, but incomplete. Actually building
  that suite surfaced a real architectural gap: `ClosedLoopExecutor` graded the
  *final* response only, at the very end of the loop. Any transport-successful
  response — right or wrong — was treated as terminal success the instant it arrived,
  so `RecoveryAction.ESCALATE` could never fire through the real closed-loop path no
  matter what a benchmark injected; a quality-shaped fault would have been silently
  accepted as "success" mid-loop. Fixed: `ClosedLoopExecutor` now grades every attempt
  in-loop by default (`validate_quality=True`), classifying a failed grade into
  `FailureCategory.SCHEMA_FAILURE` (JSON_SCHEMA-graded cases) or `INVALID_OUTPUT`
  (every other grader kind — a deterministic grader has no basis to distinguish
  `REASONING_FAILURE` from a generic wrong answer, so this repo never fabricates that
  distinction) and feeding it through the same recovery path a transport failure uses.
  `validate_quality=False` restores the exact old behavior, specifically so
  `routing.escalation_benchmark` can compare the two directly.

  **Actual results** (`routing.escalation_benchmark`, 20 JSON-schema tasks, 60%
  quality-fault rate on one candidate, verified reproducible): the `no_validation` arm
  reaches 0.350 task success (13/20 permanently wrong, `escalation_rate=0.0`, every
  unrecovered failure's category reads `"unknown"` — the arm never even classified
  *why* it failed, since it never looked); the `validation_and_escalation` arm reaches
  1.000 task success, `escalation_rate=0.65`, zero unrecovered failures, at a real,
  reported cost of +0.65 average attempts per task (added latency was negligible
  against `MockProvider`, as expected — real-provider latency would differ and is not
  claimed here). This is the honest, non-trivial demonstration the Phase E.5 spec
  asked for: validating and escalating on quality failures has a real, positive,
  measured effect through the actual closed-loop path, at a real, measured cost.
- **RESOLVED in Phase E.5**: `RecoveryPolicy.decide` previously did not read
  `RecoveryContext.health_snapshots` at all. `recovery.health.classify_recovery_health`
  now advises `select_fallback`'s candidate *ordering* (demote DEGRADED, skip
  UNHEALTHY unless it's the only option, never penalize UNKNOWN/cold-start) from four
  explicit, independently-documented signals — EMA success rate, EMA timeout rate,
  `HealthTracker.latency_drift_ratio` (now also snapshotted onto `RecoveryContext
  .latency_drift`), and a consecutive-failure streak — combined through a documented
  rule cascade, never a blended composite score. See `recovery.health`'s module
  docstring for why this is deliberately narrower than both
  `routing.reliability.ReliabilityAwareRouter` (initial-routing ranking on one signal)
  and `CircuitBreaker` (hard state gating, left untouched as a separate abstraction).
  **What's still a real limitation**: the four signals are combined by a fixed rule
  order (cold-start check, then streak, then floor, then ceiling-with-flags), not a
  tunable weighting — a candidate with a borderline EMA success rate and a mild
  latency drift is always `DEGRADED`, never independently distinguishable from one
  with a severe drift and a strong EMA, because neither signal's *magnitude* beyond
  its threshold affects the outcome. This is a deliberate simplicity tradeoff (see
  "no magic composite score" in the design goal), not an oversight, but it does mean
  the four-tier classification is coarser than the raw signals it's built from.
- **`AgentStep.recovery_action` is unused in Phase E.** The field exists (reserved
  since Commit 25) for a future mid-trajectory recovery integration where
  `AgentExecutor` would consult a `RecoveryPolicy` after a failed step instead of
  terminating `UNRECOVERABLE_FAILURE` immediately; Phase E only wires recovery at the
  task level (`ClosedLoopExecutor`), never within a single agent trajectory's steps.
  `AgentStep.fault_id` *is* wired (tool-level faults only — provider-level faults
  reaching an agent step are visible in the resulting `TerminationReason`/response
  error, but not linked back to their originating `FaultEvent`'s id on the step
  itself unless the fault was a tool fault).
- **Wall-clock latency is not part of the determinism guarantee.** Every
  seed-derived field (success/failure, recovery actions taken, attempt counts, costs,
  candidate selections) is bit-for-bit reproducible across independent process runs
  (verified by `scripts/phase_e_resilience_experiment.py --verify-determinism` and by
  running the full script twice as separate processes and diffing the CSV output);
  `median_latency_ms`/`p95_latency_ms` are measured via real `time.perf_counter()`
  calls inside `MockProvider.complete` and are excluded from every determinism
  comparison in this repo for exactly that reason.
- **The `select_fallback` depth-accounting bug found and fixed during this work**
  (see `recovery/fallback.py`'s inline comment and
  `test_select_fallback_depth_counts_distinct_candidates_not_total_attempts`/
  `test_select_fallback_depth_permits_exactly_max_fallback_depth_distinct_candidates`
  in `test_recovery_policy.py`) means any `RecoveryPolicy` comparison run before this
  commit that mixed retry and fallback in the same chain under-counted available
  fallback depth; no such comparison was ever published as a result prior to this fix,
  so there is nothing to retract, but a reader diffing this commit against Commit 27's
  original `fallback.py` should not assume the old behavior was equivalent.

### Phase E.5 specifics (2026-09-30)

- **The sustained-outage scenario (`routing.sustained_outage_benchmark`) shows what
  `resilience_v1`'s i.i.d. fault model structurally could not.** At default parameters
  (300 steps, one provider degraded to 85% failure for steps 100-199, two healthy
  baseline providers throughout), the actual observed numbers: `A-no_recovery`'s
  success rate *during the outage window* collapses to 0.190 (consistent with ~15%
  expected success at 85% failure); `B-retry_only` partially compensates, reaching
  0.480 — still failing more than half the time, because retrying the *same* degraded
  candidate up to 3 total times against an 85%-failure stream is not enough;
  `C-retry_fallback`, `D-retry_fallback_circuit_breaker`, and `E-full_policy` all
  reach 1.000 outage-window success by switching to a healthy backup provider. This is
  the honest confirmation `resilience_benchmark`'s own docstring predicted would *not*
  show up under i.i.d. faults: fallback meaningfully beats retry-only specifically
  under a correlated/sustained failure, not a momentary one.
- **Circuit breaker's benefit here shows up as efficiency, not success rate.** `D`
  and `E` tie `C`'s 1.000 outage success but do it in fewer average attempts (1.29 vs.
  1.48) — the breaker opened 4 times across the 300-step run, each time avoiding a
  wasted retry/probe against the degraded primary that fallback-only (`C`) would still
  have attempted once per task. Reported plainly rather than inflated into "the
  breaker improves success," which the numbers do not support here.
- **`time_to_recover` measures something narrower than it sounds.** Every simulated
  "step" in this scenario is an independent task with its own fresh `StaticRouter`
  decision (always the primary) — there is no cross-task "currently avoiding the
  primary" state outside of `CircuitBreaker`'s own OPEN/HALF_OPEN gating. So
  `time_to_recover == 0` for every fallback-capable config in the default run does not
  mean "recovery is instant/trivial" — it means the first post-outage task's *first
  attempt* at the primary already succeeds (the fault schedule itself has already
  returned to 1% baseline), and by then `CircuitBreaker`'s cooldown (5 virtual
  seconds, many cycles within the 100-step outage) has typically already let the
  primary back in well before `recovered_start`. This is a real, honest property of
  the per-task routing design (Commit 28: "the router decides once per task"), not a
  benchmark artifact to paper over — a future scenario that wants to observe a
  *slower*, state-carrying recovery would need a router/executor design with
  cross-task "recently avoided" memory, which this repo does not have.
- **A deterministic virtual clock, not the real one, drives `CircuitBreaker.cooldown_s`
  in this scenario** (`clock=lambda: virtual_clock["t"]`, one virtual second per
  simulated step) — reusing `CircuitBreaker`'s existing injectable-clock design. Using
  the real `time.monotonic` clock here would make cooldown/probe timing depend on how
  fast this process happens to execute 300 simulated steps (well under a second),
  making the result non-reproducible in a way unrelated to the scenario itself.

This file will grow with specific, dated entries as each subsystem is implemented.
