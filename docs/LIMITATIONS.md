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
  defect. A future suite would need to inject `FailureCategory.SCHEMA_FAILURE`/
  `INVALID_OUTPUT`/`REASONING_FAILURE`-shaped faults to exercise it end-to-end.
- **`RecoveryPolicy.decide` does not read `RecoveryContext.health_snapshots`.**
  `ClosedLoopExecutor` populates it on every call (and feeds `HealthTracker` on every
  attempt), but the only health-*derived* signal the current `RecoveryPolicy`
  implementation actually consults is `CircuitBreaker` state (`state_for`), which is
  updated from the same success/failure stream but is a separate, coarser signal
  (open/half-open/closed) than the rolling EMA `health_snapshots` carries. Wiring a
  health-aware decision (e.g. preferring a fallback candidate with a better recent
  success rate over routing order) is future work, not implemented in Phase E.
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

This file will grow with specific, dated entries as each subsystem is implemented.
