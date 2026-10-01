# Reproducibility

This document explains what "reproducible" means for each kind of result
ParetoGuard produces, and exactly what a `RunManifest` does and does not
guarantee.

## The `RunManifest`

Every persisted run (`ExperimentStore.record_run`) carries a `RunManifest`:
`run_id`, `paretoguard_version`, `git_sha`, `created_at`, `os`,
`python_version`, `seed`, `suite_name`/`suite_version`, `router_name`/
`router_config`, `pricing_config_version`, `task_count`, `repetitions`,
`environment` (non-secret metadata only), and `label` (`SIMULATION`/`LIVE`,
Commit 31). A `reports.BenchmarkReport`'s "Run identity" section is built
directly from this — never a hand-typed summary.

## Simulation runs: fully reproducible

Every offline/`MockProvider`-backed benchmark in this repo is deterministic
given the same (seed, suite version, code version):

- Task generation: every suite's `build_suite(seed, ...)` is a pure function
  of its seed — no suite is loaded from a fixture file that could silently
  drift.
- Fault injection: `chaos.FaultInjector` draws from a fresh
  `random.Random(f"{seed}:{task_id}:{step}:{fault_id}")` per decision —
  never global random state, never wall-clock time.
- Request identity: every multi-attempt/multi-step `request_id` is derived
  via `core.ids.deterministic_request_id` (UUID5), not `uuid4()` — the
  Phase D integrity audit's own finding (a random default silently broke
  reproducibility) is why this discipline exists at all.
- Bootstrap resampling: `statistics.bootstrap` seeds a
  `numpy.random.Generator` explicitly per call.

**Explicitly verified, not assumed**: `scripts/phase_e_resilience_experiment.py
--verify-determinism` and `scripts/phase_e5_experiment.py` both run their
scenarios twice — once in-process, once again as two fully independent
process invocations — and diff every field except latency. This is a
deliberate habit from the Phase D audit: "reproducible" is a claim to check,
not assume.

## The one deliberate exception: wall-clock latency

`latency_ms`/`total_latency_ms` are measured via real `time.perf_counter()`
calls — including inside `MockProvider`, whose own module docstring's "fully
deterministic" claim does not extend to this one field (see
`docs/LIMITATIONS.md`). Every determinism check in this repo excludes
latency fields for exactly this reason. `RecoveryContext.latency_drift`
(Phase E.5's health-aware fallback signal, derived from `HealthTracker`'s
latency EMA) is consequently opt-in on `ClosedLoopExecutor`
(`include_latency_drift_in_recovery`, default `False`) rather than wired in
by default — using it would make a *decision*, not just a reported number,
depend on non-deterministic timing.

## Live-provider runs: not bit-for-bit reproducible, and never claimed to be

A run against a real provider (OpenAI/Anthropic/Gemini) is not
reproducible in the way a simulation run is: providers change model
weights, safety behavior, and latency characteristics over time without
notice, and the same prompt can legitimately return a different response
on a later call even at temperature 0. `RunManifest.label = "LIVE"` marks
this explicitly. `statistics.regression.check_run_compatibility` refuses to
compare a `SIMULATION` run against a `LIVE` one (or either against an
unlabeled run) unless `RegressionPolicy(allow_simulation_vs_live=True)` is
set explicitly — silently mixing the two would let a real provider's
day-to-day variance be mistaken for a detected regression, or vice versa.

## Pricing and cost

Cost figures come from a versioned `PricingTable`
(`pricing_config_version` on the manifest); estimated cost can drift from
actual provider billing if prices change after that version's effective
date. `CostRecord.basis` (`ESTIMATED`/`SIMULATED`) distinguishes a
real-pricing-table estimate from a purely synthetic figure — a report never
presents one as the other.

## What a `git_sha` is for

`RunManifest.git_sha` (when populated by a caller) pins the exact code
version that produced a run — useful for correlating a regression back to
the commit that introduced it. It is not currently auto-populated by any
benchmark runner in this repo; a caller that cares about this should set it
explicitly when constructing a `RunManifest`.
