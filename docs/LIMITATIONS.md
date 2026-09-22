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

## LLM-as-judge

- Where used (optional, never the sole grading path for primary metrics), the judge
  model and version are recorded, and judge-based results are kept visually and
  numerically distinct from deterministic grading results.

## Chaos / agent simulation

- Simulated tool environments are simplified stand-ins for production tools; recovery
  behavior observed in simulation is evidence about the recovery *logic*, not a
  guarantee about production tool failure modes.

This file will grow with specific, dated entries as each subsystem is implemented.
