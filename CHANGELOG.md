# Changelog

Notable, deliberate breaking changes to public APIs. Routine additive changes
are not tracked here — see `git log` / `docs/BUILD_PLAN.md` for full history.

## Unreleased (Phase D)

- **Breaking:** `CostRecord` and `PricingEntry` gained a required/defaulted
  `basis: CostBasis` field (`ESTIMATED` for real-provider pricing, `SIMULATED`
  for synthetic/offline model profiles) so a cost figure can never be silently
  read as more authoritative than it is.
- **Breaking:** `Provider` implementations (including `MockProvider`) no
  longer set `InferenceResponse.cost` themselves. Cost is now computed by
  `paretoguard.runtime.Runtime` from an optional injected `PricingTable`
  (`paretoguard.runtime.cost.estimate_cost`), so it's derived from one
  versioned, config-driven source instead of being duplicated (or, for real
  providers, silently absent) per adapter. A `Runtime`/`BenchmarkRunner`
  constructed without a `pricing_table` behaves as before: `cost` stays `None`.
