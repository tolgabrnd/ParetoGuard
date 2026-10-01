# Changelog

Notable, deliberate breaking changes to public APIs. Routine additive changes
are not tracked here — see `git log` / `docs/BUILD_PLAN.md` for full history.

## Unreleased (Phase E.5 / F)

- **Breaking:** `ClosedLoopExecutor`'s `validate_quality` parameter now defaults to
  `True` (was effectively "off," with no parameter to turn it on). Previously, any
  transport-successful response was treated as terminal success regardless of
  content; by default it is now graded in-loop, so a transport-successful-but-wrong
  response is classified as a quality failure and can trigger retry/fallback/
  escalation. Pass `validate_quality=False` to restore the exact prior behavior.
- **Breaking:** `ExperimentStore` now opens one transaction per store lifetime
  (`__init__` calls `begin()`) instead of autocommitting every write. Writes are no
  longer guaranteed visible/durable to another connection until `commit()` or
  `close()` is called. This was necessary to avoid a measured ~0.2s/task persistence
  overhead at larger benchmark scales; see `docs/LIMITATIONS.md`'s Phase F entry.

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
