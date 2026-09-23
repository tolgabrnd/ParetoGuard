"""Cost estimation: turns reported token usage into a `CostRecord` using a
versioned `PricingEntry` — never a hard-coded price.

This is deliberately a pure function with no knowledge of providers or routing:
`Runtime` calls it once per successful attempt when a `PricingTable` is
configured (see `paretoguard.runtime.executor.Runtime`). If no pricing entry
exists for a (provider, model) pair, the caller gets `None` back rather than a
fabricated number — see `paretoguard.core.config.PricingTable.price_for`.
"""

from paretoguard.core.models import CostRecord, PricingEntry, TokenUsage


def estimate_cost(usage: TokenUsage, entry: PricingEntry) -> CostRecord:
    """Computes a `CostRecord` from reported `usage` and a specific `entry`.

    `entry.basis` is copied onto the result unchanged: an ESTIMATED entry
    (real provider pricing) produces an ESTIMATED cost, a SIMULATED entry
    (synthetic/offline model profile) produces a SIMULATED cost. Cached input
    tokens are billed at the same input rate — providers that discount cached
    tokens should encode that as a separate, lower-priced pseudo-model rather
    than have this function guess a discount ratio.
    """
    input_cost = usage.input_tokens / 1_000_000 * entry.input_price_per_million_usd
    output_cost = usage.output_tokens / 1_000_000 * entry.output_price_per_million_usd
    return CostRecord(
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        pricing_version=entry.version,
        basis=entry.basis,
    )
