"""Cost, latency, and pricing types.

Pricing is data (loaded from a versioned config file), never a constant baked into
routing or cost-calculation logic — providers change prices over time.
"""

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field


class CostBasis(StrEnum):
    """How a `CostRecord` was derived, so a number is never mistaken for another.

    ESTIMATED: computed from a real provider's published pricing (a versioned
    `PricingEntry` with real, dated figures) applied to actual/reported token
    usage. Still an estimate, not a provider-issued invoice.
    SIMULATED: computed from a synthetic/offline model profile's invented price
    point (e.g. a MockProvider profile). Never to be presented as a real cost.
    """

    ESTIMATED = "estimated"
    SIMULATED = "simulated"


class TokenUsage(BaseModel):
    """Normalized token accounting for one inference call."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class CostRecord(BaseModel):
    """Cost of one inference call, computed from a specific pricing version.

    Always carries `basis` so a simulated/offline number can never be silently
    read as a real provider cost (see `CostBasis`).
    """

    input_cost_usd: float = Field(ge=0)
    output_cost_usd: float = Field(ge=0)
    pricing_version: str
    basis: CostBasis
    currency: str = "USD"

    @property
    def total_cost_usd(self) -> float:
        return self.input_cost_usd + self.output_cost_usd


class LatencyRecord(BaseModel):
    """Normalized latency accounting for one inference call."""

    total_latency_ms: float = Field(ge=0)
    time_to_first_token_ms: float | None = Field(default=None, ge=0)
    queued_ms: float = Field(default=0.0, ge=0)


class PricingEntry(BaseModel):
    """One versioned price point for a (provider, model) pair.

    Multiple entries may exist for the same (provider, model) with different
    `effective_date`s; consumers should select the most recent entry at or before
    the date of interest (see `PricingTable.price_for`).
    """

    provider: str
    model: str
    input_price_per_million_usd: float = Field(ge=0)
    output_price_per_million_usd: float = Field(ge=0)
    version: str
    effective_date: date
    basis: CostBasis = Field(
        default=CostBasis.ESTIMATED,
        description=(
            "ESTIMATED for real-provider prices copied from a pricing page; "
            "SIMULATED for synthetic/offline model profiles with invented prices."
        ),
    )
