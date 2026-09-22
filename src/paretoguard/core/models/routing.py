"""Routing constraint and decision types, shared by every router implementation."""

from pydantic import BaseModel, Field


class RoutingConstraints(BaseModel):
    """Hard constraints a routing decision must satisfy."""

    max_cost_usd: float | None = Field(default=None, ge=0)
    max_latency_ms: float | None = Field(default=None, ge=0)
    min_predicted_success: float | None = Field(default=None, ge=0, le=1)
    required_capabilities: list[str] = Field(default_factory=list)
    excluded_models: list[str] = Field(default_factory=list)


class RoutingDecision(BaseModel):
    """The output of a router: what it chose and why, in a machine-readable form."""

    selected_model: str
    candidate_scores: dict[str, float] = Field(default_factory=dict)
    predicted_success: float | None = Field(default=None, ge=0, le=1)
    expected_cost_usd: float | None = Field(default=None, ge=0)
    expected_latency_ms: float | None = Field(default=None, ge=0)
    confidence: float | None = Field(default=None, ge=0, le=1)
    explanation: str
    fallback_order: list[str] = Field(default_factory=list)
    excluded_candidates: dict[str, str] = Field(
        default_factory=dict, description="model name -> human-readable exclusion reason"
    )
