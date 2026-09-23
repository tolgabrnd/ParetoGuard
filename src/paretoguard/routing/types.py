"""Shared input/eligibility types for every router implementation.

Kept separate from `paretoguard.core.models.routing` (which holds the
*output* contract, `RoutingConstraints`/`RoutingDecision`, shared with
storage) because these types are routing-implementation detail: how a
request becomes features, how candidates are filtered, how measured
history is looked up. `evals` never imports this module (see
docs/ARCHITECTURE.md: "evals ... Must not assume a specific router").
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from paretoguard.core.models import InferenceRequest, ModelSpec, RoutingConstraints
from paretoguard.telemetry.health import ModelHealth

TASK_FAMILY_METADATA_KEY = "task_family"
"""InferenceRequest.metadata key a caller may set to identify the task family
(e.g. suite name) a request belongs to, for task-family-scoped rules/profiles.
Optional: absent unless a caller (typically an EvalCase's metadata, which is
passed through verbatim to InferenceRequest.metadata) sets it."""


def candidate_key(provider: str, model: str) -> str:
    """Canonical lookup key for a (provider, model) pair, used to key both
    `RoutingRequest.health` and `RoutingRequest.profiles`."""
    return f"{provider}:{model}"


@dataclass(frozen=True)
class TaskFeatures:
    """Cheap, deterministic features extracted from an `InferenceRequest` at
    routing time — never anything only known after execution (see the
    leakage rules in `paretoguard.routing` for the learned router, Commit 21).
    """

    input_tokens_estimate: int
    max_output_tokens: int | None
    context_tokens_estimate: int
    requires_structured_output: bool
    requires_tool_use: bool
    tool_count: int
    task_family: str | None


def extract_task_features(request: InferenceRequest) -> TaskFeatures:
    """Whitespace-split word count as a token estimate — the same cheap,
    dependency-free heuristic `MockProvider` uses for its own token counts
    (see `paretoguard.providers.mock`), good enough for *routing-time*
    eligibility decisions, not for billing."""
    input_tokens_estimate = max(1, sum(len(m.content.split()) for m in request.messages))
    max_output = request.max_output_tokens
    return TaskFeatures(
        input_tokens_estimate=input_tokens_estimate,
        max_output_tokens=max_output,
        context_tokens_estimate=input_tokens_estimate + (max_output or 0),
        requires_structured_output=request.structured_output_schema is not None,
        requires_tool_use=len(request.tools) > 0,
        tool_count=len(request.tools),
        task_family=request.metadata.get(TASK_FAMILY_METADATA_KEY),
    )


@dataclass(frozen=True)
class CandidateProfile:
    """Measured or simulated historical performance for one candidate model,
    optionally scoped to a task family. Router-agnostic: the same shape is
    produced by offline matrix evaluation (Commit 21) or hand-authored for
    tests/synthetic model profiles — the router doesn't care which."""

    predicted_success: float | None = None
    mean_cost_usd: float | None = None
    mean_latency_ms: float | None = None
    task_family: str | None = None
    simulated: bool = False
    """True if this profile comes from a synthetic/offline model profile
    rather than measured real-provider history — carried through to routing
    explanations so a simulated number is never read as measured."""


@dataclass(frozen=True)
class RoutingRequest:
    """Everything a `Router` needs to make and explain one decision."""

    request: InferenceRequest
    features: TaskFeatures
    candidates: Sequence[ModelSpec]
    constraints: RoutingConstraints = field(default_factory=RoutingConstraints)
    health: Mapping[str, ModelHealth] = field(default_factory=dict)
    """Keyed by `candidate_key(provider, model)`. Empty when no `HealthTracker`
    is wired in (e.g. a cold start, or a non-reliability-aware benchmark)."""
    profiles: Mapping[str, CandidateProfile] = field(default_factory=dict)
    """Keyed by `candidate_key(provider, model)`. Empty when no measured/
    simulated profile data is available for this run."""


class NoEligibleCandidateError(RuntimeError):
    """Raised when every candidate was excluded by a hard constraint and the
    router has no escalation/fallback policy of its own (that's Commit 24's
    confidence-aware escalation layer, not this one). Routers that fail
    loudly here rather than returning a constraint-violating `RoutingDecision`
    are following CLAUDE.md's "never silently violate user constraints"."""


def filter_eligible(
    routing_request: RoutingRequest,
) -> tuple[list[ModelSpec], dict[str, str]]:
    """Applies hard eligibility (context window, structured-output support,
    tool support, explicit exclusions) shared by every non-StaticRouter.
    Soft/estimate-based constraints (max_cost_usd, max_latency_ms,
    min_predicted_success) are applied separately by callers that have
    profile data, since eligibility here must hold regardless of whether any
    profile is available.

    Returns (eligible_candidates_in_input_order, {model_name: reason})."""
    features = routing_request.features
    constraints = routing_request.constraints
    eligible: list[ModelSpec] = []
    excluded: dict[str, str] = {}

    for candidate in routing_request.candidates:
        if candidate.name in constraints.excluded_models:
            excluded[candidate.name] = "explicitly excluded by routing constraints"
            continue
        if candidate.context_window < features.context_tokens_estimate:
            excluded[candidate.name] = (
                f"context window {candidate.context_window} < required "
                f"{features.context_tokens_estimate} (estimated)"
            )
            continue
        if features.requires_structured_output and not candidate.supports_structured_output:
            excluded[candidate.name] = "task requires structured output; model does not support it"
            continue
        if features.requires_tool_use and not candidate.supports_tools:
            excluded[candidate.name] = "task requires tool use; model does not support it"
            continue
        missing_capabilities = set(constraints.required_capabilities) - set(candidate.capabilities)
        if missing_capabilities:
            excluded[candidate.name] = (
                f"missing required capabilities: {sorted(missing_capabilities)}"
            )
            continue
        eligible.append(candidate)

    return eligible, excluded
