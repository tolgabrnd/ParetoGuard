"""Shared input/eligibility types for every router implementation.

Kept separate from `paretoguard.core.models.routing` (which holds the
*output* contract, `RoutingConstraints`/`RoutingDecision`, shared with
storage) because these types are routing-implementation detail: how
candidates are filtered, how measured history is looked up. Feature
extraction itself (`TaskFeatures`/`extract_task_features`) lives in
`paretoguard.core.features`, not here — both `routing` and `evals` depend on
it, so it can't live in either without the other depending on it too (see
docs/ARCHITECTURE.md: "evals ... Must not assume a specific router"). This
module re-exports it for convenience so existing `from
paretoguard.routing.types import ...` / `from paretoguard.routing import
...` call sites don't need to know about the split.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from paretoguard.core.features import (
    TASK_FAMILY_METADATA_KEY as TASK_FAMILY_METADATA_KEY,
)
from paretoguard.core.features import (
    TaskFeatures as TaskFeatures,
)
from paretoguard.core.features import (
    extract_task_features as extract_task_features,
)
from paretoguard.core.models import InferenceRequest, ModelSpec, RoutingConstraints
from paretoguard.telemetry.health import ModelHealth


def candidate_key(provider: str, model: str) -> str:
    """Canonical lookup key for a (provider, model) pair, used to key both
    `RoutingRequest.health` and `RoutingRequest.profiles`."""
    return f"{provider}:{model}"


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


def apply_soft_constraints(
    routing_request: RoutingRequest, candidates: Sequence[ModelSpec]
) -> tuple[list[ModelSpec], dict[str, str]]:
    """Excludes candidates whose measured/simulated profile violates a soft
    (estimate-based) constraint: `max_cost_usd`, `max_latency_ms`,
    `min_predicted_success`. A candidate with no profile data can't be
    checked against these and is kept — soft constraints only exclude on
    positive evidence of violation, never on absence of data. Shared by
    `RuleRouter` and `ParetoRouter` so both apply the same "subject to"
    semantics for `RoutingConstraints`.
    """
    constraints = routing_request.constraints
    kept: list[ModelSpec] = []
    excluded: dict[str, str] = {}

    for candidate in candidates:
        profile = routing_request.profiles.get(candidate_key(candidate.provider, candidate.name))
        if profile is None:
            kept.append(candidate)
            continue

        if (
            constraints.max_cost_usd is not None
            and profile.mean_cost_usd is not None
            and profile.mean_cost_usd > constraints.max_cost_usd
        ):
            excluded[candidate.name] = (
                f"expected cost ${profile.mean_cost_usd:.4f} exceeds "
                f"max_cost_usd ${constraints.max_cost_usd:.4f}"
            )
            continue
        if (
            constraints.max_latency_ms is not None
            and profile.mean_latency_ms is not None
            and profile.mean_latency_ms > constraints.max_latency_ms
        ):
            excluded[candidate.name] = (
                f"expected latency {profile.mean_latency_ms:.0f}ms exceeds "
                f"max_latency_ms {constraints.max_latency_ms:.0f}ms"
            )
            continue
        if (
            constraints.min_predicted_success is not None
            and profile.predicted_success is not None
            and profile.predicted_success < constraints.min_predicted_success
        ):
            excluded[candidate.name] = (
                f"predicted success {profile.predicted_success:.3f} below "
                f"min_predicted_success {constraints.min_predicted_success:.3f}"
            )
            continue
        kept.append(candidate)

    return kept, excluded
