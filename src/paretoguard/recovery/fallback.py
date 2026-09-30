"""Fallback candidate selection: which *different* candidate to try next,
respecting eligibility, budget, prior attempts, and circuit state.

Deliberately reuses `paretoguard.routing.types.filter_eligible` for the
hard-constraint check (context window, structured-output/tool support,
explicit exclusions) rather than re-implementing it — a candidate ineligible
for a fresh routing decision is equally ineligible as a fallback target.
"""

from dataclasses import dataclass

from paretoguard.core.models import InferenceRequest, Message, ModelSpec, Role
from paretoguard.recovery.circuit_breaker import CircuitBreaker, CircuitState
from paretoguard.recovery.context import RecoveryContext
from paretoguard.routing.types import RoutingRequest, candidate_key, filter_eligible


@dataclass(frozen=True)
class FallbackPolicy:
    max_fallback_depth: int = 3
    """Maximum number of *distinct* candidates a single recovery chain may
    try in total (including the original) before giving up — prevents an
    unbounded fallback cascade even if every candidate keeps failing."""

    def __post_init__(self) -> None:
        if self.max_fallback_depth < 1:
            raise ValueError("max_fallback_depth must be >= 1")


@dataclass(frozen=True)
class FallbackSelection:
    candidate: ModelSpec | None
    is_probe: bool
    """True if `candidate`'s circuit is HALF_OPEN — this selection is a
    deliberate probe of a previously-failing candidate, not a fresh pick."""
    excluded: dict[str, str]


def select_fallback(
    context: RecoveryContext,
    candidates: list[ModelSpec],
    policy: FallbackPolicy,
    *,
    circuit_breaker: CircuitBreaker | None = None,
) -> FallbackSelection:
    """Picks the next candidate to try: the first hard-eligible, not-yet-
    attempted-in-this-chain, not-circuit-OPEN candidate, in the given order.
    A candidate whose circuit is HALF_OPEN is still selectable (as a probe,
    `is_probe=True`) — a circuit breaker's whole point is to occasionally
    let a probe through; refusing it here would defeat that. Returns
    `candidate=None` if depth is exhausted or nothing remains.
    """
    if len(context.attempted) + 1 >= policy.max_fallback_depth:
        return FallbackSelection(candidate=None, is_probe=False, excluded={})

    attempted_keys = {candidate_key(a.provider, a.model) for a in context.attempted}
    if context.current_key() is not None:
        attempted_keys.add(context.current_key())  # type: ignore[arg-type]

    # A placeholder request/features pair purely to satisfy RoutingRequest's
    # shape — filter_eligible only reads `.features` (the real task's, from
    # `context`) and `.constraints`/`.candidates`, never `.request` itself.
    placeholder_request = InferenceRequest(
        provider="", model="", messages=[Message(role=Role.USER, content="")]
    )
    routing_request = RoutingRequest(
        request=placeholder_request,
        features=context.task_features,
        candidates=candidates,
        constraints=context.constraints,
    )
    eligible, excluded = filter_eligible(routing_request)

    for candidate in eligible:
        key = candidate_key(candidate.provider, candidate.name)
        if key in attempted_keys:
            excluded.setdefault(candidate.name, "already attempted in this recovery chain")
            continue
        if circuit_breaker is not None:
            state = circuit_breaker.state_for(key)
            if state == CircuitState.OPEN:
                excluded.setdefault(candidate.name, "circuit breaker is OPEN")
                continue
            if state == CircuitState.HALF_OPEN:
                return FallbackSelection(candidate=candidate, is_probe=True, excluded=excluded)
        return FallbackSelection(candidate=candidate, is_probe=False, excluded=excluded)

    return FallbackSelection(candidate=None, is_probe=False, excluded=excluded)
