"""Routing strategies: static, round-robin, rule-based, Pareto, reliability-aware,
and learned. `routing` never calls a provider directly (see docs/ARCHITECTURE.md)
— a `Router` only produces a `RoutingDecision`; dispatch happens elsewhere.
"""

from paretoguard.routing.pareto import ObjectiveVector, dominates, pareto_frontier
from paretoguard.routing.pareto_router import ParetoObjective, ParetoRouter
from paretoguard.routing.protocol import Router
from paretoguard.routing.round_robin import RoundRobinRouter
from paretoguard.routing.rule import RuleRouter
from paretoguard.routing.static import StaticRouter
from paretoguard.routing.types import (
    CandidateProfile,
    NoEligibleCandidateError,
    RoutingRequest,
    TaskFeatures,
    apply_soft_constraints,
    candidate_key,
    extract_task_features,
    filter_eligible,
)

__all__ = [
    "CandidateProfile",
    "NoEligibleCandidateError",
    "ObjectiveVector",
    "ParetoObjective",
    "ParetoRouter",
    "RoundRobinRouter",
    "Router",
    "RoutingRequest",
    "RuleRouter",
    "StaticRouter",
    "TaskFeatures",
    "apply_soft_constraints",
    "candidate_key",
    "dominates",
    "extract_task_features",
    "filter_eligible",
    "pareto_frontier",
]
