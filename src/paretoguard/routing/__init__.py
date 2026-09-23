"""Routing strategies: static, round-robin, rule-based, Pareto, reliability-aware,
and learned. `routing` never calls a provider directly (see docs/ARCHITECTURE.md)
— a `Router` only produces a `RoutingDecision`; dispatch happens elsewhere.
"""

from paretoguard.routing.dataset import (
    CANDIDATE_COLUMNS,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    build_dataset,
    feature_matrix,
    labels,
)
from paretoguard.routing.evaluation import (
    RoutingEvalSummary,
    best_fixed_model,
    build_profiles_from_train_split,
    cheapest_fixed_model,
    cost_at_target_success,
    evaluate_router_offline,
    fixed_model_stats,
    oracle_upper_bound,
    success_at_fixed_budget,
)
from paretoguard.routing.learned import (
    CalibrationMetrics,
    LearnedRouter,
    LearnedRouterModel,
    evaluate_calibration,
    train_learned_router_model,
)
from paretoguard.routing.pareto import ObjectiveVector, dominates, pareto_frontier
from paretoguard.routing.pareto_router import ParetoObjective, ParetoRouter
from paretoguard.routing.protocol import Router
from paretoguard.routing.reliability import (
    HealthStatus,
    ReliabilityAwareRouter,
    ReliabilityThresholds,
)
from paretoguard.routing.reliability_simulation import (
    SimulationSummary,
    run_degradation_recovery_simulation,
    two_model_degradation_schedule,
)
from paretoguard.routing.round_robin import RoundRobinRouter
from paretoguard.routing.rule import RuleRouter
from paretoguard.routing.splits import (
    SplitRatios,
    assign_splits,
    default_group_key,
    group_aware_split,
    verify_no_group_leakage,
)
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
    "CANDIDATE_COLUMNS",
    "FEATURE_COLUMNS",
    "LABEL_COLUMNS",
    "CalibrationMetrics",
    "CandidateProfile",
    "HealthStatus",
    "LearnedRouter",
    "LearnedRouterModel",
    "NoEligibleCandidateError",
    "ObjectiveVector",
    "ParetoObjective",
    "ParetoRouter",
    "ReliabilityAwareRouter",
    "ReliabilityThresholds",
    "RoundRobinRouter",
    "Router",
    "RoutingEvalSummary",
    "RoutingRequest",
    "RuleRouter",
    "SimulationSummary",
    "SplitRatios",
    "StaticRouter",
    "TaskFeatures",
    "apply_soft_constraints",
    "assign_splits",
    "best_fixed_model",
    "build_dataset",
    "build_profiles_from_train_split",
    "candidate_key",
    "cheapest_fixed_model",
    "cost_at_target_success",
    "default_group_key",
    "dominates",
    "evaluate_calibration",
    "evaluate_router_offline",
    "extract_task_features",
    "feature_matrix",
    "filter_eligible",
    "fixed_model_stats",
    "group_aware_split",
    "labels",
    "oracle_upper_bound",
    "pareto_frontier",
    "run_degradation_recovery_simulation",
    "success_at_fixed_budget",
    "train_learned_router_model",
    "two_model_degradation_schedule",
    "verify_no_group_leakage",
]
