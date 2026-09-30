"""Confidence intervals, hypothesis tests, effect sizes, Pareto analysis,
experiment comparison, and regression detection.

See `docs/STATISTICS.md` for the methodology: why each test was chosen,
bootstrap procedure/limitations, and how multiple-comparison correction is
applied.
"""

from paretoguard.statistics.bootstrap import (
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_N_RESAMPLES,
    MIN_RELIABLE_N,
    BootstrapResult,
    bootstrap_ci,
    bootstrap_proportion_ci,
)
from paretoguard.statistics.comparison import (
    MIN_SAMPLE_SIZE_FOR_EVIDENCE,
    ComparisonConclusion,
    ExperimentComparison,
    compare_binary_outcomes,
    compare_continuous_outcomes,
)
from paretoguard.statistics.effect_size import (
    ContinuousDelta,
    OddsRatio,
    RiskDifference,
    continuous_delta,
    odds_ratio,
    risk_difference,
)
from paretoguard.statistics.pareto_frontier import ParetoAnalysis, analyze_pareto_frontier
from paretoguard.statistics.significance import (
    TestResult,
    fisher_exact_test,
    holm_bonferroni,
    mann_whitney_u_test,
    mcnemar_exact_test,
    wilcoxon_signed_rank_test,
)

__all__ = [
    "DEFAULT_CONFIDENCE_LEVEL",
    "DEFAULT_N_RESAMPLES",
    "MIN_RELIABLE_N",
    "MIN_SAMPLE_SIZE_FOR_EVIDENCE",
    "BootstrapResult",
    "ComparisonConclusion",
    "ContinuousDelta",
    "ExperimentComparison",
    "OddsRatio",
    "ParetoAnalysis",
    "RiskDifference",
    "TestResult",
    "analyze_pareto_frontier",
    "bootstrap_ci",
    "bootstrap_proportion_ci",
    "compare_binary_outcomes",
    "compare_continuous_outcomes",
    "continuous_delta",
    "fisher_exact_test",
    "holm_bonferroni",
    "mann_whitney_u_test",
    "mcnemar_exact_test",
    "odds_ratio",
    "risk_difference",
    "wilcoxon_signed_rank_test",
]
