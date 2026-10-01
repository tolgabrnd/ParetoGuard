"""Reproducible benchmark report and chart generation.

Reports (`reports.report`) are generated entirely from persisted
`ExperimentStore` data, never hand-entered values. Charts (`reports.charts`)
are pure functions of explicitly-typed data, each returning `None` rather
than a fabricated plot when the required data isn't available.
"""

from paretoguard.reports.charts import (
    CalibrationBin,
    ChartResult,
    HealthTimelinePoint,
    ResilienceCurvePoint,
    plot_baseline_vs_candidate,
    plot_calibration,
    plot_failure_taxonomy,
    plot_health_timeline,
    plot_pareto_frontier,
    plot_recovery_rate_by_level,
    plot_resilience_curve,
    plot_selection_distribution,
)
from paretoguard.reports.report import (
    SIMULATION_BANNER,
    BenchmarkReport,
    FailureTaxonomyEntry,
    generate_report,
    render_json,
    render_markdown,
)

__all__ = [
    "SIMULATION_BANNER",
    "BenchmarkReport",
    "CalibrationBin",
    "ChartResult",
    "FailureTaxonomyEntry",
    "HealthTimelinePoint",
    "ResilienceCurvePoint",
    "generate_report",
    "plot_baseline_vs_candidate",
    "plot_calibration",
    "plot_failure_taxonomy",
    "plot_health_timeline",
    "plot_pareto_frontier",
    "plot_recovery_rate_by_level",
    "plot_resilience_curve",
    "plot_selection_distribution",
    "render_json",
    "render_markdown",
]
