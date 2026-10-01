"""Unit tests for reports.charts: chart integrity — every ChartResult's
data must correspond to what was actually plotted, and missing/insufficient
data must yield None, never a fabricated chart."""

from pathlib import Path

from paretoguard.reports.charts import (
    CalibrationBin,
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
from paretoguard.statistics.comparison import compare_binary_outcomes
from paretoguard.statistics.pareto_frontier import analyze_pareto_frontier


def test_pareto_frontier_chart_is_created_and_reports_correct_ids(tmp_path: Path) -> None:
    analysis = analyze_pareto_frontier({"a": 0.9, "b": 0.5}, {"a": 0.02, "b": 0.01})
    result = plot_pareto_frontier(analysis, output_path=tmp_path / "pareto.png", run_id="run-1")
    assert result.path.exists()
    assert result.path.stat().st_size > 0
    assert result.run_ids == ("run-1",)
    assert set(result.sample_sizes) == {"a", "b"}


def test_resilience_curve_chart_none_when_no_points(tmp_path: Path) -> None:
    result = plot_resilience_curve([], output_path=tmp_path / "curve.png")
    assert result is None
    assert not (tmp_path / "curve.png").exists()


def test_resilience_curve_chart_sample_sizes_match_input(tmp_path: Path) -> None:
    points = [
        ResilienceCurvePoint(0.0, "A", 1.0, 24),
        ResilienceCurvePoint(0.3, "A", 0.6, 24),
    ]
    result = plot_resilience_curve(points, output_path=tmp_path / "curve.png")
    assert result is not None
    assert result.path.exists()
    assert result.sample_sizes == {"A@0.0": 24, "A@0.3": 24}


def test_recovery_rate_chart_skips_none_values_and_returns_none_if_all_none(
    tmp_path: Path,
) -> None:
    points = [ResilienceCurvePoint(0.0, "A", 1.0, 10)]
    result = plot_recovery_rate_by_level(
        points, {("A", 0.0): None}, output_path=tmp_path / "recovery.png"
    )
    assert result is None


def test_recovery_rate_chart_plots_available_values(tmp_path: Path) -> None:
    points = [ResilienceCurvePoint(0.15, "A", 1.0, 24)]
    result = plot_recovery_rate_by_level(
        points, {("A", 0.15): 0.8}, output_path=tmp_path / "recovery.png"
    )
    assert result is not None
    assert result.path.exists()


def test_failure_taxonomy_chart_none_when_empty(tmp_path: Path) -> None:
    result = plot_failure_taxonomy({}, output_path=tmp_path / "taxonomy.png")
    assert result is None


def test_failure_taxonomy_chart_sample_size_matches_total_count(tmp_path: Path) -> None:
    result = plot_failure_taxonomy(
        {"timeout": 5, "schema_failure": 2}, output_path=tmp_path / "taxonomy.png"
    )
    assert result is not None
    assert result.sample_sizes["total"] == 7


def test_selection_distribution_chart_none_when_empty(tmp_path: Path) -> None:
    result = plot_selection_distribution({}, output_path=tmp_path / "sel.png")
    assert result is None


def test_selection_distribution_chart_basic(tmp_path: Path) -> None:
    result = plot_selection_distribution(
        {"model-a": 10, "model-b": 3}, output_path=tmp_path / "sel.png"
    )
    assert result is not None
    assert result.path.exists()
    assert result.sample_sizes["total"] == 13


def test_health_timeline_chart_none_when_no_data() -> None:
    """The repo does not persist a health time series by default — this
    must return None, never a fabricated flat line."""
    result = plot_health_timeline([], output_path=Path("unused.png"))
    assert result is None


def test_health_timeline_chart_with_data(tmp_path: Path) -> None:
    points = [
        HealthTimelinePoint(step=0, provider="mock", model="a", ema_success_rate=0.9),
        HealthTimelinePoint(step=1, provider="mock", model="a", ema_success_rate=0.85),
    ]
    result = plot_health_timeline(points, output_path=tmp_path / "health.png")
    assert result is not None
    assert result.path.exists()
    assert result.sample_sizes == {"mock:a": 2}


def test_baseline_vs_candidate_chart_reports_both_run_ids(tmp_path: Path) -> None:
    ids = [f"t{i}" for i in range(20)]
    baseline = [True] * 10 + [False] * 10
    candidate = [True] * 20
    comparison = compare_binary_outcomes(
        ids, baseline, ids, candidate, baseline_run_id="base-run", candidate_run_id="cand-run"
    )
    result = plot_baseline_vs_candidate(comparison, output_path=tmp_path / "compare.png")
    assert result.path.exists()
    assert result.run_ids == ("base-run", "cand-run")


def test_calibration_chart_none_when_no_bins(tmp_path: Path) -> None:
    result = plot_calibration([], output_path=tmp_path / "calib.png")
    assert result is None


def test_calibration_chart_with_bins(tmp_path: Path) -> None:
    bins = [CalibrationBin(0.5, 0.4, 10), CalibrationBin(0.9, 0.85, 15)]
    result = plot_calibration(bins, output_path=tmp_path / "calib.png")
    assert result is not None
    assert result.path.exists()
    assert result.sample_sizes == {"bin_0": 10, "bin_1": 15}


def test_charts_create_parent_directories(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c" / "chart.png"
    result = plot_selection_distribution({"x": 1}, output_path=nested)
    assert result is not None
    assert nested.exists()


def test_simulation_label_is_baked_into_the_rendered_pixels_not_only_metadata(
    tmp_path: Path,
) -> None:
    """A PNG opened on its own (outside a report) must still be
    self-describing about simulation/live provenance — regression test for
    a real gap: charts previously only recorded the label in
    `ChartResult.labels`, never in the image itself. Two otherwise-identical
    charts built with different labels must produce different image bytes;
    a chart built with no label must differ from one that has one."""
    data = {"model-a": 10, "model-b": 3}
    sim = plot_selection_distribution(data, output_path=tmp_path / "sim.png", label="SIMULATION")
    live = plot_selection_distribution(data, output_path=tmp_path / "live.png", label="LIVE")
    unlabeled = plot_selection_distribution(data, output_path=tmp_path / "none.png", label="")
    assert sim is not None and live is not None and unlabeled is not None

    sim_bytes = sim.path.read_bytes()
    live_bytes = live.path.read_bytes()
    unlabeled_bytes = unlabeled.path.read_bytes()
    assert sim_bytes != live_bytes
    assert sim_bytes != unlabeled_bytes
    assert live_bytes != unlabeled_bytes


def test_baseline_vs_candidate_chart_label_defaults_to_simulation_but_is_overridable(
    tmp_path: Path,
) -> None:
    ids = [f"t{i}" for i in range(20)]
    baseline = [True] * 10 + [False] * 10
    candidate = [True] * 20
    comparison = compare_binary_outcomes(ids, baseline, ids, candidate)

    default_result = plot_baseline_vs_candidate(comparison, output_path=tmp_path / "default.png")
    live_result = plot_baseline_vs_candidate(
        comparison, output_path=tmp_path / "live.png", label="LIVE"
    )
    assert default_result.labels == ("SIMULATION",)
    assert live_result.labels == ("LIVE",)
    assert default_result.path.read_bytes() != live_result.path.read_bytes()
