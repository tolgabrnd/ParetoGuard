"""Unit tests for reports.report: generated entirely from persisted
ExperimentStore data, never hand-entered values."""

from uuid import uuid4

import pytest

from paretoguard.core.models import FailureCategory, RunManifest
from paretoguard.evals.models import EvalResult, GraderKind
from paretoguard.reports.report import (
    SIMULATION_BANNER,
    generate_report,
    render_json,
    render_markdown,
)
from paretoguard.storage import ExperimentStore


def _seed(
    store: ExperimentStore,
    run_id: str,
    *,
    num_cases: int = 20,
    fail_first_n: int = 0,
    cost_usd: float | None = 0.01,
    label: str | None = "SIMULATION",
    suite_name: str = "s1",
) -> None:
    store.record_run(
        RunManifest(
            run_id=run_id,
            paretoguard_version="0.1.0",
            os="Windows",
            python_version="3.12",
            suite_name=suite_name,
            suite_version="1.0.0",
            label=label,
            seed=0,
        )
    )
    for i in range(num_cases):
        succeeded = i >= fail_first_n
        store.record_eval_result(
            EvalResult(
                result_id=uuid4(),
                case_id=f"c{i}",
                request_id=uuid4(),
                repetition=0,
                sequence=i,
                succeeded=succeeded,
                score=1.0 if succeeded else 0.0,
                grader_kind=GraderKind.EXACT_MATCH,
                explanation="t",
                latency_ms=12.0,
                cost_usd=cost_usd if succeeded else None,
                total_tokens=5,
                response_error_category=None if succeeded else FailureCategory.TIMEOUT,
            ),
            run_id=run_id,
        )


@pytest.fixture
def store():
    with ExperimentStore(":memory:") as s:
        yield s


def test_raises_for_unknown_run(store: ExperimentStore) -> None:
    with pytest.raises(ValueError, match="no run found"):
        generate_report(store, "does-not-exist")


def test_report_with_no_eval_results_is_explicit_not_fabricated(store: ExperimentStore) -> None:
    store.record_run(
        RunManifest(
            run_id="empty",
            paretoguard_version="0.1.0",
            os="Windows",
            python_version="3.12",
            label="SIMULATION",
        )
    )
    report = generate_report(store, "empty")
    assert report.sample_size == 0
    assert report.success_rate is None
    assert "No EvalResult rows found for this run." in report.limitations


def test_basic_report_computes_real_metrics(store: ExperimentStore) -> None:
    _seed(store, "r1", num_cases=20, fail_first_n=3)
    report = generate_report(store, "r1")
    assert report.sample_size == 20
    assert report.success_rate == pytest.approx(0.85)
    assert report.success_rate_ci is not None
    assert report.cost_per_success_usd == pytest.approx(0.01)
    assert len(report.failure_taxonomy) == 1
    assert report.failure_taxonomy[0].category == "timeout"
    assert report.failure_taxonomy[0].count == 3


def test_low_sample_size_is_flagged_unreliable(store: ExperimentStore) -> None:
    _seed(store, "r1", num_cases=10)
    report = generate_report(store, "r1")
    assert report.success_rate_ci is not None
    assert not report.success_rate_ci.reliable


def test_no_cost_data_is_reported_as_unavailable_not_zero(store: ExperimentStore) -> None:
    _seed(store, "r1", cost_usd=None)
    report = generate_report(store, "r1")
    assert report.cost_per_success_usd is None
    assert any("cost data" in lim for lim in report.limitations)


def test_unlabeled_run_is_flagged_in_limitations(store: ExperimentStore) -> None:
    _seed(store, "r1", label=None)
    report = generate_report(store, "r1")
    assert any("no SIMULATION/LIVE label" in lim for lim in report.limitations)


def test_comparison_against_a_compatible_baseline(store: ExperimentStore) -> None:
    _seed(store, "baseline", fail_first_n=0)
    _seed(store, "candidate", fail_first_n=8)
    report = generate_report(store, "candidate", baseline_run_id="baseline")
    assert report.comparison is not None
    assert report.comparison.baseline_run_id == "baseline"
    assert report.comparison.candidate_run_id == "candidate"


def test_comparison_omitted_for_incompatible_baseline(store: ExperimentStore) -> None:
    _seed(store, "baseline", suite_name="suite_a")
    _seed(store, "candidate", suite_name="suite_b")
    report = generate_report(store, "candidate", baseline_run_id="baseline")
    assert report.comparison is None
    assert any("incompatible" in lim.lower() for lim in report.limitations)


def test_comparison_omitted_for_missing_baseline(store: ExperimentStore) -> None:
    _seed(store, "candidate")
    report = generate_report(store, "candidate", baseline_run_id="does-not-exist")
    assert report.comparison is None
    assert any("not found" in lim for lim in report.limitations)


def test_pareto_analysis_passthrough(store: ExperimentStore) -> None:
    from paretoguard.statistics.pareto_frontier import analyze_pareto_frontier

    _seed(store, "r1")
    pa = analyze_pareto_frontier({"a": 0.9}, {"a": 0.01})
    report = generate_report(store, "r1", pareto_analysis=pa)
    assert report.pareto_analysis is pa
    assert not any("Pareto" in lim for lim in report.limitations)


def test_render_markdown_includes_simulation_banner(store: ExperimentStore) -> None:
    _seed(store, "r1", label="SIMULATION")
    report = generate_report(store, "r1")
    markdown = render_markdown(report)
    assert SIMULATION_BANNER in markdown
    assert report.run_id in markdown


def test_render_markdown_never_fabricates_unavailable_fields(store: ExperimentStore) -> None:
    _seed(store, "r1", cost_usd=None)
    report = generate_report(store, "r1")
    markdown = render_markdown(report)
    assert "not available" in markdown
    assert "$0.00" not in markdown  # never a fabricated zero


def test_render_json_round_trips_key_fields(store: ExperimentStore) -> None:
    _seed(store, "r1", fail_first_n=2)
    report = generate_report(store, "r1")
    data = render_json(report)
    assert data["run_id"] == "r1"
    assert data["sample_size"] == 20
    assert data["success_rate"] == pytest.approx(0.9)
    assert data["success_rate_ci"]["confidence_level"] == pytest.approx(0.95)
    assert isinstance(data["failure_taxonomy"], list)


def test_recovery_summary_absent_without_outcome_events(store: ExperimentStore) -> None:
    _seed(store, "r1")
    report = generate_report(store, "r1")
    assert report.recovery_summary is None
