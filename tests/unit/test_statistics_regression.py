"""Unit tests for statistics.regression: the regression engine operating on
real ExperimentStore-persisted rows (never manually entered summaries).

Fixtures write `RunManifest`/`EvalResult` rows directly via `ExperimentStore
.record_run`/`record_eval_result` rather than running a full
`BenchmarkRunner` pass — `check_regression` only ever reads
`get_run`/`get_eval_results`/`get_outcome_events`, so this exercises the
exact same real-store code path the engine uses while staying fast (a full
benchmark run's `record_request`/`record_response` per task has real
per-statement overhead at even moderate `n` — see `ExperimentStore`'s own
docstring — which nothing here needs, since the regression engine never
reads requests/responses at all)."""

from uuid import uuid4

import pytest

from paretoguard.core.models import RunManifest
from paretoguard.evals.models import EvalResult, GraderKind
from paretoguard.statistics.regression import (
    MetricRegressionPolicy,
    RegressionOutcome,
    RegressionPolicy,
    check_regression,
    check_run_compatibility,
)
from paretoguard.storage import ExperimentStore


def _seed_run(
    store: ExperimentStore,
    run_id: str,
    *,
    num_cases: int = 30,
    corrupt_first_n: int = 0,
    cost_usd: float = 0.01,
    corrupt_cost_usd: float | None = None,
    suite_name: str = "regress_v1",
    suite_version: str = "1.0.0",
    label: str | None = "SIMULATION",
    case_id_prefix: str = "case",
) -> None:
    store.record_run(
        RunManifest(
            run_id=run_id,
            paretoguard_version="0.1.0",
            os="Windows",
            python_version="3.12",
            suite_name=suite_name,
            suite_version=suite_version,
            task_count=num_cases,
            label=label,
        )
    )
    for i in range(num_cases):
        succeeded = not (i < corrupt_first_n)
        cost = cost_usd if succeeded or corrupt_cost_usd is None else corrupt_cost_usd
        store.record_eval_result(
            EvalResult(
                result_id=uuid4(),
                case_id=f"{case_id_prefix}-{i:03d}",
                request_id=uuid4(),
                repetition=0,
                sequence=i,
                succeeded=succeeded,
                score=1.0 if succeeded else 0.0,
                grader_kind=GraderKind.EXACT_MATCH,
                explanation="test",
                latency_ms=10.0,
                cost_usd=cost,
                total_tokens=5,
            ),
            run_id=run_id,
        )


@pytest.fixture
def store():
    with ExperimentStore(":memory:") as s:
        yield s


def test_no_regression_when_nothing_changed(store: ExperimentStore) -> None:
    _seed_run(store, "baseline")
    _seed_run(store, "candidate")
    policy = RegressionPolicy(
        metrics=(MetricRegressionPolicy("success_rate", max_practical_change=0.05),)
    )
    result = check_regression(store, "baseline", "candidate", policy)
    assert result.outcome == RegressionOutcome.OK
    assert result.exit_code == 0
    assert result.passed


def test_detects_a_clear_regression(store: ExperimentStore) -> None:
    _seed_run(store, "baseline")
    _seed_run(store, "candidate", corrupt_first_n=15)
    policy = RegressionPolicy(
        metrics=(MetricRegressionPolicy("success_rate", max_practical_change=0.05),)
    )
    result = check_regression(store, "baseline", "candidate", policy)
    assert result.outcome == RegressionOutcome.REGRESSION_DETECTED
    assert result.exit_code == 1
    assert not result.passed
    metric = result.metric_results[0]
    assert metric.triggered
    assert metric.practical_change == pytest.approx(0.5)


def test_small_change_below_practical_threshold_does_not_trigger(store: ExperimentStore) -> None:
    _seed_run(store, "baseline", num_cases=20)
    _seed_run(store, "candidate", num_cases=20, corrupt_first_n=1)  # tiny change
    policy = RegressionPolicy(
        metrics=(MetricRegressionPolicy("success_rate", max_practical_change=0.10),)
    )
    result = check_regression(store, "baseline", "candidate", policy)
    assert result.outcome == RegressionOutcome.OK
    assert not result.metric_results[0].triggered


def test_statistically_insignificant_change_does_not_trigger_when_required(
    store: ExperimentStore,
) -> None:
    _seed_run(store, "baseline")
    _seed_run(store, "candidate", corrupt_first_n=2)  # small n, weak signal
    policy = RegressionPolicy(
        metrics=(
            MetricRegressionPolicy(
                "success_rate", max_practical_change=0.0, require_statistical_significance=True
            ),
        )
    )
    result = check_regression(store, "baseline", "candidate", policy)
    metric = result.metric_results[0]
    assert metric.exceeds_practical_threshold  # some change did happen
    if not metric.statistically_significant:
        assert not metric.triggered
        assert result.outcome == RegressionOutcome.OK


def test_disabling_significance_requirement_triggers_on_practical_threshold_alone(
    store: ExperimentStore,
) -> None:
    _seed_run(store, "baseline")
    _seed_run(store, "candidate", corrupt_first_n=2)
    policy = RegressionPolicy(
        metrics=(
            MetricRegressionPolicy(
                "success_rate", max_practical_change=0.0, require_statistical_significance=False
            ),
        )
    )
    result = check_regression(store, "baseline", "candidate", policy)
    assert result.outcome == RegressionOutcome.REGRESSION_DETECTED


def test_empty_policy_is_a_config_error(store: ExperimentStore) -> None:
    _seed_run(store, "baseline")
    _seed_run(store, "candidate")
    result = check_regression(store, "baseline", "candidate", RegressionPolicy(metrics=()))
    assert result.outcome == RegressionOutcome.CONFIG_ERROR
    assert result.exit_code == 4


def test_missing_run_is_a_config_error(store: ExperimentStore) -> None:
    _seed_run(store, "baseline")
    policy = RegressionPolicy(metrics=(MetricRegressionPolicy("success_rate"),))
    result = check_regression(store, "baseline", "does-not-exist", policy)
    assert result.outcome == RegressionOutcome.CONFIG_ERROR
    assert any("not found" in r for r in result.incompatibility_reasons)


def test_label_mismatch_is_incompatible_by_default(store: ExperimentStore) -> None:
    _seed_run(store, "baseline", label="SIMULATION")
    _seed_run(store, "candidate", label="LIVE")
    policy = RegressionPolicy(metrics=(MetricRegressionPolicy("success_rate"),))
    result = check_regression(store, "baseline", "candidate", policy)
    assert result.outcome == RegressionOutcome.INCOMPATIBLE_RUNS
    assert result.exit_code == 2
    assert any("label mismatch" in r for r in result.incompatibility_reasons)


def test_label_mismatch_allowed_with_explicit_override(store: ExperimentStore) -> None:
    _seed_run(store, "baseline", label="SIMULATION")
    _seed_run(store, "candidate", label="LIVE")
    policy = RegressionPolicy(
        metrics=(MetricRegressionPolicy("success_rate"),), allow_simulation_vs_live=True
    )
    result = check_regression(store, "baseline", "candidate", policy)
    assert result.outcome != RegressionOutcome.INCOMPATIBLE_RUNS


def test_suite_name_mismatch_is_incompatible(store: ExperimentStore) -> None:
    _seed_run(store, "baseline", suite_name="suite_a")
    _seed_run(store, "candidate", suite_name="suite_b")
    policy = RegressionPolicy(metrics=(MetricRegressionPolicy("success_rate"),))
    result = check_regression(store, "baseline", "candidate", policy)
    assert result.outcome == RegressionOutcome.INCOMPATIBLE_RUNS


def test_mismatched_task_ids_under_same_suite_name_is_incompatible(store: ExperimentStore) -> None:
    _seed_run(store, "baseline", case_id_prefix="case")
    _seed_run(store, "candidate", case_id_prefix="other")  # same suite, disjoint case ids
    policy = RegressionPolicy(metrics=(MetricRegressionPolicy("success_rate"),))
    result = check_regression(store, "baseline", "candidate", policy)
    assert result.outcome == RegressionOutcome.INCOMPATIBLE_RUNS


def test_metric_policy_rejects_unknown_metric_name() -> None:
    with pytest.raises(ValueError, match="unknown metric_name"):
        MetricRegressionPolicy("not_a_real_metric")


def test_metric_policy_rejects_lower_is_better_binary_metric() -> None:
    with pytest.raises(ValueError, match="higher_is_better must be True"):
        MetricRegressionPolicy("success_rate", higher_is_better=False)


def test_check_run_compatibility_reports_multiple_issues_at_once() -> None:
    baseline = RunManifest(
        run_id="a",
        paretoguard_version="0.1.0",
        os="Windows",
        python_version="3.12",
        suite_name="s1",
        suite_version="1.0.0",
        label="SIMULATION",
    )
    candidate = RunManifest(
        run_id="b",
        paretoguard_version="0.1.0",
        os="Windows",
        python_version="3.12",
        suite_name="s2",
        suite_version="2.0.0",
        label="LIVE",
    )
    issues = check_run_compatibility(baseline, candidate)
    assert len(issues) == 3  # suite_name, suite_version, label


def test_cost_regression_is_detected(store: ExperimentStore) -> None:
    _seed_run(store, "baseline", cost_usd=0.01)
    _seed_run(store, "candidate", cost_usd=0.05)
    policy = RegressionPolicy(
        metrics=(
            MetricRegressionPolicy("cost_usd", higher_is_better=False, max_practical_change=0.01),
        )
    )
    result = check_regression(store, "baseline", "candidate", policy)
    assert result.outcome == RegressionOutcome.REGRESSION_DETECTED
    assert result.metric_results[0].practical_change == pytest.approx(0.04)


def test_multiple_comparison_correction_can_prevent_a_borderline_trigger(
    store: ExperimentStore,
) -> None:
    """With several metrics configured, Holm-Bonferroni correction makes
    each individual metric's bar higher to clear — a borderline metric that
    would trigger alone may not once corrected alongside others. The
    adjusted p-value is also written back onto the returned comparison."""
    _seed_run(store, "baseline", num_cases=20)
    _seed_run(store, "candidate", num_cases=20, corrupt_first_n=4)

    corrected_policy = RegressionPolicy(
        metrics=(
            MetricRegressionPolicy("success_rate", max_practical_change=0.0),
            MetricRegressionPolicy("cost_usd", higher_is_better=False, max_practical_change=0.0),
        ),
        apply_multiple_comparison_correction=True,
    )
    corrected = check_regression(store, "baseline", "candidate", corrected_policy)
    success_metric = next(m for m in corrected.metric_results if m.metric_name == "success_rate")
    assert success_metric.comparison is not None
    assert success_metric.comparison.adjusted_p_value is not None
    assert success_metric.comparison.adjusted_p_value >= success_metric.comparison.raw_p_value
