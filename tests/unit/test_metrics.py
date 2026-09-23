"""Unit tests for paretoguard.evals.metrics: pass@k formula and compute_metrics."""

from uuid import uuid4

import pytest

from paretoguard.evals.metrics import compute_metrics, pass_at_k
from paretoguard.evals.models import EvalResult, GraderKind


def _result(
    case_id: str,
    *,
    succeeded: bool,
    grader_kind: GraderKind = GraderKind.EXACT_MATCH,
    cost_usd: float | None = None,
    latency_ms: float = 10.0,
    repetition: int = 0,
    sequence: int = 0,
) -> EvalResult:
    return EvalResult(
        case_id=case_id,
        request_id=uuid4(),
        repetition=repetition,
        sequence=sequence,
        succeeded=succeeded,
        score=1.0 if succeeded else 0.0,
        grader_kind=grader_kind,
        explanation="test",
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        total_tokens=1,
    )


# -- pass_at_k -----------------------------------------------------------------


def test_pass_at_1_equals_naive_success_rate() -> None:
    assert pass_at_k(n=5, c=2, k=1) == pytest.approx(0.4)


def test_pass_at_k_is_one_when_k_exceeds_failures() -> None:
    # n=5, c=2 (3 failures); selecting all 5 (k=5) is guaranteed to include a
    # success since c >= 1.
    assert pass_at_k(n=5, c=2, k=5) == 1.0


def test_pass_at_k_is_zero_when_nothing_succeeded() -> None:
    assert pass_at_k(n=5, c=0, k=3) == 0.0


def test_pass_at_k_matches_known_example() -> None:
    # n=10, c=5, k=3: by the eq.1 estimator, 1 - C(5,3)/C(10,3) = 1 - 10/120
    assert pass_at_k(n=10, c=5, k=3) == pytest.approx(1 - 10 / 120)


def test_pass_at_k_is_monotonically_non_decreasing_in_k() -> None:
    n, c = 8, 3
    values = [pass_at_k(n, c, k) for k in range(1, n + 1)]
    assert values == sorted(values)


def test_pass_at_k_rejects_k_greater_than_n() -> None:
    with pytest.raises(ValueError, match="k"):
        pass_at_k(n=3, c=1, k=4)


def test_pass_at_k_rejects_c_greater_than_n() -> None:
    with pytest.raises(ValueError, match="c"):
        pass_at_k(n=3, c=4, k=1)


# -- compute_metrics -------------------------------------------------------------


def test_compute_metrics_raises_on_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        compute_metrics([])


def test_compute_metrics_hand_verified_values() -> None:
    # case "a": 3/4 succeeded; case "b": 1/4 succeeded.
    results = [
        _result("a", succeeded=True, repetition=0),
        _result("a", succeeded=True, repetition=1),
        _result("a", succeeded=True, repetition=2),
        _result("a", succeeded=False, repetition=3),
        _result("b", succeeded=True, repetition=0),
        _result("b", succeeded=False, repetition=1),
        _result("b", succeeded=False, repetition=2),
        _result("b", succeeded=False, repetition=3),
    ]
    summary = compute_metrics(results)

    assert summary.task_count == 2
    assert summary.total_results == 8
    assert summary.success_rate == pytest.approx(0.5)
    assert summary.pass_at_1 == pytest.approx(0.5)
    assert summary.consistency == pytest.approx(0.75)
    assert summary.mean_variance == pytest.approx(0.25)
    assert summary.failure_probability == pytest.approx(0.5)
    assert summary.average_repetitions == pytest.approx(4.0)


def test_pass_at_1_can_differ_from_pooled_success_rate_with_unequal_repetitions() -> None:
    # case "a": 1 repetition, succeeded. case "b": 3 repetitions, all failed.
    # Pooled success rate = 1/4 = 0.25, but pass@1 averaged per-task = mean(1.0, 0.0) = 0.5.
    results = [
        _result("a", succeeded=True),
        _result("b", succeeded=False, repetition=0),
        _result("b", succeeded=False, repetition=1),
        _result("b", succeeded=False, repetition=2),
    ]
    summary = compute_metrics(results)
    assert summary.success_rate == pytest.approx(0.25)
    assert summary.pass_at_1 == pytest.approx(0.5)


def test_pass_at_k_skips_tasks_with_fewer_than_k_repetitions() -> None:
    results = [
        _result("a", succeeded=True, repetition=0),  # only 1 repetition
        _result("b", succeeded=True, repetition=0),
        _result("b", succeeded=False, repetition=1),
    ]
    summary = compute_metrics(results, k=2)
    # Only "b" has >= 2 repetitions; pass@2 for b: n=2, c=1 -> 1 - C(1,2)/C(2,2)... n-c=1 < k=2 -> 1.0
    assert summary.pass_at_k == pytest.approx(1.0)


def test_pass_at_k_is_none_when_no_task_has_enough_repetitions() -> None:
    results = [_result("a", succeeded=True), _result("b", succeeded=True)]
    summary = compute_metrics(results, k=5)
    assert summary.pass_at_k is None


def test_mean_variance_is_zero_when_no_task_has_repeated_trials() -> None:
    results = [_result("a", succeeded=True), _result("b", succeeded=False)]
    summary = compute_metrics(results)
    assert summary.mean_variance == 0.0


def test_cost_per_success_ignores_results_without_cost_data() -> None:
    results = [
        _result("a", succeeded=True, cost_usd=0.01),
        _result("b", succeeded=True, cost_usd=0.03),
        _result("c", succeeded=True, cost_usd=None),
        _result("d", succeeded=False, cost_usd=0.5),  # failed: excluded
    ]
    summary = compute_metrics(results)
    assert summary.cost_per_success_usd == pytest.approx(0.02)


def test_cost_per_success_is_none_when_no_cost_data_present() -> None:
    results = [_result("a", succeeded=True, cost_usd=None)]
    summary = compute_metrics(results)
    assert summary.cost_per_success_usd is None


def test_mean_latency_per_success_only_counts_successes() -> None:
    results = [
        _result("a", succeeded=True, latency_ms=10.0),
        _result("b", succeeded=True, latency_ms=30.0),
        _result("c", succeeded=False, latency_ms=1000.0),
    ]
    summary = compute_metrics(results)
    assert summary.mean_latency_ms_per_success == pytest.approx(20.0)


def test_mean_latency_is_none_when_no_successes() -> None:
    results = [_result("a", succeeded=False)]
    summary = compute_metrics(results)
    assert summary.mean_latency_ms_per_success is None


def test_structured_output_validity_rate_only_counts_json_schema_results() -> None:
    results = [
        _result("a", succeeded=True, grader_kind=GraderKind.JSON_SCHEMA),
        _result("b", succeeded=False, grader_kind=GraderKind.JSON_SCHEMA),
        _result("c", succeeded=True, grader_kind=GraderKind.EXACT_MATCH),
    ]
    summary = compute_metrics(results)
    assert summary.structured_output_validity_rate == pytest.approx(0.5)


def test_structured_output_validity_rate_is_none_without_json_schema_results() -> None:
    results = [_result("a", succeeded=True, grader_kind=GraderKind.EXACT_MATCH)]
    summary = compute_metrics(results)
    assert summary.structured_output_validity_rate is None


def test_always_successful_task_has_zero_variance_and_full_consistency() -> None:
    results = [_result("a", succeeded=True, repetition=i) for i in range(5)]
    summary = compute_metrics(results)
    assert summary.consistency == 1.0
    assert summary.mean_variance == 0.0
    assert summary.success_rate == 1.0
