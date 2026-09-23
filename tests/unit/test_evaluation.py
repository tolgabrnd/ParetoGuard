"""Unit tests for offline router evaluation against the exhaustive matrix."""

import pytest

from paretoguard.core.features import TaskFeatures
from paretoguard.core.models import ModelSpec
from paretoguard.evals.matrix import MatrixRow
from paretoguard.routing.dataset import build_dataset
from paretoguard.routing.evaluation import (
    best_fixed_model,
    build_profiles_from_train_split,
    cheapest_fixed_model,
    cost_at_target_success,
    evaluate_router_offline,
    fixed_model_stats,
    oracle_upper_bound,
    success_at_fixed_budget,
)
from paretoguard.routing.static import StaticRouter


def _features(task_family: str) -> TaskFeatures:
    return TaskFeatures(
        input_tokens_estimate=10,
        max_output_tokens=50,
        context_tokens_estimate=60,
        requires_structured_output=False,
        requires_tool_use=False,
        tool_count=0,
        task_family=task_family,
        schema_complexity=0,
        numeric_density=0.1,
        expected_output_length=50,
    )


def _row(
    task_id: str, model: str, succeeded: bool, cost: float, task_family: str = "fam"
) -> MatrixRow:
    return MatrixRow(
        task_id=task_id,
        repetition=0,
        provider="mock",
        model=model,
        task_features=_features(task_family),
        succeeded=succeeded,
        score=1.0 if succeeded else 0.0,
        cost_usd=cost,
        latency_ms=100.0,
        total_tokens=10,
        response_error_category=None,
        template_id=None,
    )


def _dataset_two_models(n: int = 150):
    # "good" always succeeds at higher cost; "cheap" succeeds half the time
    # at lower cost — a genuine fixed-model trade-off.
    rows = []
    for i in range(n):
        rows.append(_row(f"t-{i:03d}", "good", True, 0.05))
        rows.append(_row(f"t-{i:03d}", "cheap", i % 2 == 0, 0.01))
    return build_dataset(rows, seed=0)


def _candidates() -> list[ModelSpec]:
    return [
        ModelSpec(name="good", provider="mock", context_window=100_000),
        ModelSpec(name="cheap", provider="mock", context_window=100_000),
    ]


def test_fixed_model_stats_reports_both_models() -> None:
    df = _dataset_two_models()
    stats = fixed_model_stats(df, split="test")
    models = set(stats["model"].to_list())
    assert models <= {"good", "cheap"}


def test_best_fixed_model_is_the_always_succeeding_one() -> None:
    df = _dataset_two_models()
    assert best_fixed_model(df, split="test") == "good"


def test_cheapest_fixed_model_is_the_low_cost_one() -> None:
    df = _dataset_two_models()
    assert cheapest_fixed_model(df, split="test") == "cheap"


def test_cost_at_target_success_finds_cheapest_model_meeting_bar() -> None:
    df = _dataset_two_models()
    # "good" always succeeds (rate 1.0), so a target of 1.0 should require it.
    cost = cost_at_target_success(df, target_success=1.0, split="test")
    assert cost == pytest.approx(0.05)


def test_cost_at_target_success_none_when_unreachable() -> None:
    df = _dataset_two_models()
    assert cost_at_target_success(df, target_success=1.1, split="test") is None


def test_success_at_fixed_budget_finds_best_within_budget() -> None:
    df = _dataset_two_models()
    success = success_at_fixed_budget(df, budget_usd=0.02, split="test")
    # Only "cheap" (0.01) fits a 0.02 budget.
    assert success is not None
    assert success < 1.0


def test_success_at_fixed_budget_none_when_nothing_fits() -> None:
    df = _dataset_two_models()
    assert success_at_fixed_budget(df, budget_usd=0.001, split="test") is None


def test_oracle_upper_bound_is_at_least_any_fixed_model_success() -> None:
    df = _dataset_two_models()
    oracle = oracle_upper_bound(df, _candidates(), split="test")
    best = fixed_model_stats(df, split="test")["success_rate"].max()
    assert oracle >= best


def test_build_profiles_from_train_split_has_both_candidates() -> None:
    df = _dataset_two_models()
    profiles = build_profiles_from_train_split(df)
    assert len(profiles) == 2
    for profile in profiles.values():
        assert profile.simulated is True
        assert profile.predicted_success is not None


def test_evaluate_router_offline_scores_static_router_against_recorded_outcomes() -> None:
    df = _dataset_two_models()
    router = StaticRouter(provider="mock", model="good")
    summary = evaluate_router_offline(router, df, _candidates(), split="test")
    assert summary.router_name == "static"
    assert summary.success_rate == pytest.approx(1.0)  # "good" always succeeds
    assert summary.selection_distribution == {"good": summary.n_tasks}
    assert summary.regret_vs_oracle == pytest.approx(0.0)  # matches the oracle every time


def test_evaluate_router_offline_regret_is_positive_for_a_worse_router() -> None:
    df = _dataset_two_models()
    router = StaticRouter(provider="mock", model="cheap")
    summary = evaluate_router_offline(router, df, _candidates(), split="test")
    assert summary.success_rate < 1.0
    assert summary.regret_vs_oracle > 0.0


def test_evaluate_router_offline_rejects_missing_split() -> None:
    df = _dataset_two_models()
    router = StaticRouter(provider="mock", model="good")
    with pytest.raises(ValueError, match="no rows"):
        evaluate_router_offline(router, df, _candidates(), split="nonexistent")
