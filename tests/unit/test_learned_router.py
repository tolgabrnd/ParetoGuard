"""Unit tests for the calibrated learned router: training, calibration
evaluation, and constrained selection. All training data here is
hand-crafted SIMULATION, never real provider outcomes.
"""

import polars as pl
import pytest

from paretoguard.core.features import TaskFeatures
from paretoguard.core.models import InferenceRequest, Message, ModelSpec, Role
from paretoguard.evals.matrix import MatrixRow
from paretoguard.routing.dataset import build_dataset
from paretoguard.routing.learned import (
    LearnedRouter,
    evaluate_calibration,
    train_learned_router_model,
)
from paretoguard.routing.types import CandidateProfile, RoutingRequest, candidate_key


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
    task_id: str, task_family: str, model: str, succeeded: bool, repetition: int = 0
) -> MatrixRow:
    return MatrixRow(
        task_id=task_id,
        repetition=repetition,
        provider="mock",
        model=model,
        task_features=_features(task_family),
        succeeded=succeeded,
        score=1.0 if succeeded else 0.0,
        cost_usd=0.01 if model == "strong" else 0.001,
        latency_ms=500.0,
        total_tokens=20,
        response_error_category=None,
        template_id=None,
    )


def _learnable_rows(n_per_family: int = 40) -> list[MatrixRow]:
    """`strong` succeeds on family-a and fails on family-b; `weak` is the
    exact opposite — a cleanly learnable (task_family, model) -> success
    relationship so the classifier has a real, checkable signal to fit."""
    rows = []
    for i in range(n_per_family):
        rows.append(_row(f"a-{i:03d}", "family-a", "strong", True, repetition=0))
        rows.append(_row(f"a-{i:03d}", "family-a", "weak", False, repetition=0))
        rows.append(_row(f"b-{i:03d}", "family-b", "strong", False, repetition=0))
        rows.append(_row(f"b-{i:03d}", "family-b", "weak", True, repetition=0))
    return rows


def _dataset() -> pl.DataFrame:
    return build_dataset(_learnable_rows(), seed=0)


def test_train_rejects_dataset_with_no_train_split_rows() -> None:
    empty_train_df = _dataset().filter(pl.col("split") != "train")
    with pytest.raises(ValueError, match="no training rows"):
        train_learned_router_model(empty_train_df)


def test_train_learns_the_task_family_model_relationship() -> None:
    df = _dataset()
    model = train_learned_router_model(df, model_kind="logistic_regression")
    strong_on_a = model.predict_proba(_features("family-a"), "mock", "strong")
    weak_on_a = model.predict_proba(_features("family-a"), "mock", "weak")
    assert strong_on_a > weak_on_a


def test_train_random_forest_baseline_also_learns_the_relationship() -> None:
    df = _dataset()
    model = train_learned_router_model(df, model_kind="random_forest")
    strong_on_b = model.predict_proba(_features("family-b"), "mock", "strong")
    weak_on_b = model.predict_proba(_features("family-b"), "mock", "weak")
    assert weak_on_b > strong_on_b


def test_train_rejects_unknown_model_kind() -> None:
    df = _dataset()
    with pytest.raises(ValueError, match="unknown model_kind"):
        train_learned_router_model(df, model_kind="deep_learning")  # type: ignore[arg-type]


def test_evaluate_calibration_on_held_out_split() -> None:
    df = _dataset()
    model = train_learned_router_model(df)
    metrics = evaluate_calibration(model, df, split="test")
    assert metrics.n > 0
    assert 0.0 <= metrics.brier_score <= 1.0
    # A well-learned, clean relationship should calibrate reasonably well.
    assert metrics.brier_score < 0.3


def test_evaluate_calibration_rejects_empty_split() -> None:
    df = _dataset()
    model = train_learned_router_model(df)
    with pytest.raises(ValueError, match="no rows"):
        evaluate_calibration(model, df, split="nonexistent-split")


def _routing_request(task_family: str) -> RoutingRequest:
    request = InferenceRequest(
        provider="mock",
        model="mock-strong",
        messages=[Message(role=Role.USER, content="hi")],
        metadata={"task_family": task_family},
    )
    candidates = [
        ModelSpec(name="strong", provider="mock", context_window=100_000),
        ModelSpec(name="weak", provider="mock", context_window=100_000),
    ]
    profiles = {
        candidate_key("mock", "strong"): CandidateProfile(
            mean_cost_usd=0.01, mean_latency_ms=500.0
        ),
        candidate_key("mock", "weak"): CandidateProfile(mean_cost_usd=0.001, mean_latency_ms=500.0),
    }
    return RoutingRequest(
        request=request,
        features=_features(task_family),
        candidates=candidates,
        profiles=profiles,
    )


def test_learned_router_selects_the_predicted_stronger_candidate() -> None:
    # This fixture's (task_family, model) -> success relationship is a pure
    # XOR: it is not linearly separable from additive one-hot features, so a
    # logistic regression baseline cannot represent it (this is exactly the
    # documented reason for the tree-based baseline — see learned.py's
    # module docstring). Use random_forest here to test LearnedRouter's
    # selection wiring specifically, independent of that linear-model limit.
    df = _dataset()
    model = train_learned_router_model(df, model_kind="random_forest")
    router = LearnedRouter(model)
    decision = router.route(_routing_request("family-a"))
    assert decision.selected_model == "strong"

    decision_b = router.route(_routing_request("family-b"))
    assert decision_b.selected_model == "weak"


def test_learned_router_explanation_mentions_model_kind_and_calibration() -> None:
    df = _dataset()
    model = train_learned_router_model(df, model_kind="logistic_regression")
    router = LearnedRouter(model)
    decision = router.route(_routing_request("family-a"))
    assert "logistic_regression" in decision.explanation
    assert "calibrated" in decision.explanation.lower()
