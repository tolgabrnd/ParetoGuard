"""Unit tests for the optional PyTorch router baseline.

Skipped entirely if torch isn't installed (it's an optional extra — see
paretoguard.routing.torch_router's module docstring) rather than failing
the suite; `uv sync --extra torch` installs it for CI/dev machines that
want this coverage.
"""

import pytest

torch = pytest.importorskip("torch")

import polars as pl  # noqa: E402

from paretoguard.core.features import TaskFeatures  # noqa: E402
from paretoguard.core.models import InferenceRequest, Message, ModelSpec, Role  # noqa: E402
from paretoguard.evals.matrix import MatrixRow  # noqa: E402
from paretoguard.routing.dataset import build_dataset  # noqa: E402
from paretoguard.routing.learned import evaluate_calibration  # noqa: E402
from paretoguard.routing.torch_router import (  # noqa: E402
    TorchRouter,
    TorchRouterModel,
    train_torch_router_model,
)
from paretoguard.routing.types import CandidateProfile, RoutingRequest, candidate_key  # noqa: E402


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


def _row(task_id: str, task_family: str, model: str, succeeded: bool) -> MatrixRow:
    return MatrixRow(
        task_id=task_id,
        repetition=0,
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


def _dataset() -> pl.DataFrame:
    # A simple, linearly-separable-enough relationship (unlike the XOR
    # fixture used to stress-test the sklearn baselines): "strong" mostly
    # succeeds, "weak" mostly fails, regardless of family.
    rows = []
    for i in range(60):
        rows.append(_row(f"t-{i:03d}", "family-a" if i % 2 == 0 else "family-b", "strong", True))
        rows.append(_row(f"t-{i:03d}", "family-a" if i % 2 == 0 else "family-b", "weak", False))
    return build_dataset(rows, seed=0)


def test_train_torch_router_model_rejects_no_train_rows() -> None:
    empty_train_df = _dataset().filter(pl.col("split") != "train")
    with pytest.raises(ValueError, match="no training rows"):
        train_torch_router_model(empty_train_df, epochs=5)


def test_train_torch_router_model_learns_strong_beats_weak() -> None:
    df = _dataset()
    model = train_torch_router_model(df, epochs=100, seed=0)
    assert isinstance(model, TorchRouterModel)
    strong = model.predict_proba(_features("family-a"), "mock", "strong")
    weak = model.predict_proba(_features("family-a"), "mock", "weak")
    assert 0.0 <= strong <= 1.0
    assert 0.0 <= weak <= 1.0
    assert strong > weak


def test_torch_router_model_reports_uncalibrated() -> None:
    df = _dataset()
    model = train_torch_router_model(df, epochs=20)
    assert model.calibrated is False
    assert model.model_kind == "torch_mlp"


def test_evaluate_calibration_works_with_torch_model() -> None:
    """evaluate_calibration (built for LearnedRouterModel) must also accept
    a TorchRouterModel — they share the same predict_proba_record interface."""
    df = _dataset()
    model = train_torch_router_model(df, epochs=100)
    metrics = evaluate_calibration(model, df, split="test")
    assert metrics.n > 0
    assert 0.0 <= metrics.brier_score <= 1.0


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
        request=request, features=_features(task_family), candidates=candidates, profiles=profiles
    )


def test_torch_router_selects_the_stronger_candidate() -> None:
    df = _dataset()
    model = train_torch_router_model(df, epochs=150, seed=0)
    router = TorchRouter(model)
    decision = router.route(_routing_request("family-a"))
    assert decision.selected_model == "strong"
    assert "torch_mlp" in decision.explanation
