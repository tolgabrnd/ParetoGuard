"""A calibrated supervised model predicting P(success | task, candidate),
plus the `LearnedRouter` that uses it.

Two interpretable scikit-learn baselines, chosen at training time:
logistic regression (the default — simple, calibratable, a sane starting
point) and a shallow random forest (a justified tree-based baseline: task
features interact non-additively — e.g. "structured output AND low
context window" is worse than either alone — which a linear model can't
represent without hand-built interaction terms). Both are deliberately
simple; Commit 23's optional PyTorch baseline is compared against these,
not a replacement for them.

No pandas: this repo's core paths use Polars (see docs/BUILD_PLAN.md).
Categorical columns (task_family/provider/model) are one-hot encoded via
`sklearn.preprocessing.OneHotEncoder` operating on plain string arrays —
no DataFrame-with-named-columns dependency needed for that step.
"""

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import polars as pl
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from paretoguard.core.features import TaskFeatures
from paretoguard.core.models import RoutingDecision
from paretoguard.routing.dataset import CANDIDATE_COLUMNS, FEATURE_COLUMNS
from paretoguard.routing.pareto_router import ParetoObjective, ParetoRouter
from paretoguard.routing.protocol import Router
from paretoguard.routing.types import CandidateProfile, RoutingRequest, candidate_key

ModelKind = Literal["logistic_regression", "random_forest"]

_NUMERIC_FEATURE_COLUMNS: tuple[str, ...] = (
    "input_tokens_estimate",
    "max_output_tokens",
    "context_tokens_estimate",
    "requires_structured_output",
    "requires_tool_use",
    "tool_count",
    "schema_complexity",
    "numeric_density",
    "expected_output_length",
    "expected_step_count",
)
_CATEGORICAL_COLUMNS: tuple[str, ...] = ("task_family", "provider", "model")
_UNKNOWN_CATEGORY = "__unknown__"


def _numeric_row(record: dict[str, Any]) -> list[float]:
    return [
        float(record[col]) if record.get(col) is not None else 0.0
        for col in _NUMERIC_FEATURE_COLUMNS
    ]


def _categorical_row(record: dict[str, Any]) -> list[str]:
    return [
        str(record[col]) if record.get(col) is not None else _UNKNOWN_CATEGORY
        for col in _CATEGORICAL_COLUMNS
    ]


@dataclass
class LearnedRouterModel:
    """A fitted classifier plus its fitted categorical encoder — the pair
    must always travel together, since predictions depend on both."""

    classifier: ClassifierMixin
    encoder: Any  # sklearn.preprocessing.OneHotEncoder, fitted
    model_kind: ModelKind
    calibrated: bool

    def _design_row(self, record: dict[str, Any]) -> np.ndarray[Any, Any]:
        numeric = np.array(_numeric_row(record), dtype=float)
        categorical = self.encoder.transform([_categorical_row(record)]).toarray()[0]
        return np.concatenate([numeric, categorical])

    def predict_proba_record(self, record: dict[str, Any]) -> float:
        """Predicts P(success) from a flat record with `FEATURE_COLUMNS` +
        `CANDIDATE_COLUMNS` keys — the same shape `build_dataset` produces."""
        row = self._design_row(record).reshape(1, -1)
        return float(self.classifier.predict_proba(row)[0, 1])

    def predict_proba(self, features: TaskFeatures, provider: str, model: str) -> float:
        record = {**features.as_feature_dict(), "provider": provider, "model": model}
        return self.predict_proba_record(record)


def train_learned_router_model(
    df: pl.DataFrame,
    *,
    model_kind: ModelKind = "logistic_regression",
    calibrate: bool = True,
    seed: int = 0,
) -> LearnedRouterModel:
    """Fits on `df.filter(split == "train")` only. Passing a full,
    unsplit dataset would train on rows this function can't tell apart
    from held-out ones — callers must build `df` via
    `paretoguard.routing.dataset.build_dataset`, which always assigns splits.
    """
    from sklearn.preprocessing import OneHotEncoder

    train_df = df.filter(pl.col("split") == "train")
    if train_df.height == 0:
        raise ValueError("no training rows (split == 'train') in the given dataset")

    records = train_df.select([*FEATURE_COLUMNS, *CANDIDATE_COLUMNS]).to_dicts()
    y = train_df["succeeded"].to_numpy().astype(int)

    encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_rows = [_categorical_row(r) for r in records]
    encoder.fit(categorical_rows)

    numeric = np.array([_numeric_row(r) for r in records], dtype=float)
    categorical = encoder.transform(categorical_rows).toarray()
    X = np.hstack([numeric, categorical])

    base: ClassifierMixin
    if model_kind == "logistic_regression":
        base = LogisticRegression(max_iter=1000, random_state=seed)
    elif model_kind == "random_forest":
        base = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=seed)
    else:
        raise ValueError(f"unknown model_kind: {model_kind!r}")

    class_counts = np.bincount(y)
    min_class_count = int(class_counts.min())
    can_calibrate = calibrate and len(class_counts) > 1 and min_class_count >= 3
    if can_calibrate:
        cv = min(3, min_class_count)
        classifier = cast(ClassifierMixin, CalibratedClassifierCV(base, method="sigmoid", cv=cv))
        classifier.fit(X, y)
        calibrated = True
    else:
        base.fit(X, y)
        classifier = base
        calibrated = False

    return LearnedRouterModel(
        classifier=classifier, encoder=encoder, model_kind=model_kind, calibrated=calibrated
    )


@dataclass(frozen=True)
class CalibrationMetrics:
    """Evaluated on a held-out split only (never the rows the model was
    fit on — see `evaluate_calibration`'s docstring)."""

    n: int
    brier_score: float
    mean_predicted_success: float
    mean_actual_success: float


def evaluate_calibration(
    model: LearnedRouterModel, df: pl.DataFrame, *, split: str = "test"
) -> CalibrationMetrics:
    """Brier score (mean squared error between predicted probability and
    the binary outcome — lower is better, 0 is a perfect probabilistic
    forecast) plus a coarse over/under-confidence check
    (mean_predicted_success vs mean_actual_success). Always call with
    `split="test"` (or `"val"` for model selection) — evaluating on
    `split="train"` would silently defeat the point of holding data out.
    """
    eval_df = df.filter(pl.col("split") == split)
    if eval_df.height == 0:
        raise ValueError(f"no rows for split={split!r}")

    records = eval_df.select([*FEATURE_COLUMNS, *CANDIDATE_COLUMNS]).to_dicts()
    y_true = eval_df["succeeded"].to_numpy().astype(int)
    y_pred = np.array([model.predict_proba_record(r) for r in records])

    return CalibrationMetrics(
        n=eval_df.height,
        brier_score=float(brier_score_loss(y_true, y_pred)),
        mean_predicted_success=float(y_pred.mean()),
        mean_actual_success=float(y_true.mean()),
    )


class LearnedRouter(Router):
    """Predicts P(success | task, candidate) with a fitted
    `LearnedRouterModel`, then delegates constrained selection to
    `ParetoRouter` — this router's only job is producing a calibrated
    `predicted_success` per candidate; it reuses the already-tested
    constrained-objective/Pareto-frontier selection logic rather than
    inventing a second one. Cost/latency estimates still come from
    `RoutingRequest.profiles` (this model doesn't predict them); a
    candidate with no cost/latency profile is excluded by the inner
    `ParetoRouter`, same as it would be if called directly.
    """

    def __init__(
        self,
        model: LearnedRouterModel,
        *,
        objective: ParetoObjective = ParetoObjective.MAXIMIZE_SUCCESS,
    ) -> None:
        self.name = "learned"
        self._model = model
        self._pareto = ParetoRouter(objective)

    def route(self, routing_request: RoutingRequest) -> RoutingDecision:
        enriched_profiles = dict(routing_request.profiles)
        for candidate in routing_request.candidates:
            key = candidate_key(candidate.provider, candidate.name)
            predicted = self._model.predict_proba(
                routing_request.features, candidate.provider, candidate.name
            )
            base = enriched_profiles.get(key, CandidateProfile())
            enriched_profiles[key] = CandidateProfile(
                predicted_success=predicted,
                mean_cost_usd=base.mean_cost_usd,
                mean_latency_ms=base.mean_latency_ms,
                task_family=routing_request.features.task_family,
                simulated=base.simulated,
            )

        inner_request = RoutingRequest(
            request=routing_request.request,
            features=routing_request.features,
            candidates=routing_request.candidates,
            constraints=routing_request.constraints,
            health=routing_request.health,
            profiles=enriched_profiles,
        )
        decision = self._pareto.route(inner_request)
        return decision.model_copy(
            update={
                "explanation": (
                    f"LearnedRouter ({self._model.model_kind}, "
                    f"calibrated={self._model.calibrated}): predicted_success "
                    f"from the fitted model, then constrained selection via "
                    f"ParetoRouter — {decision.explanation}"
                )
            }
        )
