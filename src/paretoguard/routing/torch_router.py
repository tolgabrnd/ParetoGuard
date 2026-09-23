"""An optional PyTorch MLP baseline for P(success | task, candidate),
benchmarked against (never a replacement for) the scikit-learn baselines in
`paretoguard.routing.learned`.

**Not imported by `paretoguard.routing.__init__`** — unlike every other
router in this package, this module is reached only by importing it
directly (`from paretoguard.routing.torch_router import ...`). That's
deliberate: PyTorch is an optional dependency (`pyproject.toml`'s `torch`
extra) specifically so the default `paretoguard` install stays lightweight;
making `import paretoguard.routing` transitively require torch would defeat
that. `torch` itself is imported lazily inside `train_torch_router_model`/
`TorchRouterModel`, not at this module's top level, so even importing this
file without torch installed only fails when you actually try to train or
predict, with a clear message (`_require_torch`).

**Compact and defensible, not deep-learning-for-its-own-sake**: engineered
task/candidate features (the same `paretoguard.routing.encoding` pipeline
the scikit-learn baselines use) feed a small 2-hidden-layer MLP producing
one per-candidate success probability — no embeddings, no architecture
search, no GPU requirement. If it performs worse than logistic regression/
random forest on a given dataset (measured via `evaluate_calibration`, the
same Brier-score function used for the sklearn baselines), that is a valid,
reportable outcome — this module does not tune itself to always "win".
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from paretoguard.core.features import TaskFeatures
from paretoguard.routing.dataset import CANDIDATE_COLUMNS, FEATURE_COLUMNS
from paretoguard.routing.encoding import categorical_row, design_row, numeric_row
from paretoguard.routing.learned import LearnedRouter

if TYPE_CHECKING:
    import torch.nn as nn

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for paretoguard.routing.torch_router but is not "
            "installed. Install it with the optional extra: `uv sync --extra torch` "
            "(or `pip install paretoguard[torch]`)."
        )


def _build_mlp(input_dim: int, hidden_sizes: tuple[int, ...]) -> "nn.Module":
    _require_torch()
    layers: list[nn.Module] = []
    in_dim = input_dim
    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, 1))
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


@dataclass
class TorchRouterModel:
    """Mirrors `paretoguard.routing.learned.LearnedRouterModel`'s public
    interface (`predict_proba`/`predict_proba_record`, `model_kind`,
    `calibrated`) so `LearnedRouter`/`TorchRouter` can use either
    interchangeably without knowing which backend produced it.

    `calibrated` is always `False`: this MLP outputs a raw sigmoid, with no
    equivalent applied here to `CalibratedClassifierCV`'s post-hoc
    calibration — an honest, documented gap, not a claim of calibration
    this module doesn't back up. Check `evaluate_calibration`'s Brier score
    before trusting these probabilities as well-calibrated.
    """

    net: Any  # torch.nn.Module, in eval mode
    encoder: Any  # sklearn.preprocessing.OneHotEncoder, fitted
    feature_mean: np.ndarray[Any, Any]
    feature_std: np.ndarray[Any, Any]
    model_kind: str = "torch_mlp"
    calibrated: bool = False

    def predict_proba_record(self, record: dict[str, Any]) -> float:
        _require_torch()
        row = design_row(record, self.encoder)
        standardized = (row - self.feature_mean) / self.feature_std
        with torch.no_grad():
            tensor = torch.tensor(standardized, dtype=torch.float32).reshape(1, -1)
            return float(self.net(tensor).item())

    def predict_proba(self, features: TaskFeatures, provider: str, model: str) -> float:
        record = {**features.as_feature_dict(), "provider": provider, "model": model}
        return self.predict_proba_record(record)


def train_torch_router_model(
    df: pl.DataFrame,
    *,
    hidden_sizes: tuple[int, ...] = (32, 16),
    epochs: int = 200,
    lr: float = 0.01,
    seed: int = 0,
) -> TorchRouterModel:
    """Fits on `df.filter(split == "train")` only — same contract as
    `paretoguard.routing.learned.train_learned_router_model`. Numeric
    features are standardized (mean/std from the training split, reused at
    prediction time) since raw scales (e.g. token counts in the hundreds
    vs. binary flags) otherwise slow gradient-based training.
    """
    _require_torch()
    from sklearn.preprocessing import OneHotEncoder

    train_df = df.filter(pl.col("split") == "train")
    if train_df.height == 0:
        raise ValueError("no training rows (split == 'train') in the given dataset")

    records = train_df.select([*FEATURE_COLUMNS, *CANDIDATE_COLUMNS]).to_dicts()
    y = train_df["succeeded"].to_numpy().astype(np.float32)

    encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_rows = [categorical_row(r) for r in records]
    encoder.fit(categorical_rows)

    numeric = np.array([numeric_row(r) for r in records], dtype=float)
    categorical = encoder.transform(categorical_rows).toarray()
    X = np.hstack([numeric, categorical]).astype(np.float32)

    feature_mean = X.mean(axis=0)
    feature_std = X.std(axis=0)
    feature_std[feature_std == 0] = 1.0  # avoid divide-by-zero for constant columns
    X_standardized = (X - feature_mean) / feature_std

    torch.manual_seed(seed)
    net = _build_mlp(X_standardized.shape[1], hidden_sizes)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    X_tensor = torch.tensor(X_standardized, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    net.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        predictions = net(X_tensor)
        loss = loss_fn(predictions, y_tensor)
        loss.backward()
        optimizer.step()
    net.eval()

    return TorchRouterModel(
        net=net, encoder=encoder, feature_mean=feature_mean, feature_std=feature_std
    )


class TorchRouter(LearnedRouter):
    """`LearnedRouter` specialized for a `TorchRouterModel` — identical
    constrained-selection behavior (delegates to `ParetoRouter`, same as
    `LearnedRouter`), kept as its own named class for discoverability and
    parity with every other named router in this package."""
