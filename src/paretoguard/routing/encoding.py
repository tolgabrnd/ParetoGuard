"""Shared numeric design-matrix encoding for every learned router backend
(scikit-learn in `paretoguard.routing.learned`, optional PyTorch in
`paretoguard.routing.torch_router`).

Kept in one place so both backends featurize a `(TaskFeatures, provider,
model)` row identically — a mismatch here would mean the two backends
aren't actually comparable, defeating the point of benchmarking one against
the other.
"""

from typing import Any

import numpy as np

NUMERIC_FEATURE_COLUMNS: tuple[str, ...] = (
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
CATEGORICAL_COLUMNS: tuple[str, ...] = ("task_family", "provider", "model")
UNKNOWN_CATEGORY = "__unknown__"


def numeric_row(record: dict[str, Any]) -> list[float]:
    return [
        float(record[col]) if record.get(col) is not None else 0.0
        for col in NUMERIC_FEATURE_COLUMNS
    ]


def categorical_row(record: dict[str, Any]) -> list[str]:
    return [
        str(record[col]) if record.get(col) is not None else UNKNOWN_CATEGORY
        for col in CATEGORICAL_COLUMNS
    ]


def design_row(record: dict[str, Any], encoder: Any) -> np.ndarray[Any, Any]:
    """`encoder` is a fitted `sklearn.preprocessing.OneHotEncoder` over
    `categorical_row`-shaped rows — shared so a PyTorch model's input
    tensor is built from the exact same one-hot columns, in the same
    order, as the scikit-learn baselines."""
    numeric = np.array(numeric_row(record), dtype=float)
    categorical = encoder.transform([categorical_row(record)]).toarray()[0]
    return np.concatenate([numeric, categorical])
