"""Builds an ML-ready `polars.DataFrame` from `MatrixRunner` output, with
group-aware train/val/test split labels attached.

This is the single place `MatrixRow` objects become model input — the
feature/label/candidate-identity column split below is the leakage
contract for everything downstream (Commit 22's learned router, Commit 23's
optional Torch baseline): only `FEATURE_COLUMNS` + `CANDIDATE_COLUMNS` may
ever be used as model input. `LABEL_COLUMNS` are the outcome being
predicted — using one as an input feature is exactly the "never use
information unavailable at routing time" leak this repo's build plan
prohibits.
"""

import polars as pl

from paretoguard.evals.matrix import MatrixRow
from paretoguard.routing.splits import (
    SplitRatios,
    assign_splits,
    default_group_key,
    verify_no_group_leakage,
)

FEATURE_COLUMNS: tuple[str, ...] = (
    "input_tokens_estimate",
    "max_output_tokens",
    "context_tokens_estimate",
    "requires_structured_output",
    "requires_tool_use",
    "tool_count",
    "task_family",
    "schema_complexity",
    "numeric_density",
    "expected_output_length",
    "expected_step_count",
)
"""Every column here comes straight from `TaskFeatures.as_feature_dict()` —
routing-time-safe by construction (see `paretoguard.core.features`)."""

CANDIDATE_COLUMNS: tuple[str, ...] = ("provider", "model")
"""The candidate identity being scored — a legitimate input, since the
learned router predicts P(success | task, candidate), not P(success | task)
alone. Not a "feature" of the task itself, so kept separate from
FEATURE_COLUMNS."""

LABEL_COLUMNS: tuple[str, ...] = (
    "succeeded",
    "score",
    "cost_usd",
    "latency_ms",
    "total_tokens",
    "response_error_category",
)
"""Only known after the candidate actually ran this task — never a model
input. `succeeded` is Commit 22's prediction target."""

NON_FEATURE_METADATA_COLUMNS: tuple[str, ...] = ("task_id", "repetition", "template_id", "split")
"""Bookkeeping columns: identify/group/split rows, but are neither features
nor labels and must not be fed to a model either (`task_id`/`template_id`
would let a model memorize per-task/per-template outcomes directly, the
same failure mode grouped splitting exists to prevent)."""


def build_dataset(
    rows: list[MatrixRow], *, ratios: SplitRatios | None = None, seed: int = 0
) -> pl.DataFrame:
    """Flattens `rows` into a DataFrame with one row per (task, repetition,
    candidate), verifies the resulting split has no group leakage, and
    raises rather than silently returning a leaky or empty dataset.
    """
    if not rows:
        raise ValueError("cannot build a dataset from zero rows")

    def group_key(row: MatrixRow) -> str:
        return default_group_key(row.task_id, row.template_id)

    splits = assign_splits(rows, group_key_fn=group_key, ratios=ratios, seed=seed)
    verify_no_group_leakage(rows, splits, group_key_fn=group_key)

    records = [
        {
            "task_id": row.task_id,
            "repetition": row.repetition,
            "template_id": row.template_id,
            "provider": row.provider,
            "model": row.model,
            **row.task_features.as_feature_dict(),
            "succeeded": row.succeeded,
            "score": row.score,
            "cost_usd": row.cost_usd,
            "latency_ms": row.latency_ms,
            "total_tokens": row.total_tokens,
            "response_error_category": row.response_error_category,
            "split": split,
        }
        for row, split in zip(rows, splits, strict=True)
    ]
    # infer_schema_length=None scans every row rather than a sample: several
    # columns (template_id, cost_usd, response_error_category, ...) are None
    # for most rows in a typical dataset, and a short sample can miss the
    # rows that reveal a column's real (str/float) type.
    return pl.DataFrame(records, infer_schema_length=None)


def feature_matrix(df: pl.DataFrame, *, split: str | None = None) -> pl.DataFrame:
    """Returns only `FEATURE_COLUMNS` + `CANDIDATE_COLUMNS`, optionally
    filtered to one split — the columns Commit 22's learned router is
    allowed to see as input. Never includes `LABEL_COLUMNS` or
    `NON_FEATURE_METADATA_COLUMNS`."""
    frame = df if split is None else df.filter(pl.col("split") == split)
    return frame.select(list(FEATURE_COLUMNS) + list(CANDIDATE_COLUMNS))


def labels(df: pl.DataFrame, *, split: str | None = None, column: str = "succeeded") -> pl.Series:
    frame = df if split is None else df.filter(pl.col("split") == split)
    return frame[column]
