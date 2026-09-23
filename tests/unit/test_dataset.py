"""Unit tests for the ML dataset builder (routing/dataset.py)."""

import pytest

from paretoguard.core.features import TaskFeatures
from paretoguard.evals.matrix import MatrixRow
from paretoguard.routing.dataset import (
    CANDIDATE_COLUMNS,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    build_dataset,
    feature_matrix,
    labels,
)


def _features(task_family: str = "numeric_reasoning_v1") -> TaskFeatures:
    return TaskFeatures(
        input_tokens_estimate=10,
        max_output_tokens=50,
        context_tokens_estimate=60,
        requires_structured_output=False,
        requires_tool_use=False,
        tool_count=0,
        task_family=task_family,
        schema_complexity=0,
        numeric_density=0.2,
        expected_output_length=50,
    )


def _rows(n: int = 30) -> list[MatrixRow]:
    rows = []
    for i in range(n):
        rows.append(
            MatrixRow(
                task_id=f"task-{i:03d}",
                repetition=0,
                provider="mock",
                model="mock-a",
                task_features=_features(),
                succeeded=i % 2 == 0,
                score=1.0 if i % 2 == 0 else 0.0,
                cost_usd=0.001,
                latency_ms=100.0,
                total_tokens=20,
                response_error_category=None,
                template_id=None,
            )
        )
    return rows


def test_build_dataset_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="zero rows"):
        build_dataset([])


def test_build_dataset_has_one_row_per_matrix_row() -> None:
    df = build_dataset(_rows(30))
    assert df.height == 30


def test_build_dataset_assigns_a_split_to_every_row() -> None:
    df = build_dataset(_rows(30))
    assert set(df["split"].unique().to_list()) <= {"train", "val", "test"}
    assert df["split"].null_count() == 0


def test_build_dataset_keeps_same_task_id_in_one_split() -> None:
    # Two repetitions of the same task_id must land in the same split.
    row_0 = MatrixRow(
        task_id="shared-task",
        repetition=0,
        provider="mock",
        model="mock-a",
        task_features=_features(),
        succeeded=True,
        score=1.0,
        cost_usd=0.001,
        latency_ms=100.0,
        total_tokens=20,
        response_error_category=None,
        template_id=None,
    )
    row_1 = row_0.__class__(**{**row_0.__dict__, "repetition": 1})
    df = build_dataset([row_0, row_1, *_rows(20)])
    shared = df.filter(df["task_id"] == "shared-task")
    assert shared["split"].n_unique() == 1


def test_feature_matrix_excludes_labels_and_bookkeeping_columns() -> None:
    df = build_dataset(_rows(30))
    features = feature_matrix(df)
    assert set(features.columns) == set(FEATURE_COLUMNS) | set(CANDIDATE_COLUMNS)
    for leaky_column in LABEL_COLUMNS:
        assert leaky_column not in features.columns
    assert "task_id" not in features.columns
    assert "template_id" not in features.columns


def test_feature_matrix_can_filter_by_split() -> None:
    df = build_dataset(_rows(60))
    train_features = feature_matrix(df, split="train")
    assert train_features.height <= df.height
    assert train_features.height == df.filter(df["split"] == "train").height


def test_labels_returns_succeeded_by_default() -> None:
    df = build_dataset(_rows(10))
    y = labels(df)
    assert y.to_list() == df["succeeded"].to_list()
