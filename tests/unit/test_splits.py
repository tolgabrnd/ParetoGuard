"""Unit tests for group-aware train/val/test splitting."""

import pytest

from paretoguard.routing.splits import (
    SplitRatios,
    assign_splits,
    default_group_key,
    group_aware_split,
    verify_no_group_leakage,
)


def test_default_group_key_prefers_template_id() -> None:
    assert default_group_key("task-1", "tmpl-a") == "tmpl-a"


def test_default_group_key_falls_back_to_task_id() -> None:
    assert default_group_key("task-1", None) == "task-1"


def test_split_ratios_rejects_non_summing_to_one() -> None:
    with pytest.raises(ValueError):
        SplitRatios(train=0.5, val=0.5, test=0.5)


def test_group_aware_split_covers_every_group_exactly_once() -> None:
    groups = [f"g{i}" for i in range(20)]
    assignment = group_aware_split(groups, seed=1)
    assert set(assignment) == set(groups)
    assert set(assignment.values()) <= {"train", "val", "test"}


def test_group_aware_split_is_deterministic_given_same_seed() -> None:
    groups = [f"g{i}" for i in range(30)]
    a = group_aware_split(groups, seed=42)
    b = group_aware_split(groups, seed=42)
    assert a == b


def test_group_aware_split_roughly_matches_configured_ratios() -> None:
    groups = [f"g{i}" for i in range(1000)]
    assignment = group_aware_split(groups, ratios=SplitRatios(train=0.8, val=0.1, test=0.1), seed=0)
    counts = {"train": 0, "val": 0, "test": 0}
    for split in assignment.values():
        counts[split] += 1
    assert 750 < counts["train"] < 850
    assert 50 < counts["val"] < 150
    assert 50 < counts["test"] < 150


def test_assign_splits_keeps_same_group_together() -> None:
    rows = [("task-1", 0), ("task-1", 1), ("task-1", 2), ("task-2", 0)]
    labels = assign_splits(rows, group_key_fn=lambda r: r[0], seed=0)
    task_1_labels = {labels[i] for i, r in enumerate(rows) if r[0] == "task-1"}
    assert len(task_1_labels) == 1  # all 3 repetitions of task-1 landed in the same split


def test_assign_splits_groups_by_template_not_individual_task_id() -> None:
    """Near-duplicate tasks sharing a template must never split across
    train/test — the core leakage guard this module exists for."""
    rows = [(f"numeric-{i:03d}", "numeric:multiply_add") for i in range(20)]
    labels = assign_splits(rows, group_key_fn=lambda r: r[1], seed=0)
    assert len(set(labels)) == 1  # the whole template group is one split


def test_verify_no_group_leakage_passes_for_a_clean_split() -> None:
    rows = [("task-1", 0), ("task-1", 1), ("task-2", 0)]
    labels = assign_splits(rows, group_key_fn=lambda r: r[0], seed=0)
    verify_no_group_leakage(rows, labels, group_key_fn=lambda r: r[0])  # must not raise


def test_verify_no_group_leakage_detects_a_hand_rolled_leak() -> None:
    rows = [("task-1", 0), ("task-1", 1)]
    bad_labels = ["train", "test"]  # same task_id split across train and test
    with pytest.raises(ValueError, match="leaking"):
        verify_no_group_leakage(rows, bad_labels, group_key_fn=lambda r: r[0])


def test_verify_no_group_leakage_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        verify_no_group_leakage([("a", 0)], [], group_key_fn=lambda r: r[0])
