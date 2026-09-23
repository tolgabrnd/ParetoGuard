"""Task-aware / group-aware train/validation/test splitting for the offline
learned-router dataset (Commit 21/22).

**Why group-aware, not random, splitting**: a plain random split over
individual `MatrixRow`s can put two near-duplicate tasks — the same
underlying template with only its literal numbers/names changed — on
opposite sides of the train/test boundary. A model can then "solve" a test
case not by generalizing but by having memorized its template's behavior
from an almost-identical training case, making evaluation artificially
easy. Grouping by template/record family and assigning whole groups to one
split closes that hole.

**What actually needs grouping in this repo's suites, and what doesn't**:
of the four built-in suites (`paretoguard.evals.suites`), two —
`numeric_reasoning_v1` (three distinguishable arithmetic-operation
templates) and `tool_use_v1` (three distinguishable tool kinds) — generate
cases from a small, enumerable set of *distinguishable* templates, so they
tag `EvalCase.metadata["template_id"]` (see those suites' `build_suite`).
`structured_extraction_v1` and `long_context_retrieval_v1` generate every
case from one single template with only randomized field values — that
level of variation is the ordinary, expected difference between any two
training/test examples of the same task, not a duplication risk, so they
are not tagged and simply group by `task_id` (which still correctly keeps
every repetition of the *same* case together on one side of the split).

**Known limitation**: this scheme is only as good as `template_id` tagging.
A future suite with real near-duplicate structure that fails to tag it will
not be protected here — this module cannot detect duplication it isn't told
about. Document any new suite's template structure in its own docstring, as
the two suites above do.
"""

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

SplitName = Literal["train", "val", "test"]


def default_group_key(task_id: str, template_id: str | None) -> str:
    """`template_id` when the task's suite provided one, else `task_id`
    itself (see the module docstring for which suites need which)."""
    return template_id if template_id is not None else task_id


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.7
    val: float = 0.15
    test: float = 0.15

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"train+val+test must sum to 1.0, got {total}")
        if min(self.train, self.val, self.test) < 0:
            raise ValueError("ratios must be >= 0")


def group_aware_split(
    group_keys: Sequence[str], *, ratios: SplitRatios | None = None, seed: int = 0
) -> dict[str, SplitName]:
    """Deterministically assigns every *distinct* group key to exactly one
    of train/val/test.

    Uses a seeded hash of each group key (not `random.shuffle`, which would
    require materializing and sorting the whole group list identically
    every call) to rank groups, then cuts the ranked list at the configured
    ratios — the same `(group_key, seed)` always yields the same
    assignment, independent of input order or how many times this is
    called, with no external state to persist.
    """
    ratios = ratios or SplitRatios()
    unique_groups = sorted(set(group_keys))

    def _rank(key: str) -> str:
        return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()

    ordered = sorted(unique_groups, key=_rank)
    n = len(ordered)
    n_train = round(n * ratios.train)
    n_val = round(n * ratios.val)
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    assignment: dict[str, SplitName] = {}
    for i, group in enumerate(ordered):
        if i < n_train:
            assignment[group] = "train"
        elif i < n_train + n_val:
            assignment[group] = "val"
        else:
            assignment[group] = "test"
    return assignment


def assign_splits[T](
    rows: Sequence[T],
    *,
    group_key_fn: Callable[[T], str],
    ratios: SplitRatios | None = None,
    seed: int = 0,
) -> list[SplitName]:
    """Returns one split label per row, in the same order as `rows`. Every
    row sharing a group (per `group_key_fn`) always receives the same
    label — that's the entire leakage guard this module provides.
    """
    group_keys = [group_key_fn(row) for row in rows]
    assignment = group_aware_split(group_keys, ratios=ratios, seed=seed)
    return [assignment[key] for key in group_keys]


def verify_no_group_leakage[T](
    rows: Sequence[T], labels: Sequence[SplitName], *, group_key_fn: Callable[[T], str]
) -> None:
    """Raises `ValueError` if any group's rows span more than one split
    label. Intended as a cheap assertion after `assign_splits`/any
    hand-rolled split, for tests and dataset-build sanity checks — this is
    the thing "document this carefully" ultimately has to be backed by."""
    if len(rows) != len(labels):
        raise ValueError(f"rows ({len(rows)}) and labels ({len(labels)}) must be the same length")
    group_to_splits: dict[str, set[SplitName]] = {}
    for row, label in zip(rows, labels, strict=True):
        key = group_key_fn(row)
        group_to_splits.setdefault(key, set()).add(label)

    leaking = {key: splits for key, splits in group_to_splits.items() if len(splits) > 1}
    if leaking:
        raise ValueError(
            f"group(s) span more than one split, leaking across the boundary: {leaking}"
        )
