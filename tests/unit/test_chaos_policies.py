"""Unit tests for fault scheduling (ConstantProbability/StepRangeSchedule)."""

import pytest

from paretoguard.chaos.policies import ConstantProbability, StepRangeSchedule


def test_constant_probability_is_the_same_at_every_step() -> None:
    schedule = ConstantProbability(0.3)
    assert schedule.probability_at(0) == 0.3
    assert schedule.probability_at(1000) == 0.3


def test_constant_probability_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="probability"):
        ConstantProbability(1.5)
    with pytest.raises(ValueError, match="probability"):
        ConstantProbability(-0.1)


def test_step_range_schedule_matches_the_canonical_example() -> None:
    schedule = StepRangeSchedule(((0, 0.01), (100, 0.25), (200, 0.01)))
    assert schedule.probability_at(0) == 0.01
    assert schedule.probability_at(99) == 0.01
    assert schedule.probability_at(100) == 0.25
    assert schedule.probability_at(199) == 0.25
    assert schedule.probability_at(200) == 0.01
    assert schedule.probability_at(10_000) == 0.01


def test_step_range_schedule_rejects_unsorted_breakpoints() -> None:
    with pytest.raises(ValueError, match="sorted"):
        StepRangeSchedule(((100, 0.1), (0, 0.2)))


def test_step_range_schedule_rejects_duplicate_start_steps() -> None:
    with pytest.raises(ValueError, match="repeat"):
        StepRangeSchedule(((0, 0.1), (0, 0.2)))


def test_step_range_schedule_rejects_empty_breakpoints() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        StepRangeSchedule(())


def test_step_range_schedule_rejects_out_of_range_probability() -> None:
    with pytest.raises(ValueError, match="probability"):
        StepRangeSchedule(((0, 1.5),))
