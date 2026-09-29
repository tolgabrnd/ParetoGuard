"""Unit tests for deterministic request-id derivation."""

from paretoguard.core.ids import deterministic_request_id


def test_same_parts_yield_the_same_id() -> None:
    a = deterministic_request_id("task-1", "0", "sim", "profile-a")
    b = deterministic_request_id("task-1", "0", "sim", "profile-a")
    assert a == b


def test_different_parts_yield_different_ids() -> None:
    a = deterministic_request_id("task-1", "0", "sim", "profile-a")
    b = deterministic_request_id("task-1", "1", "sim", "profile-a")
    c = deterministic_request_id("task-2", "0", "sim", "profile-a")
    d = deterministic_request_id("task-1", "0", "sim", "profile-b")
    assert len({a, b, c, d}) == 4
