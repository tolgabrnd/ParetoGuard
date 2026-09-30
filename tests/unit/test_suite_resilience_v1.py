"""Tests for the resilience_v1 suite: determinism, fixed difficulty."""

from paretoguard.evals import BenchmarkRunner
from paretoguard.evals.suites import resilience_v1 as suite_module
from paretoguard.providers.mock import MockProvider


def test_build_suite_is_deterministic_given_same_seed() -> None:
    suite_a = suite_module.build_suite(seed=3, num_cases=20)
    suite_b = suite_module.build_suite(seed=3, num_cases=20)
    assert [c.ground_truth.text for c in suite_a.cases] == [
        c.ground_truth.text for c in suite_b.cases
    ]


def test_case_count_matches_requested() -> None:
    suite = suite_module.build_suite(seed=0, num_cases=17)
    assert suite.case_count == 17


def test_case_ids_are_unique_even_beyond_topic_count() -> None:
    suite = suite_module.build_suite(seed=0, num_cases=40)  # > len(_TOPICS)
    assert len({c.case_id for c in suite.cases}) == 40


async def test_every_case_succeeds_deterministically_against_mock_provider() -> None:
    """The whole point of this suite: absent any injected chaos, every task
    succeeds with certainty — failure in the resilience benchmark must come
    only from injected faults, never from task difficulty."""
    suite = suite_module.build_suite(seed=0, num_cases=24)
    runner = BenchmarkRunner(MockProvider(), model="mock-strong")
    result = await runner.run(suite)
    assert len(result.results) == 24
    assert all(r.succeeded for r in result.results)
