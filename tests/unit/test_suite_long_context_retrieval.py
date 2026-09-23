"""Tests for the long_context_retrieval_v1 suite: determinism, difficulty knob, grading."""

from paretoguard.evals import BenchmarkRunner
from paretoguard.evals.suites import long_context_retrieval_v1 as suite_module
from paretoguard.providers.mock import MockProvider


def test_build_suite_is_deterministic_given_same_seed() -> None:
    suite_a = suite_module.build_suite(seed=7, cases_per_difficulty=2)
    suite_b = suite_module.build_suite(seed=7, cases_per_difficulty=2)
    assert [c.ground_truth.expected_substrings for c in suite_a.cases] == [
        c.ground_truth.expected_substrings for c in suite_b.cases
    ]


def test_all_difficulty_levels_are_represented() -> None:
    suite = suite_module.build_suite(seed=1, cases_per_difficulty=2)
    tags = {tag for case in suite.cases for tag in case.tags}
    assert set(suite_module.DIFFICULTIES) <= tags


def test_difficulty_controls_document_length() -> None:
    suite = suite_module.build_suite(seed=1, cases_per_difficulty=1)
    by_level = {
        tag: case for case in suite.cases for tag in case.tags if tag in suite_module.DIFFICULTIES
    }
    short_len = len(by_level["short"].messages[0].content)
    long_len = len(by_level["long"].messages[0].content)
    assert long_len > short_len


def test_needle_code_appears_exactly_once_in_document() -> None:
    suite = suite_module.build_suite(seed=1, cases_per_difficulty=1)
    for case in suite.cases:
        code = (case.ground_truth.expected_substrings or [""])[0]
        document = case.messages[0].content
        assert document.count(code) == 1


async def test_end_to_end_run_against_mock_provider_succeeds() -> None:
    suite = suite_module.build_suite(seed=1, cases_per_difficulty=2)
    runner = BenchmarkRunner(MockProvider(), model="mock-strong")
    result = await runner.run(suite)
    assert len(result.results) == suite.case_count
    assert all(r.succeeded for r in result.results)
