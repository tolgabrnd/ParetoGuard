"""Tests for the numeric_reasoning_v1 suite: determinism, variation, grading."""

from paretoguard.evals import BenchmarkRunner
from paretoguard.evals.suites import numeric_reasoning_v1
from paretoguard.providers.mock import MockProvider


def test_build_suite_is_deterministic_given_same_seed() -> None:
    suite_a = numeric_reasoning_v1.build_suite(seed=7, num_cases=10)
    suite_b = numeric_reasoning_v1.build_suite(seed=7, num_cases=10)
    assert [c.ground_truth.number for c in suite_a.cases] == [
        c.ground_truth.number for c in suite_b.cases
    ]
    assert [c.messages[0].content for c in suite_a.cases] == [
        c.messages[0].content for c in suite_b.cases
    ]


def test_answers_are_not_all_identical() -> None:
    # Pins that this isn't a trivially-solvable suite where every case has the
    # same answer (a naive constant-output heuristic would then "solve" it).
    suite = numeric_reasoning_v1.build_suite(seed=1, num_cases=20)
    answers = {c.ground_truth.number for c in suite.cases}
    assert len(answers) > 10


def test_prompts_include_a_distractor_sentence() -> None:
    suite = numeric_reasoning_v1.build_suite(seed=1, num_cases=10)
    for case in suite.cases:
        assert "no bearing on this problem" in case.messages[0].content


def test_multiple_problem_templates_are_used() -> None:
    suite = numeric_reasoning_v1.build_suite(seed=1, num_cases=30)
    has_boxes = any("boxes" in c.messages[0].content for c in suite.cases)
    has_crates = any("crates" in c.messages[0].content for c in suite.cases)
    has_factory = any("factory" in c.messages[0].content for c in suite.cases)
    assert has_boxes and has_crates and has_factory


async def test_end_to_end_run_against_mock_provider_succeeds() -> None:
    suite = numeric_reasoning_v1.build_suite(seed=1, num_cases=5)
    runner = BenchmarkRunner(MockProvider(), model="mock-strong")
    result = await runner.run(suite)
    assert len(result.results) == 5
    assert all(r.succeeded for r in result.results)
