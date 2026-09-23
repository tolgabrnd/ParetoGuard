"""Tests for the structured_extraction_v1 suite: determinism and end-to-end grading."""

from paretoguard.evals import BenchmarkRunner
from paretoguard.evals.suites import structured_extraction_v1
from paretoguard.providers.mock import MockProvider


def test_build_suite_is_deterministic_given_same_seed() -> None:
    suite_a = structured_extraction_v1.build_suite(seed=7, num_cases=5)
    suite_b = structured_extraction_v1.build_suite(seed=7, num_cases=5)
    assert [c.ground_truth.expected_fields for c in suite_a.cases] == [
        c.ground_truth.expected_fields for c in suite_b.cases
    ]


def test_build_suite_different_seeds_differ() -> None:
    suite_a = structured_extraction_v1.build_suite(seed=1, num_cases=5)
    suite_b = structured_extraction_v1.build_suite(seed=2, num_cases=5)
    assert (
        suite_a.cases[0].ground_truth.expected_fields
        != suite_b.cases[0].ground_truth.expected_fields
    )


def test_suite_metadata() -> None:
    suite = structured_extraction_v1.build_suite(seed=1, num_cases=3)
    assert suite.name == "structured_extraction_v1"
    assert suite.version == "1.0.0"
    assert suite.seed == 1
    assert suite.case_count == 3


def test_every_case_has_required_fields_in_schema() -> None:
    suite = structured_extraction_v1.build_suite(seed=1, num_cases=3)
    for case in suite.cases:
        assert case.structured_output_schema is not None
        required = case.structured_output_schema["required"]
        assert set(required) == set(case.ground_truth.expected_fields or {})


async def test_end_to_end_run_against_mock_provider_succeeds() -> None:
    suite = structured_extraction_v1.build_suite(seed=1, num_cases=5)
    runner = BenchmarkRunner(MockProvider(), model="mock-strong")
    result = await runner.run(suite)
    assert len(result.results) == 5
    assert all(r.succeeded for r in result.results)
    assert all(r.score == 1.0 for r in result.results)
