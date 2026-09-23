"""Tests for the tool_use_v1 suite: determinism, tool coverage, grading."""

from paretoguard.evals import BenchmarkRunner
from paretoguard.evals.suites import tool_use_v1 as suite_module
from paretoguard.providers.mock import MockProvider


def test_build_suite_is_deterministic_given_same_seed() -> None:
    suite_a = suite_module.build_suite(seed=7, num_cases=15)
    suite_b = suite_module.build_suite(seed=7, num_cases=15)
    assert [c.ground_truth.expected_tool_calls for c in suite_a.cases] == [
        c.ground_truth.expected_tool_calls for c in suite_b.cases
    ]


def test_all_three_tools_appear_across_enough_cases() -> None:
    suite = suite_module.build_suite(seed=1, num_cases=30)
    used_tools = {
        c.ground_truth.expected_tool_calls[0].name
        for c in suite.cases
        if c.ground_truth.expected_tool_calls
    }
    assert used_tools == {"calculator", "inventory_lookup", "order_lookup"}


def test_every_case_offers_all_three_tool_specs() -> None:
    suite = suite_module.build_suite(seed=1, num_cases=5)
    for case in suite.cases:
        assert {t.name for t in case.tools} == {"calculator", "inventory_lookup", "order_lookup"}


def test_tool_schemas_have_no_shell_or_arbitrary_execution() -> None:
    for tool in suite_module.TOOLS:
        assert "shell" not in tool.description.lower()
        assert "exec" not in tool.name.lower()
        assert "eval" not in tool.name.lower()


async def test_end_to_end_run_against_mock_provider_succeeds() -> None:
    suite = suite_module.build_suite(seed=1, num_cases=10)
    runner = BenchmarkRunner(MockProvider(), model="mock-strong")
    result = await runner.run(suite)
    assert len(result.results) == 10
    assert all(r.succeeded for r in result.results)


async def test_wrong_tool_call_is_graded_as_failure() -> None:
    suite = suite_module.build_suite(seed=1, num_cases=1)
    case = suite.cases[0]
    # Deliberately point the mock at the wrong tool to confirm grading catches it.
    wrong_tool = (
        "order_lookup"
        if case.ground_truth.expected_tool_calls[0].name != "order_lookup"
        else "calculator"
    )
    broken_case = case.model_copy(
        update={
            "metadata": {
                "mock_scenario": "success",
                "mock_tool_calls": [{"name": wrong_tool, "arguments": {}}],
            }
        }
    )
    broken_suite = suite.model_copy(update={"cases": [broken_case]})
    runner = BenchmarkRunner(MockProvider(), model="mock-strong")
    result = await runner.run(broken_suite)
    assert not result.results[0].succeeded
