"""Unit tests for BenchmarkRunner: execution, ordering, persistence, failure handling."""

import pytest

from paretoguard.core.models import Message, Role, TraceEventType
from paretoguard.evals import (
    BenchmarkConfig,
    BenchmarkRunner,
    EvalCase,
    EvalSuite,
    GraderConfig,
    GraderKind,
    GroundTruth,
)
from paretoguard.providers.mock import SCENARIO_METADATA_KEY, MockProvider, MockScenario
from paretoguard.storage import ExperimentStore


def _case(case_id: str, *, expected: str = "4", scenario: MockScenario | None = None) -> EvalCase:
    metadata: dict[str, object] = {"mock_answer_text": expected}
    if scenario is not None:
        metadata[SCENARIO_METADATA_KEY] = scenario.value
    return EvalCase(
        case_id=case_id,
        messages=[Message(role=Role.USER, content="what is 2+2?")],
        grader=GraderConfig(kind=GraderKind.EXACT_MATCH),
        ground_truth=GroundTruth(text="4"),
        metadata=metadata,
    )


def _suite(cases: list[EvalCase], seed: int = 1) -> EvalSuite:
    return EvalSuite(name="demo_v1", version="1.0.0", description="demo", seed=seed, cases=cases)


async def test_run_produces_one_result_per_case() -> None:
    runner = BenchmarkRunner(MockProvider(), model="mock-strong")
    suite = _suite([_case("a"), _case("b"), _case("c")])
    result = await runner.run(suite)
    assert len(result.results) == 3
    assert all(r.succeeded for r in result.results)


async def test_run_with_repetitions_produces_repetitions_per_case() -> None:
    runner = BenchmarkRunner(
        MockProvider(), model="mock-strong", config=BenchmarkConfig(repetitions=3)
    )
    suite = _suite([_case("a"), _case("b")])
    result = await runner.run(suite)
    assert len(result.results) == 6
    assert result.manifest.repetitions == 3
    assert result.manifest.task_count == 2


async def test_results_are_in_deterministic_case_then_repetition_order() -> None:
    runner = BenchmarkRunner(
        MockProvider(),
        model="mock-strong",
        config=BenchmarkConfig(repetitions=2, max_concurrency=8),
    )
    suite = _suite([_case("a"), _case("b"), _case("c")])
    result = await runner.run(suite)
    ordered = [(r.case_id, r.repetition) for r in result.results]
    assert ordered == [
        ("a", 0),
        ("a", 1),
        ("b", 0),
        ("b", 1),
        ("c", 0),
        ("c", 1),
    ]
    assert [r.sequence for r in result.results] == list(range(6))


async def test_manifest_captures_suite_metadata() -> None:
    runner = BenchmarkRunner(MockProvider(), model="mock-strong")
    suite = _suite([_case("a")], seed=42)
    result = await runner.run(suite)
    assert result.manifest.suite_name == "demo_v1"
    assert result.manifest.suite_version == "1.0.0"
    assert result.manifest.seed == 42
    assert result.manifest.router_config == {"provider": "mock", "model": "mock-strong"}


async def test_failed_inference_produces_a_failing_result_not_a_dropped_case() -> None:
    runner = BenchmarkRunner(MockProvider(), model="mock-strong")
    suite = _suite([_case("a", scenario=MockScenario.RATE_LIMIT)])
    result = await runner.run(suite)
    assert len(result.results) == 1
    r = result.results[0]
    assert not r.succeeded
    assert r.response_error_category is not None
    assert r.response_error_category.value == "rate_limit"
    assert "inference failed" in r.explanation


async def test_mixed_success_and_failure_cases_all_remain_inspectable() -> None:
    runner = BenchmarkRunner(MockProvider(), model="mock-strong")
    suite = _suite(
        [
            _case("ok"),
            _case("bad-answer", expected="wrong"),
            _case("infra-fail", scenario=MockScenario.SERVER_ERROR),
        ]
    )
    result = await runner.run(suite)
    by_id = {r.case_id: r for r in result.results}
    assert by_id["ok"].succeeded
    assert not by_id["bad-answer"].succeeded
    assert by_id["bad-answer"].response_error_category is None  # graded, not an inference failure
    assert not by_id["infra-fail"].succeeded
    assert by_id["infra-fail"].response_error_category is not None


async def test_persists_run_requests_responses_and_eval_results() -> None:
    with ExperimentStore(":memory:") as store:
        runner = BenchmarkRunner(MockProvider(), model="mock-strong", store=store)
        suite = _suite([_case("a"), _case("b")])
        result = await runner.run(suite, run_id="run-test-1")

        manifest = store.get_run("run-test-1")
        assert manifest is not None
        assert manifest.suite_name == "demo_v1"

        stored_results = store.get_eval_results("run-test-1")
        assert len(stored_results) == 2
        assert [r.case_id for r in stored_results] == ["a", "b"]

        requests_df = store.requests_df(run_id="run-test-1")
        responses_df = store.responses_df(run_id="run-test-1")
        assert requests_df.height == 2
        assert responses_df.height == 2
        assert len(result.results) == 2


async def test_trace_events_are_persisted_when_store_is_given() -> None:
    with ExperimentStore(":memory:") as store:
        runner = BenchmarkRunner(MockProvider(), model="mock-strong", store=store)
        suite = _suite([_case("a")])
        await runner.run(suite, run_id="run-trace")
        events = store.trace_events_df(run_id="run-trace")
        assert events.height >= 1
        event_types = set(events["event_type"].to_list())
        assert TraceEventType.PROVIDER_CALL.value in event_types


async def test_grading_error_does_not_crash_the_run() -> None:
    # NUMERIC grader requires ground_truth.number; this case's grader/ground_truth
    # are mismatched on purpose to trigger the grader's own ValueError.
    case = EvalCase(
        case_id="misconfigured",
        messages=[Message(role=Role.USER, content="2+2?")],
        grader=GraderConfig(kind=GraderKind.NUMERIC),
        ground_truth=GroundTruth(text="4"),  # missing .number
        metadata={"mock_answer_text": "4"},
    )
    runner = BenchmarkRunner(MockProvider(), model="mock-strong")
    result = await runner.run(_suite([case]))
    assert len(result.results) == 1
    assert not result.results[0].succeeded
    assert "grading error" in result.results[0].explanation


async def test_benchmark_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="repetitions"):
        BenchmarkConfig(repetitions=0)
    with pytest.raises(ValueError, match="max_concurrency"):
        BenchmarkConfig(max_concurrency=0)
