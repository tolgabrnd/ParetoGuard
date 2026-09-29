"""Unit tests for MatrixRunner: the offline every-task-x-every-candidate
training-matrix generator."""

import pytest

from paretoguard.core.models import Message, ModelSpec, Role
from paretoguard.evals.matrix import MatrixConfig, MatrixRunner
from paretoguard.evals.models import EvalCase, EvalSuite, GraderConfig, GraderKind, GroundTruth
from paretoguard.providers.mock import SCENARIO_METADATA_KEY, MockProvider, MockScenario
from paretoguard.providers.profiled_mock import ProfiledMockProvider, SimulatedModelProfile
from paretoguard.storage import ExperimentStore


def _suite() -> EvalSuite:
    cases = [
        EvalCase(
            case_id="case-1",
            messages=[Message(role=Role.USER, content="say hello")],
            grader=GraderConfig(kind=GraderKind.EXACT_MATCH),
            ground_truth=GroundTruth(text="hello"),
            metadata={"mock_answer_text": "hello"},
        ),
        EvalCase(
            case_id="case-2",
            messages=[Message(role=Role.USER, content="say goodbye")],
            grader=GraderConfig(kind=GraderKind.EXACT_MATCH),
            ground_truth=GroundTruth(text="goodbye"),
            metadata={"mock_answer_text": "goodbye"},
        ),
    ]
    return EvalSuite(name="matrix-test", version="1.0.0", description="", seed=0, cases=cases)


def _candidates() -> list[ModelSpec]:
    return [
        ModelSpec(name="mock-a", provider="mock", context_window=10_000),
        ModelSpec(name="mock-b", provider="mock", context_window=10_000),
    ]


async def test_matrix_runner_produces_one_row_per_case_and_candidate() -> None:
    runner = MatrixRunner(_candidates(), {"mock": MockProvider()})
    result = await runner.run(_suite())
    assert len(result.rows) == 2 * 2  # 2 cases x 2 candidates x 1 repetition

    pairs = {(r.task_id, r.model) for r in result.rows}
    assert pairs == {
        ("case-1", "mock-a"),
        ("case-1", "mock-b"),
        ("case-2", "mock-a"),
        ("case-2", "mock-b"),
    }


async def test_matrix_runner_grades_each_candidate_independently() -> None:
    runner = MatrixRunner(_candidates(), {"mock": MockProvider()})
    result = await runner.run(_suite())
    for row in result.rows:
        assert row.succeeded  # MockProvider echoes mock_answer_text exactly
        assert row.score == 1.0


async def test_matrix_runner_task_features_identical_across_candidates_for_same_task() -> None:
    """The defining leakage guard: task_features must be a pure function of
    the task, never of which candidate happened to run it."""
    runner = MatrixRunner(_candidates(), {"mock": MockProvider()})
    result = await runner.run(_suite())
    case_1_rows = [r for r in result.rows if r.task_id == "case-1"]
    assert len({r.task_features for r in case_1_rows}) == 1


async def test_matrix_runner_records_failure_without_dropping_the_row() -> None:
    cases = [
        EvalCase(
            case_id="fails",
            messages=[Message(role=Role.USER, content="x")],
            grader=GraderConfig(kind=GraderKind.EXACT_MATCH),
            ground_truth=GroundTruth(text="whatever"),
            metadata={SCENARIO_METADATA_KEY: MockScenario.SERVER_ERROR.value},
        )
    ]
    suite = EvalSuite(name="fail-test", version="1.0.0", description="", seed=0, cases=cases)
    runner = MatrixRunner(_candidates(), {"mock": MockProvider()})
    result = await runner.run(suite)
    assert len(result.rows) == 2  # both candidates still produce a row
    assert all(not r.succeeded for r in result.rows)
    assert all(r.response_error_category == "provider_failure" for r in result.rows)


async def test_matrix_runner_respects_repetitions() -> None:
    runner = MatrixRunner(
        _candidates(), {"mock": MockProvider()}, config=MatrixConfig(repetitions=3)
    )
    result = await runner.run(_suite())
    assert len(result.rows) == 2 * 2 * 3


def test_matrix_runner_rejects_candidate_with_no_configured_provider() -> None:
    candidates = [ModelSpec(name="orphan", provider="ghost-provider", context_window=1000)]
    with pytest.raises(ValueError, match="ghost-provider"):
        MatrixRunner(candidates, {"mock": MockProvider()})


async def test_matrix_run_result_manifest_labels_it_as_matrix() -> None:
    runner = MatrixRunner(_candidates(), {"mock": MockProvider()})
    result = await runner.run(_suite())
    assert result.manifest.router_name == "__matrix__"
    assert result.manifest.suite_name == "matrix-test"


async def test_matrix_runner_persists_through_the_normal_experiment_pipeline() -> None:
    with ExperimentStore(":memory:") as store:
        runner = MatrixRunner(_candidates(), {"mock": MockProvider()}, store=store)
        result = await runner.run(_suite())

        manifest = store.get_run(result.run_id)
        assert manifest is not None
        assert manifest.router_name == "__matrix__"

        eval_results = store.get_eval_results(result.run_id)
        assert len(eval_results) == len(result.rows) == 4  # 2 cases x 2 candidates

        requests_df = store.requests_df(run_id=result.run_id)
        responses_df = store.responses_df(run_id=result.run_id)
        assert requests_df.height == 4
        assert responses_df.height == 4

        # case_id embeds the candidate since (task_id, repetition) alone is
        # no longer unique across a matrix's multiple candidates per task.
        case_ids = {r.case_id for r in eval_results}
        assert case_ids == {"case-1::mock-a", "case-1::mock-b", "case-2::mock-a", "case-2::mock-b"}


async def test_matrix_runner_is_reproducible_across_independent_runs() -> None:
    """Regression test: ProfiledMockProvider's outcome is a function of
    request_id, which previously defaulted to a fresh random UUID4 per
    InferenceRequest — meaning the *same* (seed, suite, candidates) produced
    a *different* simulated outcome on every process run. Two completely
    independent MatrixRunner/ProfiledMockProvider instances (same seed) must
    now produce byte-identical succeeded/score/cost/latency rows."""
    candidates = [ModelSpec(name="profile-a", provider="sim", context_window=10_000)]
    profile = {"profile-a": SimulatedModelProfile(default_success_probability=0.5)}

    async def run_once() -> list[tuple[str, str, bool, float]]:
        runner = MatrixRunner(
            candidates,
            {"sim": ProfiledMockProvider(profile, seed=3)},
            config=MatrixConfig(repetitions=5),
        )
        result = await runner.run(_suite())
        return [(r.task_id, r.model, r.succeeded, r.latency_ms) for r in result.rows]

    first = await run_once()
    second = await run_once()
    assert first == second
