"""Unit tests for RoutedBenchmarkRunner: Task -> features -> Router ->
RoutingDecision -> selected Provider/Model -> Runtime -> Grader -> EvalResult.
"""

from paretoguard.core.models import Message, ModelSpec, Role
from paretoguard.evals.models import EvalCase, EvalSuite, GraderConfig, GraderKind, GroundTruth
from paretoguard.providers.mock import MockProvider
from paretoguard.routing.execution import RoutedBenchmarkRunner
from paretoguard.routing.static import StaticRouter
from paretoguard.routing.types import CandidateProfile, candidate_key
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
    return EvalSuite(name="routed-test", version="1.0.0", description="", seed=0, cases=cases)


def _candidates() -> list[ModelSpec]:
    return [
        ModelSpec(name="mock-a", provider="mock", context_window=10_000),
        ModelSpec(name="mock-b", provider="mock", context_window=10_000),
    ]


async def test_routes_every_case_through_the_router() -> None:
    router = StaticRouter(provider="mock", model="mock-a")
    runner = RoutedBenchmarkRunner(router, _candidates(), {"mock": MockProvider()})
    result = await runner.run(_suite())

    assert len(result.results) == 2
    assert len(result.decisions) == 2
    assert all(r.succeeded for r in result.results)
    assert all(d.selected_model == "mock-a" for d in result.decisions)


async def test_manifest_records_router_name() -> None:
    router = StaticRouter(provider="mock", model="mock-a")
    runner = RoutedBenchmarkRunner(router, _candidates(), {"mock": MockProvider()})
    result = await runner.run(_suite())
    assert result.manifest.router_name == "static"


async def test_rejects_candidate_with_no_configured_provider() -> None:
    import pytest

    candidates = [ModelSpec(name="orphan", provider="ghost", context_window=1000)]
    with pytest.raises(ValueError, match="ghost"):
        RoutedBenchmarkRunner(
            StaticRouter(provider="mock", model="orphan"), candidates, {"mock": MockProvider()}
        )


async def test_persists_routing_decisions_to_the_store() -> None:
    router = StaticRouter(provider="mock", model="mock-a")
    with ExperimentStore(":memory:") as store:
        runner = RoutedBenchmarkRunner(router, _candidates(), {"mock": MockProvider()}, store=store)
        result = await runner.run(_suite())
        decisions_df = store.routing_decisions_df(run_id=result.run_id)
        assert decisions_df.height == 2
        assert set(decisions_df["selected_model"].to_list()) == {"mock-a"}


async def test_dispatches_to_the_correct_provider_for_the_selected_candidate() -> None:
    """Two different providers under two different names — confirms the
    router's selection actually determines which Runtime executes."""
    provider_x = MockProvider(name="provider-x")
    provider_y = MockProvider(name="provider-y")
    candidates = [
        ModelSpec(name="model-x", provider="provider-x", context_window=10_000),
        ModelSpec(name="model-y", provider="provider-y", context_window=10_000),
    ]
    router = StaticRouter(provider="provider-y", model="model-y")
    runner = RoutedBenchmarkRunner(
        router, candidates, {"provider-x": provider_x, "provider-y": provider_y}
    )
    result = await runner.run(_suite())
    assert all(d.selected_model == "model-y" for d in result.decisions)
    assert all(r.succeeded for r in result.results)


async def test_profiles_are_passed_through_to_the_router() -> None:
    """A RuleRouter given task-family-scoped profiles should honor them
    end-to-end through the routed pipeline."""
    from paretoguard.routing.rule import RuleRouter

    profiles = {
        candidate_key("mock", "mock-a"): CandidateProfile(predicted_success=0.1),
        candidate_key("mock", "mock-b"): CandidateProfile(predicted_success=0.9),
    }
    runner = RoutedBenchmarkRunner(
        RuleRouter(), _candidates(), {"mock": MockProvider()}, profiles=profiles
    )
    result = await runner.run(_suite())
    assert all(d.selected_model == "mock-b" for d in result.decisions)
