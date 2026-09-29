"""Unit tests for deterministic agent-trajectory grading and AgentSimulator."""

from paretoguard.agents.executor import AgentExecutor, AgentLimits
from paretoguard.agents.simulator import (
    AgentSimulator,
    AgentTask,
    ExpectedTrajectory,
    finalize_trajectory,
    grade_trajectory,
)
from paretoguard.agents.state import AgentStep, AgentTrajectory, TerminationReason
from paretoguard.agents.tools import OrderLookupTool, ShipmentLookupTool
from paretoguard.core.models import (
    FinishReason,
    InferenceRequest,
    InferenceResponse,
    LatencyRecord,
    TokenUsage,
    ToolCall,
    ToolResult,
)
from paretoguard.providers.base import Provider
from paretoguard.runtime import RetryPolicy, Runtime


def _trajectory(
    steps: list[AgentStep], termination: TerminationReason, final_text: str | None = None
) -> AgentTrajectory:
    return AgentTrajectory(
        task_id="t",
        provider="mock",
        model="m",
        steps=steps,
        termination_reason=termination,
        final_output_text=final_text,
    )


def test_grade_trajectory_succeeds_on_correct_sequence_and_answer() -> None:
    steps = [
        AgentStep(
            step_index=0,
            requested_tool="order_lookup",
            arguments_valid=True,
            tool_result=ToolResult(tool_call_id="c1", output={"shipment_id": "SHIP-5001"}),
            latency_ms=1.0,
        ),
    ]
    trajectory = _trajectory(steps, TerminationReason.FINAL_ANSWER, "shipment is SHIP-5001")
    expected = ExpectedTrajectory(
        expected_tool_sequence=["order_lookup"], expected_answer_substring="SHIP-5001"
    )
    outcome = grade_trajectory(trajectory, expected)
    assert outcome.succeeded
    assert outcome.correct_tool_sequence
    assert outcome.correct_tool_selection


def test_grade_trajectory_fails_on_wrong_tool_sequence() -> None:
    steps = [
        AgentStep(
            step_index=0, requested_tool="shipment_lookup", arguments_valid=True, latency_ms=1.0
        )
    ]
    trajectory = _trajectory(steps, TerminationReason.FINAL_ANSWER, "some answer")
    expected = ExpectedTrajectory(expected_tool_sequence=["order_lookup"])
    outcome = grade_trajectory(trajectory, expected)
    assert not outcome.succeeded
    assert not outcome.correct_tool_sequence


def test_grade_trajectory_fails_on_invalid_arguments() -> None:
    steps = [
        AgentStep(
            step_index=0, requested_tool="order_lookup", arguments_valid=False, latency_ms=1.0
        )
    ]
    trajectory = _trajectory(steps, TerminationReason.INVALID_TOOL_ARGUMENTS)
    expected = ExpectedTrajectory(expected_tool_sequence=["order_lookup"])
    outcome = grade_trajectory(trajectory, expected)
    assert not outcome.succeeded
    assert not outcome.all_arguments_valid


def test_grade_trajectory_fails_on_wrong_final_field() -> None:
    trajectory = AgentTrajectory(
        task_id="t",
        provider="mock",
        model="m",
        steps=[],
        termination_reason=TerminationReason.FINAL_ANSWER,
        final_structured_output={"status": "wrong"},
    )
    expected = ExpectedTrajectory(expected_final_field="status", expected_final_value="shipped")
    outcome = grade_trajectory(trajectory, expected)
    assert not outcome.succeeded


def test_grade_trajectory_never_succeeds_on_non_final_termination() -> None:
    trajectory = _trajectory([], TerminationReason.STEP_LIMIT)
    outcome = grade_trajectory(trajectory, ExpectedTrajectory())
    assert not outcome.succeeded


def test_finalize_trajectory_upgrades_final_answer_to_success() -> None:
    trajectory = _trajectory([], TerminationReason.FINAL_ANSWER, "ok")
    expected = ExpectedTrajectory(expected_answer_substring="ok")
    outcome = grade_trajectory(trajectory, expected)
    finalized = finalize_trajectory(trajectory, outcome)
    assert finalized.termination_reason == TerminationReason.SUCCESS


def test_finalize_trajectory_leaves_failed_trajectory_unchanged() -> None:
    trajectory = _trajectory([], TerminationReason.STEP_LIMIT)
    outcome = grade_trajectory(trajectory, ExpectedTrajectory())
    finalized = finalize_trajectory(trajectory, outcome)
    assert finalized.termination_reason == TerminationReason.STEP_LIMIT


class _ScriptedProvider(Provider):
    def __init__(self, responses: list[InferenceResponse]) -> None:
        self.name = "mock"
        self._responses = responses
        self.call_count = 0

    async def complete(self, request: InferenceRequest) -> InferenceResponse:
        response = self._responses[min(self.call_count, len(self._responses) - 1)]
        self.call_count += 1
        return response.model_copy(update={"request_id": request.request_id})


async def test_agent_simulator_runs_and_grades_a_batch() -> None:
    from uuid import uuid4

    responses = [
        InferenceResponse(
            request_id=uuid4(),
            provider="mock",
            model="m",
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=[ToolCall(id="c1", name="order_lookup", arguments={"order_id": "ORD-1001"})],
            token_usage=TokenUsage(input_tokens=3, output_tokens=2),
            latency=LatencyRecord(total_latency_ms=1.0),
        ),
        InferenceResponse(
            request_id=uuid4(),
            provider="mock",
            model="m",
            output_text="the shipment id is SHIP-5001",
            finish_reason=FinishReason.STOP,
            token_usage=TokenUsage(input_tokens=3, output_tokens=2),
            latency=LatencyRecord(total_latency_ms=1.0),
        ),
    ]
    runtime = Runtime(_ScriptedProvider(responses), retry_policy=RetryPolicy(max_attempts=1))
    executor = AgentExecutor(
        runtime, [OrderLookupTool(), ShipmentLookupTool()], limits=AgentLimits()
    )
    simulator = AgentSimulator(executor, provider="mock", model="m")

    task = AgentTask(
        task_id="find-shipment",
        user_prompt="find the shipment id for order ORD-1001",
        expected=ExpectedTrajectory(
            expected_tool_sequence=["order_lookup"], expected_answer_substring="SHIP-5001"
        ),
    )
    results = await simulator.run([task])
    assert len(results) == 1
    assert results[0].outcome.succeeded
    assert results[0].trajectory.termination_reason == TerminationReason.SUCCESS
