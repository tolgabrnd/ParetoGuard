"""Unit tests for AgentExecutor: the hard-bounded, deterministic agent loop."""

from uuid import uuid4

import pytest

from paretoguard.agents import AgentExecutor, AgentLimits, TerminationReason
from paretoguard.agents.tools import CalculatorTool, InventoryLookupTool
from paretoguard.core.models import (
    ErrorInfo,
    FailureCategory,
    FinishReason,
    InferenceRequest,
    InferenceResponse,
    LatencyRecord,
    TokenUsage,
    ToolCall,
)
from paretoguard.providers.base import Provider
from paretoguard.runtime import RetryPolicy, Runtime


class _ScriptedProvider(Provider):
    """Returns pre-scripted responses in order, one per `complete()` call —
    simulates a multi-turn conversation deterministically."""

    def __init__(self, responses: list[InferenceResponse], *, name: str = "mock") -> None:
        self.name = name
        self._responses = responses
        self.call_count = 0

    async def complete(self, request: InferenceRequest) -> InferenceResponse:
        response = self._responses[min(self.call_count, len(self._responses) - 1)]
        self.call_count += 1
        return response.model_copy(update={"request_id": request.request_id})


def _final_answer_response(text: str = "the answer is 42") -> InferenceResponse:
    return InferenceResponse(
        request_id=uuid4(),
        provider="mock",
        model="mock-model",
        output_text=text,
        finish_reason=FinishReason.STOP,
        token_usage=TokenUsage(input_tokens=5, output_tokens=5),
        latency=LatencyRecord(total_latency_ms=10.0),
    )


def _tool_call_response(tool_name: str, arguments: dict) -> InferenceResponse:
    return InferenceResponse(
        request_id=uuid4(),
        provider="mock",
        model="mock-model",
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=[ToolCall(id="call-1", name=tool_name, arguments=arguments)],
        token_usage=TokenUsage(input_tokens=5, output_tokens=2),
        latency=LatencyRecord(total_latency_ms=10.0),
    )


def _failed_response() -> InferenceResponse:
    return InferenceResponse(
        request_id=uuid4(),
        provider="mock",
        model="mock-model",
        finish_reason=FinishReason.ERROR,
        token_usage=TokenUsage(input_tokens=1, output_tokens=0),
        latency=LatencyRecord(total_latency_ms=5.0),
        error=ErrorInfo(category=FailureCategory.PROVIDER_FAILURE, message="boom", retryable=False),
    )


def _executor(
    responses: list[InferenceResponse], *, limits: AgentLimits | None = None
) -> AgentExecutor:
    provider = _ScriptedProvider(responses)
    runtime = Runtime(provider, retry_policy=RetryPolicy(max_attempts=1))
    tools = [CalculatorTool(), InventoryLookupTool()]
    return AgentExecutor(runtime, tools, limits=limits)


async def test_one_step_success() -> None:
    executor = _executor([_final_answer_response("hello")])
    trajectory = await executor.run(
        task_id="t1", provider="mock", model="mock-model", user_prompt="say hello"
    )
    assert trajectory.termination_reason == TerminationReason.FINAL_ANSWER
    assert trajectory.final_output_text == "hello"
    assert trajectory.step_count == 1


async def test_multi_step_success_with_tool_call() -> None:
    executor = _executor(
        [
            _tool_call_response("calculator", {"expression": "2 + 2"}),
            _final_answer_response("the result is 4"),
        ]
    )
    trajectory = await executor.run(
        task_id="t2", provider="mock", model="mock-model", user_prompt="what is 2+2?"
    )
    assert trajectory.termination_reason == TerminationReason.FINAL_ANSWER
    assert trajectory.step_count == 2
    assert trajectory.tool_call_count == 1
    assert trajectory.steps[0].tool_result is not None
    assert trajectory.steps[0].tool_result.output == 4.0
    assert trajectory.steps[0].fault_id is None
    assert trajectory.steps[0].recovery_action is None


async def test_invalid_tool_terminates_explicitly() -> None:
    executor = _executor([_tool_call_response("nonexistent_tool", {})])
    trajectory = await executor.run(
        task_id="t3", provider="mock", model="mock-model", user_prompt="do something"
    )
    assert trajectory.termination_reason == TerminationReason.INVALID_TOOL
    assert trajectory.failure_category == FailureCategory.TOOL_SELECTION_FAILURE


async def test_invalid_tool_arguments_terminates_explicitly() -> None:
    executor = _executor([_tool_call_response("calculator", {"wrong_field": "oops"})])
    trajectory = await executor.run(
        task_id="t4", provider="mock", model="mock-model", user_prompt="compute something"
    )
    assert trajectory.termination_reason == TerminationReason.INVALID_TOOL_ARGUMENTS
    assert trajectory.failure_category == FailureCategory.TOOL_ARGUMENT_FAILURE
    assert trajectory.steps[0].arguments_valid is False


async def test_step_limit_terminates_a_never_ending_conversation() -> None:
    # The scripted provider always returns a tool call, so without a step
    # limit this would never terminate on its own.
    executor = _executor(
        [_tool_call_response("calculator", {"expression": "1"})],
        limits=AgentLimits(max_steps=3, max_tool_calls=100),
    )
    trajectory = await executor.run(
        task_id="t5", provider="mock", model="mock-model", user_prompt="loop forever"
    )
    assert trajectory.termination_reason == TerminationReason.STEP_LIMIT
    assert trajectory.step_count == 3


async def test_tool_call_limit_terminates_before_step_limit() -> None:
    executor = _executor(
        [_tool_call_response("calculator", {"expression": "1"})],
        limits=AgentLimits(max_steps=100, max_tool_calls=2),
    )
    trajectory = await executor.run(
        task_id="t6", provider="mock", model="mock-model", user_prompt="loop forever"
    )
    assert trajectory.termination_reason == TerminationReason.TOOL_CALL_LIMIT
    # tool_call_count counts every *attempted* call (the enforcement point,
    # a conservative safety limit); the 3rd attempt is the one that breached
    # the limit and was rejected before executing, hence only 2 executed.
    assert trajectory.tool_call_count == 3
    executed = sum(1 for s in trajectory.steps if s.tool_result is not None)
    assert executed == 2


async def test_budget_exceeded_stops_before_step_limit() -> None:
    expensive_response = _final_answer_response("done").model_copy(
        update={
            "finish_reason": FinishReason.TOOL_CALLS,
            "tool_calls": [ToolCall(id="c1", name="calculator", arguments={"expression": "1"})],
        }
    )
    # Manually attach a cost via a real CostRecord-free response is awkward;
    # instead assert budget check using cost=None means no accumulation —
    # verify explicitly with a runtime pricing table instead.
    from datetime import date

    from paretoguard.core.config import PricingTable
    from paretoguard.core.models import CostBasis, PricingEntry

    pricing = PricingTable(
        entries=[
            PricingEntry(
                provider="mock",
                model="mock-model",
                input_price_per_million_usd=1_000_000.0,
                output_price_per_million_usd=1_000_000.0,
                version="test",
                effective_date=date(2026, 1, 1),
                basis=CostBasis.SIMULATED,
            )
        ]
    )
    provider = _ScriptedProvider([expensive_response])
    runtime = Runtime(provider, retry_policy=RetryPolicy(max_attempts=1), pricing_table=pricing)
    executor = AgentExecutor(
        runtime, [CalculatorTool()], limits=AgentLimits(max_steps=100, max_cost_usd=0.5)
    )
    trajectory = await executor.run(
        task_id="t7", provider="mock", model="mock-model", user_prompt="spend money"
    )
    assert trajectory.termination_reason == TerminationReason.BUDGET_EXCEEDED
    assert trajectory.failure_category == FailureCategory.BUDGET_EXCEEDED


async def test_provider_failure_terminates_as_unrecoverable() -> None:
    executor = _executor([_failed_response()])
    trajectory = await executor.run(
        task_id="t8", provider="mock", model="mock-model", user_prompt="anything"
    )
    assert trajectory.termination_reason == TerminationReason.UNRECOVERABLE_FAILURE
    assert trajectory.failure_category == FailureCategory.UNRECOVERABLE_STATE


async def test_timeout_terminates_explicitly() -> None:
    executor = _executor(
        [_tool_call_response("calculator", {"expression": "1"})],
        limits=AgentLimits(max_steps=1_000_000, max_tool_calls=1_000_000, max_wall_time_s=0.0001),
    )
    trajectory = await executor.run(
        task_id="t9", provider="mock", model="mock-model", user_prompt="take forever"
    )
    assert trajectory.termination_reason == TerminationReason.TIMEOUT


async def test_run_is_deterministic_given_the_same_script() -> None:
    responses = [
        _tool_call_response("inventory_lookup", {"sku": "SKU-1001"}),
        _final_answer_response("42 units in stock"),
    ]
    executor_a = _executor(list(responses))
    executor_b = _executor(list(responses))
    trajectory_a = await executor_a.run(
        task_id="same-task", provider="mock", model="mock-model", user_prompt="check stock"
    )
    trajectory_b = await executor_b.run(
        task_id="same-task", provider="mock", model="mock-model", user_prompt="check stock"
    )
    assert trajectory_a.termination_reason == trajectory_b.termination_reason
    assert trajectory_a.final_output_text == trajectory_b.final_output_text
    assert [s.tool_result.output if s.tool_result else None for s in trajectory_a.steps] == [
        s.tool_result.output if s.tool_result else None for s in trajectory_b.steps
    ]


def test_agent_limits_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        AgentLimits(max_steps=0)
    with pytest.raises(ValueError, match="max_tool_calls"):
        AgentLimits(max_tool_calls=-1)
    with pytest.raises(ValueError, match="max_cost_usd"):
        AgentLimits(max_cost_usd=-1.0)
    with pytest.raises(ValueError, match="max_wall_time_s"):
        AgentLimits(max_wall_time_s=0.0)
