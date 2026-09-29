"""Typed agent termination taxonomy and trajectory records.

`TerminationReason` is deliberately its own enum, not folded into
`FailureCategory` (`core.models.common`): it describes *how the agent loop
stopped*, which includes two non-failure states (`SUCCESS`, `FINAL_ANSWER`)
`FailureCategory` has no slot for, since that enum exists purely to
classify failures. Every reason that *is* a failure maps onto the existing
`FailureCategory` via `failure_category_for` instead of inventing parallel
semantics — see that function for the mapping.
"""

from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from paretoguard.core.models import FailureCategory, ToolResult


class TerminationReason(StrEnum):
    """Every possible way an agent run ends. Exactly one always applies —
    the executor (`paretoguard.agents.executor.AgentExecutor`) guarantees
    this by construction (see its module docstring for the loop's hard
    limits, which make an unterminated run impossible).
    """

    SUCCESS = "success"
    """Assigned by grading (`paretoguard.agents.simulator`), never by the
    executor itself — the executor has no ground truth to judge against.
    A trajectory that stops with FINAL_ANSWER may be upgraded to SUCCESS
    once grading confirms the final state/output was correct."""
    FINAL_ANSWER = "final_answer"
    STEP_LIMIT = "step_limit"
    TOOL_CALL_LIMIT = "tool_call_limit"
    BUDGET_EXCEEDED = "budget_exceeded"
    TIMEOUT = "timeout"
    INVALID_TOOL = "invalid_tool"
    INVALID_TOOL_ARGUMENTS = "invalid_tool_arguments"
    UNRECOVERABLE_FAILURE = "unrecoverable_failure"


_TERMINATION_TO_FAILURE_CATEGORY: dict[TerminationReason, FailureCategory | None] = {
    TerminationReason.SUCCESS: None,
    TerminationReason.FINAL_ANSWER: None,
    TerminationReason.STEP_LIMIT: FailureCategory.STEP_LIMIT,
    TerminationReason.TOOL_CALL_LIMIT: FailureCategory.STEP_LIMIT,
    TerminationReason.BUDGET_EXCEEDED: FailureCategory.BUDGET_EXCEEDED,
    TerminationReason.TIMEOUT: FailureCategory.TIMEOUT,
    TerminationReason.INVALID_TOOL: FailureCategory.TOOL_SELECTION_FAILURE,
    TerminationReason.INVALID_TOOL_ARGUMENTS: FailureCategory.TOOL_ARGUMENT_FAILURE,
    TerminationReason.UNRECOVERABLE_FAILURE: FailureCategory.UNRECOVERABLE_STATE,
}


def failure_category_for(reason: TerminationReason) -> FailureCategory | None:
    """The shared `FailureCategory` a given termination reason corresponds
    to, or `None` for the two non-failure reasons (SUCCESS, FINAL_ANSWER).
    `TOOL_CALL_LIMIT` reuses `STEP_LIMIT`: both are "a hard iteration bound
    was hit", just counted differently (LLM turns vs. tool invocations) —
    not a meaningfully different failure mode for recovery/telemetry."""
    return _TERMINATION_TO_FAILURE_CATEGORY[reason]


class AgentStep(BaseModel):
    """One iteration of the agent loop."""

    step_index: int = Field(ge=0)
    model_output_text: str | None = None
    requested_tool: str | None = None
    requested_arguments: dict[str, Any] | None = None
    arguments_valid: bool | None = None
    """None if no tool was requested this step; True/False once validated."""
    tool_result: ToolResult | None = None
    fault_id: str | None = None
    """Set by the executor when `paretoguard.chaos` (Commit 26) injected a
    fault at this step — the fault's own id, for cross-referencing its full
    `FaultEvent` record elsewhere. Not embedded here, to keep `agents` free
    of a hard dependency on `chaos`."""
    recovery_action: str | None = None
    """Set when `paretoguard.recovery` (Commit 27) intervened at this step
    — the `RecoveryAction`'s string value, same reasoning as `fault_id`."""
    latency_ms: float = Field(ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class AgentTrajectory(BaseModel):
    """A complete, inspectable record of one agent run. Never holds
    secrets: messages/tool arguments here are exactly what the benchmark
    task and deterministic tools produced, nothing pulled from environment
    variables or credentials."""

    trajectory_id: UUID = Field(default_factory=uuid4)
    task_id: str
    provider: str
    model: str
    steps: list[AgentStep] = Field(default_factory=list)
    termination_reason: TerminationReason
    final_output_text: str | None = None
    final_structured_output: dict[str, Any] | None = None

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def tool_call_count(self) -> int:
        """Every *attempted* tool call, including one rejected by
        `AgentLimits.max_tool_calls`/`INVALID_TOOL`/`INVALID_TOOL_ARGUMENTS`
        (`tool_result` is `None` for those). This is the count the executor
        enforces `max_tool_calls` against — a conservative safety limit on
        attempts, not just successful executions. Count steps with
        `tool_result is not None` instead for calls that actually ran."""
        return sum(1 for s in self.steps if s.requested_tool is not None)

    @property
    def unnecessary_tool_call_count(self) -> int:
        """Tool calls whose result was an error — a proxy for "the agent
        called a tool it shouldn't have, or with bad arguments/target"."""
        return sum(1 for s in self.steps if s.tool_result is not None and s.tool_result.is_error)

    @property
    def total_cost_usd(self) -> float | None:
        costs = [s.cost_usd for s in self.steps if s.cost_usd is not None]
        return sum(costs) if costs else None

    @property
    def total_latency_ms(self) -> float:
        return sum(s.latency_ms for s in self.steps)

    @property
    def failure_category(self) -> FailureCategory | None:
        return failure_category_for(self.termination_reason)
