"""Deterministic, state-based grading for agent trajectories, and a small
batch runner over `AgentExecutor` — no LLM judge for the core agent suite
(see CLAUDE.md: "Prefer deterministic evaluation").
"""

from dataclasses import dataclass, field
from typing import Any

from paretoguard.agents.executor import AgentExecutor
from paretoguard.agents.state import AgentTrajectory, TerminationReason


@dataclass(frozen=True)
class ExpectedTrajectory:
    """Ground truth for one agent task."""

    expected_tool_sequence: list[str] = field(default_factory=list)
    """Ordered tool names the agent should call, e.g.
    `["order_lookup", "shipment_lookup"]`. Empty for a task solvable without
    tools."""
    expected_final_field: str | None = None
    """Key to check in `final_structured_output`, for tasks expecting a
    structured final answer."""
    expected_final_value: Any = None
    expected_answer_substring: str | None = None
    """Substring `final_output_text` should contain, for free-text tasks.
    Mutually exclusive with `expected_final_field` in practice, though
    nothing enforces that — a task should set one or the other."""


@dataclass(frozen=True)
class AgentGradeOutcome:
    """Everything `paretoguard.evals.graders.GradeOutcome` reports for a
    single-turn eval case, plus the agent-specific measures Commit 25's
    spec calls for (tool selection/sequence correctness, unnecessary calls,
    step count)."""

    succeeded: bool
    correct_tool_selection: bool
    """Same *set* of tools called, regardless of order."""
    correct_tool_sequence: bool
    """Same tools called in the same order."""
    all_arguments_valid: bool
    unnecessary_tool_calls: int
    step_count: int
    termination_reason: TerminationReason
    explanation: str


def grade_trajectory(
    trajectory: AgentTrajectory, expected: ExpectedTrajectory
) -> AgentGradeOutcome:
    """Pure function of the trajectory's own recorded state — never
    re-executes anything, never calls a judge model."""
    actual_sequence = [
        step.requested_tool for step in trajectory.steps if step.requested_tool is not None
    ]
    correct_tool_sequence = actual_sequence == expected.expected_tool_sequence
    correct_tool_selection = set(actual_sequence) == set(expected.expected_tool_sequence)
    all_arguments_valid = all(step.arguments_valid is not False for step in trajectory.steps)

    final_state_correct = True
    if expected.expected_final_field is not None:
        final_state_correct = (
            trajectory.final_structured_output is not None
            and trajectory.final_structured_output.get(expected.expected_final_field)
            == expected.expected_final_value
        )
    elif expected.expected_answer_substring is not None:
        final_state_correct = (
            trajectory.final_output_text is not None
            and expected.expected_answer_substring in trajectory.final_output_text
        )

    succeeded = (
        trajectory.termination_reason in (TerminationReason.FINAL_ANSWER, TerminationReason.SUCCESS)
        and correct_tool_sequence
        and all_arguments_valid
        and final_state_correct
    )

    explanation_parts = [f"termination={trajectory.termination_reason.value}"]
    if not correct_tool_sequence:
        explanation_parts.append(
            f"tool sequence {actual_sequence} != expected {expected.expected_tool_sequence}"
        )
    if not all_arguments_valid:
        explanation_parts.append("one or more tool calls had invalid arguments")
    if not final_state_correct:
        explanation_parts.append("final state did not match expected")

    return AgentGradeOutcome(
        succeeded=succeeded,
        correct_tool_selection=correct_tool_selection,
        correct_tool_sequence=correct_tool_sequence,
        all_arguments_valid=all_arguments_valid,
        unnecessary_tool_calls=trajectory.unnecessary_tool_call_count,
        step_count=trajectory.step_count,
        termination_reason=trajectory.termination_reason,
        explanation="; ".join(explanation_parts),
    )


def finalize_trajectory(trajectory: AgentTrajectory, outcome: AgentGradeOutcome) -> AgentTrajectory:
    """Upgrades a `FINAL_ANSWER`-terminated trajectory to `SUCCESS` once
    grading confirms it was correct — the executor itself has no ground
    truth to make this call (see `TerminationReason.SUCCESS`'s docstring).
    A no-op for every other termination reason."""
    if outcome.succeeded and trajectory.termination_reason == TerminationReason.FINAL_ANSWER:
        return trajectory.model_copy(update={"termination_reason": TerminationReason.SUCCESS})
    return trajectory


@dataclass(frozen=True)
class AgentTask:
    task_id: str
    user_prompt: str
    expected: ExpectedTrajectory
    system_prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentTaskResult:
    task: AgentTask
    trajectory: AgentTrajectory
    outcome: AgentGradeOutcome


class AgentSimulator:
    """Runs a batch of `AgentTask`s through one `AgentExecutor`
    (provider/model fixed for the batch), grading each with
    `grade_trajectory`."""

    def __init__(self, executor: AgentExecutor, *, provider: str, model: str) -> None:
        self._executor = executor
        self._provider = provider
        self._model = model

    async def run(self, tasks: list[AgentTask]) -> list[AgentTaskResult]:
        results = []
        for task in tasks:
            trajectory = await self._executor.run(
                task_id=task.task_id,
                provider=self._provider,
                model=self._model,
                user_prompt=task.user_prompt,
                system_prompt=task.system_prompt,
                metadata=task.metadata,
            )
            outcome = grade_trajectory(trajectory, task.expected)
            trajectory = finalize_trajectory(trajectory, outcome)
            results.append(AgentTaskResult(task=task, trajectory=trajectory, outcome=outcome))
        return results
