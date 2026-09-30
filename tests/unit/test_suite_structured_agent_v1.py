"""Tests for the structured_agent_v1 suite: task coverage, determinism, and
end-to-end execution against its own ScriptedAgentProvider through the real
AgentExecutor/AgentSimulator (Commit 25)."""

from paretoguard.agents.executor import AgentExecutor, AgentLimits
from paretoguard.agents.simulator import AgentSimulator
from paretoguard.agents.state import TerminationReason
from paretoguard.agents.tools import default_tools
from paretoguard.evals.suites import structured_agent_v1 as suite_module
from paretoguard.runtime import RetryPolicy, Runtime


def test_build_tasks_is_deterministic() -> None:
    tasks_a = suite_module.build_tasks()
    tasks_b = suite_module.build_tasks()
    assert [t.task_id for t in tasks_a] == [t.task_id for t in tasks_b]


def test_every_task_id_is_unique() -> None:
    tasks = suite_module.build_tasks()
    assert len({t.task_id for t in tasks}) == len(tasks)


def test_scripted_provider_has_a_script_for_every_task() -> None:
    tasks = suite_module.build_tasks()
    provider = suite_module.ScriptedAgentProvider()
    for task in tasks:
        assert task.task_id in provider._scripts


async def test_full_suite_runs_end_to_end_and_grades_deterministically() -> None:
    tasks = suite_module.build_tasks()
    provider = suite_module.ScriptedAgentProvider()
    runtime = Runtime(provider, retry_policy=RetryPolicy(max_attempts=1))
    executor = AgentExecutor(runtime, default_tools(), limits=AgentLimits())
    simulator = AgentSimulator(executor, provider="mock", model="m")

    results = await simulator.run(tasks)
    assert len(results) == len(tasks)

    by_task = {r.task.task_id: r for r in results}
    # The two deliberately-malformed tasks are expected failures (see
    # suite_module's module docstring) — every other task should succeed.
    for task_id, task_result in by_task.items():
        outcome = task_result.outcome
        if task_id.endswith("invalid-args"):
            assert outcome.termination_reason == TerminationReason.INVALID_TOOL_ARGUMENTS
            assert not outcome.succeeded
        elif task_id.endswith("invalid-tool-name"):
            assert outcome.termination_reason == TerminationReason.INVALID_TOOL
            assert not outcome.succeeded
        else:
            assert outcome.succeeded, f"{task_id} failed: {outcome.explanation}"
            # finalize_trajectory upgrades FINAL_ANSWER -> SUCCESS on the
            # returned trajectory once grading confirms correctness; the
            # AgentGradeOutcome itself still reports the pre-upgrade reason.
            assert task_result.trajectory.termination_reason == TerminationReason.SUCCESS


async def test_multihop_task_calls_tools_in_expected_order() -> None:
    tasks = [t for t in suite_module.build_tasks() if "multihop" in t.task_id]
    assert len(tasks) == 1
    provider = suite_module.ScriptedAgentProvider()
    runtime = Runtime(provider, retry_policy=RetryPolicy(max_attempts=1))
    executor = AgentExecutor(runtime, default_tools(), limits=AgentLimits())
    simulator = AgentSimulator(executor, provider="mock", model="m")

    results = await simulator.run(tasks)
    trajectory = results[0].trajectory
    called_tools = [s.requested_tool for s in trajectory.steps if s.requested_tool is not None]
    assert called_tools == ["order_lookup", "shipment_lookup"]
