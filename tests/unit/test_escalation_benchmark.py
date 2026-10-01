"""Unit tests for routing.escalation_benchmark: the Phase E.5 quality-
failure escalation scenario, exercised through the real closed-loop path."""

from paretoguard.routing.escalation_benchmark import run_escalation_scenario
from paretoguard.storage import ExperimentStore


async def test_no_validation_arm_never_escalates_and_suffers_lower_success() -> None:
    result = await run_escalation_scenario(num_cases=20, quality_fault_rate=0.6, seed=0)
    assert result.no_validation.escalation_rate == 0.0
    assert result.no_validation.task_success_rate < 1.0


async def test_validation_and_escalation_arm_recovers_every_quality_failure() -> None:
    result = await run_escalation_scenario(num_cases=20, quality_fault_rate=0.6, seed=0)
    assert result.validation_and_escalation.escalation_rate > 0.0
    assert result.validation_and_escalation.task_success_rate == 1.0
    assert result.validation_and_escalation.unrecovered_failure_distribution == {}


async def test_validation_and_escalation_never_does_worse_than_no_validation() -> None:
    result = await run_escalation_scenario(num_cases=20, quality_fault_rate=0.6, seed=0)
    assert result.success_rate_delta >= 0.0
    assert (
        result.validation_and_escalation.task_success_rate >= result.no_validation.task_success_rate
    )


async def test_escalation_has_an_attempt_cost() -> None:
    """Escalation isn't free — recovering a quality failure costs at least
    one extra attempt on average; the comparison should show that
    honestly, not hide it."""
    result = await run_escalation_scenario(num_cases=20, quality_fault_rate=0.6, seed=0)
    assert result.added_average_attempts > 0.0


async def test_zero_fault_rate_means_no_difference_between_arms() -> None:
    result = await run_escalation_scenario(num_cases=10, quality_fault_rate=0.0, seed=0)
    assert result.no_validation.task_success_rate == 1.0
    assert result.validation_and_escalation.task_success_rate == 1.0
    assert result.success_rate_delta == 0.0


async def test_determinism_two_runs_produce_identical_metrics() -> None:
    result_a = await run_escalation_scenario(num_cases=20, quality_fault_rate=0.6, seed=0)
    result_b = await run_escalation_scenario(num_cases=20, quality_fault_rate=0.6, seed=0)
    assert result_a.no_validation.task_success_rate == result_b.no_validation.task_success_rate
    assert (
        result_a.validation_and_escalation.escalation_rate
        == result_b.validation_and_escalation.escalation_rate
    )
    assert result_a.success_rate_delta == result_b.success_rate_delta
    assert result_a.added_average_attempts == result_b.added_average_attempts


async def test_persists_a_run_manifest_for_each_arm() -> None:
    """Regression test: a real end-to-end bug (scripts/phase_f_report.py)
    found that this orchestrator (and the two sibling resilience/sustained-
    outage orchestrators) persisted every per-task row but never registered
    a RunManifest, so statistics.regression.check_regression/
    reports.generate_report (both of which require store.get_run to
    succeed) could never find these runs."""
    with ExperimentStore(":memory:") as store:
        await run_escalation_scenario(num_cases=15, quality_fault_rate=0.6, seed=0, store=store)
        for run_id in ("escalation-no_validation", "escalation-validation_and_escalation"):
            manifest = store.get_run(run_id)
            assert manifest is not None
            assert manifest.label == "SIMULATION"
            assert manifest.task_count == 15
