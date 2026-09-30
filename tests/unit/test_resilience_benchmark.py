"""Unit tests for routing.resilience_benchmark: the Phase E flagship
recovery-policy comparison (Commit 29)."""

from paretoguard.core.models import OutcomeEvent
from paretoguard.evals.suites import resilience_v1
from paretoguard.routing.resilience_benchmark import (
    CANDIDATE_PROVIDERS,
    recovery_config_specs,
    run_resilience_benchmark,
)
from paretoguard.storage import ExperimentStore


def _small_suite(num_cases: int = 8):
    return resilience_v1.build_suite(seed=0, num_cases=num_cases)


async def test_zero_fault_level_never_needs_recovery_for_any_config() -> None:
    suite = _small_suite()
    result = await run_resilience_benchmark(suite, fault_levels=[0.0], seed=0)
    assert len(result.rows) == len(recovery_config_specs())
    for row in result.rows:
        assert row.fault_level == 0.0
        assert row.metrics.task_success_rate == 1.0
        assert row.metrics.average_attempts == 1.0


async def test_no_recovery_config_never_retries_or_falls_back() -> None:
    suite = _small_suite()
    result = await run_resilience_benchmark(suite, fault_levels=[0.30], seed=0)
    no_recovery_row = next(r for r in result.rows if r.recovery_config == "A-no_recovery")
    assert no_recovery_row.metrics.retry_rate == 0.0
    assert no_recovery_row.metrics.fallback_rate == 0.0
    assert no_recovery_row.metrics.average_attempts == 1.0
    # At a nonzero fault level, no-recovery's raw failures are never
    # recovered by definition (there is no recovery to do it).
    if no_recovery_row.metrics.raw_failure_rate > 0:
        assert no_recovery_row.metrics.recovery_rate == 0.0


async def test_recovery_enabled_configs_never_recover_worse_than_no_recovery() -> None:
    """Every recovery-enabled config (B-E) has at least as much attempt
    budget as A — it must never end up with a *lower* task success rate at
    the same fault level, though it is not required to do strictly better
    (see resilience_benchmark's module docstring: this is not tuned to
    guarantee any config "wins")."""
    suite = _small_suite(num_cases=16)
    result = await run_resilience_benchmark(suite, fault_levels=[0.30], seed=0)
    by_config = {r.recovery_config: r.metrics.task_success_rate for r in result.rows}
    baseline = by_config["A-no_recovery"]
    for name, success_rate in by_config.items():
        if name != "A-no_recovery":
            assert success_rate >= baseline, f"{name} did worse than no-recovery baseline"


async def test_same_fault_level_uses_the_identical_fault_stream_across_configs() -> None:
    """The experimental-design invariant the module docstring promises:
    every config at a given fault level must face the same fault draws —
    verified indirectly by asserting every config observes the exact same
    number of raw first-attempt failures (since the world is otherwise
    fully deterministic and identical for every config)."""
    suite = _small_suite(num_cases=16)
    result = await run_resilience_benchmark(suite, fault_levels=[0.30], seed=0)
    raw_failure_counts = {round(r.metrics.raw_failure_rate * r.metrics.n) for r in result.rows}
    assert len(raw_failure_counts) == 1  # identical across every config


async def test_escalation_rate_is_zero_for_this_suites_transient_only_faults() -> None:
    suite = _small_suite()
    result = await run_resilience_benchmark(suite, fault_levels=[0.30], seed=0)
    for row in result.rows:
        assert row.metrics.escalation_rate == 0.0


async def test_persists_outcome_events_through_the_normal_pipeline() -> None:
    suite = _small_suite(num_cases=4)
    with ExperimentStore(":memory:") as store:
        result = await run_resilience_benchmark(suite, fault_levels=[0.0], seed=0, store=store)
        row = result.rows[0]
        run_id = f"resilience-{suite.name}-{row.fault_level}-{row.recovery_config}"
        events = store.get_outcome_events(run_id)
        assert len(events) == 4
        assert all(isinstance(e, OutcomeEvent) for e in events)


def test_candidate_providers_are_independent_fault_targets() -> None:
    assert len(CANDIDATE_PROVIDERS) == 3
    assert len(set(CANDIDATE_PROVIDERS)) == 3


_LATENCY_FIELDS = {"median_latency_ms", "p95_latency_ms"}
"""Excluded from determinism comparison: latency is real wall-clock time
(`time.perf_counter()` inside `MockProvider.complete`), not seed-derived —
every other field here is a pure function of (seed, fault level, config)."""


def _stable_metrics(row) -> dict:  # type: ignore[no-untyped-def]
    return {k: v for k, v in row.metrics.model_dump().items() if k not in _LATENCY_FIELDS}


async def test_determinism_two_runs_produce_identical_metrics() -> None:
    suite = _small_suite(num_cases=10)
    result_a = await run_resilience_benchmark(suite, fault_levels=[0.15], seed=0)
    result_b = await run_resilience_benchmark(suite, fault_levels=[0.15], seed=0)
    metrics_a = [(r.recovery_config, _stable_metrics(r)) for r in result_a.rows]
    metrics_b = [(r.recovery_config, _stable_metrics(r)) for r in result_b.rows]
    assert metrics_a == metrics_b
