"""Unit tests for routing.sustained_outage_benchmark: the Phase E.5
correlated-outage scenario."""

from paretoguard.routing.sustained_outage_benchmark import run_sustained_outage_scenario


def _small_scenario_kwargs() -> dict:
    return {
        "total_steps": 60,
        "degraded_start": 20,
        "recovered_start": 40,
        "baseline_probability": 0.01,
        "degraded_probability": 0.9,
        "seed": 0,
    }


async def test_no_recovery_suffers_during_the_outage_window() -> None:
    results = await run_sustained_outage_scenario(**_small_scenario_kwargs())
    by_config = {r.recovery_config: r.metrics for r in results}
    no_recovery = by_config["A-no_recovery"]
    assert no_recovery.task_success_rate_during_outage < 0.5


async def test_fallback_enabled_configs_beat_retry_only_under_correlated_outage() -> None:
    """The central claim this scenario exists to test: under a *sustained*
    outage, fallback-capable configs should recover more of the outage
    window than retry-only — unlike resilience_v1's i.i.d. fault model,
    where they tie (see routing.resilience_benchmark's module docstring)."""
    results = await run_sustained_outage_scenario(**_small_scenario_kwargs())
    by_config = {r.recovery_config: r.metrics for r in results}
    retry_only = by_config["B-retry_only"].task_success_rate_during_outage
    retry_fallback = by_config["C-retry_fallback"].task_success_rate_during_outage
    assert retry_fallback > retry_only


async def test_fallback_configs_never_do_worse_than_no_recovery() -> None:
    results = await run_sustained_outage_scenario(**_small_scenario_kwargs())
    by_config = {r.recovery_config: r.metrics.task_success_rate_during_outage for r in results}
    baseline = by_config["A-no_recovery"]
    for name, rate in by_config.items():
        if name != "A-no_recovery":
            assert rate >= baseline, f"{name} did worse than no-recovery during the outage"


async def test_circuit_breaker_configs_record_fallback_and_open_transitions() -> None:
    results = await run_sustained_outage_scenario(**_small_scenario_kwargs())
    by_config = {r.recovery_config: r.metrics for r in results}
    for name in ("D-retry_fallback_circuit_breaker", "E-full_policy"):
        assert by_config[name].fallback_count > 0
    # no-recovery and retry-only never invoke a fallback at all
    assert by_config["A-no_recovery"].fallback_count == 0
    assert by_config["B-retry_only"].fallback_count == 0


async def test_time_to_reroute_is_set_for_fallback_configs_and_none_for_retry_only() -> None:
    results = await run_sustained_outage_scenario(**_small_scenario_kwargs())
    by_config = {r.recovery_config: r.metrics for r in results}
    assert by_config["C-retry_fallback"].time_to_reroute is not None
    assert by_config["B-retry_only"].time_to_reroute is None
    assert by_config["A-no_recovery"].time_to_reroute is None


async def test_no_unnecessary_reroutes_outside_the_outage_window() -> None:
    """Candidates are stable (1% baseline) outside the degraded window —
    a well-behaved recovery policy should never reroute away from a
    healthy primary there."""
    results = await run_sustained_outage_scenario(**_small_scenario_kwargs())
    for r in results:
        assert r.metrics.unnecessary_reroute_count == 0


async def test_determinism_two_runs_produce_identical_metrics() -> None:
    kwargs = _small_scenario_kwargs()
    results_a = await run_sustained_outage_scenario(**kwargs)
    results_b = await run_sustained_outage_scenario(**kwargs)
    # Exclude the two wall-clock-latency-derived fields, same rationale as
    # resilience_benchmark's own determinism test.
    _LATENCY_FIELDS = {"mean_latency_ms_baseline", "mean_latency_ms_outage"}

    def stable(r):  # type: ignore[no-untyped-def]
        d = r.metrics.__dict__
        return (r.recovery_config, {k: v for k, v in d.items() if k not in _LATENCY_FIELDS})

    assert [stable(r) for r in results_a] == [stable(r) for r in results_b]


async def test_no_pricing_means_no_fabricated_cost_field() -> None:
    """Cost overhead is deliberately not reported by this scenario (no
    PricingTable is configured) — asserted here as a guard against a future
    change silently adding a fabricated cost number."""
    results = await run_sustained_outage_scenario(**_small_scenario_kwargs())
    for r in results:
        assert not hasattr(r.metrics, "cost_overhead_usd")
