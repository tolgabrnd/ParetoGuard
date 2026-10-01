"""Sustained/correlated provider outage scenario (Phase E.5, item 2).

**SIMULATION.** Every failure here is a synthetic, injected
`chaos.FaultInjector` fault against `MockProvider` — never a real
provider's actual reliability (see CLAUDE.md).

**Why this exists**: `routing.resilience_benchmark`'s flagship comparison
uses an i.i.d.-per-attempt fault model, which its own module docstring
states plainly does not exercise circuit-breaker's strongest use case (a
sustained, correlated outage) or give fallback a fundamentally different
expected outcome from retry-only. This module is the scenario that fills
that gap: one candidate provider (`CANDIDATE_PROVIDERS[0]`) goes through a
deterministic healthy -> degraded -> recovered timeline —

    steps   0-99:  1% failure probability   (healthy baseline)
    steps 100-199: 85% failure probability  (sustained outage)
    steps  200+:   1% failure probability   (recovered)

— reusing `chaos.scenarios.provider_degradation_schedule` (the same
schedule shape Phase D's `routing.reliability_simulation` established,
generalized to a `FaultPolicy`) rather than hand-rolling a new one. 85% is a
representative point within the 60-100% "sustained outage" band the Phase
E.5 spec describes — a fixed value, not a range, so the scenario stays
exactly reproducible; see this module's own result artifacts for the actual
observed numbers. The other two candidate providers stay at a flat 1%
baseline for the entire run — a genuine, healthy fallback target, not a
strictly-worse option, so a recovery policy has a real reason to prefer
them during the outage (mirroring `two_model_degradation_schedule`'s
`stable_model` design choice in Phase D).

**Timeline vs. attempt**: `ClosedLoopExecutor.execute`'s `chaos_step`
normally resets to 0 for every task (it means "attempt number within this
task"). A sustained-outage scenario needs `step` to instead mean "time
since the scenario began" so the schedule's degraded window actually lines
up with wall-clock-like progress across many sequential tasks — this is
exactly what Commit `chaos_step_offset` (added in this same commit) is for:
each outer step `t`'s task is executed with `chaos_step_offset=t`, so its
first attempt draws at schedule-step `t` and any retries within that task
advance forward from there, never resetting to 0.

**Deterministic virtual clock for the circuit breaker**: this scenario
spans a simulated timeline of `total_steps` sequential tasks that execute
in well under a second of real wall-clock time. A `CircuitBreaker` using
the real `time.monotonic` clock would see its `cooldown_s` window elapse
almost instantly relative to real time regardless of how many *simulated*
steps have passed — making HALF_OPEN probe timing an artifact of how fast
this process happens to run, not of the scenario. A virtual clock
(`t -> float(t)`, one "virtual second" per simulated step) makes cooldown
timing a deterministic function of the simulated timeline instead — reusing
`CircuitBreaker`'s existing injectable-clock design (built in Commit 27
specifically so "no sleep() in tests" could hold; this is the same
principle applied to a multi-step scenario instead of a unit test).
"""

import platform
from dataclasses import dataclass

from paretoguard import __version__
from paretoguard.chaos.injector import FaultInjector, FaultyProvider
from paretoguard.chaos.policies import ConstantProbability, FaultPolicy
from paretoguard.chaos.scenarios import provider_degradation_schedule
from paretoguard.core.models import Message, Role, RunManifest
from paretoguard.evals.metrics import compute_resilience_metrics
from paretoguard.evals.models import EvalCase, GraderConfig, GraderKind, GroundTruth
from paretoguard.providers.base import Provider
from paretoguard.providers.mock import MockProvider
from paretoguard.recovery.circuit_breaker import CircuitState
from paretoguard.recovery.context import RecoveryAction
from paretoguard.routing.execution import ClosedLoopExecutor
from paretoguard.routing.resilience_benchmark import (
    CANDIDATE_PROVIDERS,
    RecoveryConfigSpec,
    candidates,
    recovery_config_specs,
)
from paretoguard.routing.static import StaticRouter
from paretoguard.runtime import RetryPolicy
from paretoguard.storage import ExperimentStore
from paretoguard.telemetry.health import HealthTracker

DEFAULT_TOTAL_STEPS = 300
DEFAULT_DEGRADED_START = 100
DEFAULT_RECOVERED_START = 200
DEFAULT_BASELINE_PROBABILITY = 0.01
DEFAULT_DEGRADED_PROBABILITY = 0.85

_NO_RUNTIME_RETRY = RetryPolicy(max_attempts=1)

_PRIMARY_PROVIDER = CANDIDATE_PROVIDERS[0]


def _step_case(t: int) -> EvalCase:
    return EvalCase(
        case_id=f"sustained-outage-{t:04d}",
        messages=[Message(role=Role.USER, content="What is the capital of France?")],
        grader=GraderConfig(kind=GraderKind.NORMALIZED_TEXT_MATCH),
        ground_truth=GroundTruth(text="Paris"),
        tags=["sustained_outage"],
        metadata={"mock_scenario": "success", "mock_answer_text": "Paris"},
    )


def _fault_injector(
    *,
    degraded_start: int,
    recovered_start: int,
    baseline_probability: float,
    degraded_probability: float,
    seed: int,
) -> FaultInjector:
    primary_policy = provider_degradation_schedule(
        fault_id=f"sustained-outage-{_PRIMARY_PROVIDER}",
        target_provider=_PRIMARY_PROVIDER,
        target_model=candidates()[0].name,
        degraded_start=degraded_start,
        recovered_start=recovered_start,
        baseline_probability=baseline_probability,
        degraded_probability=degraded_probability,
    )
    baseline_policies = [
        FaultPolicy(
            fault_id=f"sustained-outage-baseline-{provider}",
            category=primary_policy.category,
            kind=primary_policy.kind,
            schedule=ConstantProbability(baseline_probability),
            target_provider=provider,
        )
        for provider in CANDIDATE_PROVIDERS
        if provider != _PRIMARY_PROVIDER
    ]
    return FaultInjector([primary_policy, *baseline_policies], seed=seed)


def _providers(injector: FaultInjector) -> dict[str, Provider]:
    return {
        provider: FaultyProvider(inner=MockProvider(provider, seed=0), injector=injector)
        for provider in CANDIDATE_PROVIDERS
    }


@dataclass(frozen=True)
class SustainedOutageMetrics:
    """Every field defined precisely — no metric here is left to
    interpretation (Phase E.5 spec's explicit measurement list)."""

    task_success_rate: float
    task_success_rate_during_outage: float
    """Success rate restricted to `[degraded_start, recovered_start)`."""
    average_attempts: float
    fallback_count: int
    """Tasks whose `recovery_actions` include FALLBACK_MODEL or FALLBACK_PROVIDER."""
    circuit_open_transitions: int
    """Number of times the primary candidate's circuit transitioned into
    OPEN across the run (0 for a config with no circuit breaker)."""
    half_open_probe_count: int
    """Tasks whose `recovery_actions` include PROBE."""
    time_to_detect: int | None
    """Steps between `degraded_start` and the first task where recovery was
    invoked at all (attempt_count > 1 or any recovery_actions) — "the
    system noticed something was wrong", not necessarily that it fixed it.
    None if never detected."""
    time_to_reroute: int | None
    """Steps between `degraded_start` and the first task whose
    `final_provider` differs from the primary — "the system successfully
    switched away". None if it never did (e.g. a no-recovery config)."""
    time_to_recover: int | None
    """Steps between `recovered_start` and the first task whose
    `final_provider` returns to the primary, for a config that had
    previously switched away. None if it never switched back (or never
    switched away in the first place)."""
    unnecessary_reroute_count: int
    """Switches of `final_provider` away from the primary while *outside*
    the degraded window — not justified by the injected schedule."""
    mean_latency_ms_baseline: float
    mean_latency_ms_outage: float
    """Latency overhead comparison: mean total_latency_ms for steps inside
    vs. outside the degraded window. Cost overhead is not reported — no
    `PricingTable` is configured for this scenario (`MockProvider` produces
    no cost data without one), so there is nothing real to compare; showing
    a fabricated $0.00 would violate CLAUDE.md's "never fabricate... cost"
    rule more than omitting the field does."""


@dataclass(frozen=True)
class SustainedOutageResult:
    recovery_config: str
    metrics: SustainedOutageMetrics


async def run_sustained_outage_scenario(
    *,
    configs: list[RecoveryConfigSpec] | None = None,
    total_steps: int = DEFAULT_TOTAL_STEPS,
    degraded_start: int = DEFAULT_DEGRADED_START,
    recovered_start: int = DEFAULT_RECOVERED_START,
    baseline_probability: float = DEFAULT_BASELINE_PROBABILITY,
    degraded_probability: float = DEFAULT_DEGRADED_PROBABILITY,
    seed: int = 0,
    store: ExperimentStore | None = None,
) -> list[SustainedOutageResult]:
    """Runs `total_steps` sequential tasks through each config's own fresh
    `ClosedLoopExecutor`, against the *same* healthy/degraded/recovered
    fault stream every config faces (the injector's seed depends only on
    the scenario parameters, never on which config is running — the same
    fairness discipline `resilience_benchmark` uses). Sequential, never
    concurrent, for the same reason `resilience_benchmark` is: mutation
    order of `HealthTracker`/`CircuitBreaker` must be deterministic.
    """
    virtual_clock = {"t": 0.0}
    clock = lambda: virtual_clock["t"]  # noqa: E731 - closure needs to see virtual_clock by reference
    configs = configs if configs is not None else recovery_config_specs(clock=clock)

    results: list[SustainedOutageResult] = []
    for spec in configs:
        virtual_clock["t"] = 0.0
        injector = _fault_injector(
            degraded_start=degraded_start,
            recovered_start=recovered_start,
            baseline_probability=baseline_probability,
            degraded_probability=degraded_probability,
            seed=seed,
        )
        recovery_policy, circuit_breaker = spec.build()
        executor = ClosedLoopExecutor(
            StaticRouter(provider=_PRIMARY_PROVIDER, model=candidates()[0].name),
            candidates(),
            _providers(injector),
            recovery_policy=recovery_policy,
            circuit_breaker=circuit_breaker,
            health_tracker=HealthTracker(),
            store=store,
            retry_policy=_NO_RUNTIME_RETRY,
            max_attempts=spec.max_attempts,
        )
        run_id = f"sustained-outage-{spec.name}"
        if store is not None:
            store.record_run(
                RunManifest(
                    run_id=run_id,
                    paretoguard_version=__version__,
                    os=platform.system(),
                    python_version=platform.python_version(),
                    seed=seed,
                    suite_name="sustained_outage",
                    suite_version="1.0.0",
                    router_name="static",
                    router_config={"recovery_config": spec.name, "total_steps": total_steps},
                    task_count=total_steps,
                    label="SIMULATION",
                )
            )
        primary_key = f"{_PRIMARY_PROVIDER}:{candidates()[0].name}"

        previous_final_provider: str | None = None
        switched_away = False
        reroute_count = 0
        unnecessary_reroute_count = 0
        time_to_detect: int | None = None
        time_to_reroute: int | None = None
        time_to_recover: int | None = None
        circuit_open_transitions = 0
        previous_circuit_state = CircuitState.CLOSED
        outcomes = []
        latencies_baseline: list[float] = []
        latencies_outage: list[float] = []

        for t in range(total_steps):
            virtual_clock["t"] = float(t)
            _, outcome = await executor.execute(_step_case(t), run_id=run_id, chaos_step_offset=t)
            outcomes.append(outcome)
            in_outage_window = degraded_start <= t < recovered_start
            (latencies_outage if in_outage_window else latencies_baseline).append(
                outcome.total_latency_ms
            )

            if degraded_start <= t and time_to_detect is None:
                detected = outcome.attempt_count > 1 or bool(outcome.recovery_actions)
                if detected:
                    time_to_detect = t - degraded_start

            current_final_provider = outcome.final_provider
            if (
                previous_final_provider is not None
                and current_final_provider != previous_final_provider
            ):
                reroute_count += 1
                if current_final_provider != _PRIMARY_PROVIDER and not in_outage_window:
                    unnecessary_reroute_count += 1

            if (
                in_outage_window
                and current_final_provider != _PRIMARY_PROVIDER
                and not switched_away
            ):
                switched_away = True
                if time_to_reroute is None:
                    time_to_reroute = t - degraded_start

            if (
                t >= recovered_start
                and switched_away
                and current_final_provider == _PRIMARY_PROVIDER
                and time_to_recover is None
            ):
                time_to_recover = t - recovered_start

            previous_final_provider = current_final_provider

            if circuit_breaker is not None:
                state = circuit_breaker.state_for(primary_key)
                if state == CircuitState.OPEN and previous_circuit_state != CircuitState.OPEN:
                    circuit_open_transitions += 1
                previous_circuit_state = state

        metrics_all = compute_resilience_metrics(outcomes)
        outage_outcomes = outcomes[degraded_start:recovered_start]
        success_rate_outage = (
            sum(o.succeeded for o in outage_outcomes) / len(outage_outcomes)
            if outage_outcomes
            else 0.0
        )
        fallback_count = sum(
            (RecoveryAction.FALLBACK_MODEL.value in o.recovery_actions)
            or (RecoveryAction.FALLBACK_PROVIDER.value in o.recovery_actions)
            for o in outcomes
        )
        half_open_probe_count = sum(
            RecoveryAction.PROBE.value in o.recovery_actions for o in outcomes
        )

        results.append(
            SustainedOutageResult(
                recovery_config=spec.name,
                metrics=SustainedOutageMetrics(
                    task_success_rate=metrics_all.task_success_rate,
                    task_success_rate_during_outage=success_rate_outage,
                    average_attempts=metrics_all.average_attempts,
                    fallback_count=fallback_count,
                    circuit_open_transitions=circuit_open_transitions,
                    half_open_probe_count=half_open_probe_count,
                    time_to_detect=time_to_detect,
                    time_to_reroute=time_to_reroute,
                    time_to_recover=time_to_recover,
                    unnecessary_reroute_count=unnecessary_reroute_count,
                    mean_latency_ms_baseline=(
                        sum(latencies_baseline) / len(latencies_baseline)
                        if latencies_baseline
                        else 0.0
                    ),
                    mean_latency_ms_outage=(
                        sum(latencies_outage) / len(latencies_outage) if latencies_outage else 0.0
                    ),
                ),
            )
        )

    return results
