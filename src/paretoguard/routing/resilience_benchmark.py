"""Phase E flagship resilience/recovery comparison (Commit 29).

**SIMULATION.** Every task in `resilience_v1` succeeds deterministically
under `MockProvider`'s default SUCCESS scenario; every failure this
benchmark observes comes from `chaos.FaultInjector`-injected, synthetic
provider faults — nothing here is a measurement of any real provider (see
CLAUDE.md: never fabricate reliability numbers).

**Experimental design.** For each of `FAULT_LEVELS` (0%, 5%, 15%, 30%
per-attempt fault probability), all five `RECOVERY_CONFIGS` (A-E, see
`recovery_config_specs`) run against the exact same fault stream: the
`FaultInjector`'s seed is a pure function of the fault level alone, never of
which recovery config is being evaluated, so what varies between A-E at a
given level is *only* the recovery policy, never the world it faces. This
is deliberate: the Phase E spec requires that this benchmark not be tuned to
guarantee any particular config wins, and holding the fault stream constant
across configs is what makes "which config recovered more tasks" a fair
question rather than a foregone conclusion.

Three independent candidate providers (`CANDIDATE_PROVIDERS`) each get their
own `FaultPolicy` with a provider-qualified `fault_id` — `FaultInjector`'s
RNG key is `(seed, task_id, step, fault_id)`, deliberately *not*
provider-qualified by default (see `chaos.injector`'s docstring), so without
distinct fault ids every candidate would see the identical fire/no-fire
decision at a given attempt and fallback could never demonstrate any value.
Each attempt within one task's recovery chain also gets its own `chaos_step`
(`= attempt_number - 1`, wired by `ClosedLoopExecutor` itself), so a retry
of the same candidate draws an independent decision from the retry it's
retrying.

**Known limitation, stated plainly**: this fault model is i.i.d. per
attempt (a `ConstantProbability` schedule) — it does not model a sustained,
correlated outage the way `routing.reliability_simulation`'s degradation
schedule does. Under an i.i.d. fault stream, retry-same and fallback face
statistically the *same* expected per-attempt success probability, so
fallback is not expected to systematically outperform retry-only here; if
it doesn't, that is the correct, honest result for this fault model, not a
benchmark bug (see CLAUDE.md and the Phase E spec: report a negative result
rather than hide it). Circuit-breaker's benefit under i.i.d. faults is
narrower too: avoided wasted attempts against a candidate currently on a
losing streak, not "detecting and escaping a sustained outage" (that story
is `routing.reliability_simulation`'s, at the routing layer, not recovery).
Similarly, `resilience_v1`'s injected faults are all transient/provider-
category (`FailureCategory.TIMEOUT`/`RATE_LIMIT`/`PROVIDER_FAILURE`), never
a quality-category failure (`SCHEMA_FAILURE`/`INVALID_OUTPUT`/
`REASONING_FAILURE`) — `RecoveryAction.ESCALATE` is wired and tested
(`test_recovery_policy.py`), but this particular benchmark is not expected
to exercise it; an escalation_rate of 0.0 in this benchmark's output is
expected, not a defect.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from paretoguard.chaos.faults import ExpectedRecoverability, FaultCategory, ProviderFaultKind
from paretoguard.chaos.injector import FaultInjector, FaultyProvider
from paretoguard.chaos.policies import ConstantProbability, FaultPolicy
from paretoguard.core.models import ModelSpec, OutcomeEvent
from paretoguard.evals.metrics import ResilienceMetricsSummary, compute_resilience_metrics
from paretoguard.evals.models import EvalSuite
from paretoguard.providers.base import Provider
from paretoguard.providers.mock import MockProvider
from paretoguard.recovery.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from paretoguard.recovery.fallback import FallbackPolicy
from paretoguard.recovery.policy import RecoveryPolicy
from paretoguard.recovery.retry import RecoveryRetryPolicy
from paretoguard.routing.execution import ClosedLoopExecutor
from paretoguard.routing.static import StaticRouter
from paretoguard.runtime import RetryPolicy
from paretoguard.storage import ExperimentStore
from paretoguard.telemetry.health import HealthTracker

FAULT_LEVELS: tuple[float, ...] = (0.0, 0.05, 0.15, 0.30)

CANDIDATE_PROVIDERS: tuple[str, ...] = ("resilience-a", "resilience-b", "resilience-c")

# Runtime's own default RetryPolicy (3 attempts, real backoff) would consume
# a fault-injected failure before the recovery layer ever saw it — the same
# lesson Commit 28's tests already encoded (see test_closed_loop_executor.py).
_NO_RUNTIME_RETRY = RetryPolicy(max_attempts=1)


def candidates() -> list[ModelSpec]:
    return [
        ModelSpec(name=f"model-{provider[-1]}", provider=provider, context_window=100_000)
        for provider in CANDIDATE_PROVIDERS
    ]


def _fault_injector(fault_level: float, *, seed: int) -> FaultInjector:
    policies = [
        FaultPolicy(
            fault_id=f"resilience-{provider}",
            category=FaultCategory.PROVIDER,
            kind=ProviderFaultKind.SERVER_ERROR.value,
            schedule=ConstantProbability(fault_level),
            target_provider=provider,
            expected_recoverability=ExpectedRecoverability.RECOVERABLE,
        )
        for provider in CANDIDATE_PROVIDERS
    ]
    return FaultInjector(policies, seed=seed)


def _providers(injector: FaultInjector) -> dict[str, Provider]:
    return {
        provider: FaultyProvider(inner=MockProvider(provider, seed=0), injector=injector)
        for provider in CANDIDATE_PROVIDERS
    }


@dataclass(frozen=True)
class RecoveryConfigSpec:
    """One point in the A-E comparison. `build` is a zero-arg factory
    (rather than a pre-built `RecoveryPolicy`/`CircuitBreaker` pair) so every
    `(fault_level, config)` cell gets its own fresh, un-shared
    `CircuitBreaker` instance — a breaker's state must not leak between
    independent comparison cells."""

    name: str
    description: str
    max_attempts: int
    build: Callable[[], tuple[RecoveryPolicy, CircuitBreaker | None]]


def _no_recovery() -> tuple[RecoveryPolicy, CircuitBreaker | None]:
    return (
        RecoveryPolicy(
            retry_policy=RecoveryRetryPolicy(max_same_candidate_attempts=0),
            fallback_policy=FallbackPolicy(max_fallback_depth=1),
            allow_escalation=False,
        ),
        None,
    )


def _retry_only() -> tuple[RecoveryPolicy, CircuitBreaker | None]:
    return (
        RecoveryPolicy(
            retry_policy=RecoveryRetryPolicy(max_same_candidate_attempts=2),
            fallback_policy=FallbackPolicy(max_fallback_depth=1),
            allow_escalation=False,
        ),
        None,
    )


def _retry_fallback() -> tuple[RecoveryPolicy, CircuitBreaker | None]:
    return (
        RecoveryPolicy(
            retry_policy=RecoveryRetryPolicy(max_same_candidate_attempts=1),
            fallback_policy=FallbackPolicy(max_fallback_depth=3),
            allow_escalation=False,
        ),
        None,
    )


def _retry_fallback_circuit_breaker(
    clock: Callable[[], float] | None = None,
) -> tuple[RecoveryPolicy, CircuitBreaker | None]:
    breaker = CircuitBreaker(
        CircuitBreakerConfig(failure_threshold=2, cooldown_s=5.0), clock=clock or time.monotonic
    )
    return (
        RecoveryPolicy(
            retry_policy=RecoveryRetryPolicy(max_same_candidate_attempts=1),
            fallback_policy=FallbackPolicy(max_fallback_depth=3),
            circuit_breaker=breaker,
            allow_escalation=False,
        ),
        breaker,
    )


def _full_policy(
    clock: Callable[[], float] | None = None,
) -> tuple[RecoveryPolicy, CircuitBreaker | None]:
    breaker = CircuitBreaker(
        CircuitBreakerConfig(failure_threshold=2, cooldown_s=5.0), clock=clock or time.monotonic
    )
    return (
        RecoveryPolicy(
            retry_policy=RecoveryRetryPolicy(max_same_candidate_attempts=1),
            fallback_policy=FallbackPolicy(max_fallback_depth=3),
            circuit_breaker=breaker,
            allow_escalation=True,
        ),
        breaker,
    )


def recovery_config_specs(clock: Callable[[], float] | None = None) -> list[RecoveryConfigSpec]:
    """The five points of the flagship comparison, in the order the Phase E
    spec names them. `max_attempts` is set per config to exactly the room
    its own policy could ever use (1 initial + retry budget, or 1 initial +
    retry budget per candidate across the full fallback depth) — a config
    is never handicapped by `ClosedLoopExecutor`'s own hard cap running out
    before its *policy's* budget would.

    `clock` (Phase E.5): passed through to the two circuit-breaker-enabled
    configs' `CircuitBreaker`. `None` (the default here, used by
    `run_resilience_benchmark`'s i.i.d.-fault comparison) means the real
    `time.monotonic` clock — fine for that comparison, which runs in well
    under a second of real wall-clock time regardless of fault level. A
    multi-step scenario spanning a long *simulated* timeline (e.g.
    `routing.sustained_outage_benchmark`) needs a deterministic virtual
    clock instead — real wall-clock cooldown timing would otherwise depend
    on how fast this process happens to execute, making circuit-breaker
    cooldown/probe behavior non-reproducible and effectively untested across
    the simulated timeline (300 simulated steps can execute in well under
    the real cooldown_s)."""
    return [
        RecoveryConfigSpec(
            "A-no_recovery",
            "No retry, no fallback, no escalation: one attempt, then stop.",
            max_attempts=1,
            build=_no_recovery,
        ),
        RecoveryConfigSpec(
            "B-retry_only",
            "Retry the same candidate up to 2 additional times; never falls back.",
            max_attempts=3,
            build=_retry_only,
        ),
        RecoveryConfigSpec(
            "C-retry_fallback",
            "One retry per candidate, falls back across all 3 candidates.",
            max_attempts=6,
            build=_retry_fallback,
        ),
        RecoveryConfigSpec(
            "D-retry_fallback_circuit_breaker",
            "Same as C, plus a circuit breaker excluding a tripped candidate.",
            max_attempts=6,
            build=lambda: _retry_fallback_circuit_breaker(clock),
        ),
        RecoveryConfigSpec(
            "E-full_policy",
            "Same as D, plus escalation on quality-category failures.",
            max_attempts=6,
            build=lambda: _full_policy(clock),
        ),
    ]


@dataclass(frozen=True)
class ResilienceComparisonRow:
    fault_level: float
    recovery_config: str
    metrics: ResilienceMetricsSummary


@dataclass(frozen=True)
class ResilienceBenchmarkResult:
    rows: list[ResilienceComparisonRow]
    outcomes: dict[tuple[float, str], list[OutcomeEvent]]
    """Keyed by `(fault_level, recovery_config)` — the raw `OutcomeEvent`s
    each `ResilienceComparisonRow.metrics` was computed from, for anyone who
    wants to recompute or inspect beyond the aggregate summary."""


async def run_resilience_benchmark(
    suite: EvalSuite,
    *,
    fault_levels: Sequence[float] = FAULT_LEVELS,
    configs: Sequence[RecoveryConfigSpec] | None = None,
    seed: int = 0,
    store: ExperimentStore | None = None,
) -> ResilienceBenchmarkResult:
    """Runs every `(fault_level, config)` cell of the flagship comparison
    over `suite`'s cases, sequentially within each cell (never
    `asyncio.gather`) — `HealthTracker`/`CircuitBreaker` mutate on every
    attempt, and this benchmark's own determinism guarantee (verified by
    `scripts/phase_e_resilience_experiment.py` running twice and diffing)
    depends on those mutations happening in one fixed, seed-derived order,
    not whatever order an event loop happens to schedule concurrent
    coroutines in.
    """
    configs = list(configs) if configs is not None else recovery_config_specs()
    rows: list[ResilienceComparisonRow] = []
    outcomes: dict[tuple[float, str], list[OutcomeEvent]] = {}

    for fault_level in fault_levels:
        injector = _fault_injector(fault_level, seed=seed * 1_000 + round(fault_level * 1000))
        for spec in configs:
            recovery_policy, circuit_breaker = spec.build()
            executor = ClosedLoopExecutor(
                StaticRouter(provider=CANDIDATE_PROVIDERS[0], model=candidates()[0].name),
                candidates(),
                _providers(injector),
                recovery_policy=recovery_policy,
                circuit_breaker=circuit_breaker,
                health_tracker=HealthTracker(),
                store=store,
                retry_policy=_NO_RUNTIME_RETRY,
                max_attempts=spec.max_attempts,
            )
            run_id = f"resilience-{suite.name}-{fault_level}-{spec.name}"
            cell_outcomes = []
            for case in suite.cases:
                _, outcome = await executor.execute(case, run_id=run_id)
                cell_outcomes.append(outcome)
            outcomes[(fault_level, spec.name)] = cell_outcomes
            rows.append(
                ResilienceComparisonRow(
                    fault_level=fault_level,
                    recovery_config=spec.name,
                    metrics=compute_resilience_metrics(cell_outcomes),
                )
            )

    return ResilienceBenchmarkResult(rows=rows, outcomes=outcomes)
