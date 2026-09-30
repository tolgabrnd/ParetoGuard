"""Quality-failure escalation scenario (Phase E.5, item 3).

**SIMULATION.** Every failure here is a synthetic, injected
`chaos.FaultInjector` fault against `MockProvider` — never a real
provider's actual output quality (see CLAUDE.md).

**Why this exists**: `resilience_v1`'s flagship comparison only injects
transient/provider-category faults, so `RecoveryAction.ESCALATE` — which
exists specifically for quality-category failures
(`recovery.escalation.QUALITY_FAILURE_CATEGORIES`) — never fired there.
Worse, until this same commit, `ClosedLoopExecutor` had no path *at all* for
a transport-successful-but-wrong response to reach recovery: any response
without a transport error was treated as terminal success regardless of
content, so escalation could never fire through the real closed-loop path
no matter what a benchmark injected (see `routing.execution`'s
`validate_quality` parameter and docs/LIMITATIONS.md's Phase E.5 section for
the fix). This module is both the fix's exercise and its evidence.

**The fault**: reuses `chaos.injector.FaultyProvider` with
`ProviderFaultKind.MALFORMED_STRUCTURED_OUTPUT` — a transport-*successful*
response whose `structured_output` has a required key dropped — targeting
one candidate provider at a configurable rate. Every case here is
JSON_SCHEMA-graded against a schema that requires the dropped key, so the
corruption deterministically fails grading. This is not a new fault
mechanism; Commit 26 already built `MALFORMED_STRUCTURED_OUTPUT` for
exactly this shape of corruption.

**The comparison**: two `ClosedLoopExecutor` configurations against the
identical fault stream (same seed, independent of which arm is running,
the same fairness discipline `resilience_benchmark` uses) —

- `no_validation`: `validate_quality=False` — the pre-Phase-E.5 behavior.
  A malformed response is accepted as "successful" the instant the
  provider call itself doesn't error; recovery is never invoked; the final
  `EvalResult` still correctly grades it as failed (grading always happens
  at the very end regardless), but nothing tried to fix it along the way.
- `validation_and_escalation`: `validate_quality=True` (the default) with
  `RecoveryPolicy(allow_escalation=True)` — the fixed behavior. The
  malformed response is graded in-loop, classified `SCHEMA_FAILURE`, and
  (since that's a quality category, not a transient one) escalated to a
  healthy fallback candidate.

This isolates exactly the variable the Phase E.5 spec asks about: not
"does escalation fire" in isolation (already unit-tested in
`test_recovery_policy.py`), but "does *validating and escalating on*
quality failures, through the real closed-loop path, actually change task
outcomes" — compared against a control arm that skips validation entirely,
not against some other recovery mechanism.
"""

from dataclasses import dataclass

from paretoguard.chaos.faults import ExpectedRecoverability, FaultCategory, ProviderFaultKind
from paretoguard.chaos.injector import FaultInjector, FaultyProvider
from paretoguard.chaos.policies import ConstantProbability, FaultPolicy
from paretoguard.core.models import Message, Role
from paretoguard.evals.metrics import ResilienceMetricsSummary, compute_resilience_metrics
from paretoguard.evals.models import EvalCase, EvalSuite, GraderConfig, GraderKind, GroundTruth
from paretoguard.providers.base import Provider
from paretoguard.providers.mock import MockProvider
from paretoguard.recovery.fallback import FallbackPolicy
from paretoguard.recovery.policy import RecoveryPolicy
from paretoguard.recovery.retry import RecoveryRetryPolicy
from paretoguard.routing.execution import ClosedLoopExecutor
from paretoguard.routing.resilience_benchmark import CANDIDATE_PROVIDERS, candidates
from paretoguard.routing.static import StaticRouter
from paretoguard.runtime import RetryPolicy
from paretoguard.storage import ExperimentStore
from paretoguard.telemetry.health import HealthTracker

DEFAULT_NUM_CASES = 20
DEFAULT_QUALITY_FAULT_RATE = 0.6
_PRIMARY_PROVIDER = CANDIDATE_PROVIDERS[0]
_DROP_KEY = "name"

_NO_RUNTIME_RETRY = RetryPolicy(max_attempts=1)

_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
}
_ANSWER = {"name": "Ada", "age": 30}


def build_suite(num_cases: int = DEFAULT_NUM_CASES) -> EvalSuite:
    """Deterministic, fixed-difficulty JSON-schema tasks — every case
    succeeds with certainty absent injected chaos (same "fixed task load"
    discipline `resilience_v1` uses), so failure here comes only from the
    injected quality fault."""
    cases = [
        EvalCase(
            case_id=f"escalation-{i:03d}",
            messages=[Message(role=Role.USER, content="Return a record with name and age.")],
            grader=GraderConfig(kind=GraderKind.JSON_SCHEMA),
            ground_truth=GroundTruth(json_schema=_SCHEMA),
            tags=["escalation"],
            metadata={"mock_scenario": "exact_json", "mock_json_answer": dict(_ANSWER)},
        )
        for i in range(num_cases)
    ]
    return EvalSuite(
        name="escalation_v1",
        version="1.0.0",
        description=(
            "SIMULATION: fixed-difficulty JSON-schema tasks for the Phase E.5 "
            "quality-failure escalation scenario."
        ),
        seed=0,
        cases=cases,
    )


def _fault_injector(*, quality_fault_rate: float, seed: int) -> FaultInjector:
    policy = FaultPolicy(
        fault_id=f"escalation-{_PRIMARY_PROVIDER}",
        category=FaultCategory.PROVIDER,
        kind=ProviderFaultKind.MALFORMED_STRUCTURED_OUTPUT.value,
        schedule=ConstantProbability(quality_fault_rate),
        target_provider=_PRIMARY_PROVIDER,
        parameters={"drop_key": _DROP_KEY},
        expected_recoverability=ExpectedRecoverability.RECOVERABLE,
    )
    return FaultInjector([policy], seed=seed)


def _providers(injector: FaultInjector) -> dict[str, Provider]:
    return {
        provider: FaultyProvider(inner=MockProvider(provider, seed=0), injector=injector)
        for provider in CANDIDATE_PROVIDERS
    }


@dataclass(frozen=True)
class EscalationComparisonResult:
    no_validation: ResilienceMetricsSummary
    validation_and_escalation: ResilienceMetricsSummary
    success_rate_delta: float
    """`validation_and_escalation.task_success_rate - no_validation.task_success_rate`."""
    added_average_attempts: float
    """`validation_and_escalation.average_attempts - no_validation.average_attempts`
    — the attempt-count cost of validating and escalating."""
    added_mean_latency_ms: float
    """Mean `total_latency_ms` delta (validation_and_escalation - no_validation),
    computed directly from raw outcomes — not `median`/`p95`, since the
    interesting question here is total added work across the whole run, not
    the shape of the latency distribution. Cost is not reported: no
    `PricingTable` is configured (`MockProvider` produces no cost data
    without one), and CLAUDE.md forbids reporting a fabricated $0.00 in its
    place."""


async def run_escalation_scenario(
    *,
    num_cases: int = DEFAULT_NUM_CASES,
    quality_fault_rate: float = DEFAULT_QUALITY_FAULT_RATE,
    seed: int = 0,
    store: ExperimentStore | None = None,
) -> EscalationComparisonResult:
    suite = build_suite(num_cases)
    injector = _fault_injector(quality_fault_rate=quality_fault_rate, seed=seed)

    no_validation_policy = RecoveryPolicy(
        retry_policy=RecoveryRetryPolicy(max_same_candidate_attempts=0),
        fallback_policy=FallbackPolicy(max_fallback_depth=1),
        allow_escalation=False,
    )
    no_validation_executor = ClosedLoopExecutor(
        StaticRouter(provider=_PRIMARY_PROVIDER, model=candidates()[0].name),
        candidates(),
        _providers(injector),
        recovery_policy=no_validation_policy,
        health_tracker=HealthTracker(),
        store=store,
        retry_policy=_NO_RUNTIME_RETRY,
        max_attempts=1,
        validate_quality=False,
    )

    validation_policy = RecoveryPolicy(
        retry_policy=RecoveryRetryPolicy(max_same_candidate_attempts=0),
        fallback_policy=FallbackPolicy(max_fallback_depth=3),
        allow_escalation=True,
    )
    validation_executor = ClosedLoopExecutor(
        StaticRouter(provider=_PRIMARY_PROVIDER, model=candidates()[0].name),
        candidates(),
        _providers(injector),
        recovery_policy=validation_policy,
        health_tracker=HealthTracker(),
        store=store,
        retry_policy=_NO_RUNTIME_RETRY,
        max_attempts=3,
        validate_quality=True,
    )

    no_validation_outcomes = []
    for case in suite.cases:
        _, outcome = await no_validation_executor.execute(case, run_id="escalation-no_validation")
        no_validation_outcomes.append(outcome)

    validation_outcomes = []
    for case in suite.cases:
        _, outcome = await validation_executor.execute(
            case, run_id="escalation-validation_and_escalation"
        )
        validation_outcomes.append(outcome)

    no_validation_metrics = compute_resilience_metrics(no_validation_outcomes)
    validation_metrics = compute_resilience_metrics(validation_outcomes)

    mean_latency_no_validation = sum(o.total_latency_ms for o in no_validation_outcomes) / len(
        no_validation_outcomes
    )
    mean_latency_validation = sum(o.total_latency_ms for o in validation_outcomes) / len(
        validation_outcomes
    )

    return EscalationComparisonResult(
        no_validation=no_validation_metrics,
        validation_and_escalation=validation_metrics,
        success_rate_delta=(
            validation_metrics.task_success_rate - no_validation_metrics.task_success_rate
        ),
        added_average_attempts=(
            validation_metrics.average_attempts - no_validation_metrics.average_attempts
        ),
        added_mean_latency_ms=mean_latency_validation - mean_latency_no_validation,
    )
