"""RoutedBenchmarkRunner: the router-aware benchmark integration point.

    Task -> feature extraction -> Router -> RoutingDecision
         -> selected Provider/Model -> Runtime -> Grader -> EvalResult

This module is the **single, deliberate exception** to "`routing` must not
call providers directly" (docs/ARCHITECTURE.md's module boundaries): every
actual routing *strategy* (`StaticRouter`, `RuleRouter`, `ParetoRouter`,
`ReliabilityAwareRouter`, `LearnedRouter`, `EscalationRouter`, ...) still
never imports `providers` or `runtime` and only ever produces a
`RoutingDecision` — that invariant is exactly what makes those strategies
independently unit-testable without a provider. This file is the one place
that decision actually gets *dispatched*, composing `paretoguard.evals`'s
pieces (`Runtime`, `evals.graders.grade_case`) rather than embedding
dispatch logic inside a `Router` implementation. It equally does not modify
`BenchmarkRunner` (`paretoguard.evals.runner`) — the fixed-model path there
stays untouched, per that module's own docstring — nor
`paretoguard.evals.matrix.MatrixRunner` (which runs *every* candidate per
task for offline training data, not one router-selected candidate per task
the way this module does).

Also home to `ClosedLoopExecutor` (Phase E, Commit 28): the same dispatch
responsibility, extended with automatic recovery — retry/fallback/escalate/
probe/abstain via `paretoguard.recovery`, with every outcome automatically
fed back into `HealthTracker`, `CircuitBreaker`, and `EscalationRouter`
(closing the manual-`record_outcome()` gap Phase D left open). See its own
docstring for the full lifecycle.

**Not imported by `paretoguard.routing.__init__`**, for the same reason
`torch_router` isn't: it pulls in `evals`/`providers`/`runtime`/`storage`/
`telemetry`/`recovery`, and `import paretoguard.routing` (e.g. just to get
`StaticRouter`) should not have to load that whole stack. Import it
directly: `from paretoguard.routing.execution import RoutedBenchmarkRunner`
or `ClosedLoopExecutor`.
"""

import asyncio
import platform
from collections.abc import Callable
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from paretoguard import __version__
from paretoguard.core.config import PricingTable
from paretoguard.core.features import extract_task_features
from paretoguard.core.ids import deterministic_request_id
from paretoguard.core.models import (
    InferenceRequest,
    ModelSpec,
    OutcomeEvent,
    RoutingDecision,
    RunManifest,
    TraceEvent,
    TraceEventType,
)
from paretoguard.evals.graders import grade_case
from paretoguard.evals.models import EvalCase, EvalResult, EvalSuite
from paretoguard.evals.runner import BenchmarkConfig
from paretoguard.providers.base import Provider
from paretoguard.recovery.circuit_breaker import CircuitBreaker
from paretoguard.recovery.context import AttemptRecord, RecoveryAction, RecoveryContext
from paretoguard.recovery.policy import RecoveryPolicy
from paretoguard.routing.escalation import EscalationRouter
from paretoguard.routing.protocol import Router
from paretoguard.routing.types import CandidateProfile, RoutingRequest, candidate_key
from paretoguard.runtime import BudgetGuard, RetryPolicy, Runtime
from paretoguard.storage import ExperimentStore
from paretoguard.telemetry.health import HealthTracker, ModelHealth


@dataclass
class RoutedBenchmarkRunResult:
    run_id: str
    manifest: RunManifest
    results: list[EvalResult] = field(default_factory=list)
    decisions: list[RoutingDecision] = field(default_factory=list)
    """Parallel to `results`, same (case, repetition) order — `decisions[i]`
    is the `RoutingDecision` that produced `results[i]`."""


class RoutedBenchmarkRunner:
    """Runs every (case, repetition) pair in a suite through `router` to
    select one candidate, then through that candidate's `Runtime` — unlike
    `paretoguard.evals.matrix.MatrixRunner`, which runs every candidate for
    every task (that's the offline training-matrix path); this executes
    exactly one candidate per task, the way a real deployment would.
    """

    def __init__(
        self,
        router: Router,
        candidates: list[ModelSpec],
        providers: dict[str, Provider],
        *,
        config: BenchmarkConfig | None = None,
        store: ExperimentStore | None = None,
        retry_policy: RetryPolicy | None = None,
        budget_guard: BudgetGuard | None = None,
        pricing_table: PricingTable | None = None,
        pricing_config_version: str | None = None,
        health_tracker: HealthTracker | None = None,
        profiles: dict[str, CandidateProfile] | None = None,
    ) -> None:
        missing = {c.provider for c in candidates} - set(providers)
        if missing:
            raise ValueError(f"no Provider configured for candidate provider(s): {sorted(missing)}")
        self._router = router
        self._candidates = candidates
        self._providers = providers
        self._config = config or BenchmarkConfig()
        self._store = store
        self._retry_policy = retry_policy
        self._budget_guard = budget_guard
        self._pricing_table = pricing_table
        self._pricing_config_version = pricing_config_version
        self._health_tracker = health_tracker
        self._profiles = profiles or {}

    async def run(self, suite: EvalSuite, *, run_id: str | None = None) -> RoutedBenchmarkRunResult:
        run_id = run_id or f"routed-{uuid4()}"
        manifest = RunManifest(
            run_id=run_id,
            paretoguard_version=__version__,
            os=platform.system(),
            python_version=platform.python_version(),
            seed=suite.seed,
            suite_name=suite.name,
            suite_version=suite.version,
            router_name=self._router.name,
            router_config={"candidates": [c.name for c in self._candidates]},
            pricing_config_version=self._pricing_config_version,
            task_count=suite.case_count,
            repetitions=self._config.repetitions,
        )
        if self._store is not None:
            self._store.record_run(manifest)

        runtimes = {
            name: Runtime(
                provider,
                retry_policy=self._retry_policy,
                budget_guard=self._budget_guard,
                pricing_table=self._pricing_table,
                max_concurrency=self._config.max_concurrency,
                timeout_s=self._config.timeout_s,
                on_trace_event=self._trace_handler(run_id),
                run_id=run_id,
            )
            for name, provider in self._providers.items()
        }

        tasks = [
            self._run_one(runtimes, case, case_index, repetition, run_id)
            for case_index, case in enumerate(suite.cases)
            for repetition in range(self._config.repetitions)
        ]
        outcomes = list(await asyncio.gather(*tasks))
        results = [result for result, _decision, _request_id in outcomes]
        decisions = [decision for _result, decision, _request_id in outcomes]

        if self._store is not None:
            for result, decision, request_id in outcomes:
                self._store.record_eval_result(result, run_id=run_id)
                self._store.record_routing_decision(request_id, decision, run_id=run_id)

        return RoutedBenchmarkRunResult(
            run_id=run_id, manifest=manifest, results=results, decisions=decisions
        )

    def _trace_handler(self, run_id: str) -> Callable[[TraceEvent], None] | None:
        if self._store is None and self._health_tracker is None:
            return None
        store = self._store
        health_tracker = self._health_tracker

        def handler(event: TraceEvent) -> None:
            if store is not None:
                store.record_trace_event(event, run_id=run_id)
            if health_tracker is not None:
                health_tracker.record_trace_event(event)

        return handler

    async def _run_one(
        self,
        runtimes: dict[str, Runtime],
        case: EvalCase,
        case_index: int,
        repetition: int,
        run_id: str,
    ) -> tuple[EvalResult, RoutingDecision, UUID]:
        probe_request = InferenceRequest(
            task_id=case.case_id,
            provider="",
            model="",
            messages=case.messages,
            tools=case.tools,
            structured_output_schema=case.structured_output_schema,
            max_output_tokens=case.max_output_tokens,
            temperature=case.temperature,
            metadata=case.metadata,
        )
        features = extract_task_features(probe_request)
        health_snapshot = {}
        if self._health_tracker is not None:
            for candidate in self._candidates:
                health = self._health_tracker.get(candidate.provider, candidate.name)
                if health is not None:
                    health_snapshot[candidate_key(candidate.provider, candidate.name)] = health

        routing_request = RoutingRequest(
            request=probe_request,
            features=features,
            candidates=self._candidates,
            health=health_snapshot,
            profiles=self._profiles,
        )
        decision = self._router.route(routing_request)
        selected = next(c for c in self._candidates if c.name == decision.selected_model)
        # See core.ids.deterministic_request_id's docstring: the router's
        # choice isn't known until after routing, so the probe_request's
        # random default request_id (only ever used to extract features)
        # gets overridden here with one derived from the case/repetition and
        # the *actual* selected candidate, once that's known.
        request = probe_request.model_copy(
            update={
                "provider": selected.provider,
                "model": selected.name,
                "request_id": deterministic_request_id(
                    case.case_id, str(repetition), selected.provider, selected.name
                ),
            }
        )

        response = await runtimes[selected.provider].run(request)

        if self._store is not None:
            self._store.record_request(request, run_id=run_id)
            self._store.record_response(response, run_id=run_id)

        sequence = case_index * self._config.repetitions + repetition

        if not response.succeeded:
            error = response.error
            result = EvalResult(
                case_id=case.case_id,
                request_id=request.request_id,
                repetition=repetition,
                sequence=sequence,
                succeeded=False,
                score=0.0,
                grader_kind=case.grader.kind,
                explanation=f"inference failed: {error.message if error else 'unknown error'}",
                response_error_category=error.category if error else None,
                latency_ms=response.latency.total_latency_ms,
                cost_usd=response.cost.total_cost_usd if response.cost else None,
                total_tokens=response.token_usage.total_tokens,
            )
            return result, decision, request.request_id

        outcome = grade_case(case, response)
        result = EvalResult(
            case_id=case.case_id,
            request_id=request.request_id,
            repetition=repetition,
            sequence=sequence,
            succeeded=outcome.succeeded,
            score=outcome.score,
            grader_kind=case.grader.kind,
            explanation=outcome.explanation,
            details=outcome.details,
            latency_ms=response.latency.total_latency_ms,
            cost_usd=response.cost.total_cost_usd if response.cost else None,
            total_tokens=response.token_usage.total_tokens,
        )
        return result, decision, request.request_id


_TERMINAL_RECOVERY_ACTIONS = frozenset({RecoveryAction.ABSTAIN, RecoveryAction.FAIL})


class ClosedLoopExecutor:
    """The Phase E control plane:

        RoutingDecision -> Execution -> Validation/failure classification
        -> Outcome event -> Health update -> Escalation/recovery feedback
        -> optional next execution attempt -> terminal result

    This is what makes `EscalationRouter.record_outcome` automatic (Phase D
    required a caller to invoke it manually) and what lets `HealthTracker`'s
    signals actually influence recovery — without coupling `HealthTracker`,
    `CircuitBreaker`, or `RecoveryPolicy` to each other or to any `Router`.
    Each stays independently usable; this class is the one place that
    *observes* an execution outcome and *feeds* every one of them, then asks
    `RecoveryPolicy` — a pure function of the resulting immutable
    `RecoveryContext` snapshot — what should happen next. The `Router`
    itself is consulted exactly once per task, to pick the *first*
    candidate; every subsequent switch within the same task is a
    `RecoveryDecision`, not a fresh routing decision, matching "the router
    decides, the execution/control plane owns lifecycle state."
    """

    def __init__(
        self,
        router: Router,
        candidates: list[ModelSpec],
        providers: dict[str, Provider],
        *,
        recovery_policy: RecoveryPolicy | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        health_tracker: HealthTracker | None = None,
        store: ExperimentStore | None = None,
        retry_policy: RetryPolicy | None = None,
        pricing_table: PricingTable | None = None,
        profiles: dict[str, CandidateProfile] | None = None,
        max_attempts: int = 5,
    ) -> None:
        missing = {c.provider for c in candidates} - set(providers)
        if missing:
            raise ValueError(f"no Provider configured for candidate provider(s): {sorted(missing)}")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._router = router
        self._candidates = candidates
        self._candidates_by_name = {c.name: c for c in candidates}
        self._recovery_policy = recovery_policy or RecoveryPolicy()
        self._circuit_breaker = circuit_breaker
        self._health_tracker = health_tracker
        self._store = store
        self._profiles = dict(profiles or {})
        self._max_attempts = max_attempts
        self._runtimes = {
            name: Runtime(provider, retry_policy=retry_policy, pricing_table=pricing_table)
            for name, provider in providers.items()
        }

    async def execute(
        self, case: EvalCase, *, run_id: str | None = None
    ) -> tuple[EvalResult, OutcomeEvent]:
        """Runs one task through the full closed loop to a terminal
        `EvalResult` + `OutcomeEvent`. Never raises for an execution
        failure — an exhausted recovery chain is a normal `succeeded=False`
        result, not an exception (matching every other runner in this repo:
        no case is ever silently dropped)."""
        probe_request = InferenceRequest(
            task_id=case.case_id,
            provider="",
            model="",
            messages=case.messages,
            tools=case.tools,
            structured_output_schema=case.structured_output_schema,
            max_output_tokens=case.max_output_tokens,
            temperature=case.temperature,
            metadata=case.metadata,
        )
        features = extract_task_features(probe_request)
        routing_request = RoutingRequest(
            request=probe_request,
            features=features,
            candidates=self._candidates,
            health=self._health_snapshot(),
            profiles=self._profiles,
        )
        initial_decision = self._router.route(routing_request)
        initial_request_id = probe_request.request_id
        if self._store is not None:
            self._store.record_routing_decision(initial_request_id, initial_decision, run_id=run_id)

        current_model = initial_decision.selected_model
        current_provider = self._candidates_by_name[current_model].provider

        attempted: list[AttemptRecord] = []
        recovery_actions: list[str] = []
        total_cost = 0.0
        total_latency = 0.0
        final_response = None
        final_request_id = initial_request_id

        for attempt_number in range(1, self._max_attempts + 1):
            request_id = deterministic_request_id(
                case.case_id, str(attempt_number), current_provider, current_model
            )
            final_request_id = request_id
            request = probe_request.model_copy(
                update={
                    "provider": current_provider,
                    "model": current_model,
                    "request_id": request_id,
                }
            )

            response = await self._runtimes[current_provider].run(request)
            if self._store is not None:
                self._store.record_request(request, run_id=run_id)
                self._store.record_response(response, run_id=run_id)

            step_cost = response.cost.total_cost_usd if response.cost else None
            total_cost += step_cost or 0.0
            total_latency += response.latency.total_latency_ms
            succeeded = response.succeeded
            failure_category = response.error.category if response.error else None
            final_response = response

            # --- automatic closed-loop feedback (never left to a caller) ---
            if self._health_tracker is not None:
                self._health_tracker.record_response(response)
            key = candidate_key(current_provider, current_model)
            if self._circuit_breaker is not None:
                if succeeded:
                    self._circuit_breaker.record_success(key)
                else:
                    self._circuit_breaker.record_failure(key)
            if isinstance(self._router, EscalationRouter) and attempt_number == 1:
                # EscalationRouter's own record keys on the *original*
                # routing request id, and reports on that first candidate's
                # own outcome — see EscalationRouter.record_outcome's
                # docstring for why this can only happen once per chain.
                self._router.record_outcome(
                    initial_request_id, succeeded=succeeded, actual_cost_usd=step_cost
                )

            attempted.append(
                AttemptRecord(current_provider, current_model, failure_category, succeeded)
            )

            if succeeded:
                break

            # `RecoveryContext.attempted` is *prior* history — attempts
            # before the one that just failed, which `current_provider`/
            # `current_model`/`failure_category` already represent. `attempted`
            # here (the full audit trail for `OutcomeEvent`) always has the
            # just-failed attempt as its last element by this point, hence
            # `[:-1]`. `select_fallback` separately excludes the *current*
            # candidate explicitly, so this never causes it to be revisited.
            context = RecoveryContext(
                current_provider=current_provider,
                current_model=current_model,
                failure_category=failure_category,
                task_features=features,
                attempted=tuple(attempted[:-1]),
                health_snapshots=self._health_snapshot(),
                circuit_snapshots=self._circuit_breaker.all_snapshots()
                if self._circuit_breaker
                else {},
                constraints=routing_request.constraints,
            )
            recovery_decision = self._recovery_policy.decide(context, self._candidates)
            recovery_actions.append(recovery_decision.action.value)
            if self._store is not None:
                self._store.record_trace_event(
                    TraceEvent(
                        run_id=run_id,
                        request_id=request_id,
                        event_type=TraceEventType.RECOVERY_ACTION,
                        payload={
                            "action": recovery_decision.action.value,
                            "reason": recovery_decision.reason,
                            "target_provider": recovery_decision.target_provider,
                            "target_model": recovery_decision.target_model,
                        },
                    ),
                    run_id=run_id,
                )

            if recovery_decision.action in _TERMINAL_RECOVERY_ACTIONS:
                break
            current_provider = recovery_decision.target_provider or current_provider
            current_model = recovery_decision.target_model or current_model

        assert final_response is not None  # loop runs >= 1 time (max_attempts >= 1)

        if not final_response.succeeded:
            error = final_response.error
            result = EvalResult(
                case_id=case.case_id,
                request_id=final_request_id,
                repetition=0,
                sequence=0,
                succeeded=False,
                score=0.0,
                grader_kind=case.grader.kind,
                explanation=(
                    f"inference failed after {len(attempted)} attempt(s): "
                    f"{error.message if error else 'unknown error'}"
                ),
                response_error_category=error.category if error else None,
                latency_ms=total_latency,
                cost_usd=total_cost or None,
                total_tokens=final_response.token_usage.total_tokens,
            )
        else:
            grade_outcome = grade_case(case, final_response)
            result = EvalResult(
                case_id=case.case_id,
                request_id=final_request_id,
                repetition=0,
                sequence=0,
                succeeded=grade_outcome.succeeded,
                score=grade_outcome.score,
                grader_kind=case.grader.kind,
                explanation=grade_outcome.explanation,
                details=grade_outcome.details,
                latency_ms=total_latency,
                cost_usd=total_cost or None,
                total_tokens=final_response.token_usage.total_tokens,
            )

        if self._store is not None:
            self._store.record_eval_result(result, run_id=run_id)

        last_failure = attempted[-1].failure_category if attempted else None
        outcome_event = OutcomeEvent(
            run_id=run_id,
            task_id=case.case_id,
            initial_routing_decision_id=initial_request_id,
            final_provider=current_provider,
            final_model=current_model,
            attempt_count=len(attempted),
            succeeded=result.succeeded,
            failure_category=last_failure.value if last_failure else None,
            total_cost_usd=total_cost or None,
            total_latency_ms=total_latency,
            recovery_actions=recovery_actions,
        )
        return result, outcome_event

    def _health_snapshot(self) -> dict[str, ModelHealth]:
        if self._health_tracker is None:
            return {}
        snapshot: dict[str, ModelHealth] = {}
        for candidate in self._candidates:
            health = self._health_tracker.get(candidate.provider, candidate.name)
            if health is not None:
                snapshot[candidate_key(candidate.provider, candidate.name)] = health
        return snapshot
