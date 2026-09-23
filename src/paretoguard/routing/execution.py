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

**Not imported by `paretoguard.routing.__init__`**, for the same reason
`torch_router` isn't: it pulls in `evals`/`providers`/`runtime`/`storage`/
`telemetry`, and `import paretoguard.routing` (e.g. just to get
`StaticRouter`) should not have to load that whole stack. Import it
directly: `from paretoguard.routing.execution import RoutedBenchmarkRunner`.
"""

import asyncio
import platform
from collections.abc import Callable
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from paretoguard import __version__
from paretoguard.core.config import PricingTable
from paretoguard.core.features import extract_task_features
from paretoguard.core.models import (
    InferenceRequest,
    ModelSpec,
    RoutingDecision,
    RunManifest,
    TraceEvent,
)
from paretoguard.evals.graders import grade_case
from paretoguard.evals.models import EvalCase, EvalResult, EvalSuite
from paretoguard.evals.runner import BenchmarkConfig
from paretoguard.providers.base import Provider
from paretoguard.routing.protocol import Router
from paretoguard.routing.types import CandidateProfile, RoutingRequest, candidate_key
from paretoguard.runtime import BudgetGuard, RetryPolicy, Runtime
from paretoguard.storage import ExperimentStore
from paretoguard.telemetry.health import HealthTracker


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
        request = probe_request.model_copy(
            update={"provider": selected.provider, "model": selected.name}
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
