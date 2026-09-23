"""Async benchmark runner: executes an EvalSuite against a Provider via Runtime,
grades each response, and (optionally) persists everything to an ExperimentStore.

Built entirely on the existing `paretoguard.providers.Provider` and
`paretoguard.runtime.Runtime` abstractions — no router exists yet (Phase D), so a
run targets one fixed (provider, model) pair, matching the "assume no specific
router" constraint on `evals` in docs/ARCHITECTURE.md.
"""

import asyncio
import platform
from collections.abc import Callable
from dataclasses import dataclass, field
from uuid import uuid4

from paretoguard import __version__
from paretoguard.core.config import PricingTable
from paretoguard.core.models import InferenceRequest, RunManifest, TraceEvent
from paretoguard.evals.graders import GradeOutcome, get_grader
from paretoguard.evals.models import EvalCase, EvalResult, EvalSuite
from paretoguard.providers.base import Provider
from paretoguard.runtime import BudgetGuard, RetryPolicy, Runtime
from paretoguard.storage import ExperimentStore


@dataclass
class BenchmarkConfig:
    """Execution parameters for one benchmark run."""

    repetitions: int = 1
    max_concurrency: int = 4
    timeout_s: float = 60.0

    def __post_init__(self) -> None:
        if self.repetitions < 1:
            raise ValueError("repetitions must be >= 1")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")


@dataclass
class BenchmarkRunResult:
    """Everything produced by one `BenchmarkRunner.run()` call."""

    run_id: str
    manifest: RunManifest
    results: list[EvalResult] = field(default_factory=list)
    """Deterministically ordered: cases in suite order, repetitions in order,
    regardless of concurrent completion timing (see `EvalResult.sequence`)."""


class BenchmarkRunner:
    """Runs every (case, repetition) pair in a suite through one `Runtime`
    instance, grades each response, and persists results if a store is given.

    A failed *inference* (provider/runtime error after retries) and a failed
    *grade* (wrong answer) are both recorded as a normal EvalResult with
    `succeeded=False` — no case is ever silently dropped from the results.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        model: str,
        config: BenchmarkConfig | None = None,
        store: ExperimentStore | None = None,
        retry_policy: RetryPolicy | None = None,
        budget_guard: BudgetGuard | None = None,
        pricing_table: PricingTable | None = None,
        pricing_config_version: str | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._config = config or BenchmarkConfig()
        self._store = store
        self._retry_policy = retry_policy
        self._budget_guard = budget_guard
        self._pricing_table = pricing_table
        self._pricing_config_version = pricing_config_version

    async def run(self, suite: EvalSuite, *, run_id: str | None = None) -> BenchmarkRunResult:
        run_id = run_id or f"run-{uuid4()}"
        manifest = RunManifest(
            run_id=run_id,
            paretoguard_version=__version__,
            os=platform.system(),
            python_version=platform.python_version(),
            seed=suite.seed,
            suite_name=suite.name,
            suite_version=suite.version,
            router_name=None,
            router_config={"provider": self._provider.name, "model": self._model},
            pricing_config_version=self._pricing_config_version,
            task_count=suite.case_count,
            repetitions=self._config.repetitions,
        )
        if self._store is not None:
            self._store.record_run(manifest)

        runtime = Runtime(
            self._provider,
            retry_policy=self._retry_policy,
            budget_guard=self._budget_guard,
            pricing_table=self._pricing_table,
            max_concurrency=self._config.max_concurrency,
            timeout_s=self._config.timeout_s,
            on_trace_event=self._trace_handler(run_id),
            run_id=run_id,
        )

        tasks = [
            self._run_one(runtime, case, case_index, repetition, run_id)
            for case_index, case in enumerate(suite.cases)
            for repetition in range(self._config.repetitions)
        ]
        # asyncio.gather preserves input order in its returned list regardless of
        # which task finishes first, so `results` is already in canonical
        # (case, repetition) order even though tasks race concurrently.
        results = list(await asyncio.gather(*tasks))

        if self._store is not None:
            for result in results:
                self._store.record_eval_result(result, run_id=run_id)

        return BenchmarkRunResult(run_id=run_id, manifest=manifest, results=results)

    def _trace_handler(self, run_id: str) -> Callable[[TraceEvent], None] | None:
        if self._store is None:
            return None
        store = self._store

        def handler(event: TraceEvent) -> None:
            store.record_trace_event(event, run_id=run_id)

        return handler

    async def _run_one(
        self, runtime: Runtime, case: EvalCase, case_index: int, repetition: int, run_id: str
    ) -> EvalResult:
        request = InferenceRequest(
            task_id=case.case_id,
            provider=self._provider.name,
            model=self._model,
            messages=case.messages,
            tools=case.tools,
            structured_output_schema=case.structured_output_schema,
            max_output_tokens=case.max_output_tokens,
            temperature=case.temperature,
            metadata=case.metadata,
        )
        response = await runtime.run(request)

        if self._store is not None:
            self._store.record_request(request, run_id=run_id)
            self._store.record_response(response, run_id=run_id)

        sequence = case_index * self._config.repetitions + repetition

        if not response.succeeded:
            error = response.error
            return EvalResult(
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

        try:
            grader = get_grader(case.grader.kind)
            outcome = grader(response, case.ground_truth, case.grader)
        except Exception as exc:
            outcome = GradeOutcome(succeeded=False, score=0.0, explanation=f"grading error: {exc}")

        return EvalResult(
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
