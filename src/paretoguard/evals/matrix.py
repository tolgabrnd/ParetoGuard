"""Offline training-matrix generation: runs every `EvalCase` against every
eligible candidate model — never just one router's selection.

This exists specifically to avoid the selection-bias trap in learned-router
training: training a router only on traffic an already-active router
selected means only the selected model's outcome is ever observed, so the
"what would have happened with a different model" counterfactual the router
is supposed to learn is never in the data. Running the exhaustive matrix
here, offline, is how every candidate's outcome on every task gets recorded.

This is strictly an offline/benchmarking path. At routing/serving time, only
the selected model is ever executed — see `paretoguard.routing` (the
`Router` protocol never calls a provider) and
`paretoguard.routing.execution.RoutedBenchmarkRunner` (executes exactly
one candidate per request, the router's choice). Conflating the two would
leak information no real deployment has.
"""

import asyncio
import platform
from dataclasses import dataclass, field
from uuid import uuid4

from paretoguard import __version__
from paretoguard.core.config import PricingTable
from paretoguard.core.features import TaskFeatures, extract_task_features
from paretoguard.core.ids import deterministic_request_id
from paretoguard.core.models import InferenceRequest, ModelSpec, RunManifest
from paretoguard.evals.graders import grade_case
from paretoguard.evals.models import EvalCase, EvalResult, EvalSuite
from paretoguard.providers.base import Provider
from paretoguard.runtime import BudgetGuard, RetryPolicy, Runtime
from paretoguard.storage import ExperimentStore


@dataclass(frozen=True)
class MatrixRow:
    """One (task, candidate) training example.

    `task_features` is computed once per task from routing-time-safe
    information only (see `TaskFeatures`) and is identical across every
    candidate's row for the same (case_id, repetition) — it must never vary
    by which candidate produced the response, since a real router would
    extract it before knowing which candidate would be chosen.
    """

    task_id: str
    repetition: int
    provider: str
    model: str
    task_features: TaskFeatures
    succeeded: bool
    score: float
    cost_usd: float | None
    latency_ms: float
    total_tokens: int
    response_error_category: str | None
    template_id: str | None
    """From `EvalCase.metadata["template_id"]`, for group-aware split
    grouping only (`paretoguard.routing.splits`) — deliberately NOT part of
    `task_features`/the model's feature vector: including which of a small,
    enumerable set of templates generated a task as a direct input feature
    would let a learned router memorize per-template outcomes instead of
    learning generalizable task structure. `None` for suites that don't tag
    it (single-template suites, where case-level grouping is already
    sufficient — see routing/splits.py's docstring for which suites need
    this and why)."""


@dataclass
class MatrixConfig:
    repetitions: int = 1
    max_concurrency: int = 4
    timeout_s: float = 60.0

    def __post_init__(self) -> None:
        if self.repetitions < 1:
            raise ValueError("repetitions must be >= 1")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")


@dataclass
class MatrixRunResult:
    run_id: str
    manifest: RunManifest
    rows: list[MatrixRow] = field(default_factory=list)


class MatrixRunner:
    """Runs every (case, repetition, candidate) triple in a suite, one
    `Runtime` per provider (candidates may span multiple providers).
    """

    def __init__(
        self,
        candidates: list[ModelSpec],
        providers: dict[str, Provider],
        *,
        config: MatrixConfig | None = None,
        pricing_table: PricingTable | None = None,
        retry_policy: RetryPolicy | None = None,
        budget_guard: BudgetGuard | None = None,
        store: ExperimentStore | None = None,
    ) -> None:
        missing = {c.provider for c in candidates} - set(providers)
        if missing:
            raise ValueError(f"no Provider configured for candidate provider(s): {sorted(missing)}")
        self._candidates = candidates
        self._providers = providers
        self._config = config or MatrixConfig()
        self._pricing_table = pricing_table
        self._retry_policy = retry_policy
        self._budget_guard = budget_guard
        self._store = store

    async def run(self, suite: EvalSuite, *, run_id: str | None = None) -> MatrixRunResult:
        run_id = run_id or f"matrix-{uuid4()}"
        manifest = RunManifest(
            run_id=run_id,
            paretoguard_version=__version__,
            os=platform.system(),
            python_version=platform.python_version(),
            seed=suite.seed,
            suite_name=suite.name,
            suite_version=suite.version,
            router_name="__matrix__",
            router_config={"candidates": [c.name for c in self._candidates]},
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
            )
            for name, provider in self._providers.items()
        }

        flattened = [
            (case, repetition, candidate)
            for case in suite.cases
            for repetition in range(self._config.repetitions)
            for candidate in self._candidates
        ]
        tasks = [
            self._run_one(runtimes, case, repetition, candidate, sequence, run_id)
            for sequence, (case, repetition, candidate) in enumerate(flattened)
        ]
        rows = list(await asyncio.gather(*tasks))
        return MatrixRunResult(run_id=run_id, manifest=manifest, rows=rows)

    async def _run_one(
        self,
        runtimes: dict[str, Runtime],
        case: EvalCase,
        repetition: int,
        candidate: ModelSpec,
        sequence: int,
        run_id: str,
    ) -> MatrixRow:
        request = InferenceRequest(
            request_id=deterministic_request_id(
                case.case_id, str(repetition), candidate.provider, candidate.name
            ),
            task_id=case.case_id,
            provider=candidate.provider,
            model=candidate.name,
            messages=case.messages,
            tools=case.tools,
            structured_output_schema=case.structured_output_schema,
            max_output_tokens=case.max_output_tokens,
            temperature=case.temperature,
            metadata=case.metadata,
        )
        task_features = extract_task_features(request)
        template_id = case.metadata.get("template_id")
        response = await runtimes[candidate.provider].run(request)

        if self._store is not None:
            self._store.record_request(request, run_id=run_id)
            self._store.record_response(response, run_id=run_id)

        if not response.succeeded:
            error = response.error
            row = MatrixRow(
                task_id=case.case_id,
                repetition=repetition,
                provider=candidate.provider,
                model=candidate.name,
                task_features=task_features,
                succeeded=False,
                score=0.0,
                cost_usd=response.cost.total_cost_usd if response.cost else None,
                latency_ms=response.latency.total_latency_ms,
                total_tokens=response.token_usage.total_tokens,
                response_error_category=error.category.value if error else None,
                template_id=template_id,
            )
            if self._store is not None:
                self._store.record_eval_result(
                    self._matrix_row_to_eval_result(row, request, sequence, case), run_id=run_id
                )
            return row

        outcome = grade_case(case, response)
        row = MatrixRow(
            task_id=case.case_id,
            repetition=repetition,
            provider=candidate.provider,
            model=candidate.name,
            task_features=task_features,
            succeeded=outcome.succeeded,
            score=outcome.score,
            cost_usd=response.cost.total_cost_usd if response.cost else None,
            latency_ms=response.latency.total_latency_ms,
            total_tokens=response.token_usage.total_tokens,
            response_error_category=None,
            template_id=template_id,
        )
        if self._store is not None:
            self._store.record_eval_result(
                self._matrix_row_to_eval_result(row, request, sequence, case), run_id=run_id
            )
        return row

    @staticmethod
    def _matrix_row_to_eval_result(
        row: MatrixRow, request: InferenceRequest, sequence: int, case: EvalCase
    ) -> EvalResult:
        """`MatrixRow` (candidate-labeled) -> `EvalResult` (the storage
        schema's shape) so matrix runs persist through the same
        `eval_results` table as every other run. `case_id` here embeds the
        candidate, since the normal (task_id, repetition) key is no longer
        unique across a matrix's multiple candidates per task; `task_id`
        (`row.task_id`) is preserved as the join key back to the task."""
        from paretoguard.core.models import FailureCategory

        return EvalResult(
            case_id=f"{row.task_id}::{row.model}",
            request_id=request.request_id,
            repetition=row.repetition,
            sequence=sequence,
            succeeded=row.succeeded,
            score=row.score,
            grader_kind=case.grader.kind,
            explanation=f"matrix row: task={row.task_id!r} candidate={row.model!r}",
            response_error_category=(
                FailureCategory(row.response_error_category)
                if row.response_error_category
                else None
            ),
            latency_ms=row.latency_ms,
            cost_usd=row.cost_usd,
            total_tokens=row.total_tokens,
        )
