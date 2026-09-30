"""Metrics derived from a set of `EvalResult`s.

Every metric here is computed from raw `EvalResult` records — never hard-coded —
and every formula assumes each case in the input was run the *same* number of
times (`repetitions`), which is how `BenchmarkRunner` always executes a suite.
Passing results from a mix of repetition counts across cases will still compute
without raising an error, but `pass_at_k`/`average_repetitions` lose their clean
interpretation in that case (documented per-field below).
"""

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from statistics import mean
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from paretoguard.core.models import OutcomeEvent
from paretoguard.evals.models import EvalResult, GraderKind

if TYPE_CHECKING:
    # Deferred for the same reason store.py's EvalResult import is: a
    # module-level `from paretoguard.recovery.context import RecoveryAction`
    # here is circular (recovery.context -> routing.types -> routing
    # (__init__ imports routing.dataset) -> evals.matrix -> evals (__init__
    # imports this module) -> back here). Only type checkers need this;
    # `compute_resilience_metrics` imports RecoveryAction locally itself,
    # where recovery.context is already fully initialized by the time this
    # module's own import machinery finishes.
    from paretoguard.agents.state import AgentTrajectory


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021, "Evaluating Large Language
    Models Trained on Code", eq. 1): the probability that at least one of `k`
    samples drawn without replacement from `n` total samples succeeds, given
    that `c` of the `n` succeeded.

    Uses the numerically stable product form (`1 - prod` rather than a ratio of
    factorials/binomial coefficients) to avoid overflow for large `n`. Requires
    `0 <= c <= n` and `1 <= k <= n`.
    """
    if not 0 <= c <= n:
        raise ValueError(f"c ({c}) must be in [0, n] (n={n})")
    if not 1 <= k <= n:
        raise ValueError(f"k ({k}) must be in [1, n] (n={n})")
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


class MetricsSummary(BaseModel):
    """Aggregate metrics over a set of `EvalResult`s. Every metric that can be
    undefined for the given results (e.g. no cost data) is `None` rather than a
    misleading 0.0."""

    task_count: int = Field(ge=0, description="Distinct case_ids in the input.")
    total_results: int = Field(ge=0, description="Total EvalResults (all repetitions pooled).")
    success_rate: float = Field(
        ge=0, le=1, description="Mean of `succeeded` across all results, all repetitions pooled."
    )
    pass_at_1: float = Field(
        ge=0,
        le=1,
        description=(
            "pass@1, averaged per task then across tasks. Equal to `success_rate` "
            "exactly when every task has the same number of repetitions."
        ),
    )
    k: int = Field(ge=1)
    pass_at_k: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="pass@k (see `k`), averaged over tasks with >= k repetitions. None if no task had >= k.",
    )
    consistency: float = Field(
        ge=0,
        le=1,
        description=(
            "Mean, over tasks with >= 1 repetition, of "
            "max(successes, failures) / repetitions — the fraction of repeated "
            "runs that agree with that task's majority outcome. 1.0 for a task "
            "that always succeeds or always fails; 0.5 for a task split evenly."
        ),
    )
    mean_variance: float = Field(
        ge=0,
        description=(
            "Mean, over tasks with >= 2 repetitions, of the sample variance "
            "(ddof=1) of the binary success indicator across that task's "
            "repetitions. 0.0 if no task had >= 2 repetitions."
        ),
    )
    failure_probability: float = Field(ge=0, le=1, description="1 - success_rate.")
    average_repetitions: float = Field(
        ge=0,
        description="total_results / task_count — the mean number of repetitions actually recorded per task.",
    )
    cost_per_success_usd: float | None = Field(
        default=None,
        ge=0,
        description="Mean cost_usd over successful results that have cost data. None if none do.",
    )
    mean_latency_ms_per_success: float | None = Field(
        default=None,
        ge=0,
        description="Mean latency_ms over successful results. None if there are none.",
    )
    structured_output_validity_rate: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description=(
            "Mean of `succeeded` over results graded with the JSON_SCHEMA grader "
            "specifically — schema *validity*, not field-content correctness "
            "(see FIELD_SCORING for that). None if no JSON_SCHEMA-graded results "
            "are present."
        ),
    )


def compute_metrics(results: Sequence[EvalResult], *, k: int = 1) -> MetricsSummary:
    """Computes `MetricsSummary` from raw results. Raises ValueError on an empty
    sequence — there is no meaningful "metrics over nothing"."""
    if not results:
        raise ValueError("cannot compute metrics over an empty result set")

    by_case: dict[str, list[EvalResult]] = defaultdict(list)
    for result in results:
        by_case[result.case_id].append(result)

    task_count = len(by_case)
    total_results = len(results)
    success_rate = sum(r.succeeded for r in results) / total_results

    pass_at_1_value = _mean_pass_at_k(by_case, 1)
    assert pass_at_1_value is not None  # every case has >= 1 result by construction
    pass_at_k_value = pass_at_1_value if k == 1 else _mean_pass_at_k(by_case, k)

    consistencies = []
    variances = []
    for case_results in by_case.values():
        n = len(case_results)
        successes = sum(r.succeeded for r in case_results)
        failures = n - successes
        consistencies.append(max(successes, failures) / n)
        if n > 1:
            outcomes = [1.0 if r.succeeded else 0.0 for r in case_results]
            outcome_mean = sum(outcomes) / n
            variances.append(sum((o - outcome_mean) ** 2 for o in outcomes) / (n - 1))

    successful = [r for r in results if r.succeeded]
    costs = [r.cost_usd for r in successful if r.cost_usd is not None]
    latencies = [r.latency_ms for r in successful]
    schema_results = [r for r in results if r.grader_kind == GraderKind.JSON_SCHEMA]

    return MetricsSummary(
        task_count=task_count,
        total_results=total_results,
        success_rate=success_rate,
        pass_at_1=pass_at_1_value,
        k=k,
        pass_at_k=pass_at_k_value,
        consistency=mean(consistencies),
        mean_variance=mean(variances) if variances else 0.0,
        failure_probability=1.0 - success_rate,
        average_repetitions=total_results / task_count,
        cost_per_success_usd=(sum(costs) / len(costs)) if costs else None,
        mean_latency_ms_per_success=(sum(latencies) / len(latencies)) if latencies else None,
        structured_output_validity_rate=(
            (sum(r.succeeded for r in schema_results) / len(schema_results))
            if schema_results
            else None
        ),
    )


def _mean_pass_at_k(by_case: dict[str, list[EvalResult]], k: int) -> float | None:
    values = []
    for case_results in by_case.values():
        n = len(case_results)
        if n < k:
            continue
        c = sum(r.succeeded for r in case_results)
        values.append(pass_at_k(n, c, k))
    return mean(values) if values else None


def _percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile over a copy of `values`, sorted ascending.
    `p` in [0, 1]; `values` must be non-empty. Deterministic and dependency-
    free (no numpy) — exact interpolation method matters far less here than
    every consumer agreeing on one, given the sample sizes this repo's
    offline benchmarks actually produce."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0, min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1))
    return ordered[rank]


class ResilienceMetricsSummary(BaseModel):
    """Aggregate metrics over a set of `OutcomeEvent`s — the Phase E
    (Commit 29) counterpart to `MetricsSummary`, one level up: an
    `OutcomeEvent` already summarizes a whole recovery chain (possibly many
    attempts), where `EvalResult`/`MetricsSummary` summarize one attempt.
    Every metric is defined precisely here rather than left to each
    caller's own interpretation (required by the Phase E spec's
    "STATISTICAL DISCIPLINE" section).
    """

    n: int = Field(ge=0, description="Total OutcomeEvents (one per task).")
    task_success_rate: float = Field(ge=0, le=1, description="Mean of `succeeded`.")
    raw_failure_rate: float = Field(
        ge=0,
        le=1,
        description=(
            "Fraction of tasks whose *first* attempt failed — i.e. `attempt_count "
            "> 1` (recovery was invoked at all) OR the single attempt made simply "
            "did not succeed. This is the failure rate a deployment with no "
            "recovery layer at all would have observed on the same task/fault "
            "stream, reconstructed from the fact that recovery is only ever "
            "invoked after a failed attempt."
        ),
    )
    recovery_rate: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description=(
            "Of the raw failures (see `raw_failure_rate`), the fraction that "
            "ended up `succeeded=True` after recovery. None if there were no "
            "raw failures to recover from."
        ),
    )
    retry_rate: float = Field(
        ge=0, le=1, description="Fraction of tasks whose recovery_actions include RETRY_SAME."
    )
    fallback_rate: float = Field(
        ge=0,
        le=1,
        description=(
            "Fraction of tasks whose recovery_actions include FALLBACK_MODEL or FALLBACK_PROVIDER."
        ),
    )
    escalation_rate: float = Field(
        ge=0, le=1, description="Fraction of tasks whose recovery_actions include ESCALATE."
    )
    probe_rate: float = Field(
        ge=0, le=1, description="Fraction of tasks whose recovery_actions include PROBE."
    )
    average_attempts: float = Field(ge=1, description="Mean of `attempt_count`.")
    average_tool_calls: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Mean tool-call count across a companion set of `AgentTrajectory`s, "
            "passed in explicitly via `compute_resilience_metrics`'s "
            "`average_tool_calls` argument (see `mean_tool_call_count`) — "
            "`OutcomeEvent` itself has no tool-call concept, so this is never "
            "derived from `outcomes`. None when not supplied (the non-agent, "
            "single-shot closed-loop comparison)."
        ),
    )
    cost_per_successful_task_usd: float | None = Field(
        default=None,
        ge=0,
        description="Mean total_cost_usd over succeeded tasks with cost data. None if none do.",
    )
    median_latency_ms: float = Field(ge=0, description="Median of total_latency_ms, all tasks.")
    p95_latency_ms: float = Field(
        ge=0, description="95th-percentile (nearest-rank) of total_latency_ms, all tasks."
    )
    unrecovered_failure_distribution: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Counts of `failure_category` (or 'unknown' if unset) among tasks "
            "that ended `succeeded=False` — i.e. failures recovery did *not* fix."
        ),
    )


def mean_tool_call_count(trajectories: Sequence["AgentTrajectory"]) -> float | None:
    """Mean `AgentTrajectory.tool_call_count` (every *attempted* call — see
    that property's docstring) over `trajectories`. None if empty, so
    callers don't need to special-case "no agent trajectories this run"."""
    return mean(t.tool_call_count for t in trajectories) if trajectories else None


def compute_resilience_metrics(
    outcomes: Sequence[OutcomeEvent], *, average_tool_calls: float | None = None
) -> ResilienceMetricsSummary:
    """Computes `ResilienceMetricsSummary` from raw `OutcomeEvent`s. Raises
    ValueError on an empty sequence, matching `compute_metrics`. Pass
    `average_tool_calls=mean_tool_call_count(trajectories)` when summarizing
    an agent suite run alongside its trajectories."""
    from paretoguard.recovery.context import RecoveryAction  # see TYPE_CHECKING note above

    if not outcomes:
        raise ValueError("cannot compute resilience metrics over an empty outcome set")

    n = len(outcomes)
    task_success_rate = sum(o.succeeded for o in outcomes) / n

    raw_failures = [o for o in outcomes if o.attempt_count > 1 or not o.succeeded]
    raw_failure_rate = len(raw_failures) / n
    recovery_rate = (
        (sum(o.succeeded for o in raw_failures) / len(raw_failures)) if raw_failures else None
    )

    def _rate(action: RecoveryAction) -> float:
        return sum(action.value in o.recovery_actions for o in outcomes) / n

    successful_costs = [
        o.total_cost_usd for o in outcomes if o.succeeded and o.total_cost_usd is not None
    ]
    latencies = [o.total_latency_ms for o in outcomes]

    unrecovered = [o for o in outcomes if not o.succeeded]
    failure_distribution = Counter(o.failure_category or "unknown" for o in unrecovered)

    return ResilienceMetricsSummary(
        n=n,
        task_success_rate=task_success_rate,
        raw_failure_rate=raw_failure_rate,
        recovery_rate=recovery_rate,
        retry_rate=_rate(RecoveryAction.RETRY_SAME),
        fallback_rate=sum(
            (RecoveryAction.FALLBACK_MODEL.value in o.recovery_actions)
            or (RecoveryAction.FALLBACK_PROVIDER.value in o.recovery_actions)
            for o in outcomes
        )
        / n,
        escalation_rate=_rate(RecoveryAction.ESCALATE),
        probe_rate=_rate(RecoveryAction.PROBE),
        average_attempts=mean(o.attempt_count for o in outcomes),
        average_tool_calls=average_tool_calls,
        cost_per_successful_task_usd=(
            sum(successful_costs) / len(successful_costs) if successful_costs else None
        ),
        median_latency_ms=_percentile(latencies, 0.5),
        p95_latency_ms=_percentile(latencies, 0.95),
        unrecovered_failure_distribution=dict(failure_distribution),
    )
