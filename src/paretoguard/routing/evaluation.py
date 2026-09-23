"""Offline router evaluation against the exhaustive training matrix.

Because `MatrixRunner` already recorded every candidate's outcome for every
task (see `paretoguard.evals.matrix`), a router's counterfactual performance
can be computed without re-executing anything: call `router.route()` to get
its selection for a task, then look up that exact (task, candidate)'s
already-recorded outcome. `evaluate_router_offline` is that closed loop.

The oracle (`oracle_upper_bound`) is **NON-DEPLOYABLE**: it picks whichever
candidate actually succeeded on each task, using outcomes no real router has
at routing time. It exists only to compute `regret_vs_oracle`, never as a
routing strategy a system could run — see CLAUDE.md's "never fabricate...
capabilities" and this module's use of it strictly as an analysis upper
bound, always labeled as such in every returned/printed value.
"""

from collections import Counter
from dataclasses import dataclass
from statistics import mean
from typing import cast

import polars as pl

from paretoguard.core.features import TaskFeatures
from paretoguard.core.models import InferenceRequest, Message, ModelSpec, Role
from paretoguard.routing.dataset import FEATURE_COLUMNS
from paretoguard.routing.protocol import Router
from paretoguard.routing.types import CandidateProfile, RoutingRequest, candidate_key


def build_profiles_from_train_split(df: pl.DataFrame) -> dict[str, CandidateProfile]:
    """Aggregates mean success/cost/latency per (provider, model) from
    `df.filter(split == "train")` only — the sole basis a real system
    would have for a candidate's expected performance before routing a
    test-split task. Passing this to RuleRouter/ParetoRouter/LearnedRouter
    in `evaluate_router_offline` keeps their cost/latency inputs
    leakage-free; only `oracle_upper_bound` is allowed to look at test
    outcomes directly, because it is explicitly non-deployable.
    """
    train_df = df.filter(pl.col("split") == "train")
    if train_df.height == 0:
        raise ValueError("no training rows (split == 'train') to build profiles from")

    stats = train_df.group_by(["provider", "model"]).agg(
        pl.col("succeeded").mean().alias("success_rate"),
        pl.col("cost_usd").mean().alias("mean_cost_usd"),
        pl.col("latency_ms").mean().alias("mean_latency_ms"),
    )
    return {
        candidate_key(row["provider"], row["model"]): CandidateProfile(
            predicted_success=row["success_rate"],
            mean_cost_usd=row["mean_cost_usd"],
            mean_latency_ms=row["mean_latency_ms"],
            simulated=True,
        )
        for row in stats.iter_rows(named=True)
    }


@dataclass(frozen=True)
class RoutingEvalSummary:
    router_name: str
    n_tasks: int
    success_rate: float
    mean_cost_usd: float | None
    mean_latency_ms: float
    selection_distribution: dict[str, int]
    regret_vs_oracle: float
    """Mean, over tasks, of (oracle_success_indicator - routed_success_indicator).
    0.0 means the router matched the oracle's outcome on every task; positive
    means it did worse. Never negative (the oracle is an upper bound)."""


def _task_frame(eval_df: pl.DataFrame) -> pl.DataFrame:
    return eval_df.select(["task_id", *FEATURE_COLUMNS]).unique(subset=["task_id"])


def evaluate_router_offline(
    router: Router,
    df: pl.DataFrame,
    candidates: list[ModelSpec],
    *,
    split: str = "test",
    profiles: dict[str, CandidateProfile] | None = None,
) -> RoutingEvalSummary:
    """Routes every distinct task in `df.filter(split == split)` through
    `router` and scores the result against that task's already-recorded
    outcome for the selected candidate — never executes anything live.
    `profiles` should come from `build_profiles_from_train_split`, not the
    same split being evaluated (that would leak test outcomes into the
    router's own inputs).
    """
    eval_df = df.filter(pl.col("split") == split)
    if eval_df.height == 0:
        raise ValueError(f"no rows for split={split!r}")

    outcomes = {(r["task_id"], r["model"]): r for r in eval_df.iter_rows(named=True)}
    tasks = _task_frame(eval_df)
    profiles = profiles or {}

    selections: list[str] = []
    successes: list[bool] = []
    costs: list[float] = []
    latencies: list[float] = []
    oracle_successes: list[bool] = []

    for row in tasks.iter_rows(named=True):
        features = TaskFeatures(**{col: row[col] for col in FEATURE_COLUMNS})
        request = InferenceRequest(
            provider="offline-eval",
            model="n/a",
            messages=[Message(role=Role.USER, content="offline evaluation probe")],
            metadata={"task_family": features.task_family} if features.task_family else {},
        )
        routing_request = RoutingRequest(
            request=request, features=features, candidates=candidates, profiles=profiles
        )
        decision = router.route(routing_request)
        selections.append(decision.selected_model)

        outcome_key = (row["task_id"], decision.selected_model)
        if outcome_key not in outcomes:
            raise ValueError(
                f"router {router.name!r} selected {decision.selected_model!r} for task "
                f"{row['task_id']!r}, but the matrix has no recorded outcome for that "
                f"(task, candidate) pair — was it run with the same candidate set?"
            )
        outcome = outcomes[outcome_key]
        successes.append(bool(outcome["succeeded"]))
        if outcome["cost_usd"] is not None:
            costs.append(float(outcome["cost_usd"]))
        latencies.append(float(outcome["latency_ms"]))

        task_candidate_outcomes = [
            bool(outcomes[(row["task_id"], c.name)]["succeeded"])
            for c in candidates
            if (row["task_id"], c.name) in outcomes
        ]
        oracle_successes.append(any(task_candidate_outcomes))

    regret = mean(int(o) - int(s) for o, s in zip(oracle_successes, successes, strict=True))

    return RoutingEvalSummary(
        router_name=router.name,
        n_tasks=len(successes),
        success_rate=mean(successes),
        mean_cost_usd=mean(costs) if costs else None,
        mean_latency_ms=mean(latencies),
        selection_distribution=dict(Counter(selections)),
        regret_vs_oracle=regret,
    )


def oracle_upper_bound(
    df: pl.DataFrame, candidates: list[ModelSpec], *, split: str = "test"
) -> float:
    """**NON-DEPLOYABLE.** Mean, over distinct tasks in `split`, of whether
    *any* candidate succeeded — the best any routing strategy (or Router
    protocol implementation, in principle) could ever achieve on this data,
    computed only for `regret_vs_oracle`/reporting. Never route with this.
    """
    eval_df = df.filter(pl.col("split") == split)
    if eval_df.height == 0:
        raise ValueError(f"no rows for split={split!r}")
    candidate_names = {c.name for c in candidates}
    per_task = (
        eval_df.filter(pl.col("model").is_in(candidate_names))
        .group_by("task_id")
        .agg(pl.col("succeeded").max().alias("any_succeeded"))
    )
    return cast(float, per_task["any_succeeded"].mean())


def fixed_model_stats(df: pl.DataFrame, *, split: str = "test") -> pl.DataFrame:
    """Per-(provider, model) success rate and mean cost/latency over
    `split` — the "fixed model" baselines: deploy exactly one model,
    always, for every task."""
    eval_df = df.filter(pl.col("split") == split)
    if eval_df.height == 0:
        raise ValueError(f"no rows for split={split!r}")
    return eval_df.group_by(["provider", "model"]).agg(
        pl.col("succeeded").mean().alias("success_rate"),
        pl.col("cost_usd").mean().alias("mean_cost_usd"),
        pl.col("latency_ms").mean().alias("mean_latency_ms"),
    )


def best_fixed_model(df: pl.DataFrame, *, split: str = "test") -> str:
    stats = fixed_model_stats(df, split=split)
    return str(stats.sort("success_rate", descending=True).row(0, named=True)["model"])


def cheapest_fixed_model(df: pl.DataFrame, *, split: str = "test") -> str:
    stats = fixed_model_stats(df, split=split).filter(pl.col("mean_cost_usd").is_not_null())
    if stats.height == 0:
        raise ValueError("no fixed model has cost data")
    return str(stats.sort("mean_cost_usd").row(0, named=True)["model"])


def cost_at_target_success(
    df: pl.DataFrame, *, target_success: float, split: str = "test"
) -> float | None:
    """Minimum mean cost among fixed models reaching `target_success`;
    `None` if no single fixed model reaches it (an adaptive router may
    still reach it — this is a fixed-model baseline only)."""
    stats = fixed_model_stats(df, split=split)
    eligible = stats.filter(
        (pl.col("success_rate") >= target_success) & pl.col("mean_cost_usd").is_not_null()
    )
    if eligible.height == 0:
        return None
    return cast(float, eligible["mean_cost_usd"].min())


def success_at_fixed_budget(
    df: pl.DataFrame, *, budget_usd: float, split: str = "test"
) -> float | None:
    """Maximum success rate among fixed models within `budget_usd` mean
    cost; `None` if every fixed model exceeds the budget."""
    stats = fixed_model_stats(df, split=split)
    eligible = stats.filter(
        pl.col("mean_cost_usd").is_not_null() & (pl.col("mean_cost_usd") <= budget_usd)
    )
    if eligible.height == 0:
        return None
    return cast(float, eligible["success_rate"].max())
