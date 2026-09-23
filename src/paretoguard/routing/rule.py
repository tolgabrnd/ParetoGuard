"""RuleRouter: interpretable, hand-authored eligibility + preference rules.

Every exclusion and every preference is attributable to a named rule in the
resulting `RoutingDecision.explanation` — there is no learned or opaque
scoring here (that's Commit 22's LearnedRouter). Task-family preferences are
read from `RoutingRequest.profiles`, which may be measured (real benchmark
history) or simulated (synthetic model profiles); either is fine here as
long as the caller populated it, but the source is never asserted by this
router.
"""

from paretoguard.core.models import ModelSpec, RoutingDecision
from paretoguard.routing.protocol import Router
from paretoguard.routing.types import (
    NoEligibleCandidateError,
    RoutingRequest,
    candidate_key,
    filter_eligible,
)


class RuleRouter(Router):
    """Filters candidates by hard eligibility and soft cost/latency/success
    constraints, then prefers the candidate with the best measured/simulated
    success for the request's task family — falling back, in order, to
    overall health success rate, then to configured candidate order, when no
    task-family-specific signal exists.
    """

    def __init__(self) -> None:
        self.name = "rule"

    def route(self, routing_request: RoutingRequest) -> RoutingDecision:
        eligible, excluded = filter_eligible(routing_request)
        eligible, soft_excluded = self._apply_soft_constraints(routing_request, eligible)
        excluded.update(soft_excluded)

        if not eligible:
            raise NoEligibleCandidateError(
                f"RuleRouter: no candidate satisfies constraints; excluded={excluded}"
            )

        scored = self._score_candidates(routing_request, eligible)
        ranked = sorted(eligible, key=lambda c: scored[c.name][0], reverse=True)
        selected = ranked[0]
        _selected_score, rule_used = scored[selected.name]
        profile = routing_request.profiles.get(candidate_key(selected.provider, selected.name))

        return RoutingDecision(
            selected_model=selected.name,
            candidate_scores={name: score for name, (score, _rule) in scored.items()},
            predicted_success=profile.predicted_success if profile else None,
            expected_cost_usd=profile.mean_cost_usd if profile else None,
            expected_latency_ms=profile.mean_latency_ms if profile else None,
            explanation=(
                f"RuleRouter: selected '{selected.name}' via rule [{rule_used}] "
                f"among {len(eligible)} eligible candidate(s) "
                f"(task_family={routing_request.features.task_family!r})."
            ),
            fallback_order=[c.name for c in ranked[1:]],
            excluded_candidates=excluded,
        )

    def _apply_soft_constraints(
        self, routing_request: RoutingRequest, candidates: list[ModelSpec]
    ) -> tuple[list[ModelSpec], dict[str, str]]:
        """Excludes candidates whose measured/simulated profile violates a
        soft (estimate-based) constraint. A candidate with no profile data
        can't be checked against these and is kept — soft constraints only
        exclude on positive evidence of violation, never on absence of data.
        """
        constraints = routing_request.constraints
        kept: list[ModelSpec] = []
        excluded: dict[str, str] = {}

        for candidate in candidates:
            profile = routing_request.profiles.get(
                candidate_key(candidate.provider, candidate.name)
            )
            if profile is None:
                kept.append(candidate)
                continue

            if (
                constraints.max_cost_usd is not None
                and profile.mean_cost_usd is not None
                and profile.mean_cost_usd > constraints.max_cost_usd
            ):
                excluded[candidate.name] = (
                    f"expected cost ${profile.mean_cost_usd:.4f} exceeds "
                    f"max_cost_usd ${constraints.max_cost_usd:.4f}"
                )
                continue
            if (
                constraints.max_latency_ms is not None
                and profile.mean_latency_ms is not None
                and profile.mean_latency_ms > constraints.max_latency_ms
            ):
                excluded[candidate.name] = (
                    f"expected latency {profile.mean_latency_ms:.0f}ms exceeds "
                    f"max_latency_ms {constraints.max_latency_ms:.0f}ms"
                )
                continue
            if (
                constraints.min_predicted_success is not None
                and profile.predicted_success is not None
                and profile.predicted_success < constraints.min_predicted_success
            ):
                excluded[candidate.name] = (
                    f"predicted success {profile.predicted_success:.3f} below "
                    f"min_predicted_success {constraints.min_predicted_success:.3f}"
                )
                continue
            kept.append(candidate)

        return kept, excluded

    def _score_candidates(
        self, routing_request: RoutingRequest, candidates: list[ModelSpec]
    ) -> dict[str, tuple[float, str]]:
        """Returns {model_name: (score, rule_name)}. Rule precedence, most to
        least specific: task-family-scoped measured success, then any
        measured success, then rolling health success rate, then a fixed
        deterministic default (configured order) when nothing is known."""
        task_family = routing_request.features.task_family
        scored: dict[str, tuple[float, str]] = {}

        for rank, candidate in enumerate(candidates):
            key = candidate_key(candidate.provider, candidate.name)
            profile = routing_request.profiles.get(key)
            health = routing_request.health.get(key)
            default_score = -float(rank) / max(1, len(candidates))

            if profile is not None and profile.predicted_success is not None:
                if task_family is not None and profile.task_family == task_family:
                    scored[candidate.name] = (
                        profile.predicted_success,
                        "task-family-scoped measured/simulated success",
                    )
                    continue
                if profile.task_family is None:
                    scored[candidate.name] = (
                        profile.predicted_success,
                        "overall measured/simulated success (no task-family profile)",
                    )
                    continue
            if health is not None and health.success_rate is not None:
                scored[candidate.name] = (
                    health.success_rate,
                    "rolling health success rate (no profile data)",
                )
                continue
            scored[candidate.name] = (
                default_score,
                "no measured signal; defaulting to configured candidate order",
            )

        return scored
