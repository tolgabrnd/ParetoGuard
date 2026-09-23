"""Confidence-aware escalation and abstention: wraps an inner `Router`
(typically `LearnedRouter`/`TorchRouter`, but any `Router` that populates
`predicted_success` works) so a low-confidence initial pick escalates to a
stronger eligible candidate, falls back to a configured safe router, or
explicitly abstains — never silently returning a decision that violates the
caller's constraints or that the router itself doesn't trust.

"Confidence" here means the selected candidate's `predicted_success` (or
`RoutingDecision.confidence` when the inner router sets it explicitly,
preferred when present) — how likely the router believes its pick is to
succeed, not a second-order uncertainty-over-the-prediction measure. That
matches how every other router in this package already populates
`RoutingDecision.predicted_success`.
"""

from dataclasses import dataclass, replace
from uuid import UUID

from paretoguard.core.models import RoutingDecision
from paretoguard.routing.protocol import Router
from paretoguard.routing.types import NoEligibleCandidateError, RoutingRequest


class NoConfidentRouteError(RuntimeError):
    """Raised when escalation is exhausted, no fallback router is
    configured (or the fallback also fails), and confidence is still below
    threshold — explicit abstention rather than silently returning a
    decision the router itself doesn't trust. See CLAUDE.md: "never
    silently violate user constraints."""


@dataclass(frozen=True)
class EscalationPolicy:
    confidence_threshold: float = 0.7
    """Below this predicted_success/confidence, the initial pick is not
    trusted and escalation is attempted."""
    max_escalations: int = 1
    """How many additional candidates to try (each excluding every
    previously-tried candidate) before falling back or abstaining."""


@dataclass(frozen=True)
class EscalationRecord:
    """One routing call's full audit trail. `actual_cost_usd`/`succeeded`
    start `None` — a `Router` only ever produces a decision, never executes
    it (see docs/ARCHITECTURE.md), so the real outcome can only be known
    after the fact; call `EscalationRouter.record_outcome` once it is."""

    request_id: UUID
    initial_selection: str
    initial_confidence: float | None
    final_selection: str
    escalation_reason: str | None
    """None if the initial pick already met the confidence threshold."""
    used_fallback: bool
    expected_cost_usd: float | None
    actual_cost_usd: float | None = None
    succeeded: bool | None = None


class EscalationRouter(Router):
    """See module docstring. Stateful (like `ReliabilityAwareRouter`): it
    keeps an `EscalationRecord` per routed request, keyed by
    `request_id`, so outcomes can be attached later via `record_outcome`.
    """

    def __init__(
        self,
        inner: Router,
        *,
        policy: EscalationPolicy | None = None,
        fallback_router: Router | None = None,
    ) -> None:
        self.name = "escalation"
        self._inner = inner
        self._policy = policy or EscalationPolicy()
        self._fallback_router = fallback_router
        self._records: dict[UUID, EscalationRecord] = {}

    def route(self, routing_request: RoutingRequest) -> RoutingDecision:
        request_id = routing_request.request.request_id
        initial = self._inner.route(routing_request)
        confidence = self._confidence_of(initial)

        if confidence is None or confidence >= self._policy.confidence_threshold:
            self._records[request_id] = EscalationRecord(
                request_id=request_id,
                initial_selection=initial.selected_model,
                initial_confidence=confidence,
                final_selection=initial.selected_model,
                escalation_reason=None,
                used_fallback=False,
                expected_cost_usd=initial.expected_cost_usd,
            )
            return initial

        best = initial
        best_confidence = confidence
        tried = {initial.selected_model}
        escalation_steps = [
            f"initial '{initial.selected_model}' confidence={confidence:.3f} "
            f"< threshold {self._policy.confidence_threshold:.3f}"
        ]

        for _ in range(self._policy.max_escalations):
            constraints = routing_request.constraints.model_copy(
                update={"excluded_models": [*routing_request.constraints.excluded_models, *tried]}
            )
            probe_request = replace(routing_request, constraints=constraints)
            try:
                candidate = self._inner.route(probe_request)
            except NoEligibleCandidateError:
                escalation_steps.append("no further eligible candidate to escalate to")
                break

            tried.add(candidate.selected_model)
            candidate_confidence = self._confidence_of(candidate)
            escalation_steps.append(
                f"escalated to '{candidate.selected_model}' confidence={candidate_confidence!r}"
            )
            if candidate_confidence is not None and (
                best_confidence is None or candidate_confidence > best_confidence
            ):
                best, best_confidence = candidate, candidate_confidence
            if best_confidence is not None and best_confidence >= self._policy.confidence_threshold:
                break

        reason = "; ".join(escalation_steps)

        if best_confidence is not None and best_confidence >= self._policy.confidence_threshold:
            decision = best.model_copy(
                update={
                    "confidence": best_confidence,
                    "explanation": f"EscalationRouter: {reason}. {best.explanation}",
                }
            )
            self._records[request_id] = EscalationRecord(
                request_id=request_id,
                initial_selection=initial.selected_model,
                initial_confidence=confidence,
                final_selection=decision.selected_model,
                escalation_reason=reason,
                used_fallback=False,
                expected_cost_usd=decision.expected_cost_usd,
            )
            return decision

        if self._fallback_router is not None:
            try:
                fallback = self._fallback_router.route(routing_request)
            except NoEligibleCandidateError:
                fallback = None
            if fallback is not None:
                decision = fallback.model_copy(
                    update={
                        "explanation": (
                            f"EscalationRouter: {reason}; confidence threshold not reached "
                            f"after escalation, deferred to fallback router "
                            f"'{self._fallback_router.name}'. {fallback.explanation}"
                        )
                    }
                )
                self._records[request_id] = EscalationRecord(
                    request_id=request_id,
                    initial_selection=initial.selected_model,
                    initial_confidence=confidence,
                    final_selection=decision.selected_model,
                    escalation_reason=reason,
                    used_fallback=True,
                    expected_cost_usd=decision.expected_cost_usd,
                )
                return decision

        raise NoConfidentRouteError(
            f"EscalationRouter: no candidate reached confidence_threshold="
            f"{self._policy.confidence_threshold} after {self._policy.max_escalations} "
            f"escalation(s), and no fallback router was configured (or it also failed). "
            f"{reason}"
        )

    def record_outcome(
        self, request_id: UUID, *, succeeded: bool, actual_cost_usd: float | None = None
    ) -> None:
        """Attaches the real outcome to a previously routed request's
        `EscalationRecord`. A `Router` never executes anything itself, so
        this must be called by whatever did — e.g. after `Runtime.run()`."""
        record = self._records.get(request_id)
        if record is None:
            raise KeyError(f"no EscalationRecord for request_id={request_id}")
        self._records[request_id] = replace(
            record, succeeded=succeeded, actual_cost_usd=actual_cost_usd
        )

    def get_record(self, request_id: UUID) -> EscalationRecord | None:
        return self._records.get(request_id)

    def escalation_log(self) -> list[EscalationRecord]:
        return list(self._records.values())

    @staticmethod
    def _confidence_of(decision: RoutingDecision) -> float | None:
        return (
            decision.confidence if decision.confidence is not None else decision.predicted_success
        )
