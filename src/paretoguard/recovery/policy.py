"""The recovery engine: ties retry, fallback, circuit-breaker, and
escalation policy together into one decision per failed attempt.

**Runtime retry vs. recovery** — the full distinction (see also
`recovery.retry`'s module docstring): by the time `RecoveryPolicy.decide` is
called, a `paretoguard.runtime.Runtime.run()` call has already exhausted its
own transient-transport retry budget and returned a final `InferenceResponse`
with `succeeded=False`. This module never retries *within* that call — it
only decides what a higher-level control plane (Commit 28) should do next:
try the same candidate again (a fresh `Runtime.run()`, coarser budget),
switch to a different candidate, escalate to a stronger one, probe a
recovering circuit, or stop (`ABSTAIN`/`FAIL`).

**Health vs. circuit-breaker vs. this policy**: `RecoveryPolicy` never reads
`HealthTracker` or mutates `CircuitBreaker` state directly — it only reads
the immutable snapshots handed to it via `RecoveryContext` (health) and asks
a `CircuitBreaker` instance for read-only state (`state_for`, never the
state-mutating `allow_request`, which is reserved for the actual dispatch
point once a decision has been made — see `recovery.fallback`'s use of the
same read-only check). This keeps `HealthTracker`/`CircuitBreaker` reusable
by other consumers without this policy being tightly coupled to either.
"""

from dataclasses import dataclass, field

from paretoguard.core.models import ModelSpec
from paretoguard.recovery.circuit_breaker import CircuitBreaker, CircuitState
from paretoguard.recovery.context import RecoveryAction, RecoveryContext, RecoveryDecision
from paretoguard.recovery.escalation import should_escalate
from paretoguard.recovery.fallback import FallbackPolicy, select_fallback
from paretoguard.recovery.retry import RecoveryRetryPolicy


@dataclass(frozen=True)
class RecoveryPolicy:
    retry_policy: RecoveryRetryPolicy = field(default_factory=RecoveryRetryPolicy)
    fallback_policy: FallbackPolicy = field(default_factory=FallbackPolicy)
    circuit_breaker: CircuitBreaker | None = None
    allow_escalation: bool = True

    def decide(self, context: RecoveryContext, candidates: list[ModelSpec]) -> RecoveryDecision:
        if self._budget_exhausted(context):
            return RecoveryDecision(RecoveryAction.ABSTAIN, reason="budget exhausted")

        current_key = context.current_key()
        permanent = context.failure_category is not None and not self.retry_policy.is_retryable(
            context.failure_category
        )
        circuit_open = (
            self.circuit_breaker is not None
            and current_key is not None
            and self.circuit_breaker.state_for(current_key) == CircuitState.OPEN
        )

        if not permanent and not circuit_open:
            same_attempts = context.attempts_on(current_key)
            if self.retry_policy.has_retry_budget(same_attempts):
                return RecoveryDecision(
                    RecoveryAction.RETRY_SAME,
                    target_provider=context.current_provider,
                    target_model=context.current_model,
                    reason=(
                        f"transient failure ({context.failure_category}) within "
                        "recovery retry budget"
                    ),
                )

        selection = select_fallback(
            context, candidates, self.fallback_policy, circuit_breaker=self.circuit_breaker
        )
        if selection.candidate is None:
            action = RecoveryAction.FAIL if permanent else RecoveryAction.ABSTAIN
            return RecoveryDecision(
                action, reason=f"no eligible fallback remains (excluded={selection.excluded})"
            )

        if selection.is_probe:
            return RecoveryDecision(
                RecoveryAction.PROBE,
                target_provider=selection.candidate.provider,
                target_model=selection.candidate.name,
                reason="target circuit is half-open; probing for recovery",
            )

        if self.allow_escalation and should_escalate(context.failure_category):
            return RecoveryDecision(
                RecoveryAction.ESCALATE,
                target_provider=selection.candidate.provider,
                target_model=selection.candidate.name,
                reason=f"quality failure ({context.failure_category}); escalating",
            )

        action = (
            RecoveryAction.FALLBACK_PROVIDER
            if selection.candidate.provider != context.current_provider
            else RecoveryAction.FALLBACK_MODEL
        )
        cause = "permanent failure" if permanent else "exhausted same-candidate retry budget"
        return RecoveryDecision(
            action,
            target_provider=selection.candidate.provider,
            target_model=selection.candidate.name,
            reason=f"falling back after {cause} ({context.failure_category})",
        )

    @staticmethod
    def _budget_exhausted(context: RecoveryContext) -> bool:
        if context.remaining_call_budget is not None and context.remaining_call_budget <= 0:
            return True
        if context.remaining_usd_budget is not None and context.remaining_usd_budget <= 0:
            return True
        return (
            context.remaining_latency_budget_ms is not None
            and context.remaining_latency_budget_ms <= 0
        )
