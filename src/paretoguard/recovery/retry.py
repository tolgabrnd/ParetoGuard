"""The recovery layer's own retry classification — deliberately not the
same thing as `paretoguard.runtime.RetryPolicy`.

**The boundary** (see `paretoguard.recovery`'s package docstring for the
full picture): `runtime.RetryPolicy` retries the *same* request against the
*same* provider/model within a single `Runtime.run()` call, for genuinely
transient transport failures (a dropped connection, a momentary 5xx) — fast,
fine-grained backoff, no awareness of budgets, other candidates, or circuit
state. `RecoveryRetryPolicy` here governs whether the *higher-level*
control plane (Commit 28) should issue a brand-new execution attempt at the
same candidate *after* `Runtime.run()` has already returned a final failed
`InferenceResponse` — a coarser, budget-aware decision that also considers
whether to retry at all versus falling back to a different candidate.
Neither one duplicates the other's job; a request that exhausts
`runtime.RetryPolicy` and still fails is exactly the input `RecoveryPolicy`
(`recovery.policy`) classifies next.
"""

from dataclasses import dataclass, field

from paretoguard.core.models import FailureCategory

_DEFAULT_RETRYABLE = frozenset(
    {
        FailureCategory.TIMEOUT,
        FailureCategory.RATE_LIMIT,
        FailureCategory.PROVIDER_FAILURE,
        FailureCategory.TRANSPORT_FAILURE,
    }
)
"""Categories worth trying again at all (at either layer). Schema/argument/
tool-selection/budget/step-limit/unrecoverable-state failures are never
retried — re-sending the identical request cannot change a validation or
configuration outcome, and retrying would just be a retry storm against a
guaranteed-to-fail cause."""


@dataclass(frozen=True)
class RecoveryRetryPolicy:
    max_same_candidate_attempts: int = 1
    """Additional attempts (beyond the one that already failed) allowed
    against the *same* candidate before recovery must fall back instead.
    `0` means "never retry the same candidate at the recovery level, always
    fall back on any failure" — a valid, conservative configuration."""
    retryable_categories: frozenset[FailureCategory] = field(
        default_factory=lambda: _DEFAULT_RETRYABLE
    )

    def __post_init__(self) -> None:
        if self.max_same_candidate_attempts < 0:
            raise ValueError("max_same_candidate_attempts must be >= 0")

    def is_retryable(self, category: FailureCategory | None) -> bool:
        """`None` (no failure, or a non-provider failure with no category)
        is treated as not retryable — there's nothing transient to retry."""
        return category is not None and category in self.retryable_categories

    def has_retry_budget(self, attempts_on_current_candidate: int) -> bool:
        return attempts_on_current_candidate < self.max_same_candidate_attempts
