"""Recovery-layer escalation: when a fallback should specifically target a
*stronger* eligible candidate rather than merely the next untried one.

Distinct from `paretoguard.routing.escalation.EscalationRouter`, which
escalates a *routing decision* before execution, based on predicted
confidence. This module escalates *after* an execution attempt has already
failed with a quality problem — see `recovery.policy`'s module docstring
for how the two connect across the full lifecycle (Commit 28).
"""

from paretoguard.core.models import FailureCategory

QUALITY_FAILURE_CATEGORIES = frozenset(
    {
        FailureCategory.SCHEMA_FAILURE,
        FailureCategory.INVALID_OUTPUT,
        FailureCategory.REASONING_FAILURE,
    }
)
"""Failures where the candidate *ran* and produced a response, but the
response itself was wrong or invalid — as opposed to an availability
failure (timeout/rate-limit/provider-down), where the same-tier next
candidate is just as likely to work. A quality failure is the specific
signal that motivates trying a *stronger* model, not merely a *different*
one at the same tier."""


def should_escalate(failure_category: FailureCategory | None) -> bool:
    return failure_category in QUALITY_FAILURE_CATEGORIES
