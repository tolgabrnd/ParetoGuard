"""Retry, fallback, circuit-breaker, and escalation policies — the
system-level response to a *failed* execution, as distinct from
`paretoguard.runtime.Runtime`'s transient-transport retry within one
execution. See `recovery.policy`'s module docstring for the exact boundary.
"""

from paretoguard.recovery.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitSnapshot,
    CircuitState,
)
from paretoguard.recovery.context import (
    AttemptRecord,
    RecoveryAction,
    RecoveryContext,
    RecoveryDecision,
)
from paretoguard.recovery.escalation import QUALITY_FAILURE_CATEGORIES, should_escalate
from paretoguard.recovery.fallback import FallbackPolicy, FallbackSelection, select_fallback
from paretoguard.recovery.policy import RecoveryPolicy
from paretoguard.recovery.retry import RecoveryRetryPolicy

__all__ = [
    "QUALITY_FAILURE_CATEGORIES",
    "AttemptRecord",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitSnapshot",
    "CircuitState",
    "FallbackPolicy",
    "FallbackSelection",
    "RecoveryAction",
    "RecoveryContext",
    "RecoveryDecision",
    "RecoveryPolicy",
    "RecoveryRetryPolicy",
    "select_fallback",
    "should_escalate",
]
