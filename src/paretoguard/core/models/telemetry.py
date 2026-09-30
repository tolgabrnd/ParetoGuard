"""Trace event types emitted throughout the request lifecycle."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class TraceEventType(StrEnum):
    """Kinds of events recorded to the trace/telemetry stream.

    Extended in later phases (chaos, recovery, evals) as those subsystems land;
    kept as an explicit enum rather than free-form strings so consumers can
    exhaustively handle known event kinds.
    """

    REQUEST_STARTED = "request_started"
    ROUTING_DECISION = "routing_decision"
    PROVIDER_CALL = "provider_call"
    FAULT_INJECTED = "fault_injected"
    RECOVERY_ACTION = "recovery_action"
    EVAL_RESULT = "eval_result"
    REQUEST_COMPLETED = "request_completed"


class TraceEvent(BaseModel):
    """One normalized, timestamped event in a request/run's trace."""

    event_id: UUID = Field(default_factory=uuid4)
    run_id: str | None = None
    request_id: UUID | None = None
    event_type: TraceEventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: dict[str, Any] = Field(default_factory=dict)


class OutcomeEvent(BaseModel):
    """One full closed-loop execution outcome: the whole
    route -> execute -> validate -> (recover)* -> terminal chain for one
    task, as one record — the summary `TraceEvent`'s per-step granularity
    can't express on its own (see `paretoguard.routing.execution
    .ClosedLoopExecutor`, which produces exactly one of these per task).
    """

    outcome_id: UUID = Field(default_factory=uuid4)
    run_id: str | None = None
    task_id: str | None = None
    initial_routing_decision_id: UUID | None = None
    """The `request_id` of the *first* `RoutingDecision` in this chain —
    later attempts (recovery-driven) reuse the same task but a fresh
    request id per attempt; this is the one to join against
    `routing_decisions` for "what did the router originally decide"."""
    final_provider: str
    final_model: str
    attempt_count: int = Field(ge=1)
    succeeded: bool
    failure_category: str | None = None
    total_cost_usd: float | None = Field(default=None, ge=0)
    total_latency_ms: float = Field(ge=0)
    recovery_actions: list[str] = Field(default_factory=list)
    """The sequence of `RecoveryAction` values taken, in order — empty if
    the first attempt succeeded and recovery was never consulted."""
    terminal: bool = True
    """Always `True` in this repo today: `ClosedLoopExecutor` always runs a
    chain to completion. Reserved for a future streaming/paused execution
    model where an outcome could be reported mid-chain."""
