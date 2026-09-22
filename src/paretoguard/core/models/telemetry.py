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
