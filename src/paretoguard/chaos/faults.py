"""Typed fault model. Every fault carries real, typed metadata — never an
anonymous string — so injected faults are inspectable and attributable
after the fact (see `FaultEvent`, persisted via `TraceEventType.FAULT_INJECTED`).
"""

from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class FaultCategory(StrEnum):
    PROVIDER = "provider"
    TOOL = "tool"
    CONTEXT = "context"


class ProviderFaultKind(StrEnum):
    """Faults applied to a provider call — see `chaos.injector.FaultyProvider`."""

    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    LATENCY_SPIKE = "latency_spike"
    MALFORMED_STRUCTURED_OUTPUT = "malformed_structured_output"
    TRUNCATED_OUTPUT = "truncated_output"


class ToolFaultKind(StrEnum):
    """Faults applied to a tool call — see `chaos.injector.FaultyTool`."""

    EXCEPTION = "exception"
    TIMEOUT = "timeout"
    STALE_RESULT = "stale_result"
    MISSING_FIELD = "missing_field"
    SCHEMA_MISMATCH = "schema_mismatch"
    PARTIAL_RESULT = "partial_result"
    TEMPORARY_UNAVAILABLE = "temporary_unavailable"


class ContextFaultKind(StrEnum):
    """Benchmark-construction-time faults applied to task content — see
    `chaos.scenarios.apply_context_fault`. Never applied at dispatch time,
    and never allowed to make the benchmark's own ground truth ambiguous:
    the authoritative source for grading is always the benchmark's
    `ground_truth`/`ExpectedTrajectory`, regardless of which record a
    contradictory-stale-record fault labels as "stale" within the prompt.
    """

    DISTRACTOR_INJECTION = "distractor_injection"
    MISSING_EVIDENCE = "missing_evidence"
    STALE_RECORD = "stale_record"
    CONTRADICTORY_STALE_RECORD = "contradictory_stale_record"


class ExpectedRecoverability(StrEnum):
    """A fault author's own expectation of whether this fault *should* be
    recoverable by a well-behaved recovery policy — recorded so a
    resilience benchmark (Commit 29) can distinguish "the system failed to
    recover from something recoverable" (a real finding) from "the fault
    was deliberately unrecoverable" (expected)."""

    RECOVERABLE = "recoverable"
    UNRECOVERABLE = "unrecoverable"
    UNKNOWN = "unknown"


class FaultEvent(BaseModel):
    """One instance of an injected fault, fully attributable."""

    fault_event_id: UUID = Field(default_factory=uuid4)
    fault_id: str
    """The stable id of the `FaultPolicy` that produced this event (never a
    random id) — used to cross-reference `AgentStep.fault_id`."""
    category: FaultCategory
    kind: str
    """The specific `ProviderFaultKind`/`ToolFaultKind`/`ContextFaultKind`
    value — kept as `str` here (rather than a union type) so `FaultEvent`
    doesn't need to know which category's enum applies."""
    target_provider: str | None = None
    target_model: str | None = None
    target_tool: str | None = None
    injection_point: str
    """Where in the pipeline this fired, e.g. `"provider_call"`,
    `"tool_execution"`."""
    seed: int
    step: int | None = None
    """The task-local step index this fired at, if applicable (agent
    faults) — `None` for a single-shot (non-agent) request."""
    parameters: dict[str, Any] = Field(default_factory=dict)
    expected_recoverability: ExpectedRecoverability = ExpectedRecoverability.UNKNOWN
