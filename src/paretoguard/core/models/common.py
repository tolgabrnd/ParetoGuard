"""Shared enums and small value types used across the domain model."""

from enum import StrEnum

from pydantic import BaseModel


class ProviderKind(StrEnum):
    """Which adapter implementation a `ProviderSpec` is served by."""

    MOCK = "mock"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class Role(StrEnum):
    """Chat message role, normalized across providers."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(StrEnum):
    """Normalized reason a provider stopped generating."""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


class FailureCategory(StrEnum):
    """Failure taxonomy shared by telemetry, recovery, and evaluation.

    Defined once in core so every subsystem classifies failures the same way
    instead of inventing ad-hoc string reasons per module.
    """

    PROVIDER_FAILURE = "provider_failure"
    TRANSPORT_FAILURE = "transport_failure"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    INVALID_OUTPUT = "invalid_output"
    SCHEMA_FAILURE = "schema_failure"
    REASONING_FAILURE = "reasoning_failure"
    TOOL_SELECTION_FAILURE = "tool_selection_failure"
    TOOL_ARGUMENT_FAILURE = "tool_argument_failure"
    TOOL_EXECUTION_FAILURE = "tool_execution_failure"
    INCOMPLETE_TRAJECTORY = "incomplete_trajectory"
    BUDGET_EXCEEDED = "budget_exceeded"
    STEP_LIMIT = "step_limit"
    UNRECOVERABLE_STATE = "unrecoverable_state"


class Message(BaseModel):
    """A single normalized chat message."""

    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None
