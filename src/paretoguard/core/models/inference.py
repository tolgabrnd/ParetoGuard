"""Inference request/response types and tool-call types.

These are the normalized boundary between `paretoguard.providers` and the rest of
the system — every adapter maps its provider's native format to `InferenceResponse`
and nothing downstream needs to know which provider produced it.
"""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from paretoguard.core.models.common import FailureCategory, FinishReason, Message
from paretoguard.core.models.cost import CostRecord, LatencyRecord, TokenUsage


class ToolSpec(BaseModel):
    """A tool made available to the model for a given request."""

    name: str
    description: str
    parameters_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for the tool's arguments."
    )


class ToolCall(BaseModel):
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """The result of executing a `ToolCall`."""

    tool_call_id: str
    output: Any = None
    is_error: bool = False
    error_message: str | None = None
    latency_ms: float | None = Field(default=None, ge=0)


class ErrorInfo(BaseModel):
    """Normalized error information attached to a failed `InferenceResponse`."""

    category: FailureCategory
    message: str
    retryable: bool = False
    provider_error_code: str | None = None


class InferenceRequest(BaseModel):
    """A normalized request to a provider/model."""

    request_id: UUID = Field(default_factory=uuid4)
    task_id: str | None = None
    provider: str
    model: str
    messages: list[Message]
    tools: list[ToolSpec] = Field(default_factory=list)
    max_output_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    structured_output_schema: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class InferenceResponse(BaseModel):
    """A normalized response from a provider/model."""

    request_id: UUID
    provider: str
    model: str
    output_text: str | None = None
    structured_output: dict[str, Any] | None = None
    finish_reason: FinishReason
    tool_calls: list[ToolCall] = Field(default_factory=list)
    token_usage: TokenUsage
    cost: CostRecord | None = None
    latency: LatencyRecord
    error: ErrorInfo | None = None
    raw_provider_metadata: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.finish_reason != FinishReason.ERROR
