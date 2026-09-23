"""Typed evaluation schema: tasks, ground truth, grading configuration, and results.

These types are specific to the evaluation subsystem (task definition, grading,
benchmark orchestration) and are deliberately not folded into `core.models`, which
holds only types shared across every subsystem. `RunManifest` (reproducibility
metadata) already lives in `core.models` and is reused as-is by the benchmark
runner rather than duplicated here.
"""

from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from paretoguard.core.models import FailureCategory, Message, ToolSpec


class GraderKind(StrEnum):
    """Deterministic grading strategies implemented in `paretoguard.evals.graders`.

    LLM-as-judge is deliberately not a member of this enum: it is not implemented
    in Phase C, and when it lands it must be a clearly separate, optional path
    (see CLAUDE.md) rather than another interchangeable grader kind.
    """

    EXACT_MATCH = "exact_match"
    NORMALIZED_TEXT_MATCH = "normalized_text_match"
    NUMERIC = "numeric"
    CLASSIFICATION = "classification"
    JSON_SCHEMA = "json_schema"
    FIELD_SCORING = "field_scoring"
    SUBSTRING_PRESENCE = "substring_presence"
    TOOL_TRAJECTORY = "tool_trajectory"


class GraderConfig(BaseModel):
    """Parameters controlling how a case is graded. Which fields apply depends on
    `kind`; unused fields are simply ignored by that grader."""

    kind: GraderKind
    tolerance: float | None = Field(default=None, ge=0, description="NUMERIC grader only")
    case_sensitive: bool = Field(
        default=False, description="Text-based graders (CLASSIFICATION, SUBSTRING_PRESENCE)"
    )


class ExpectedToolCall(BaseModel):
    """One expected tool invocation, for the TOOL_TRAJECTORY grader."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    match_arguments: bool = Field(
        default=True,
        description="If False, only the tool name and call order are checked, not exact arguments.",
    )


class GroundTruth(BaseModel):
    """The expected answer for a case. Ground truth shape is inherently
    grader-specific (a number, a label, a JSON Schema, a list of expected tool
    calls, ...); rather than one opaque `dict[str, Any]`, each shape a grader in
    this codebase understands gets its own typed, optional field. Populate the
    field(s) that match the case's `GraderConfig.kind`.
    """

    text: str | None = Field(default=None, description="EXACT_MATCH, NORMALIZED_TEXT_MATCH")
    number: float | None = Field(default=None, description="NUMERIC")
    label: str | None = Field(default=None, description="CLASSIFICATION")
    json_schema: dict[str, Any] | None = Field(default=None, description="JSON_SCHEMA")
    expected_fields: dict[str, Any] | None = Field(default=None, description="FIELD_SCORING")
    expected_substrings: list[str] | None = Field(default=None, description="SUBSTRING_PRESENCE")
    expected_tool_calls: list[ExpectedToolCall] | None = Field(
        default=None, description="TOOL_TRAJECTORY"
    )


class EvalCase(BaseModel):
    """One task instance within an `EvalSuite`."""

    case_id: str
    messages: list[Message]
    tools: list[ToolSpec] = Field(default_factory=list)
    structured_output_schema: dict[str, Any] | None = None
    max_output_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    grader: GraderConfig
    ground_truth: GroundTruth
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Passed through to InferenceRequest.metadata. May include "
            "MockProvider-recognized keys (see paretoguard.providers.mock) so a "
            "case can demonstrate a passing or failing result offline; real "
            "provider adapters ignore keys they don't recognize."
        ),
    )


class EvalSuite(BaseModel):
    """A versioned, reproducible collection of `EvalCase`s.

    Suites are generated deterministically from `seed` (see
    `paretoguard.evals.suites`) rather than stored as static fixture files, so a
    suite's cases are fully reproducible from (module version, seed) alone.
    """

    name: str
    version: str
    description: str
    seed: int
    cases: list[EvalCase]

    @property
    def case_count(self) -> int:
        return len(self.cases)


class EvalResult(BaseModel):
    """The outcome of grading one (case, repetition) execution."""

    result_id: UUID = Field(default_factory=uuid4)
    case_id: str
    request_id: UUID
    repetition: int = Field(ge=0)
    sequence: int = Field(
        ge=0,
        description=(
            "case_index * repetitions + repetition, assigned by the benchmark "
            "runner. Concurrent execution can complete out of order; this field "
            "reconstructs the canonical (case, repetition) order independent of "
            "storage/read-back order, which SQL does not otherwise guarantee."
        ),
    )
    succeeded: bool
    score: float = Field(ge=0, le=1)
    grader_kind: GraderKind
    explanation: str
    details: dict[str, Any] = Field(default_factory=dict)
    response_error_category: FailureCategory | None = Field(
        default=None,
        description="Set when the InferenceResponse itself failed (provider/runtime failure) rather than the case being graded incorrect.",
    )
    latency_ms: float = Field(ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    total_tokens: int = Field(ge=0)
