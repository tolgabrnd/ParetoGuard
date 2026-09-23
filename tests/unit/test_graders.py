"""Unit tests for every deterministic grader."""

from uuid import uuid4

import pytest

from paretoguard.core.models import (
    FinishReason,
    InferenceResponse,
    LatencyRecord,
    TokenUsage,
    ToolCall,
)
from paretoguard.evals.graders import get_grader
from paretoguard.evals.models import ExpectedToolCall, GraderConfig, GraderKind, GroundTruth

_USAGE = TokenUsage(input_tokens=1, output_tokens=1)
_LATENCY = LatencyRecord(total_latency_ms=1.0)


def _response(
    *,
    output_text: str | None = None,
    structured_output: dict | None = None,
    tool_calls: list[ToolCall] | None = None,
) -> InferenceResponse:
    return InferenceResponse(
        request_id=uuid4(),
        provider="mock",
        model="mock-strong",
        output_text=output_text,
        structured_output=structured_output,
        finish_reason=FinishReason.TOOL_CALLS if tool_calls else FinishReason.STOP,
        tool_calls=tool_calls or [],
        token_usage=_USAGE,
        latency=_LATENCY,
    )


# -- exact_match ------------------------------------------------------------


def test_exact_match_success() -> None:
    grader = get_grader(GraderKind.EXACT_MATCH)
    outcome = grader(
        _response(output_text="4"), GroundTruth(text="4"), GraderConfig(kind=GraderKind.EXACT_MATCH)
    )
    assert outcome.succeeded
    assert outcome.score == 1.0


def test_exact_match_is_case_sensitive() -> None:
    grader = get_grader(GraderKind.EXACT_MATCH)
    outcome = grader(
        _response(output_text="Paris"),
        GroundTruth(text="paris"),
        GraderConfig(kind=GraderKind.EXACT_MATCH),
    )
    assert not outcome.succeeded


def test_exact_match_requires_ground_truth_text() -> None:
    grader = get_grader(GraderKind.EXACT_MATCH)
    with pytest.raises(ValueError, match=r"ground_truth\.text"):
        grader(_response(output_text="4"), GroundTruth(), GraderConfig(kind=GraderKind.EXACT_MATCH))


# -- normalized_text_match ---------------------------------------------------


def test_normalized_text_match_ignores_case_and_whitespace() -> None:
    grader = get_grader(GraderKind.NORMALIZED_TEXT_MATCH)
    outcome = grader(
        _response(output_text="  PARIS  \n"),
        GroundTruth(text="paris"),
        GraderConfig(kind=GraderKind.NORMALIZED_TEXT_MATCH),
    )
    assert outcome.succeeded


# -- classification -----------------------------------------------------------


def test_classification_case_insensitive_by_default() -> None:
    grader = get_grader(GraderKind.CLASSIFICATION)
    outcome = grader(
        _response(output_text="Positive"),
        GroundTruth(label="positive"),
        GraderConfig(kind=GraderKind.CLASSIFICATION),
    )
    assert outcome.succeeded


def test_classification_case_sensitive_when_configured() -> None:
    grader = get_grader(GraderKind.CLASSIFICATION)
    outcome = grader(
        _response(output_text="Positive"),
        GroundTruth(label="positive"),
        GraderConfig(kind=GraderKind.CLASSIFICATION, case_sensitive=True),
    )
    assert not outcome.succeeded


# -- numeric ------------------------------------------------------------------


def test_numeric_extracts_last_number_and_matches_within_tolerance() -> None:
    grader = get_grader(GraderKind.NUMERIC)
    outcome = grader(
        _response(output_text="First I compute 10, then the final answer is 42."),
        GroundTruth(number=42.0),
        GraderConfig(kind=GraderKind.NUMERIC),
    )
    assert outcome.succeeded
    assert outcome.details["actual"] == 42.0


def test_numeric_respects_configured_tolerance() -> None:
    grader = get_grader(GraderKind.NUMERIC)
    outcome = grader(
        _response(output_text="41.95"),
        GroundTruth(number=42.0),
        GraderConfig(kind=GraderKind.NUMERIC, tolerance=0.1),
    )
    assert outcome.succeeded


def test_numeric_fails_outside_tolerance() -> None:
    grader = get_grader(GraderKind.NUMERIC)
    outcome = grader(
        _response(output_text="41.0"),
        GroundTruth(number=42.0),
        GraderConfig(kind=GraderKind.NUMERIC),
    )
    assert not outcome.succeeded


def test_numeric_handles_no_number_in_output() -> None:
    grader = get_grader(GraderKind.NUMERIC)
    outcome = grader(
        _response(output_text="I don't know"),
        GroundTruth(number=42.0),
        GraderConfig(kind=GraderKind.NUMERIC),
    )
    assert not outcome.succeeded
    assert "no numeric answer" in outcome.explanation


# -- json_schema ----------------------------------------------------------------

_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "number"}},
    "required": ["answer"],
}


def test_json_schema_valid_structured_output() -> None:
    grader = get_grader(GraderKind.JSON_SCHEMA)
    outcome = grader(
        _response(structured_output={"answer": 4}),
        GroundTruth(json_schema=_SCHEMA),
        GraderConfig(kind=GraderKind.JSON_SCHEMA),
    )
    assert outcome.succeeded


def test_json_schema_falls_back_to_parsing_output_text() -> None:
    grader = get_grader(GraderKind.JSON_SCHEMA)
    outcome = grader(
        _response(output_text='{"answer": 4}'),
        GroundTruth(json_schema=_SCHEMA),
        GraderConfig(kind=GraderKind.JSON_SCHEMA),
    )
    assert outcome.succeeded


def test_json_schema_rejects_invalid_shape() -> None:
    grader = get_grader(GraderKind.JSON_SCHEMA)
    outcome = grader(
        _response(structured_output={"answer": "not a number"}),
        GroundTruth(json_schema=_SCHEMA),
        GraderConfig(kind=GraderKind.JSON_SCHEMA),
    )
    assert not outcome.succeeded
    assert "schema validation failed" in outcome.explanation


def test_json_schema_handles_malformed_json_text() -> None:
    grader = get_grader(GraderKind.JSON_SCHEMA)
    outcome = grader(
        _response(output_text="{not valid json"),
        GroundTruth(json_schema=_SCHEMA),
        GraderConfig(kind=GraderKind.JSON_SCHEMA),
    )
    assert not outcome.succeeded


# -- field_scoring ----------------------------------------------------------------


def test_field_scoring_perfect_match() -> None:
    grader = get_grader(GraderKind.FIELD_SCORING)
    outcome = grader(
        _response(structured_output={"vendor": "Acme", "total": 10.0}),
        GroundTruth(expected_fields={"vendor": "Acme", "total": 10.0}),
        GraderConfig(kind=GraderKind.FIELD_SCORING),
    )
    assert outcome.succeeded
    assert outcome.score == 1.0


def test_field_scoring_partial_credit() -> None:
    grader = get_grader(GraderKind.FIELD_SCORING)
    outcome = grader(
        _response(structured_output={"vendor": "Acme", "total": 999.0}),
        GroundTruth(expected_fields={"vendor": "Acme", "total": 10.0}),
        GraderConfig(kind=GraderKind.FIELD_SCORING),
    )
    assert not outcome.succeeded
    assert 0.0 < outcome.score < 1.0
    assert outcome.details["correct_field_count"] == 1


def test_field_scoring_no_structured_output() -> None:
    grader = get_grader(GraderKind.FIELD_SCORING)
    outcome = grader(
        _response(output_text="I can't extract that"),
        GroundTruth(expected_fields={"vendor": "Acme"}),
        GraderConfig(kind=GraderKind.FIELD_SCORING),
    )
    assert not outcome.succeeded
    assert outcome.score == 0.0


# -- substring_presence -------------------------------------------------------------


def test_substring_presence_all_found() -> None:
    grader = get_grader(GraderKind.SUBSTRING_PRESENCE)
    outcome = grader(
        _response(output_text="The secret code is ALPHA-1234."),
        GroundTruth(expected_substrings=["ALPHA-1234"]),
        GraderConfig(kind=GraderKind.SUBSTRING_PRESENCE),
    )
    assert outcome.succeeded
    assert outcome.score == 1.0


def test_substring_presence_partial_credit_when_some_missing() -> None:
    grader = get_grader(GraderKind.SUBSTRING_PRESENCE)
    outcome = grader(
        _response(output_text="Only found ALPHA-1234"),
        GroundTruth(expected_substrings=["ALPHA-1234", "BRAVO-5678"]),
        GraderConfig(kind=GraderKind.SUBSTRING_PRESENCE),
    )
    assert not outcome.succeeded
    assert outcome.score == 0.5
    assert outcome.details["missing"] == ["BRAVO-5678"]


def test_substring_presence_case_insensitive_by_default() -> None:
    grader = get_grader(GraderKind.SUBSTRING_PRESENCE)
    outcome = grader(
        _response(output_text="the code is alpha-1234"),
        GroundTruth(expected_substrings=["ALPHA-1234"]),
        GraderConfig(kind=GraderKind.SUBSTRING_PRESENCE),
    )
    assert outcome.succeeded


# -- tool_trajectory ----------------------------------------------------------------


def test_tool_trajectory_exact_match() -> None:
    grader = get_grader(GraderKind.TOOL_TRAJECTORY)
    outcome = grader(
        _response(
            tool_calls=[ToolCall(id="1", name="calculator", arguments={"expression": "2+2"})]
        ),
        GroundTruth(
            expected_tool_calls=[
                ExpectedToolCall(name="calculator", arguments={"expression": "2+2"})
            ]
        ),
        GraderConfig(kind=GraderKind.TOOL_TRAJECTORY),
    )
    assert outcome.succeeded


def test_tool_trajectory_wrong_tool_name() -> None:
    grader = get_grader(GraderKind.TOOL_TRAJECTORY)
    outcome = grader(
        _response(tool_calls=[ToolCall(id="1", name="wrong_tool", arguments={})]),
        GroundTruth(expected_tool_calls=[ExpectedToolCall(name="calculator", arguments={})]),
        GraderConfig(kind=GraderKind.TOOL_TRAJECTORY),
    )
    assert not outcome.succeeded
    assert outcome.details["mismatches"][0]["expected_name"] == "calculator"


def test_tool_trajectory_wrong_call_count() -> None:
    grader = get_grader(GraderKind.TOOL_TRAJECTORY)
    outcome = grader(
        _response(tool_calls=[]),
        GroundTruth(expected_tool_calls=[ExpectedToolCall(name="calculator", arguments={})]),
        GraderConfig(kind=GraderKind.TOOL_TRAJECTORY),
    )
    assert not outcome.succeeded
    assert outcome.score == 0.0


def test_tool_trajectory_ignores_arguments_when_match_arguments_false() -> None:
    grader = get_grader(GraderKind.TOOL_TRAJECTORY)
    outcome = grader(
        _response(
            tool_calls=[ToolCall(id="1", name="calculator", arguments={"expression": "anything"})]
        ),
        GroundTruth(
            expected_tool_calls=[
                ExpectedToolCall(name="calculator", arguments={}, match_arguments=False)
            ]
        ),
        GraderConfig(kind=GraderKind.TOOL_TRAJECTORY),
    )
    assert outcome.succeeded


def test_get_grader_raises_for_unregistered_kind() -> None:
    # All GraderKind members are registered; this pins that get_grader fails
    # loudly rather than silently for anything not in the registry.
    from paretoguard.evals.graders.base import _REGISTRY

    for kind in GraderKind:
        assert kind in _REGISTRY
