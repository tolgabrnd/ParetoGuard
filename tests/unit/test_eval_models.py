"""Unit tests for the eval domain model."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from paretoguard.core.models import Message, Role
from paretoguard.evals import (
    EvalCase,
    EvalResult,
    EvalSuite,
    ExpectedToolCall,
    GraderConfig,
    GraderKind,
    GroundTruth,
)


def _case(case_id: str = "case-1") -> EvalCase:
    return EvalCase(
        case_id=case_id,
        messages=[Message(role=Role.USER, content="2+2?")],
        grader=GraderConfig(kind=GraderKind.EXACT_MATCH),
        ground_truth=GroundTruth(text="4"),
    )


def test_eval_case_defaults() -> None:
    case = _case()
    assert case.tools == []
    assert case.tags == []
    assert case.metadata == {}


def test_eval_suite_case_count() -> None:
    suite = EvalSuite(
        name="demo_v1", version="1.0.0", description="demo", seed=1, cases=[_case("a"), _case("b")]
    )
    assert suite.case_count == 2


def test_ground_truth_allows_multiple_optional_shapes() -> None:
    gt = GroundTruth(
        number=4.0,
        expected_substrings=["hi"],
        expected_tool_calls=[ExpectedToolCall(name="calculator", arguments={"a": 1})],
    )
    assert gt.text is None
    assert gt.number == 4.0
    assert gt.expected_tool_calls is not None
    assert gt.expected_tool_calls[0].match_arguments is True


def test_grader_config_tolerance_must_be_non_negative() -> None:
    with pytest.raises(ValidationError):
        GraderConfig(kind=GraderKind.NUMERIC, tolerance=-1.0)


def test_eval_result_score_bounds() -> None:
    with pytest.raises(ValidationError):
        EvalResult(
            case_id="c",
            request_id=uuid4(),
            repetition=0,
            sequence=0,
            succeeded=True,
            score=1.5,
            grader_kind=GraderKind.EXACT_MATCH,
            explanation="bad score",
            latency_ms=1.0,
            total_tokens=1,
        )


def test_eval_result_defaults() -> None:
    result = EvalResult(
        case_id="c",
        request_id=uuid4(),
        repetition=0,
        sequence=0,
        succeeded=True,
        score=1.0,
        grader_kind=GraderKind.EXACT_MATCH,
        explanation="ok",
        latency_ms=12.0,
        total_tokens=5,
    )
    assert result.details == {}
    assert result.response_error_category is None
    assert result.cost_usd is None
