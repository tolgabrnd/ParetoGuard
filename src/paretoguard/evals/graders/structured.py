"""Structured-output graders: JSON Schema validation and field-level scoring."""

import json
from typing import Any

import jsonschema

from paretoguard.core.models import InferenceResponse
from paretoguard.evals.graders.base import GradeOutcome, register, require
from paretoguard.evals.models import GraderConfig, GraderKind, GroundTruth


def _candidate_json(response: InferenceResponse) -> dict[str, Any] | None:
    """Prefers a native `structured_output`; falls back to parsing `output_text`
    as JSON (some providers/mock scenarios return only text)."""
    if response.structured_output is not None:
        return response.structured_output
    if response.output_text:
        try:
            parsed = json.loads(response.output_text)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


@register(GraderKind.JSON_SCHEMA)
def grade_json_schema(
    response: InferenceResponse, ground_truth: GroundTruth, config: GraderConfig
) -> GradeOutcome:
    """Validates the response's structured output against `ground_truth.json_schema`.
    This checks *shape* validity, not content correctness — pair with
    FIELD_SCORING (or a stricter schema) to also check field values."""
    schema = require(ground_truth.json_schema, "json_schema requires ground_truth.json_schema")
    candidate = _candidate_json(response)
    if candidate is None:
        return GradeOutcome(
            succeeded=False,
            score=0.0,
            explanation="no structured output or parseable JSON text to validate",
            details={"raw_output": response.output_text},
        )
    try:
        jsonschema.validate(candidate, schema)
    except jsonschema.ValidationError as exc:
        return GradeOutcome(
            succeeded=False,
            score=0.0,
            explanation=f"schema validation failed: {exc.message}",
            details={"path": list(exc.absolute_path), "candidate": candidate},
        )
    return GradeOutcome(
        succeeded=True,
        score=1.0,
        explanation="valid against schema",
        details={"candidate": candidate},
    )


@register(GraderKind.FIELD_SCORING)
def grade_field_scoring(
    response: InferenceResponse, ground_truth: GroundTruth, config: GraderConfig
) -> GradeOutcome:
    """Field-level precision/recall/F1 between the response's structured output
    and `ground_truth.expected_fields`. A field counts as correct only on an
    exact value match (`candidate[key] == expected_value`); no partial credit
    within a single field."""
    expected = require(
        ground_truth.expected_fields, "field_scoring requires ground_truth.expected_fields"
    )
    candidate = _candidate_json(response)
    if candidate is None:
        return GradeOutcome(
            succeeded=False,
            score=0.0,
            explanation="no structured output to score",
            details={"expected_fields": expected},
        )

    correct = sum(1 for key, value in expected.items() if candidate.get(key) == value)
    total_expected = len(expected)
    total_predicted = len(candidate)
    precision = correct / total_predicted if total_predicted else 0.0
    recall = correct / total_expected if total_expected else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    succeeded = f1 == 1.0

    return GradeOutcome(
        succeeded=succeeded,
        score=f1,
        explanation=f"field F1={f1:.2f} (precision={precision:.2f}, recall={recall:.2f})",
        details={
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "correct_field_count": correct,
            "expected_field_count": total_expected,
            "predicted_field_count": total_predicted,
            "candidate": candidate,
        },
    )
