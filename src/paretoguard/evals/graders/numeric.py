"""Numeric-answer grading with a configurable tolerance."""

import re

from paretoguard.core.models import InferenceResponse
from paretoguard.evals.graders.base import GradeOutcome, register, require
from paretoguard.evals.models import GraderConfig, GraderKind, GroundTruth

_DEFAULT_TOLERANCE = 1e-6
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_final_number(text: str) -> float | None:
    """Extracts the last number in the text, by convention the model's final
    answer (as opposed to numbers appearing earlier in its reasoning)."""
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    return float(matches[-1])


@register(GraderKind.NUMERIC)
def grade_numeric(
    response: InferenceResponse, ground_truth: GroundTruth, config: GraderConfig
) -> GradeOutcome:
    """Extracts the last number in `response.output_text` and compares it to
    `ground_truth.number` within `config.tolerance` (absolute difference;
    defaults to 1e-6 if unset)."""
    expected = require(ground_truth.number, "numeric requires ground_truth.number")
    tolerance = config.tolerance if config.tolerance is not None else _DEFAULT_TOLERANCE

    text = response.output_text or ""
    actual = _extract_final_number(text)
    if actual is None:
        return GradeOutcome(
            succeeded=False,
            score=0.0,
            explanation="no numeric answer found in output",
            details={"expected": expected, "actual_text": text},
        )

    diff = abs(actual - expected)
    succeeded = diff <= tolerance
    return GradeOutcome(
        succeeded=succeeded,
        score=float(succeeded),
        explanation=(f"expected {expected} (tolerance {tolerance}), got {actual} (diff {diff})"),
        details={"expected": expected, "actual": actual, "diff": diff, "tolerance": tolerance},
    )
