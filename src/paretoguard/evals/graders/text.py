"""Text-based graders: exact match, normalized text match, classification."""

from paretoguard.core.models import InferenceResponse
from paretoguard.evals.graders.base import GradeOutcome, register, require
from paretoguard.evals.models import GraderConfig, GraderKind, GroundTruth


def _normalize(text: str) -> str:
    """Lowercase, strip, and collapse internal whitespace to single spaces."""
    return " ".join(text.lower().split())


@register(GraderKind.EXACT_MATCH)
def grade_exact_match(
    response: InferenceResponse, ground_truth: GroundTruth, config: GraderConfig
) -> GradeOutcome:
    """Byte-for-byte comparison of `response.output_text` to `ground_truth.text`."""
    expected = require(ground_truth.text, "exact_match requires ground_truth.text")
    actual = response.output_text
    succeeded = actual == expected
    return GradeOutcome(
        succeeded=succeeded,
        score=float(succeeded),
        explanation="exact match" if succeeded else f"expected {expected!r}, got {actual!r}",
        details={"expected": expected, "actual": actual},
    )


@register(GraderKind.NORMALIZED_TEXT_MATCH)
def grade_normalized_text_match(
    response: InferenceResponse, ground_truth: GroundTruth, config: GraderConfig
) -> GradeOutcome:
    """Case/whitespace-insensitive comparison, for free-text answers where exact
    formatting shouldn't matter (e.g. "Paris" vs "paris" vs " Paris\\n")."""
    expected = require(ground_truth.text, "normalized_text_match requires ground_truth.text")
    actual = response.output_text or ""
    succeeded = _normalize(actual) == _normalize(expected)
    return GradeOutcome(
        succeeded=succeeded,
        score=float(succeeded),
        explanation="normalized match" if succeeded else f"expected {expected!r}, got {actual!r}",
        details={"expected": expected, "actual": actual},
    )


@register(GraderKind.CLASSIFICATION)
def grade_classification(
    response: InferenceResponse, ground_truth: GroundTruth, config: GraderConfig
) -> GradeOutcome:
    """Compares `response.output_text` to a single expected class label.
    Case-insensitive by default; set `config.case_sensitive=True` to require an
    exact case match."""
    expected = require(ground_truth.label, "classification requires ground_truth.label")
    actual = (response.output_text or "").strip()
    succeeded = actual == expected if config.case_sensitive else actual.lower() == expected.lower()
    return GradeOutcome(
        succeeded=succeeded,
        score=float(succeeded),
        explanation="label match" if succeeded else f"expected label {expected!r}, got {actual!r}",
        details={"expected": expected, "actual": actual, "case_sensitive": config.case_sensitive},
    )
