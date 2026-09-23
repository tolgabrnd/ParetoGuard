"""Evidence-presence grading, for long-context retrieval tasks."""

from paretoguard.core.models import InferenceResponse
from paretoguard.evals.graders.base import GradeOutcome, register, require
from paretoguard.evals.models import GraderConfig, GraderKind, GroundTruth


@register(GraderKind.SUBSTRING_PRESENCE)
def grade_substring_presence(
    response: InferenceResponse, ground_truth: GroundTruth, config: GraderConfig
) -> GradeOutcome:
    """Checks how many of `ground_truth.expected_substrings` appear in
    `response.output_text`. `score` is the fraction found (partial credit);
    `succeeded` requires all of them to be present. Case-insensitive by default."""
    needles = require(
        ground_truth.expected_substrings,
        "substring_presence requires ground_truth.expected_substrings",
    )
    actual = response.output_text or ""
    haystack = actual if config.case_sensitive else actual.lower()
    search_needles = needles if config.case_sensitive else [n.lower() for n in needles]

    found = [
        needle for needle, search in zip(needles, search_needles, strict=True) if search in haystack
    ]
    score = len(found) / len(needles) if needles else 0.0
    succeeded = len(found) == len(needles)

    return GradeOutcome(
        succeeded=succeeded,
        score=score,
        explanation=f"{len(found)}/{len(needles)} expected substrings found",
        details={"found": found, "missing": [n for n in needles if n not in found]},
    )
