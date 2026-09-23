"""Tool-call trajectory grading.

Scoped to what a single `InferenceResponse` can contain: the ordered list of tool
calls the model returned in one turn (a real, well-established benchmark
category — e.g. parallel/sequenced function-calling correctness). Grading a
*multi-turn* agent trajectory against an executed, stateful tool simulator
(final-state grading after actually running the tools) is out of scope here: that
requires the tool-execution loop that `paretoguard.agents` will provide (planned,
not yet implemented — see docs/BUILD_PLAN.md Phase E). Implementing it in `evals`
ahead of that module existing would duplicate its responsibility and require
inventing simulator semantics this package has no authority over.
"""

from paretoguard.core.models import InferenceResponse, ToolCall
from paretoguard.evals.graders.base import GradeOutcome, register, require
from paretoguard.evals.models import ExpectedToolCall, GraderConfig, GraderKind, GroundTruth


def _matches(actual: ToolCall, expected: ExpectedToolCall) -> bool:
    if actual.name != expected.name:
        return False
    return not (expected.match_arguments and actual.arguments != expected.arguments)


@register(GraderKind.TOOL_TRAJECTORY)
def grade_tool_trajectory(
    response: InferenceResponse, ground_truth: GroundTruth, config: GraderConfig
) -> GradeOutcome:
    """Compares `response.tool_calls`, in order, against
    `ground_truth.expected_tool_calls`. Requires the same number of calls in the
    same order; `score` is the fraction of positions that matched (partial
    credit), `succeeded` requires all of them to match."""
    expected_calls = require(
        ground_truth.expected_tool_calls,
        "tool_trajectory requires ground_truth.expected_tool_calls",
    )
    actual_calls = response.tool_calls

    if len(actual_calls) != len(expected_calls):
        return GradeOutcome(
            succeeded=False,
            score=0.0,
            explanation=f"expected {len(expected_calls)} tool call(s), got {len(actual_calls)}",
            details={
                "expected": [c.model_dump() for c in expected_calls],
                "actual": [c.model_dump() for c in actual_calls],
            },
        )

    mismatches = []
    matches = 0
    for index, (actual, expected) in enumerate(zip(actual_calls, expected_calls, strict=True)):
        if _matches(actual, expected):
            matches += 1
        else:
            mismatches.append(
                {
                    "index": index,
                    "expected_name": expected.name,
                    "actual_name": actual.name,
                    "expected_arguments": expected.arguments if expected.match_arguments else None,
                    "actual_arguments": actual.arguments,
                }
            )

    score = matches / len(expected_calls) if expected_calls else 0.0
    succeeded = matches == len(expected_calls)
    return GradeOutcome(
        succeeded=succeeded,
        score=score,
        explanation=f"{matches}/{len(expected_calls)} tool calls matched",
        details={"mismatches": mismatches},
    )
