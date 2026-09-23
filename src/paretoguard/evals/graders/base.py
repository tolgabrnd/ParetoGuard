"""The grader contract: a pure function from (response, ground truth, config) to a
typed, inspectable outcome.

Graders never raise for an ordinary grading mismatch (wrong answer, invalid JSON,
missing fields) — that's a normal `GradeOutcome(succeeded=False, ...)`, not an
exception. They may raise `ValueError` for a genuine misconfiguration (e.g. a
NUMERIC case with no `ground_truth.number` set), which the benchmark runner
catches and turns into a failing `EvalResult` rather than crashing the run.
"""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from paretoguard.core.models import InferenceResponse
from paretoguard.evals.models import GraderConfig, GraderKind, GroundTruth


class GradeOutcome(BaseModel):
    """The result of grading one response. `score` is continuous where a grader
    has a natural notion of partial credit (e.g. field-level F1); otherwise it is
    exactly 0.0 or 1.0 and equal to `float(succeeded)`."""

    succeeded: bool
    score: float = Field(ge=0, le=1)
    explanation: str
    details: dict[str, Any] = Field(default_factory=dict)


GraderFn = Callable[[InferenceResponse, GroundTruth, GraderConfig], GradeOutcome]
"""A grader is a plain function, not a class hierarchy — every grader in this
package is a stateless, pure computation over its three inputs."""


def require(value: Any, message: str) -> Any:
    """Raise ValueError with a clear message if a required ground-truth field is
    missing, rather than failing later with a confusing AttributeError/KeyError."""
    if value is None:
        raise ValueError(message)
    return value


_REGISTRY: dict[GraderKind, GraderFn] = {}


def register(kind: GraderKind) -> Callable[[GraderFn], GraderFn]:
    def decorator(fn: GraderFn) -> GraderFn:
        _REGISTRY[kind] = fn
        return fn

    return decorator


def get_grader(kind: GraderKind) -> GraderFn:
    try:
        return _REGISTRY[kind]
    except KeyError:
        raise ValueError(f"no grader registered for {kind!r}") from None
