"""Task-feature extraction: the single function that turns an
`InferenceRequest` into cheap, deterministic, routing-time-safe features.

Lives in `core` (not `routing` or `evals`) because both depend on it without
either depending on the other: every `Router` uses it to make a live
decision (`paretoguard.routing`), and offline training-matrix generation
(`paretoguard.evals.matrix`) uses the exact same function to label training
rows — using one function both online and offline is what guarantees the
learned router (Commit 22) is trained on exactly the information actually
available at routing time, with no online/offline skew. Neither `routing`
nor `evals` may assume the other exists (see docs/ARCHITECTURE.md's module
boundaries); this module lets both use feature extraction without either
importing the other.
"""

from dataclasses import dataclass

from paretoguard.core.models import InferenceRequest

TASK_FAMILY_METADATA_KEY = "task_family"
"""InferenceRequest.metadata key a caller may set to identify the task family
(e.g. suite name) a request belongs to, for task-family-scoped rules/profiles.
Optional: absent unless a caller (typically an EvalCase's metadata, which is
passed through verbatim to InferenceRequest.metadata) sets it."""


@dataclass(frozen=True)
class TaskFeatures:
    """Cheap, deterministic features extracted from an `InferenceRequest` at
    routing time — never anything only known after execution.

    `expected_step_count` is `None` — this codebase has no step-count
    predictor yet (Phase E's agent simulator may add one); it is never
    fabricated as a fixed/guessed number.
    """

    input_tokens_estimate: int
    max_output_tokens: int | None
    context_tokens_estimate: int
    requires_structured_output: bool
    requires_tool_use: bool
    tool_count: int
    task_family: str | None
    schema_complexity: int
    """Recursive count of `properties`/`items` schema nodes in
    `structured_output_schema`. 0 when there is no schema."""
    numeric_density: float
    """Fraction, in [0, 1], of input word-tokens that parse as a number."""
    expected_output_length: int | None
    """Alias of `max_output_tokens`, named to match the ML feature list in
    docs/BUILD_PLAN.md — the same value, not a second measurement."""
    expected_step_count: int | None = None

    def as_feature_dict(self) -> dict[str, int | float | bool | str | None]:
        """Flat, ML-pipeline-friendly view: every field, `task_family`
        included as a raw categorical string (the dataset builder in
        `paretoguard.routing.dataset` is responsible for encoding it)."""
        return {
            "input_tokens_estimate": self.input_tokens_estimate,
            "max_output_tokens": self.max_output_tokens,
            "context_tokens_estimate": self.context_tokens_estimate,
            "requires_structured_output": self.requires_structured_output,
            "requires_tool_use": self.requires_tool_use,
            "tool_count": self.tool_count,
            "task_family": self.task_family,
            "schema_complexity": self.schema_complexity,
            "numeric_density": self.numeric_density,
            "expected_output_length": self.expected_output_length,
            "expected_step_count": self.expected_step_count,
        }


def _schema_complexity(schema: dict[str, object] | None) -> int:
    """Recursively counts `properties`/`items` nodes in a JSON Schema as a
    cheap proxy for structural complexity. Not a full JSON Schema walker —
    good enough to rank "flat object" vs "deeply nested" tasks, not intended
    as a precise complexity metric."""
    if schema is None:
        return 0

    count = 0
    properties = schema.get("properties")
    if isinstance(properties, dict):
        count += len(properties)
        for value in properties.values():
            if isinstance(value, dict):
                count += _schema_complexity(value)
    items = schema.get("items")
    if isinstance(items, dict):
        count += 1 + _schema_complexity(items)
    return count


def _numeric_density(words: list[str]) -> float:
    if not words:
        return 0.0
    numeric_count = sum(1 for w in words if _looks_numeric(w))
    return numeric_count / len(words)


def _looks_numeric(word: str) -> bool:
    stripped = word.strip(".,;:%()$")
    if not stripped:
        return False
    try:
        float(stripped)
    except ValueError:
        return False
    return True


def extract_task_features(request: InferenceRequest) -> TaskFeatures:
    """Whitespace-split word count as a token estimate — the same cheap,
    dependency-free heuristic `MockProvider` uses for its own token counts
    (see `paretoguard.providers.mock`), good enough for *routing-time*
    eligibility decisions, not for billing."""
    words = [w for m in request.messages for w in m.content.split()]
    input_tokens_estimate = max(1, len(words))
    max_output = request.max_output_tokens
    return TaskFeatures(
        input_tokens_estimate=input_tokens_estimate,
        max_output_tokens=max_output,
        context_tokens_estimate=input_tokens_estimate + (max_output or 0),
        requires_structured_output=request.structured_output_schema is not None,
        requires_tool_use=len(request.tools) > 0,
        tool_count=len(request.tools),
        task_family=request.metadata.get(TASK_FAMILY_METADATA_KEY),
        schema_complexity=_schema_complexity(request.structured_output_schema),
        numeric_density=_numeric_density(words),
        expected_output_length=max_output,
    )
