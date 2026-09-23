"""Unit tests for shared routing types: feature extraction and eligibility filtering."""

from paretoguard.core.models import (
    InferenceRequest,
    Message,
    ModelSpec,
    Role,
    RoutingConstraints,
    ToolSpec,
)
from paretoguard.routing.types import (
    TASK_FAMILY_METADATA_KEY,
    RoutingRequest,
    candidate_key,
    extract_task_features,
    filter_eligible,
)


def _request(**kwargs: object) -> InferenceRequest:
    defaults: dict[str, object] = {
        "provider": "mock",
        "model": "mock-strong",
        "messages": [Message(role=Role.USER, content="hello there general kenobi")],
    }
    defaults.update(kwargs)
    return InferenceRequest(**defaults)  # type: ignore[arg-type]


def _model(
    name: str,
    *,
    provider: str = "mock",
    context_window: int = 1000,
    supports_tools: bool = False,
    supports_structured_output: bool = False,
    capabilities: list[str] | None = None,
) -> ModelSpec:
    return ModelSpec(
        name=name,
        provider=provider,
        context_window=context_window,
        supports_tools=supports_tools,
        supports_structured_output=supports_structured_output,
        capabilities=capabilities or [],
    )


def test_candidate_key_is_provider_and_model() -> None:
    assert candidate_key("openai", "gpt-x") == "openai:gpt-x"


def test_extract_task_features_counts_words_as_token_estimate() -> None:
    features = extract_task_features(_request())
    assert features.input_tokens_estimate == 4


def test_extract_task_features_detects_structured_output() -> None:
    features = extract_task_features(_request(structured_output_schema={"type": "object"}))
    assert features.requires_structured_output is True


def test_extract_task_features_detects_tools() -> None:
    tool = ToolSpec(name="calc", description="calculator")
    features = extract_task_features(_request(tools=[tool]))
    assert features.requires_tool_use is True
    assert features.tool_count == 1


def test_extract_task_features_reads_task_family_from_metadata() -> None:
    features = extract_task_features(_request(metadata={TASK_FAMILY_METADATA_KEY: "numeric"}))
    assert features.task_family == "numeric"


def test_extract_task_features_task_family_defaults_to_none() -> None:
    features = extract_task_features(_request())
    assert features.task_family is None


def test_extract_task_features_context_estimate_includes_max_output() -> None:
    features = extract_task_features(_request(max_output_tokens=100))
    assert features.context_tokens_estimate == features.input_tokens_estimate + 100


def test_filter_eligible_excludes_model_with_too_small_context_window() -> None:
    request = _request(messages=[Message(role=Role.USER, content=" ".join(["word"] * 50))])
    candidates = [_model("small", context_window=10), _model("large", context_window=1000)]
    routing_request = RoutingRequest(
        request=request, features=extract_task_features(request), candidates=candidates
    )
    eligible, excluded = filter_eligible(routing_request)
    assert [c.name for c in eligible] == ["large"]
    assert "small" in excluded


def test_filter_eligible_excludes_model_without_structured_output_support() -> None:
    request = _request(structured_output_schema={"type": "object"})
    candidates = [
        _model("no-structured", supports_structured_output=False),
        _model("has-structured", supports_structured_output=True),
    ]
    routing_request = RoutingRequest(
        request=request, features=extract_task_features(request), candidates=candidates
    )
    eligible, excluded = filter_eligible(routing_request)
    assert [c.name for c in eligible] == ["has-structured"]
    assert "no-structured" in excluded


def test_filter_eligible_excludes_model_without_tool_support() -> None:
    request = _request(tools=[ToolSpec(name="calc", description="calculator")])
    candidates = [
        _model("no-tools", supports_tools=False),
        _model("has-tools", supports_tools=True),
    ]
    routing_request = RoutingRequest(
        request=request, features=extract_task_features(request), candidates=candidates
    )
    eligible, excluded = filter_eligible(routing_request)
    assert [c.name for c in eligible] == ["has-tools"]
    assert "no-tools" in excluded


def test_filter_eligible_respects_explicit_exclusion() -> None:
    request = _request()
    candidates = [_model("a"), _model("b")]
    routing_request = RoutingRequest(
        request=request,
        features=extract_task_features(request),
        candidates=candidates,
        constraints=RoutingConstraints(excluded_models=["a"]),
    )
    eligible, excluded = filter_eligible(routing_request)
    assert [c.name for c in eligible] == ["b"]
    assert "explicitly excluded" in excluded["a"]


def test_filter_eligible_requires_missing_capabilities() -> None:
    request = _request()
    candidates = [
        _model("plain", capabilities=[]),
        _model("vision", capabilities=["vision"]),
    ]
    routing_request = RoutingRequest(
        request=request,
        features=extract_task_features(request),
        candidates=candidates,
        constraints=RoutingConstraints(required_capabilities=["vision"]),
    )
    eligible, excluded = filter_eligible(routing_request)
    assert [c.name for c in eligible] == ["vision"]
    assert "vision" in excluded["plain"]
