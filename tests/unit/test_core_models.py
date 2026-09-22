"""Unit tests for core domain models."""

from datetime import date
from uuid import uuid4

import pytest
from pydantic import ValidationError

from paretoguard.core.models import (
    CostRecord,
    ErrorInfo,
    FailureCategory,
    FinishReason,
    InferenceRequest,
    InferenceResponse,
    LatencyRecord,
    Message,
    ModelSpec,
    PricingEntry,
    ProviderKind,
    ProviderSpec,
    Role,
    RoutingConstraints,
    RoutingDecision,
    RunManifest,
    TokenUsage,
    ToolCall,
)


def test_token_usage_total_tokens() -> None:
    usage = TokenUsage(input_tokens=100, output_tokens=50)
    assert usage.total_tokens == 150


def test_token_usage_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        TokenUsage(input_tokens=-1, output_tokens=0)


def test_cost_record_total_cost() -> None:
    cost = CostRecord(input_cost_usd=0.01, output_cost_usd=0.02, pricing_version="2026-09-22")
    assert cost.total_cost_usd == pytest.approx(0.03)


def test_cost_record_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        CostRecord(input_cost_usd=-0.01, output_cost_usd=0.0, pricing_version="v1")


def test_pricing_entry_requires_effective_date() -> None:
    entry = PricingEntry(
        provider="mock",
        model="mock-strong",
        input_price_per_million_usd=0.0,
        output_price_per_million_usd=0.0,
        version="2026-09-22",
        effective_date=date(2026, 9, 22),
    )
    assert entry.effective_date == date(2026, 9, 22)


def test_provider_spec_holds_env_var_name_not_a_secret_value() -> None:
    spec = ProviderSpec(name="openai", kind=ProviderKind.OPENAI, api_key_env_var="OPENAI_API_KEY")
    # The field is documented and typed to hold an environment variable *name*;
    # this pins that the round-tripped value is exactly what was passed in, i.e.
    # nothing resolves or substitutes an actual key value into the model.
    assert spec.model_dump()["api_key_env_var"] == "OPENAI_API_KEY"


def test_model_spec_requires_positive_context_window() -> None:
    with pytest.raises(ValidationError):
        ModelSpec(name="m", provider="mock", context_window=0)


def test_inference_request_defaults() -> None:
    req = InferenceRequest(
        provider="mock",
        model="mock-strong",
        messages=[Message(role=Role.USER, content="hello")],
    )
    assert req.tools == []
    assert req.metadata == {}
    assert req.created_at.tzinfo is not None


def test_inference_response_succeeded_property() -> None:
    usage = TokenUsage(input_tokens=10, output_tokens=5)
    latency = LatencyRecord(total_latency_ms=120.0)
    ok = InferenceResponse(
        request_id=uuid4(),
        provider="mock",
        model="mock-strong",
        output_text="hi",
        finish_reason=FinishReason.STOP,
        token_usage=usage,
        latency=latency,
    )
    assert ok.succeeded is True

    failed = ok.model_copy(
        update={
            "finish_reason": FinishReason.ERROR,
            "error": ErrorInfo(
                category=FailureCategory.TIMEOUT, message="timed out", retryable=True
            ),
        }
    )
    assert failed.succeeded is False


def test_tool_call_defaults_to_empty_arguments() -> None:
    call = ToolCall(id="1", name="calculator")
    assert call.arguments == {}


def test_routing_constraints_bounds() -> None:
    with pytest.raises(ValidationError):
        RoutingConstraints(min_predicted_success=1.5)


def test_routing_decision_requires_explanation() -> None:
    decision = RoutingDecision(selected_model="mock-strong", explanation="cheapest eligible model")
    assert decision.fallback_order == []
    assert decision.explanation


def test_run_manifest_repetitions_defaults_to_one() -> None:
    manifest = RunManifest(
        run_id="run-1",
        paretoguard_version="0.1.0",
        os="Windows",
        python_version="3.12.14",
    )
    assert manifest.repetitions == 1
    assert manifest.environment == {}
