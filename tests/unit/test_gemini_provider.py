"""Unit tests for the Gemini adapter (Interactions API), entirely via httpx.MockTransport."""

import json
from collections.abc import Callable

import httpx
import pytest

from paretoguard.core.models import (
    FinishReason,
    InferenceRequest,
    Message,
    ProviderKind,
    Role,
    ToolSpec,
)
from paretoguard.core.models.provider import ProviderSpec
from paretoguard.providers.gemini import GeminiProvider
from paretoguard.providers.http import ProviderConfigError

_SPEC = ProviderSpec(name="gemini", kind=ProviderKind.GEMINI, api_key_env_var="GEMINI_API_KEY")


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _request(**kwargs: object) -> InferenceRequest:
    defaults: dict[str, object] = {
        "provider": "gemini",
        "model": "gemini-test",
        "messages": [Message(role=Role.USER, content="what is 2+2?")],
    }
    defaults.update(kwargs)
    return InferenceRequest(**defaults)  # type: ignore[arg-type]


def _success_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "v1_abc123",
            "object": "interaction",
            "status": "completed",
            "model": "gemini-test",
            "steps": [{"type": "model_output", "content": [{"type": "text", "text": "4"}]}],
            "usage": {
                "total_input_tokens": 10,
                "total_output_tokens": 3,
                "total_tokens": 13,
                "total_cached_tokens": 1,
            },
        },
    )


async def test_successful_completion() -> None:
    provider = GeminiProvider(
        _SPEC, api_key="fake-gemini-key", http_client=_client(_success_handler)
    )
    response = await provider.complete(_request())
    assert response.succeeded
    assert response.output_text == "4"
    assert response.token_usage.input_tokens == 10
    assert response.token_usage.output_tokens == 3
    assert response.token_usage.cached_input_tokens == 1
    await provider.aclose()


async def test_request_hits_the_interactions_endpoint_with_header_auth() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        return _success_handler(request)

    provider = GeminiProvider(_SPEC, api_key="fake-gemini-key-xyz", http_client=_client(handler))
    await provider.complete(_request())
    assert str(captured["url"]).endswith("/interactions")
    assert "fake-gemini-key-xyz" not in str(captured["url"])
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers.get("x-goog-api-key") == "fake-gemini-key-xyz"


async def test_request_payload_is_stateless_with_content_items() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _success_handler(request)

    provider = GeminiProvider(_SPEC, api_key="fake-key", http_client=_client(handler))
    await provider.complete(_request(temperature=0.2, max_output_tokens=50))
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "gemini-test"
    assert body["store"] is False
    assert "previous_interaction_id" not in body
    assert body["input"] == [{"type": "text", "text": "what is 2+2?"}]
    assert body["generation_config"] == {"max_output_tokens": 50, "temperature": 0.2}


async def test_system_messages_become_system_instruction_string() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _success_handler(request)

    provider = GeminiProvider(_SPEC, api_key="fake-key", http_client=_client(handler))
    await provider.complete(
        _request(
            messages=[
                Message(role=Role.SYSTEM, content="be terse"),
                Message(role=Role.USER, content="what is 2+2?"),
            ]
        )
    )
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["system_instruction"] == "be terse"
    assert body["input"] == [{"type": "text", "text": "what is 2+2?"}]


async def test_prior_assistant_turn_replays_as_model_output_step() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _success_handler(request)

    provider = GeminiProvider(_SPEC, api_key="fake-key", http_client=_client(handler))
    await provider.complete(
        _request(
            messages=[
                Message(role=Role.USER, content="hi"),
                Message(role=Role.ASSISTANT, content="hello"),
                Message(role=Role.USER, content="what is 2+2?"),
            ]
        )
    )
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["input"][1] == {
        "type": "model_output",
        "content": [{"type": "text", "text": "hello"}],
    }


async def test_tool_result_message_becomes_function_result_step() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _success_handler(request)

    provider = GeminiProvider(_SPEC, api_key="fake-key", http_client=_client(handler))
    await provider.complete(
        _request(
            messages=[
                Message(role=Role.USER, content="what is 2+2?"),
                Message(role=Role.TOOL, content="4", tool_call_id="call_1", name="calculator"),
            ]
        )
    )
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["input"][1] == {
        "type": "function_result",
        "call_id": "call_1",
        "name": "calculator",
        "result": [{"type": "text", "text": "4"}],
    }


async def test_tool_schema_is_flat_function_type() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _success_handler(request)

    provider = GeminiProvider(_SPEC, api_key="fake-key", http_client=_client(handler))
    await provider.complete(
        _request(tools=[ToolSpec(name="calculator", description="adds", parameters_schema={})])
    )
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["tools"] == [
        {"type": "function", "name": "calculator", "description": "adds", "parameters": {}}
    ]


async def test_function_call_step_is_parsed_as_tool_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "v1_1",
                "status": "completed",
                "model": "gemini-test",
                "steps": [
                    {
                        "type": "function_call",
                        "id": "call_123",
                        "name": "calculator",
                        "arguments": {"a": 2, "b": 2},
                    }
                ],
                "usage": {"total_input_tokens": 5, "total_output_tokens": 5, "total_tokens": 10},
            },
        )

    provider = GeminiProvider(_SPEC, api_key="fake-key", http_client=_client(handler))
    response = await provider.complete(
        _request(tools=[ToolSpec(name="calculator", description="adds", parameters_schema={})])
    )
    assert response.finish_reason == FinishReason.TOOL_CALLS
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "call_123"
    assert response.tool_calls[0].name == "calculator"
    assert response.tool_calls[0].arguments == {"a": 2, "b": 2}


async def test_thought_steps_are_preserved_in_raw_metadata_not_dropped_silently() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "v1_1",
                "status": "completed",
                "model": "gemini-test",
                "steps": [
                    {"type": "thought", "signature": "abc123"},
                    {"type": "model_output", "content": [{"type": "text", "text": "4"}]},
                ],
                "usage": {"total_input_tokens": 5, "total_output_tokens": 5, "total_tokens": 10},
            },
        )

    provider = GeminiProvider(_SPEC, api_key="fake-key", http_client=_client(handler))
    response = await provider.complete(_request())
    assert response.output_text == "4"
    assert response.raw_provider_metadata is not None
    assert response.raw_provider_metadata["steps"][0]["type"] == "thought"


async def test_incomplete_status_maps_to_length_finish_reason() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "v1_1",
                "status": "incomplete",
                "model": "gemini-test",
                "steps": [{"type": "model_output", "content": [{"type": "text", "text": "trunc"}]}],
                "usage": {"total_input_tokens": 5, "total_output_tokens": 5, "total_tokens": 10},
            },
        )

    provider = GeminiProvider(_SPEC, api_key="fake-key", http_client=_client(handler))
    response = await provider.complete(_request())
    assert response.finish_reason == FinishReason.LENGTH


async def test_in_band_failed_status_maps_to_non_retryable_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "v1_1",
                "status": "failed",
                "model": "gemini-test",
                "errors": [{"code": "SOME_ERROR", "message": "generation failed"}],
            },
        )

    provider = GeminiProvider(_SPEC, api_key="fake-key", http_client=_client(handler))
    response = await provider.complete(_request())
    assert not response.succeeded
    assert response.error is not None
    assert response.error.message == "generation failed"
    assert response.error.retryable is False


async def test_rate_limit_status_maps_to_retryable_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={
                "error": {"code": 429, "message": "quota exceeded", "status": "RESOURCE_EXHAUSTED"}
            },
        )

    provider = GeminiProvider(_SPEC, api_key="fake-key", http_client=_client(handler))
    response = await provider.complete(_request())
    assert response.error is not None
    assert response.error.category.value == "rate_limit"
    assert response.error.retryable is True
    assert response.error.message == "quota exceeded"


async def test_server_error_status_maps_to_retryable_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "internal error"}})

    provider = GeminiProvider(_SPEC, api_key="fake-key", http_client=_client(handler))
    response = await provider.complete(_request())
    assert response.error is not None
    assert response.error.category.value == "provider_failure"
    assert response.error.retryable is True


async def test_timeout_maps_to_retryable_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    provider = GeminiProvider(_SPEC, api_key="fake-key", http_client=_client(handler))
    response = await provider.complete(_request())
    assert response.error is not None
    assert response.error.category.value == "timeout"
    assert response.error.retryable is True


async def test_missing_api_key_raises_config_error() -> None:
    spec = ProviderSpec(
        name="gemini", kind=ProviderKind.GEMINI, api_key_env_var="DOES_NOT_EXIST_XYZ"
    )
    with pytest.raises(ProviderConfigError):
        GeminiProvider(spec)


async def test_response_never_contains_the_api_key() -> None:
    provider = GeminiProvider(
        _SPEC, api_key="super-secret-gemini-key", http_client=_client(_success_handler)
    )
    response = await provider.complete(_request())
    serialized = response.model_dump_json()
    assert "super-secret-gemini-key" not in serialized
    await provider.aclose()
