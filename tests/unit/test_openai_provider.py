"""Unit tests for the OpenAI adapter (Responses API), entirely via httpx.MockTransport.

No network calls, no SDK, no API key required — the whole point of adapters being
REST-based and injectable is that they're testable this way.
"""

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
from paretoguard.providers.http import ProviderConfigError
from paretoguard.providers.openai import OpenAIProvider

_SPEC = ProviderSpec(name="openai", kind=ProviderKind.OPENAI, api_key_env_var="OPENAI_API_KEY")


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _request(**kwargs: object) -> InferenceRequest:
    defaults: dict[str, object] = {
        "provider": "openai",
        "model": "gpt-test",
        "messages": [Message(role=Role.USER, content="what is 2+2?")],
    }
    defaults.update(kwargs)
    return InferenceRequest(**defaults)  # type: ignore[arg-type]


def _success_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "resp_abc123",
            "model": "gpt-test",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "4"}],
                }
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 3,
                "total_tokens": 13,
                "input_tokens_details": {"cached_tokens": 2},
            },
        },
    )


async def test_successful_completion() -> None:
    provider = OpenAIProvider(
        _SPEC, api_key="sk-fake-test-key", http_client=_client(_success_handler)
    )
    response = await provider.complete(_request())
    assert response.succeeded
    assert response.output_text == "4"
    assert response.token_usage.input_tokens == 10
    assert response.token_usage.output_tokens == 3
    assert response.token_usage.cached_input_tokens == 2
    await provider.aclose()


async def test_request_hits_the_responses_endpoint_with_bearer_auth() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        return _success_handler(request)

    provider = OpenAIProvider(_SPEC, api_key="sk-fake-test-key", http_client=_client(handler))
    await provider.complete(_request())
    assert str(captured["url"]).endswith("/responses")
    assert captured["auth"] == "Bearer sk-fake-test-key"


async def test_request_payload_uses_input_and_instructions_not_messages() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _success_handler(request)

    provider = OpenAIProvider(_SPEC, api_key="sk-fake-test-key", http_client=_client(handler))
    await provider.complete(
        _request(
            messages=[
                Message(role=Role.SYSTEM, content="be terse"),
                Message(role=Role.USER, content="what is 2+2?"),
            ],
            temperature=0.2,
            max_output_tokens=50,
        )
    )
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "gpt-test"
    assert body["instructions"] == "be terse"
    assert body["input"] == [{"role": "user", "content": "what is 2+2?"}]
    assert body["temperature"] == 0.2
    assert body["max_output_tokens"] == 50
    assert "messages" not in body
    assert "max_tokens" not in body


async def test_tool_schema_is_flat_not_nested_under_function() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _success_handler(request)

    provider = OpenAIProvider(_SPEC, api_key="sk-fake-test-key", http_client=_client(handler))
    await provider.complete(
        _request(tools=[ToolSpec(name="calculator", description="adds", parameters_schema={})])
    )
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["tools"] == [
        {"type": "function", "name": "calculator", "description": "adds", "parameters": {}}
    ]


async def test_structured_output_uses_text_format_json_schema() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _success_handler(request)

    provider = OpenAIProvider(_SPEC, api_key="sk-fake-test-key", http_client=_client(handler))
    schema = {"type": "object", "properties": {"answer": {"type": "number"}}}
    await provider.complete(_request(structured_output_schema=schema))
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["text"] == {
        "format": {"type": "json_schema", "name": "response", "schema": schema, "strict": True}
    }


async def test_tool_call_output_items_are_parsed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "model": "gpt-test",
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "calculator",
                        "arguments": '{"a": 2, "b": 2}',
                    }
                ],
                "usage": {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
            },
        )

    provider = OpenAIProvider(_SPEC, api_key="sk-fake-test-key", http_client=_client(handler))
    response = await provider.complete(
        _request(tools=[ToolSpec(name="calculator", description="adds", parameters_schema={})])
    )
    assert response.finish_reason == FinishReason.TOOL_CALLS
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].name == "calculator"
    assert response.tool_calls[0].arguments == {"a": 2, "b": 2}


async def test_incomplete_status_maps_to_length_finish_reason() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "model": "gpt-test",
                "status": "incomplete",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "trunc"}],
                    }
                ],
                "usage": {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
            },
        )

    provider = OpenAIProvider(_SPEC, api_key="sk-fake-test-key", http_client=_client(handler))
    response = await provider.complete(_request())
    assert response.finish_reason == FinishReason.LENGTH


async def test_failed_status_maps_to_non_retryable_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "model": "gpt-test",
                "status": "failed",
                "error": {"message": "something went wrong"},
                "output": [],
            },
        )

    provider = OpenAIProvider(_SPEC, api_key="sk-fake-test-key", http_client=_client(handler))
    response = await provider.complete(_request())
    assert not response.succeeded
    assert response.error is not None
    assert response.error.message == "something went wrong"
    assert response.error.retryable is False


async def test_tool_result_message_becomes_function_call_output_item() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _success_handler(request)

    provider = OpenAIProvider(_SPEC, api_key="sk-fake-test-key", http_client=_client(handler))
    await provider.complete(
        _request(
            messages=[
                Message(role=Role.USER, content="what is 2+2?"),
                Message(role=Role.TOOL, content="4", tool_call_id="call_1"),
            ]
        )
    )
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["input"][1] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "4",
    }


async def test_rate_limit_status_maps_to_retryable_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    provider = OpenAIProvider(_SPEC, api_key="sk-fake-test-key", http_client=_client(handler))
    response = await provider.complete(_request())
    assert not response.succeeded
    assert response.error is not None
    assert response.error.category.value == "rate_limit"
    assert response.error.retryable is True
    assert response.error.message == "rate limited"


async def test_server_error_status_maps_to_retryable_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "internal error"}})

    provider = OpenAIProvider(_SPEC, api_key="sk-fake-test-key", http_client=_client(handler))
    response = await provider.complete(_request())
    assert response.error is not None
    assert response.error.category.value == "provider_failure"
    assert response.error.retryable is True


async def test_bad_request_status_maps_to_non_retryable_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "invalid request"}})

    provider = OpenAIProvider(_SPEC, api_key="sk-fake-test-key", http_client=_client(handler))
    response = await provider.complete(_request())
    assert response.error is not None
    assert response.error.retryable is False


async def test_timeout_maps_to_retryable_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    provider = OpenAIProvider(_SPEC, api_key="sk-fake-test-key", http_client=_client(handler))
    response = await provider.complete(_request())
    assert response.error is not None
    assert response.error.category.value == "timeout"
    assert response.error.retryable is True


async def test_missing_api_key_raises_config_error() -> None:
    spec = ProviderSpec(
        name="openai", kind=ProviderKind.OPENAI, api_key_env_var="DOES_NOT_EXIST_XYZ"
    )
    with pytest.raises(ProviderConfigError):
        OpenAIProvider(spec)


async def test_response_never_contains_the_api_key() -> None:
    provider = OpenAIProvider(
        _SPEC, api_key="sk-super-secret-value", http_client=_client(_success_handler)
    )
    response = await provider.complete(_request())
    serialized = response.model_dump_json()
    assert "sk-super-secret-value" not in serialized
    await provider.aclose()
