"""Unit tests for the OpenAI adapter, entirely via httpx.MockTransport.

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
            "id": "chatcmpl-abc123",
            "model": "gpt-test",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "4"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 2},
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


async def test_request_payload_includes_messages_and_model() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _success_handler(request)

    provider = OpenAIProvider(_SPEC, api_key="sk-fake-test-key", http_client=_client(handler))
    await provider.complete(_request(temperature=0.2, max_output_tokens=50))
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "gpt-test"
    assert body["messages"] == [{"role": "user", "content": "what is 2+2?"}]
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 50


async def test_tool_calls_are_parsed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": "gpt-test",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {
                                        "name": "calculator",
                                        "arguments": '{"a": 2, "b": 2}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5},
            },
        )

    provider = OpenAIProvider(_SPEC, api_key="sk-fake-test-key", http_client=_client(handler))
    response = await provider.complete(
        _request(tools=[ToolSpec(name="calculator", description="adds", parameters_schema={})])
    )
    assert response.finish_reason == FinishReason.TOOL_CALLS
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "calculator"
    assert response.tool_calls[0].arguments == {"a": 2, "b": 2}


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
