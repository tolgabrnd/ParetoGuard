"""Unit tests for the Anthropic adapter, entirely via httpx.MockTransport."""

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
from paretoguard.providers.anthropic import AnthropicProvider
from paretoguard.providers.http import ProviderConfigError

_SPEC = ProviderSpec(
    name="anthropic", kind=ProviderKind.ANTHROPIC, api_key_env_var="ANTHROPIC_API_KEY"
)


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _request(**kwargs: object) -> InferenceRequest:
    defaults: dict[str, object] = {
        "provider": "anthropic",
        "model": "claude-test",
        "messages": [Message(role=Role.USER, content="what is 2+2?")],
    }
    defaults.update(kwargs)
    return InferenceRequest(**defaults)  # type: ignore[arg-type]


def _success_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_abc123",
            "model": "claude-test",
            "content": [{"type": "text", "text": "4"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 3, "cache_read_input_tokens": 1},
        },
    )


async def test_successful_completion() -> None:
    provider = AnthropicProvider(
        _SPEC, api_key="sk-ant-fake-test-key", http_client=_client(_success_handler)
    )
    response = await provider.complete(_request())
    assert response.succeeded
    assert response.output_text == "4"
    assert response.token_usage.input_tokens == 10
    assert response.token_usage.cached_input_tokens == 1
    await provider.aclose()


async def test_uses_bearer_auth_header_not_legacy_x_api_key() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return _success_handler(request)

    provider = AnthropicProvider(
        _SPEC, api_key="sk-ant-fake-bearer-key", http_client=_client(handler)
    )
    await provider.complete(_request())
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers.get("authorization") == "Bearer sk-ant-fake-bearer-key"
    assert "x-api-key" not in headers


async def test_system_messages_are_extracted_into_system_field() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _success_handler(request)

    provider = AnthropicProvider(_SPEC, api_key="sk-ant-fake", http_client=_client(handler))
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
    assert body["system"] == "be terse"
    assert body["messages"] == [{"role": "user", "content": "what is 2+2?"}]


async def test_max_tokens_defaults_when_not_set_on_request() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _success_handler(request)

    provider = AnthropicProvider(_SPEC, api_key="sk-ant-fake", http_client=_client(handler))
    await provider.complete(_request())
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["max_tokens"] == 1024


async def test_tool_use_blocks_are_parsed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-test",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "calculator",
                        "input": {"a": 2, "b": 2},
                    }
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 5, "output_tokens": 5},
            },
        )

    provider = AnthropicProvider(_SPEC, api_key="sk-ant-fake", http_client=_client(handler))
    response = await provider.complete(
        _request(tools=[ToolSpec(name="calculator", description="adds", parameters_schema={})])
    )
    assert response.finish_reason == FinishReason.TOOL_CALLS
    assert response.tool_calls[0].name == "calculator"
    assert response.tool_calls[0].arguments == {"a": 2, "b": 2}


async def test_overloaded_5xx_status_maps_to_retryable_provider_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            529,
            json={"type": "error", "error": {"type": "overloaded_error", "message": "overloaded"}},
        )

    provider = AnthropicProvider(_SPEC, api_key="sk-ant-fake", http_client=_client(handler))
    response = await provider.complete(_request())
    assert response.error is not None
    assert response.error.category.value == "provider_failure"
    assert response.error.retryable is True
    assert response.error.message == "overloaded"


async def test_missing_api_key_raises_config_error() -> None:
    spec = ProviderSpec(
        name="anthropic", kind=ProviderKind.ANTHROPIC, api_key_env_var="DOES_NOT_EXIST_XYZ"
    )
    with pytest.raises(ProviderConfigError):
        AnthropicProvider(spec)


async def test_response_never_contains_the_api_key() -> None:
    provider = AnthropicProvider(
        _SPEC, api_key="sk-ant-super-secret-value", http_client=_client(_success_handler)
    )
    response = await provider.complete(_request())
    assert "sk-ant-super-secret-value" not in response.model_dump_json()
    await provider.aclose()
