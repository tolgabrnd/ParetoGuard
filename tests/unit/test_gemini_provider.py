"""Unit tests for the Gemini adapter, entirely via httpx.MockTransport."""

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
            "candidates": [
                {"content": {"parts": [{"text": "4"}]}, "finishReason": "STOP"},
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 3,
                "cachedContentTokenCount": 1,
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
    assert response.token_usage.cached_input_tokens == 1
    await provider.aclose()


async def test_api_key_sent_as_header_not_query_param() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        return _success_handler(request)

    provider = GeminiProvider(_SPEC, api_key="fake-gemini-key-xyz", http_client=_client(handler))
    await provider.complete(_request())
    assert "fake-gemini-key-xyz" not in str(captured["url"])
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers.get("x-goog-api-key") == "fake-gemini-key-xyz"


async def test_system_messages_become_system_instruction() -> None:
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
    assert body["systemInstruction"] == {"parts": [{"text": "be terse"}]}
    assert body["contents"] == [{"role": "user", "parts": [{"text": "what is 2+2?"}]}]


async def test_function_call_parts_are_parsed_as_tool_calls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"functionCall": {"name": "calculator", "args": {"a": 2, "b": 2}}}
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 5},
            },
        )

    provider = GeminiProvider(_SPEC, api_key="fake-key", http_client=_client(handler))
    response = await provider.complete(
        _request(tools=[ToolSpec(name="calculator", description="adds", parameters_schema={})])
    )
    assert response.finish_reason == FinishReason.TOOL_CALLS
    assert response.tool_calls[0].name == "calculator"
    assert response.tool_calls[0].arguments == {"a": 2, "b": 2}


async def test_no_candidates_maps_to_schema_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": []})

    provider = GeminiProvider(_SPEC, api_key="fake-key", http_client=_client(handler))
    response = await provider.complete(_request())
    assert response.error is not None
    assert response.error.category.value == "schema_failure"
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
    assert "super-secret-gemini-key" not in response.model_dump_json()
    await provider.aclose()
