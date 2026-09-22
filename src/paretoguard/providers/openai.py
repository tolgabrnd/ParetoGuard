"""OpenAI adapter: talks to the Chat Completions REST API directly over httpx.

Request/response shapes follow OpenAI's documented Chat Completions API. Provider
APIs evolve — verify field names against current OpenAI docs before relying on
this in production (see docs/LIMITATIONS.md). Tested entirely via
`httpx.MockTransport`; never calls the real API in tests.
"""

import json
import time
from typing import Any

import httpx

from paretoguard.core.models import (
    ErrorInfo,
    FailureCategory,
    FinishReason,
    InferenceRequest,
    InferenceResponse,
    LatencyRecord,
    Message,
    TokenUsage,
    ToolCall,
    ToolSpec,
)
from paretoguard.core.models.provider import ProviderSpec
from paretoguard.providers.base import Provider
from paretoguard.providers.http import call_json_endpoint, resolve_api_key

DEFAULT_BASE_URL = "https://api.openai.com/v1"

_FINISH_REASON_MAP = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALLS,
    "content_filter": FinishReason.CONTENT_FILTER,
}


class OpenAIProvider(Provider):
    """Adapter for OpenAI-compatible Chat Completions endpoints."""

    def __init__(
        self,
        spec: ProviderSpec,
        *,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = spec.name
        self._spec = spec
        self._api_key = api_key or resolve_api_key(spec)
        self._base_url = spec.base_url or DEFAULT_BASE_URL
        self._client = http_client or httpx.AsyncClient()
        self._owns_client = http_client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def complete(self, request: InferenceRequest) -> InferenceResponse:
        start = time.perf_counter()
        payload = _build_payload(request)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._base_url}/chat/completions"

        body, error = await call_json_endpoint(
            self._client,
            url,
            headers=headers,
            json_body=payload,
            timeout_s=self._spec.timeout_s,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        if error is not None:
            return _error_response(self.name, request, error, elapsed_ms)
        return _parse_response(self.name, request, body or {}, elapsed_ms)


def _build_payload(request: InferenceRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": request.model,
        "messages": [_message_to_openai(m) for m in request.messages],
    }
    if request.max_output_tokens is not None:
        payload["max_tokens"] = request.max_output_tokens
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.tools:
        payload["tools"] = [_tool_to_openai(t) for t in request.tools]
    if request.structured_output_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": request.structured_output_schema,
                "strict": True,
            },
        }
    return payload


def _message_to_openai(message: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": message.role.value, "content": message.content}
    if message.tool_call_id is not None:
        out["tool_call_id"] = message.tool_call_id
    if message.name is not None:
        out["name"] = message.name
    return out


def _tool_to_openai(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters_schema,
        },
    }


def _parse_response(
    provider_name: str, request: InferenceRequest, body: dict[str, Any], elapsed_ms: float
) -> InferenceResponse:
    choices = body.get("choices") or []
    if not choices:
        return _error_response(
            provider_name,
            request,
            ErrorInfo(
                category=FailureCategory.SCHEMA_FAILURE,
                message="response had no choices",
                retryable=False,
            ),
            elapsed_ms,
        )
    choice = choices[0]
    message = choice.get("message") or {}
    finish_reason = _FINISH_REASON_MAP.get(choice.get("finish_reason", ""), FinishReason.STOP)

    tool_calls = [
        ToolCall(
            id=tc["id"],
            name=tc["function"]["name"],
            arguments=_safe_json_loads(tc["function"].get("arguments", "{}")),
        )
        for tc in message.get("tool_calls") or []
    ]

    usage_raw = body.get("usage") or {}
    usage = TokenUsage(
        input_tokens=usage_raw.get("prompt_tokens", 0),
        output_tokens=usage_raw.get("completion_tokens", 0),
        cached_input_tokens=(usage_raw.get("prompt_tokens_details") or {}).get("cached_tokens"),
    )

    return InferenceResponse(
        request_id=request.request_id,
        provider=provider_name,
        model=body.get("model", request.model),
        output_text=message.get("content"),
        finish_reason=finish_reason,
        tool_calls=tool_calls,
        token_usage=usage,
        latency=LatencyRecord(total_latency_ms=elapsed_ms),
        raw_provider_metadata={"id": body.get("id")} if body.get("id") else None,
    )


def _safe_json_loads(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _error_response(
    provider_name: str, request: InferenceRequest, error: ErrorInfo, elapsed_ms: float
) -> InferenceResponse:
    return InferenceResponse(
        request_id=request.request_id,
        provider=provider_name,
        model=request.model,
        finish_reason=FinishReason.ERROR,
        token_usage=TokenUsage(input_tokens=0, output_tokens=0),
        latency=LatencyRecord(total_latency_ms=elapsed_ms),
        error=error,
    )
