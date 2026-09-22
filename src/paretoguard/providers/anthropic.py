"""Anthropic adapter: talks to the Messages REST API directly over httpx.

Request/response shapes follow Anthropic's documented Messages API. Provider APIs
evolve — verify field names against current Anthropic docs before relying on this
in production (see docs/PROVIDER_COMPATIBILITY.md). Tested entirely via
`httpx.MockTransport`; never calls the real API in tests.

Authenticates with `Authorization: Bearer <key>`, the header platform.claude.com's
authentication docs currently document as the recommended method (verified
2026-09-22); the legacy `x-api-key` header still works but isn't used here.
"""

import time
from typing import Any

import httpx

from paretoguard.core.models import (
    ErrorInfo,
    FinishReason,
    InferenceRequest,
    InferenceResponse,
    LatencyRecord,
    Message,
    Role,
    TokenUsage,
    ToolCall,
    ToolSpec,
)
from paretoguard.core.models.provider import ProviderSpec
from paretoguard.providers.base import Provider
from paretoguard.providers.http import call_json_endpoint, resolve_api_key

DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 1024

_STOP_REASON_MAP = {
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "max_tokens": FinishReason.LENGTH,
    "tool_use": FinishReason.TOOL_CALLS,
}


class AnthropicProvider(Provider):
    """Adapter for the Anthropic Messages API."""

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
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        url = f"{self._base_url}/messages"

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
    system_messages = [m.content for m in request.messages if m.role == Role.SYSTEM]
    conversation = [m for m in request.messages if m.role != Role.SYSTEM]

    payload: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_output_tokens or DEFAULT_MAX_TOKENS,
        "messages": [_message_to_anthropic(m) for m in conversation],
    }
    if system_messages:
        payload["system"] = "\n\n".join(system_messages)
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.tools:
        payload["tools"] = [_tool_to_anthropic(t) for t in request.tools]
    return payload


def _message_to_anthropic(message: Message) -> dict[str, Any]:
    if message.role == Role.TOOL:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.content,
                }
            ],
        }
    role = "assistant" if message.role == Role.ASSISTANT else "user"
    return {"role": role, "content": message.content}


def _tool_to_anthropic(tool: ToolSpec) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters_schema,
    }


def _parse_response(
    provider_name: str, request: InferenceRequest, body: dict[str, Any], elapsed_ms: float
) -> InferenceResponse:
    content_blocks = body.get("content") or []
    text_parts = [b.get("text", "") for b in content_blocks if b.get("type") == "text"]
    tool_calls = [
        ToolCall(id=b.get("id", ""), name=b.get("name", ""), arguments=b.get("input") or {})
        for b in content_blocks
        if b.get("type") == "tool_use"
    ]

    finish_reason = _STOP_REASON_MAP.get(body.get("stop_reason", ""), FinishReason.STOP)

    usage_raw = body.get("usage") or {}
    usage = TokenUsage(
        input_tokens=usage_raw.get("input_tokens", 0),
        output_tokens=usage_raw.get("output_tokens", 0),
        cached_input_tokens=usage_raw.get("cache_read_input_tokens"),
    )

    return InferenceResponse(
        request_id=request.request_id,
        provider=provider_name,
        model=body.get("model", request.model),
        output_text="\n".join(text_parts) if text_parts else None,
        finish_reason=finish_reason,
        tool_calls=tool_calls,
        token_usage=usage,
        latency=LatencyRecord(total_latency_ms=elapsed_ms),
        raw_provider_metadata={"id": body.get("id")} if body.get("id") else None,
    )


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
