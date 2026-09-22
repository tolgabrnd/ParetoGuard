"""Gemini adapter: talks to the generateContent REST API directly over httpx.

Request/response shapes follow Google's documented Gemini generateContent API.
Provider APIs evolve — verify field names against current Gemini docs before
relying on this in production (see docs/LIMITATIONS.md). Tested entirely via
`httpx.MockTransport`; never calls the real API in tests.

The API key is sent via the `x-goog-api-key` header rather than Gemini's
alternative `?key=` query-string form, specifically so it never ends up in a
request URL that might get logged or echoed back anywhere.
"""

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
    Role,
    TokenUsage,
    ToolCall,
    ToolSpec,
)
from paretoguard.core.models.provider import ProviderSpec
from paretoguard.providers.base import Provider
from paretoguard.providers.http import call_json_endpoint, resolve_api_key

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

_FINISH_REASON_MAP = {
    "STOP": FinishReason.STOP,
    "MAX_TOKENS": FinishReason.LENGTH,
    "SAFETY": FinishReason.CONTENT_FILTER,
    "RECITATION": FinishReason.CONTENT_FILTER,
}


class GeminiProvider(Provider):
    """Adapter for the Gemini `models/{model}:generateContent` API."""

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
            "x-goog-api-key": self._api_key,
            "content-type": "application/json",
        }
        url = f"{self._base_url}/models/{request.model}:generateContent"

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

    payload: dict[str, Any] = {"contents": [_message_to_gemini(m) for m in conversation]}
    if system_messages:
        payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_messages)}]}

    generation_config: dict[str, Any] = {}
    if request.max_output_tokens is not None:
        generation_config["maxOutputTokens"] = request.max_output_tokens
    if request.temperature is not None:
        generation_config["temperature"] = request.temperature
    if generation_config:
        payload["generationConfig"] = generation_config

    if request.tools:
        payload["tools"] = [{"functionDeclarations": [_tool_to_gemini(t) for t in request.tools]}]
    return payload


def _message_to_gemini(message: Message) -> dict[str, Any]:
    role = "model" if message.role == Role.ASSISTANT else "user"
    return {"role": role, "parts": [{"text": message.content}]}


def _tool_to_gemini(tool: ToolSpec) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters_schema,
    }


def _parse_response(
    provider_name: str, request: InferenceRequest, body: dict[str, Any], elapsed_ms: float
) -> InferenceResponse:
    candidates = body.get("candidates") or []
    if not candidates:
        return _error_response(
            provider_name,
            request,
            ErrorInfo(
                category=FailureCategory.SCHEMA_FAILURE,
                message="response had no candidates",
                retryable=False,
            ),
            elapsed_ms,
        )

    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text_parts = [p["text"] for p in parts if "text" in p]
    tool_calls = [
        ToolCall(
            id=f"gemini-call-{i}",
            name=p["functionCall"]["name"],
            arguments=p["functionCall"].get("args") or {},
        )
        for i, p in enumerate(parts)
        if "functionCall" in p
    ]

    finish_reason = _FINISH_REASON_MAP.get(candidate.get("finishReason", ""), FinishReason.STOP)
    if tool_calls:
        finish_reason = FinishReason.TOOL_CALLS

    usage_raw = body.get("usageMetadata") or {}
    usage = TokenUsage(
        input_tokens=usage_raw.get("promptTokenCount", 0),
        output_tokens=usage_raw.get("candidatesTokenCount", 0),
        cached_input_tokens=usage_raw.get("cachedContentTokenCount"),
    )

    return InferenceResponse(
        request_id=request.request_id,
        provider=provider_name,
        model=request.model,
        output_text="\n".join(text_parts) if text_parts else None,
        finish_reason=finish_reason,
        tool_calls=tool_calls,
        token_usage=usage,
        latency=LatencyRecord(total_latency_ms=elapsed_ms),
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
