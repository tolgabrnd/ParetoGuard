"""OpenAI adapter: talks to the Responses API directly over httpx.

Uses `POST /v1/responses`, not the older Chat Completions endpoint
(`/v1/chat/completions`). As of this adapter's last verification (2026-09-22,
against developers.openai.com), Chat Completions is not deprecated and still
works, but OpenAI documents Responses as "recommended for all new projects" —
particularly agentic ones, which is what ParetoGuard's tool-use/agent evaluation
work needs. See docs/PROVIDER_COMPATIBILITY.md for the verification notes and
sources.

Tested entirely via `httpx.MockTransport`; never calls the real API in tests.
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
    Role,
    TokenUsage,
    ToolCall,
    ToolSpec,
)
from paretoguard.core.models.provider import ProviderSpec
from paretoguard.providers.base import Provider
from paretoguard.providers.http import call_json_endpoint, resolve_api_key

DEFAULT_BASE_URL = "https://api.openai.com/v1"

_STATUS_FINISH_REASON_MAP = {
    "completed": FinishReason.STOP,
    "incomplete": FinishReason.LENGTH,
}


class OpenAIProvider(Provider):
    """Adapter for the OpenAI Responses API (`POST /v1/responses`)."""

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
        url = f"{self._base_url}/responses"

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
        "input": [_message_to_input_item(m) for m in conversation],
    }
    if system_messages:
        payload["instructions"] = "\n\n".join(system_messages)
    if request.max_output_tokens is not None:
        payload["max_output_tokens"] = request.max_output_tokens
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.tools:
        payload["tools"] = [_tool_to_openai(t) for t in request.tools]
    if request.structured_output_schema is not None:
        payload["text"] = {
            "format": {
                "type": "json_schema",
                "name": "response",
                "schema": request.structured_output_schema,
                "strict": True,
            }
        }
    return payload


def _message_to_input_item(message: Message) -> dict[str, Any]:
    if message.role == Role.TOOL:
        return {
            "type": "function_call_output",
            "call_id": message.tool_call_id,
            "output": message.content,
        }
    return {"role": message.role.value, "content": message.content}


def _tool_to_openai(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters_schema,
    }


def _parse_response(
    provider_name: str, request: InferenceRequest, body: dict[str, Any], elapsed_ms: float
) -> InferenceResponse:
    status = body.get("status", "completed")

    if status == "failed":
        error_body = body.get("error") or {}
        return _error_response(
            provider_name,
            request,
            ErrorInfo(
                category=FailureCategory.PROVIDER_FAILURE,
                message=str(error_body.get("message", "response generation failed")),
                retryable=False,
            ),
            elapsed_ms,
        )

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for item in body.get("output") or []:
        item_type = item.get("type")
        if item_type == "message":
            for block in item.get("content") or []:
                if block.get("type") == "output_text":
                    text_parts.append(block.get("text", ""))
        elif item_type == "function_call":
            tool_calls.append(
                ToolCall(
                    id=item.get("call_id") or item.get("id", ""),
                    name=item.get("name", ""),
                    arguments=_safe_json_loads(item.get("arguments", "{}")),
                )
            )

    finish_reason = _STATUS_FINISH_REASON_MAP.get(status, FinishReason.STOP)
    if tool_calls:
        finish_reason = FinishReason.TOOL_CALLS

    usage_raw = body.get("usage") or {}
    usage = TokenUsage(
        input_tokens=usage_raw.get("input_tokens", 0),
        output_tokens=usage_raw.get("output_tokens", 0),
        cached_input_tokens=(usage_raw.get("input_tokens_details") or {}).get("cached_tokens"),
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
