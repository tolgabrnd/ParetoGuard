"""Gemini adapter: talks to the Interactions API directly over httpx.

Migrated from `generateContent` (2026-09-22 audit) after re-verifying Google's
current docs immediately before this change: the Interactions API has been GA
since June 2026 and is documented as the recommended/default API for new Gemini
projects; `generateContent` is explicitly labeled "(Legacy)". See
docs/PROVIDER_COMPATIBILITY.md for the full trade-off analysis, sources, and the
residual schema uncertainty noted below.

**Stateless by design.** This adapter always sends `"store": false` and never
sends `previous_interaction_id` — every call resends the full conversation via
`input`, exactly like the OpenAI and Anthropic adapters. Interactions' recommended
mode (server-stored history, referenced by ID) is a genuine capability of the
provider but is deliberately not used here: it would mean part of a request's
"state" lives only in Google's servers, which the generic `Provider` interface
(one `InferenceRequest` in, one `InferenceResponse` out, no adapter-held session)
has no way to represent without leaking Gemini-specific concepts into
`paretoguard.core.models`. Google's own docs confirm stateless mode is a fully
supported first-class option, not a workaround: `store: false` accepts the same
`function_call`/`function_result` step history a stateful conversation would,
just replayed in one request instead of referenced by ID.

**Known residual uncertainty (documented rather than guessed around):** the
`function_result` step's exact field set showed minor inconsistency across
official doc pages during verification (one example omitted `call_id`, a more
detailed one included it correlating to the originating `function_call`'s `id`).
This adapter sends `call_id`, matching the more detailed source. If Gemini
rejects or ignores that field, `docs/PROVIDER_COMPATIBILITY.md` documents this so
it's the first thing to check.

Tested entirely via `httpx.MockTransport`; never calls the real API in tests.
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

_STATUS_FINISH_REASON_MAP = {
    "completed": FinishReason.STOP,
    "incomplete": FinishReason.LENGTH,
}


class GeminiProvider(Provider):
    """Adapter for the Gemini Interactions API (`POST /v1beta/interactions`),
    used in stateless mode only (`store: false`, no `previous_interaction_id`)."""

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
        url = f"{self._base_url}/interactions"

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
        "store": False,
    }
    if system_messages:
        payload["system_instruction"] = "\n\n".join(system_messages)

    generation_config: dict[str, Any] = {}
    if request.max_output_tokens is not None:
        generation_config["max_output_tokens"] = request.max_output_tokens
    if request.temperature is not None:
        generation_config["temperature"] = request.temperature
    if generation_config:
        payload["generation_config"] = generation_config

    if request.tools:
        payload["tools"] = [_tool_to_gemini(t) for t in request.tools]
    return payload


def _message_to_input_item(message: Message) -> dict[str, Any]:
    """Map a normalized Message to an Interactions `input` item.

    Plain `Content` items (`{"type": "text", ...}`) carry no role field — the API
    treats bare input items as the caller's turn, so a prior *model* turn must be
    replayed as a `model_output` Step instead, or it would be misread as another
    user turn. Tool results replay as `function_result` steps.
    """
    if message.role == Role.TOOL:
        return {
            "type": "function_result",
            "call_id": message.tool_call_id,
            "name": message.name or "",
            "result": [{"type": "text", "text": message.content}],
        }
    if message.role == Role.ASSISTANT:
        return {"type": "model_output", "content": [{"type": "text", "text": message.content}]}
    return {"type": "text", "text": message.content}


def _tool_to_gemini(tool: ToolSpec) -> dict[str, Any]:
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
        errors = body.get("errors") or []
        message = (
            str(errors[0].get("message", "interaction failed"))
            if errors and isinstance(errors[0], dict)
            else "interaction failed"
        )
        return _error_response(
            provider_name,
            request,
            ErrorInfo(category=FailureCategory.PROVIDER_FAILURE, message=message, retryable=False),
            elapsed_ms,
        )

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for step in body.get("steps") or []:
        step_type = step.get("type")
        if step_type == "model_output":
            for block in step.get("content") or []:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
        elif step_type == "function_call":
            tool_calls.append(
                ToolCall(
                    id=step.get("id", ""),
                    name=step.get("name", ""),
                    arguments=step.get("arguments") or {},
                )
            )
        # Other step types (e.g. "thought") have no normalized equivalent — they
        # are not lost, though: the full raw `steps` array is preserved below in
        # raw_provider_metadata for callers that want them.

    finish_reason = _STATUS_FINISH_REASON_MAP.get(status, FinishReason.STOP)
    if tool_calls:
        finish_reason = FinishReason.TOOL_CALLS

    usage_raw = body.get("usage") or {}
    usage = TokenUsage(
        input_tokens=usage_raw.get("total_input_tokens", 0),
        output_tokens=usage_raw.get("total_output_tokens", 0),
        cached_input_tokens=usage_raw.get("total_cached_tokens"),
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
        raw_provider_metadata={"id": body.get("id"), "steps": body.get("steps") or []},
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
