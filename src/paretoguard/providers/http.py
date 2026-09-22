"""Shared HTTP-transport helpers for REST-based provider adapters.

Each adapter (openai.py, anthropic.py, gemini.py) translates its own request/
response JSON shape by calling its provider's REST API directly over httpx rather
than through an official SDK — this keeps every adapter testable with
`httpx.MockTransport` (no network, no SDK-specific mocking) and avoids coupling to
a fast-moving SDK's client API. Transport-level error classification and API-key
resolution are identical across providers, so they live here once instead of being
duplicated three times.
"""

import os
from typing import Any

import httpx

from paretoguard.core.models import ErrorInfo, FailureCategory
from paretoguard.core.models.provider import ProviderSpec


class ProviderConfigError(RuntimeError):
    """Raised when a provider adapter is misconfigured (e.g. a missing API key).

    This is a genuine misconfiguration the caller must fix, not a retryable
    provider failure — so it's raised rather than returned as ErrorInfo.
    """


def resolve_api_key(spec: ProviderSpec) -> str:
    """Read the API key from the environment variable named by the spec.

    Never accepts or returns a literal key from configuration — only the *name*
    of the environment variable holding it, so keys never end up in committed
    config or in a ProviderSpec that might get logged or serialized.
    """
    if spec.api_key_env_var is None:
        raise ProviderConfigError(f"provider {spec.name!r} has no api_key_env_var configured")
    value = os.environ.get(spec.api_key_env_var)
    if not value:
        raise ProviderConfigError(
            f"environment variable {spec.api_key_env_var!r} is not set for provider {spec.name!r}"
        )
    return value


async def call_json_endpoint(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str],
    json_body: dict[str, Any],
    timeout_s: float,
) -> tuple[dict[str, Any] | None, ErrorInfo | None]:
    """POST JSON and return `(parsed_body, None)` on success or `(None, ErrorInfo)`
    on any transport/HTTP-level failure.

    Never raises for ordinary provider failures — see
    `paretoguard.providers.base.Provider` for why. The returned dict/ErrorInfo
    never contains `headers` or the request URL, so a caller can't accidentally
    leak the Authorization header/API key into `raw_provider_metadata`.
    """
    try:
        response = await client.post(url, headers=headers, json=json_body, timeout=timeout_s)
    except httpx.TimeoutException:
        return None, ErrorInfo(
            category=FailureCategory.TIMEOUT, message="request timed out", retryable=True
        )
    except httpx.TransportError as exc:
        return None, ErrorInfo(
            category=FailureCategory.TRANSPORT_FAILURE, message=str(exc), retryable=True
        )

    if response.status_code == 429:
        return None, ErrorInfo(
            category=FailureCategory.RATE_LIMIT,
            message=_error_message(response),
            retryable=True,
            provider_error_code=str(response.status_code),
        )
    if response.status_code >= 500:
        return None, ErrorInfo(
            category=FailureCategory.PROVIDER_FAILURE,
            message=_error_message(response),
            retryable=True,
            provider_error_code=str(response.status_code),
        )
    if response.status_code >= 400:
        return None, ErrorInfo(
            category=FailureCategory.PROVIDER_FAILURE,
            message=_error_message(response),
            retryable=False,
            provider_error_code=str(response.status_code),
        )

    try:
        body = response.json()
    except ValueError:
        return None, ErrorInfo(
            category=FailureCategory.SCHEMA_FAILURE,
            message="response body was not valid JSON",
            retryable=False,
        )
    if not isinstance(body, dict):
        return None, ErrorInfo(
            category=FailureCategory.SCHEMA_FAILURE,
            message="response body was not a JSON object",
            retryable=False,
        )
    return body, None


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:500]
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and "message" in error:
            return str(error["message"])
        if "message" in body:
            return str(body["message"])
    return str(body)[:500]
