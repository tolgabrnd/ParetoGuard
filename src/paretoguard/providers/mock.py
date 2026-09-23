"""Deterministic mock provider.

Makes the whole repository usable offline: no network calls, no API keys, and
fully reproducible given the same seed. The scenario for a request is selected via
`request.metadata["mock_scenario"]` (see `MockScenario`); it defaults to SUCCESS.
"""

import asyncio
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import UUID

from paretoguard.core.models import (
    ErrorInfo,
    FailureCategory,
    FinishReason,
    InferenceRequest,
    InferenceResponse,
    LatencyRecord,
    TokenUsage,
    ToolCall,
)
from paretoguard.providers.base import Provider

SCENARIO_METADATA_KEY = "mock_scenario"


class MockScenario(StrEnum):
    """Deterministic behaviors MockProvider can be asked to produce."""

    SUCCESS = "success"
    EXACT_JSON = "exact_json"
    DELAYED = "delayed"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    MALFORMED_JSON = "malformed_json"
    TRUNCATED = "truncated"
    INVALID_TOOL_CALL = "invalid_tool_call"
    WRONG_ANSWER = "wrong_answer"
    FLAKY = "flaky"


@dataclass
class _Outcome:
    output_text: str | None = None
    structured_output: dict[str, Any] | None = None
    finish_reason: FinishReason = FinishReason.STOP
    tool_calls: list[ToolCall] = field(default_factory=list)
    error: ErrorInfo | None = None


class MockProvider(Provider):
    """A fully deterministic, seed-controlled provider used for offline development,
    testing, and the runtime's own test suite."""

    def __init__(self, name: str = "mock", *, seed: int = 0) -> None:
        self.name = name
        self._seed = seed
        self._flaky_attempts: dict[UUID, int] = defaultdict(int)

    def _rng_for(self, request_id: UUID) -> random.Random:
        return random.Random(f"{self._seed}:{request_id}")

    @staticmethod
    def _count_tokens(text: str) -> int:
        return max(1, len(text.split()))

    async def complete(self, request: InferenceRequest) -> InferenceResponse:
        start = time.perf_counter()
        scenario = MockScenario(request.metadata.get(SCENARIO_METADATA_KEY, MockScenario.SUCCESS))
        input_text = " ".join(m.content for m in request.messages)
        input_tokens = self._count_tokens(input_text)

        outcome = await self._resolve(scenario, request, input_text)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if outcome.error is not None:
            return InferenceResponse(
                request_id=request.request_id,
                provider=self.name,
                model=request.model,
                finish_reason=FinishReason.ERROR,
                token_usage=TokenUsage(input_tokens=input_tokens, output_tokens=0),
                latency=LatencyRecord(total_latency_ms=elapsed_ms),
                error=outcome.error,
            )

        output_repr = outcome.output_text or json.dumps(outcome.structured_output or {})
        output_tokens = self._count_tokens(output_repr)

        return InferenceResponse(
            request_id=request.request_id,
            provider=self.name,
            model=request.model,
            output_text=outcome.output_text,
            structured_output=outcome.structured_output,
            finish_reason=outcome.finish_reason,
            tool_calls=outcome.tool_calls,
            token_usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            latency=LatencyRecord(total_latency_ms=elapsed_ms),
        )

    async def _resolve(
        self, scenario: MockScenario, request: InferenceRequest, input_text: str
    ) -> _Outcome:
        if scenario == MockScenario.SUCCESS:
            tool_calls_meta = request.metadata.get("mock_tool_calls")
            if tool_calls_meta:
                tool_calls = [
                    ToolCall(
                        id=f"mock-call-{i}", name=tc["name"], arguments=tc.get("arguments", {})
                    )
                    for i, tc in enumerate(tool_calls_meta)
                ]
                return _Outcome(tool_calls=tool_calls, finish_reason=FinishReason.TOOL_CALLS)
            text = request.metadata.get("mock_answer_text", f"mock response to: {input_text[:50]}")
            return _Outcome(output_text=text)

        if scenario == MockScenario.EXACT_JSON:
            return _Outcome(
                structured_output=request.metadata.get("mock_json_answer", {}), output_text=None
            )

        if scenario == MockScenario.DELAYED:
            delay_s = float(request.metadata.get("mock_delay_ms", 100.0)) / 1000
            await asyncio.sleep(delay_s)
            text = request.metadata.get(
                "mock_answer_text", f"delayed response to: {input_text[:50]}"
            )
            return _Outcome(output_text=text)

        if scenario == MockScenario.TIMEOUT:
            return _Outcome(
                error=ErrorInfo(
                    category=FailureCategory.TIMEOUT,
                    message="mock provider simulated a timeout",
                    retryable=True,
                )
            )

        if scenario == MockScenario.RATE_LIMIT:
            return _Outcome(
                error=ErrorInfo(
                    category=FailureCategory.RATE_LIMIT,
                    message="mock provider simulated a rate limit",
                    retryable=True,
                )
            )

        if scenario == MockScenario.SERVER_ERROR:
            return _Outcome(
                error=ErrorInfo(
                    category=FailureCategory.PROVIDER_FAILURE,
                    message="mock provider simulated a server error",
                    retryable=True,
                )
            )

        if scenario == MockScenario.MALFORMED_JSON:
            return _Outcome(output_text='{"incomplete": tru', finish_reason=FinishReason.STOP)

        if scenario == MockScenario.TRUNCATED:
            text = request.metadata.get("mock_answer_text", f"mock response to: {input_text[:50]}")
            return _Outcome(
                output_text=text[: max(1, len(text) // 2)], finish_reason=FinishReason.LENGTH
            )

        if scenario == MockScenario.INVALID_TOOL_CALL:
            return _Outcome(
                tool_calls=[ToolCall(id="mock-tool-1", name="nonexistent_tool", arguments={})],
                finish_reason=FinishReason.TOOL_CALLS,
            )

        if scenario == MockScenario.WRONG_ANSWER:
            text = request.metadata.get("mock_wrong_answer", "incorrect-answer")
            return _Outcome(output_text=text)

        if scenario == MockScenario.FLAKY:
            return self._resolve_flaky(request, input_text)

        raise AssertionError(f"unhandled MockScenario: {scenario}")  # pragma: no cover

    def _resolve_flaky(self, request: InferenceRequest, input_text: str) -> _Outcome:
        explicit = request.metadata.get("mock_flaky_fail_attempts")
        fail_attempts = (
            int(explicit)
            if explicit is not None
            else self._rng_for(request.request_id).randint(0, 2)
        )
        attempt = self._flaky_attempts[request.request_id]
        self._flaky_attempts[request.request_id] += 1
        if attempt < fail_attempts:
            return _Outcome(
                error=ErrorInfo(
                    category=FailureCategory.PROVIDER_FAILURE,
                    message=f"mock provider simulated a transient failure (attempt {attempt})",
                    retryable=True,
                )
            )
        text = request.metadata.get("mock_answer_text", f"mock response to: {input_text[:50]}")
        return _Outcome(output_text=text)
