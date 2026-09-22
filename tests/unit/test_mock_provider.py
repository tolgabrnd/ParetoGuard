"""Unit tests for MockProvider's deterministic scenarios."""

import json

import pytest

from paretoguard.core.models import FinishReason, InferenceRequest, Message, Role
from paretoguard.providers.mock import SCENARIO_METADATA_KEY, MockProvider, MockScenario


def _request(scenario: MockScenario | None = None, **metadata: object) -> InferenceRequest:
    meta: dict[str, object] = dict(metadata)
    if scenario is not None:
        meta[SCENARIO_METADATA_KEY] = scenario.value
    return InferenceRequest(
        provider="mock",
        model="mock-strong",
        messages=[Message(role=Role.USER, content="what is 2+2?")],
        metadata=meta,
    )


async def test_success_scenario_is_the_default() -> None:
    provider = MockProvider()
    response = await provider.complete(_request())
    assert response.succeeded
    assert response.finish_reason == FinishReason.STOP
    assert response.output_text is not None
    assert response.token_usage.input_tokens > 0
    assert response.token_usage.output_tokens > 0


async def test_success_scenario_respects_custom_answer_text() -> None:
    provider = MockProvider()
    response = await provider.complete(
        _request(MockScenario.SUCCESS, mock_answer_text="the answer is 4")
    )
    assert response.output_text == "the answer is 4"


async def test_exact_json_scenario() -> None:
    provider = MockProvider()
    response = await provider.complete(
        _request(MockScenario.EXACT_JSON, mock_json_answer={"answer": 4})
    )
    assert response.succeeded
    assert response.structured_output == {"answer": 4}
    assert response.output_text is None


async def test_delayed_scenario_actually_waits() -> None:
    # A generous margin below the requested delay: this only needs to confirm the
    # scenario really sleeps (vs. being a no-op), not pin an exact millisecond
    # boundary that Windows timer-resolution jitter can occasionally undershoot.
    provider = MockProvider()
    response = await provider.complete(_request(MockScenario.DELAYED, mock_delay_ms=50))
    assert response.succeeded
    assert response.latency.total_latency_ms >= 35


async def test_timeout_scenario_returns_retryable_error() -> None:
    provider = MockProvider()
    response = await provider.complete(_request(MockScenario.TIMEOUT))
    assert not response.succeeded
    assert response.error is not None
    assert response.error.category.value == "timeout"
    assert response.error.retryable is True


async def test_rate_limit_scenario_returns_retryable_error() -> None:
    provider = MockProvider()
    response = await provider.complete(_request(MockScenario.RATE_LIMIT))
    assert response.error is not None
    assert response.error.category.value == "rate_limit"
    assert response.error.retryable is True


async def test_server_error_scenario_returns_retryable_error() -> None:
    provider = MockProvider()
    response = await provider.complete(_request(MockScenario.SERVER_ERROR))
    assert response.error is not None
    assert response.error.category.value == "provider_failure"
    assert response.error.retryable is True


async def test_malformed_json_scenario_is_not_an_error_response() -> None:
    provider = MockProvider()
    response = await provider.complete(_request(MockScenario.MALFORMED_JSON))
    assert response.succeeded
    assert response.output_text is not None
    with pytest.raises(ValueError):
        json.loads(response.output_text)


async def test_truncated_scenario_sets_length_finish_reason() -> None:
    provider = MockProvider()
    response = await provider.complete(
        _request(MockScenario.TRUNCATED, mock_answer_text="a full untruncated sentence")
    )
    assert response.finish_reason == FinishReason.LENGTH
    assert response.output_text is not None
    assert len(response.output_text) < len("a full untruncated sentence")


async def test_invalid_tool_call_scenario() -> None:
    provider = MockProvider()
    response = await provider.complete(_request(MockScenario.INVALID_TOOL_CALL))
    assert response.finish_reason == FinishReason.TOOL_CALLS
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "nonexistent_tool"


async def test_wrong_answer_scenario_is_deterministic() -> None:
    provider = MockProvider()
    r1 = await provider.complete(_request(MockScenario.WRONG_ANSWER))
    r2 = await provider.complete(_request(MockScenario.WRONG_ANSWER))
    assert r1.output_text == r2.output_text == "incorrect-answer"


async def test_flaky_scenario_fails_then_succeeds_with_explicit_count() -> None:
    provider = MockProvider()
    request = _request(MockScenario.FLAKY, mock_flaky_fail_attempts=2)
    first = await provider.complete(request)
    second = await provider.complete(request)
    third = await provider.complete(request)
    assert not first.succeeded
    assert not second.succeeded
    assert third.succeeded


async def test_flaky_scenario_is_deterministic_given_same_seed() -> None:
    request = _request(MockScenario.FLAKY)
    provider_a = MockProvider(seed=7)
    provider_b = MockProvider(seed=7)
    outcomes_a = [(await provider_a.complete(request)).succeeded for _ in range(4)]
    outcomes_b = [(await provider_b.complete(request)).succeeded for _ in range(4)]
    assert outcomes_a == outcomes_b


async def test_seed_participates_in_flaky_rng() -> None:
    from uuid import UUID

    request_id = UUID("12345678-1234-5678-1234-567812345678")
    provider_a = MockProvider(seed=1)
    provider_b = MockProvider(seed=2)
    # Fixed request_id and seed pair, so this is deterministic, not flaky: pins that
    # the configured seed actually feeds the RNG rather than being ignored.
    assert provider_a._rng_for(request_id).randint(0, 1000) != provider_b._rng_for(
        request_id
    ).randint(0, 1000)
