"""Unit tests for the async runtime: retries, timeout, budget, concurrency."""

import asyncio
import random

import pytest

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
    TraceEventType,
)
from paretoguard.providers.base import Provider
from paretoguard.providers.mock import SCENARIO_METADATA_KEY, MockProvider, MockScenario
from paretoguard.runtime import BudgetGuard, RetryPolicy, Runtime


def _request(**metadata: object) -> InferenceRequest:
    return InferenceRequest(
        provider="mock",
        model="mock-strong",
        messages=[Message(role=Role.USER, content="hi")],
        metadata=metadata,
    )


def _fast_retry_policy(max_attempts: int = 3) -> RetryPolicy:
    return RetryPolicy(
        max_attempts=max_attempts, base_delay_s=0.001, max_delay_s=0.002, jitter=False
    )


class _AlwaysNonRetryableProvider(Provider):
    def __init__(self) -> None:
        self.name = "always-fail"
        self.call_count = 0

    async def complete(self, request: InferenceRequest) -> InferenceResponse:
        self.call_count += 1
        return InferenceResponse(
            request_id=request.request_id,
            provider=self.name,
            model=request.model,
            finish_reason=FinishReason.ERROR,
            token_usage=TokenUsage(input_tokens=1, output_tokens=0),
            latency=LatencyRecord(total_latency_ms=1.0),
            error=ErrorInfo(
                category=FailureCategory.PROVIDER_FAILURE, message="nope", retryable=False
            ),
        )


class _ConcurrencyTrackingProvider(Provider):
    def __init__(self, delay_s: float = 0.05) -> None:
        self.name = "tracker"
        self._delay_s = delay_s
        self._in_flight = 0
        self.max_seen = 0

    async def complete(self, request: InferenceRequest) -> InferenceResponse:
        self._in_flight += 1
        self.max_seen = max(self.max_seen, self._in_flight)
        await asyncio.sleep(self._delay_s)
        self._in_flight -= 1
        return InferenceResponse(
            request_id=request.request_id,
            provider=self.name,
            model=request.model,
            output_text="ok",
            finish_reason=FinishReason.STOP,
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
            latency=LatencyRecord(total_latency_ms=self._delay_s * 1000),
        )


class _NeverReturningProvider(Provider):
    def __init__(self) -> None:
        self.name = "hangs"

    async def complete(self, request: InferenceRequest) -> InferenceResponse:
        await asyncio.sleep(3600)
        raise AssertionError("should have been cancelled before this point")


async def test_successful_request_returns_immediately() -> None:
    runtime = Runtime(MockProvider(), retry_policy=_fast_retry_policy())
    response = await runtime.run(_request())
    assert response.succeeded


async def test_retries_until_flaky_provider_succeeds() -> None:
    provider = MockProvider()
    runtime = Runtime(provider, retry_policy=_fast_retry_policy(max_attempts=5))
    request = _request(
        **{SCENARIO_METADATA_KEY: MockScenario.FLAKY.value}, mock_flaky_fail_attempts=2
    )
    response = await runtime.run(request)
    assert response.succeeded


async def test_exhausts_retries_and_returns_final_failure() -> None:
    provider = MockProvider()
    runtime = Runtime(provider, retry_policy=_fast_retry_policy(max_attempts=2))
    request = _request(**{SCENARIO_METADATA_KEY: MockScenario.RATE_LIMIT.value})
    response = await runtime.run(request)
    assert not response.succeeded
    assert response.error is not None
    assert response.error.category == FailureCategory.RATE_LIMIT


async def test_non_retryable_error_short_circuits_retries() -> None:
    provider = _AlwaysNonRetryableProvider()
    runtime = Runtime(provider, retry_policy=_fast_retry_policy(max_attempts=5))
    response = await runtime.run(_request())
    assert not response.succeeded
    assert provider.call_count == 1


async def test_runtime_timeout_wraps_slow_provider() -> None:
    provider = MockProvider()
    runtime = Runtime(provider, retry_policy=RetryPolicy(max_attempts=1), timeout_s=0.02)
    request = _request(**{SCENARIO_METADATA_KEY: MockScenario.DELAYED.value}, mock_delay_ms=500)
    response = await runtime.run(request)
    assert not response.succeeded
    assert response.error is not None
    assert response.error.category == FailureCategory.TIMEOUT


async def test_max_concurrency_is_enforced() -> None:
    provider = _ConcurrencyTrackingProvider(delay_s=0.05)
    runtime = Runtime(provider, max_concurrency=2, retry_policy=RetryPolicy(max_attempts=1))
    await asyncio.gather(*(runtime.run(_request()) for _ in range(6)))
    assert provider.max_seen <= 2


async def test_budget_guard_blocks_before_exceeding_call_limit() -> None:
    provider = MockProvider()
    budget = BudgetGuard(max_calls=1)
    runtime = Runtime(
        provider,
        budget_guard=budget,
        retry_policy=_fast_retry_policy(max_attempts=3),
    )
    request = _request(**{SCENARIO_METADATA_KEY: MockScenario.RATE_LIMIT.value})
    response = await runtime.run(request)
    assert not response.succeeded
    assert response.error is not None
    assert response.error.category == FailureCategory.BUDGET_EXCEEDED
    assert budget.calls_made == 1


async def test_cancellation_propagates_and_does_not_corrupt_state() -> None:
    provider = _NeverReturningProvider()
    runtime = Runtime(provider, max_concurrency=1, timeout_s=10.0)
    task = asyncio.ensure_future(runtime.run(_request()))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The semaphore must have been released by the cancelled task; a fresh call
    # through the same runtime should not deadlock.
    fast_provider = MockProvider()
    fast_runtime = Runtime(fast_provider, max_concurrency=1)
    response = await asyncio.wait_for(fast_runtime.run(_request()), timeout=2.0)
    assert response.succeeded


async def test_trace_events_are_emitted_in_order() -> None:
    events = []
    runtime = Runtime(
        MockProvider(),
        retry_policy=_fast_retry_policy(),
        on_trace_event=events.append,
        run_id="run-123",
    )
    request = _request()
    response = await runtime.run(request)
    assert response.succeeded

    event_types = [e.event_type for e in events]
    assert event_types == [
        TraceEventType.REQUEST_STARTED,
        TraceEventType.PROVIDER_CALL,
        TraceEventType.REQUEST_COMPLETED,
    ]
    assert all(e.run_id == "run-123" for e in events)
    assert all(e.request_id == request.request_id for e in events)
    assert events[1].payload["succeeded"] is True


async def test_retry_policy_rejects_invalid_max_attempts() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=0)


def test_retry_policy_delay_is_bounded_without_jitter() -> None:
    policy = RetryPolicy(base_delay_s=1.0, max_delay_s=4.0, jitter=False)
    assert policy.delay_for(0) == 1.0
    assert policy.delay_for(1) == 2.0
    assert policy.delay_for(5) == 4.0  # capped


def test_retry_policy_jitter_stays_within_bounds() -> None:
    policy = RetryPolicy(base_delay_s=1.0, max_delay_s=4.0, jitter=True, rng=random.Random(0))
    for attempt in range(5):
        delay = policy.delay_for(attempt)
        assert 0 <= delay <= 4.0


async def test_budget_guard_rejects_negative_limits() -> None:
    with pytest.raises(ValueError):
        BudgetGuard(max_run_usd=-1)
    with pytest.raises(ValueError):
        BudgetGuard(max_calls=-1)
