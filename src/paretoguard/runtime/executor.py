"""Async execution engine: bounded concurrency, timeout, retry, and budget guards.

This is the only place that turns a Provider's single-attempt result into a final
InferenceResponse via retries — providers themselves only ever execute one attempt
(see `paretoguard.providers.base.Provider`).
"""

import asyncio
import time
from collections.abc import Callable

from paretoguard.core.models import (
    ErrorInfo,
    FailureCategory,
    FinishReason,
    InferenceRequest,
    InferenceResponse,
    LatencyRecord,
    TokenUsage,
    TraceEvent,
    TraceEventType,
)
from paretoguard.providers.base import Provider
from paretoguard.runtime.budget import BudgetExceededError, BudgetGuard
from paretoguard.runtime.retry import RetryPolicy


def _empty_error_response(
    request: InferenceRequest, provider_name: str, error: ErrorInfo, elapsed_ms: float
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


class Runtime:
    """Executes `InferenceRequest`s against a `Provider` with bounded concurrency,
    a per-attempt timeout, retry-with-backoff for retryable failures, and an
    optional budget guard.

    `on_trace_event` is the sole extensibility hook: storage and telemetry
    subscribe to the emitted `TraceEvent`s independently. The runtime does not
    import either module, keeping the dependency direction one-way.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        retry_policy: RetryPolicy | None = None,
        budget_guard: BudgetGuard | None = None,
        max_concurrency: int = 4,
        timeout_s: float = 60.0,
        on_trace_event: Callable[[TraceEvent], None] | None = None,
        run_id: str | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        self._provider = provider
        self._retry_policy = retry_policy or RetryPolicy()
        self._budget_guard = budget_guard
        self._timeout_s = timeout_s
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._on_trace_event = on_trace_event
        self._run_id = run_id

    async def run(self, request: InferenceRequest) -> InferenceResponse:
        async with self._semaphore:
            self._emit(
                TraceEvent(
                    run_id=self._run_id,
                    request_id=request.request_id,
                    event_type=TraceEventType.REQUEST_STARTED,
                    payload={"provider": self._provider.name, "model": request.model},
                )
            )

            response = await self._run_attempts(request)

            self._emit(
                TraceEvent(
                    run_id=self._run_id,
                    request_id=request.request_id,
                    event_type=TraceEventType.REQUEST_COMPLETED,
                    payload={"succeeded": response.succeeded},
                )
            )
            return response

    async def _run_attempts(self, request: InferenceRequest) -> InferenceResponse:
        response: InferenceResponse | None = None
        for attempt in range(self._retry_policy.max_attempts):
            if self._budget_guard is not None:
                try:
                    await self._budget_guard.check_before_call()
                except BudgetExceededError as exc:
                    return _empty_error_response(
                        request,
                        self._provider.name,
                        ErrorInfo(
                            category=FailureCategory.BUDGET_EXCEEDED,
                            message=str(exc),
                            retryable=False,
                        ),
                        0.0,
                    )

            response = await self._attempt(request)

            if self._budget_guard is not None:
                cost = response.cost.total_cost_usd if response.cost is not None else 0.0
                await self._budget_guard.record_call(cost)

            self._emit_provider_call_event(request, response, attempt)

            if response.succeeded or not self._is_retryable(response):
                return response

            if attempt < self._retry_policy.max_attempts - 1:
                await asyncio.sleep(self._retry_policy.delay_for(attempt))

        assert response is not None  # loop runs >= 1 time (RetryPolicy enforces max_attempts >= 1)
        return response

    async def _attempt(self, request: InferenceRequest) -> InferenceResponse:
        start = time.perf_counter()
        try:
            return await asyncio.wait_for(self._provider.complete(request), timeout=self._timeout_s)
        except TimeoutError:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return _empty_error_response(
                request,
                self._provider.name,
                ErrorInfo(
                    category=FailureCategory.TIMEOUT,
                    message=f"runtime timeout after {self._timeout_s}s",
                    retryable=True,
                ),
                elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return _empty_error_response(
                request,
                self._provider.name,
                ErrorInfo(
                    category=FailureCategory.TRANSPORT_FAILURE,
                    message=f"unexpected exception from provider: {exc}",
                    retryable=True,
                ),
                elapsed_ms,
            )

    @staticmethod
    def _is_retryable(response: InferenceResponse) -> bool:
        return response.error is not None and response.error.retryable

    def _emit(self, event: TraceEvent) -> None:
        if self._on_trace_event is not None:
            self._on_trace_event(event)

    def _emit_provider_call_event(
        self, request: InferenceRequest, response: InferenceResponse, attempt: int
    ) -> None:
        self._emit(
            TraceEvent(
                run_id=self._run_id,
                request_id=request.request_id,
                event_type=TraceEventType.PROVIDER_CALL,
                payload={
                    "provider": response.provider,
                    "model": response.model,
                    "attempt": attempt,
                    "succeeded": response.succeeded,
                    "finish_reason": response.finish_reason.value,
                    "error_category": response.error.category.value if response.error else None,
                    "latency_ms": response.latency.total_latency_ms,
                    "total_tokens": response.token_usage.total_tokens,
                },
            )
        )
