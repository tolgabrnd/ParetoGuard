"""Rolling per-(provider, model) health metrics.

Consumes the `TraceEvent`s emitted by `paretoguard.runtime.Runtime` via its
`on_trace_event` hook — this module has no dependency on `runtime`, only on the
`TraceEvent`/`FailureCategory` types they both share from `core.models`, which
keeps the dependency direction one-way (runtime -> emits events; telemetry ->
consumes them; see docs/ARCHITECTURE.md).
"""

import math
from collections import deque
from dataclasses import dataclass, field

from paretoguard.core.models import FailureCategory, InferenceResponse, TraceEvent, TraceEventType

_ERROR_CATEGORY_COUNTERS: dict[FailureCategory, str] = {
    FailureCategory.TIMEOUT: "timeout_count",
    FailureCategory.RATE_LIMIT: "rate_limit_count",
    FailureCategory.PROVIDER_FAILURE: "server_error_count",
}


@dataclass
class ModelHealth:
    """Rolling health snapshot for one (provider, model) pair."""

    provider: str
    model: str
    request_count: int = 0
    success_count: int = 0
    timeout_count: int = 0
    rate_limit_count: int = 0
    server_error_count: int = 0
    # Populated once paretoguard.recovery exists (Phase E) — None until then.
    recovery_rate: float | None = None

    @property
    def success_rate(self) -> float | None:
        if self.request_count == 0:
            return None
        return self.success_count / self.request_count


@dataclass
class _LatencyState:
    ema_ms: float | None = None
    samples: deque[float] = field(default_factory=lambda: deque(maxlen=500))


class HealthTracker:
    """Aggregates outcomes into rolling `ModelHealth` snapshots, EMA latency, and
    an approximate p95 latency over a bounded rolling window per (provider, model).
    """

    def __init__(self, *, ema_alpha: float = 0.2, latency_window: int = 500) -> None:
        if not 0 < ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1]")
        if latency_window < 1:
            raise ValueError("latency_window must be >= 1")
        self._ema_alpha = ema_alpha
        self._latency_window = latency_window
        self._health: dict[tuple[str, str], ModelHealth] = {}
        self._latency: dict[tuple[str, str], _LatencyState] = {}

    def record_outcome(
        self,
        provider: str,
        model: str,
        *,
        succeeded: bool,
        latency_ms: float,
        error_category: FailureCategory | None = None,
    ) -> None:
        key = (provider, model)
        health = self._health.setdefault(key, ModelHealth(provider=provider, model=model))
        health.request_count += 1
        if succeeded:
            health.success_count += 1
        elif error_category is not None and error_category in _ERROR_CATEGORY_COUNTERS:
            attr = _ERROR_CATEGORY_COUNTERS[error_category]
            setattr(health, attr, getattr(health, attr) + 1)

        latency = self._latency.setdefault(
            key, _LatencyState(samples=deque(maxlen=self._latency_window))
        )
        latency.samples.append(latency_ms)
        latency.ema_ms = (
            latency_ms
            if latency.ema_ms is None
            else self._ema_alpha * latency_ms + (1 - self._ema_alpha) * latency.ema_ms
        )

    def record_response(self, response: InferenceResponse) -> None:
        self.record_outcome(
            response.provider,
            response.model,
            succeeded=response.succeeded,
            latency_ms=response.latency.total_latency_ms,
            error_category=response.error.category if response.error else None,
        )

    def record_trace_event(self, event: TraceEvent) -> None:
        """Convenience adapter for wiring directly into `Runtime(on_trace_event=...)`.

        Only `PROVIDER_CALL` events carry outcome data (see the payload contract
        built in `paretoguard.runtime.executor`); other event types are ignored.
        """
        if event.event_type != TraceEventType.PROVIDER_CALL:
            return
        payload = event.payload
        error_category_raw = payload.get("error_category")
        self.record_outcome(
            str(payload["provider"]),
            str(payload["model"]),
            succeeded=bool(payload["succeeded"]),
            latency_ms=float(payload["latency_ms"]),
            error_category=FailureCategory(error_category_raw) if error_category_raw else None,
        )

    def get(self, provider: str, model: str) -> ModelHealth | None:
        return self._health.get((provider, model))

    def ema_latency_ms(self, provider: str, model: str) -> float | None:
        state = self._latency.get((provider, model))
        return None if state is None else state.ema_ms

    def p95_latency_ms(self, provider: str, model: str) -> float | None:
        state = self._latency.get((provider, model))
        if state is None or not state.samples:
            return None
        ordered = sorted(state.samples)
        index = max(0, math.ceil(0.95 * len(ordered)) - 1)
        return ordered[index]

    def snapshot(self) -> list[ModelHealth]:
        return list(self._health.values())
