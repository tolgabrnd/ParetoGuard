"""Spend/call budget guards.

Enforces `PARETOGUARD_MAX_RUN_USD` / `PARETOGUARD_MAX_CALLS` (see
`paretoguard.core.config.Settings`) so a misconfigured or runaway benchmark can't
make unbounded provider calls or spend.
"""

import asyncio

from paretoguard.core.config import Settings


class BudgetExceededError(RuntimeError):
    """Raised when a configured spend or call budget has been reached."""


class BudgetGuard:
    """Tracks calls made and (estimated) USD spent against optional limits.

    Cost isn't known until a response comes back, so this can't preemptively
    block a call that would exceed the cost budget — it blocks the *next* call
    once the budget has already been exceeded by a prior one. The call-count
    budget, by contrast, is checked before every call.
    """

    def __init__(self, *, max_run_usd: float | None = None, max_calls: int | None = None) -> None:
        if max_run_usd is not None and max_run_usd < 0:
            raise ValueError("max_run_usd must be >= 0")
        if max_calls is not None and max_calls < 0:
            raise ValueError("max_calls must be >= 0")
        self.max_run_usd = max_run_usd
        self.max_calls = max_calls
        self.calls_made = 0
        self.spent_usd = 0.0
        self._lock = asyncio.Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> "BudgetGuard":
        return cls(max_run_usd=settings.max_run_usd, max_calls=settings.max_calls)

    async def check_before_call(self) -> None:
        async with self._lock:
            if self.max_calls is not None and self.calls_made >= self.max_calls:
                raise BudgetExceededError(f"max_calls budget of {self.max_calls} reached")
            if self.max_run_usd is not None and self.spent_usd >= self.max_run_usd:
                raise BudgetExceededError(f"max_run_usd budget of {self.max_run_usd} reached")

    async def record_call(self, cost_usd: float = 0.0) -> None:
        async with self._lock:
            self.calls_made += 1
            self.spent_usd += cost_usd
