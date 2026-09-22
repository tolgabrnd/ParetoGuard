"""Retry backoff policy."""

import random
from dataclasses import dataclass, field


@dataclass
class RetryPolicy:
    """Exponential backoff with optional full jitter.

    `rng` is injectable so tests can assert exact delays; production code should
    leave it at the default (a fresh, unseeded `random.Random`).
    """

    max_attempts: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0
    jitter: bool = True
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay_s < 0:
            raise ValueError("base_delay_s must be >= 0")
        if self.max_delay_s < self.base_delay_s:
            raise ValueError("max_delay_s must be >= base_delay_s")

    def delay_for(self, attempt: int) -> float:
        """Delay before retrying, given a zero-indexed attempt number that has
        already failed (i.e. call with 0 after the first failure)."""
        capped: float = min(self.max_delay_s, self.base_delay_s * (2**attempt))
        if not self.jitter:
            return capped
        return float(self.rng.uniform(0, capped))
