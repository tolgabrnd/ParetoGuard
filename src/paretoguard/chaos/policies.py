"""Fault scheduling: seeded probabilistic injection and explicit
deterministic schedules, both reusable in non-stationary reliability
experiments (the Phase D degradation/recovery scenario's schedule shape,
generalized).
"""

from dataclasses import dataclass, field
from typing import Protocol

from paretoguard.chaos.faults import ExpectedRecoverability, FaultCategory


class FaultSchedule(Protocol):
    """A pure function from step index to injection probability in [0, 1]."""

    def probability_at(self, step: int) -> float: ...


@dataclass(frozen=True)
class ConstantProbability:
    """A fixed probability at every step — the "seeded probabilistic fault"
    form (e.g. `probability=0.15, seed=42`)."""

    probability: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")

    def probability_at(self, step: int) -> float:
        return self.probability


@dataclass(frozen=True)
class StepRangeSchedule:
    """An explicit deterministic schedule: probability changes at named
    step boundaries. E.g. steps 0-99: 1%, 100-199: 25%, 200+: 1% —
    `StepRangeSchedule(((0, 0.01), (100, 0.25), (200, 0.01)))`.
    """

    breakpoints: tuple[tuple[int, float], ...]
    """`(start_step, probability)` pairs. Need not be pre-sorted —
    validated and effectively sorted at construction."""

    def __post_init__(self) -> None:
        if not self.breakpoints:
            raise ValueError("breakpoints must be non-empty")
        starts = [start for start, _ in self.breakpoints]
        if starts != sorted(starts):
            raise ValueError("breakpoints must be sorted by start_step")
        if len(set(starts)) != len(starts):
            raise ValueError("breakpoints must not repeat a start_step")
        for _, probability in self.breakpoints:
            if not 0.0 <= probability <= 1.0:
                raise ValueError("every breakpoint probability must be in [0, 1]")

    def probability_at(self, step: int) -> float:
        current = 0.0
        for start, probability in self.breakpoints:
            if step >= start:
                current = probability
            else:
                break
        return current


@dataclass(frozen=True)
class FaultPolicy:
    """Ties a fault kind/category to *when* it fires (`schedule`) and
    *what* it targets. `target_*` fields left `None` match any value for
    that dimension — e.g. `target_tool=None` with `category=TOOL` means
    "every tool"."""

    fault_id: str
    category: FaultCategory
    kind: str
    schedule: FaultSchedule
    target_provider: str | None = None
    target_model: str | None = None
    target_tool: str | None = None
    parameters: dict[str, object] = field(default_factory=dict)
    expected_recoverability: ExpectedRecoverability = ExpectedRecoverability.UNKNOWN
