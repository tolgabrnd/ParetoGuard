"""Deterministic, seed-controlled fault injection for providers, tools, and
context (benchmark task construction).

`FaultInjector`'s decisions are a pure function of (seed, task_id, step,
fault_id) — never a random UUID, a process-dependent hash, or wall-clock
time — so a seeded fault schedule is reproducible across separate process
runs (see the Phase D audit's determinism finding, and
`core.ids.deterministic_request_id`, applied the same way here).
"""

from paretoguard.chaos.faults import (
    ContextFaultKind,
    ExpectedRecoverability,
    FaultCategory,
    FaultEvent,
    ProviderFaultKind,
    ToolFaultKind,
)
from paretoguard.chaos.injector import (
    FaultCallback,
    FaultInjector,
    FaultyProvider,
    apply_tool_fault,
    check_tool_fault,
)
from paretoguard.chaos.policies import (
    ConstantProbability,
    FaultPolicy,
    FaultSchedule,
    StepRangeSchedule,
)
from paretoguard.chaos.scenarios import (
    apply_context_fault,
    provider_degradation_schedule,
    resilience_fault_levels,
)

__all__ = [
    "ConstantProbability",
    "ContextFaultKind",
    "ExpectedRecoverability",
    "FaultCallback",
    "FaultCategory",
    "FaultEvent",
    "FaultInjector",
    "FaultPolicy",
    "FaultSchedule",
    "FaultyProvider",
    "ProviderFaultKind",
    "StepRangeSchedule",
    "ToolFaultKind",
    "apply_context_fault",
    "apply_tool_fault",
    "check_tool_fault",
    "provider_degradation_schedule",
    "resilience_fault_levels",
]
