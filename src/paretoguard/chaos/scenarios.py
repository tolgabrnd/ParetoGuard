"""Pre-built, reusable fault scenarios — parallels
`paretoguard.providers.profiled_mock.simulated_profile_suite`'s role for
routing: hand-authored, clearly-labeled-synthetic configurations rather
than ad hoc one-off fault setups per script.
"""

from random import Random

from paretoguard.chaos.faults import (
    ContextFaultKind,
    ExpectedRecoverability,
    FaultCategory,
    ProviderFaultKind,
)
from paretoguard.chaos.policies import ConstantProbability, FaultPolicy, StepRangeSchedule


def provider_degradation_schedule(
    *,
    fault_id: str = "provider-degradation",
    target_provider: str,
    target_model: str,
    degraded_start: int,
    recovered_start: int,
    baseline_probability: float = 0.01,
    degraded_probability: float = 0.25,
    kind: ProviderFaultKind = ProviderFaultKind.TIMEOUT,
) -> FaultPolicy:
    """The canonical non-stationary provider scenario — the Phase D
    degradation/recovery experiment's schedule shape
    (`paretoguard.routing.reliability_simulation.two_model_degradation_schedule`),
    generalized as a reusable `FaultPolicy`: baseline -> degraded -> recovered.
    """
    schedule = StepRangeSchedule(
        (
            (0, baseline_probability),
            (degraded_start, degraded_probability),
            (recovered_start, baseline_probability),
        )
    )
    return FaultPolicy(
        fault_id=fault_id,
        category=FaultCategory.PROVIDER,
        kind=kind.value,
        schedule=schedule,
        target_provider=target_provider,
        target_model=target_model,
        expected_recoverability=ExpectedRecoverability.RECOVERABLE,
    )


def resilience_fault_levels(
    levels: list[float],
    *,
    target_provider: str,
    kind: ProviderFaultKind = ProviderFaultKind.SERVER_ERROR,
) -> dict[float, FaultPolicy]:
    """One constant-probability `FaultPolicy` per requested fault level —
    for `resilience_v1`'s "run the same tasks under 0%/5%/15%/30% fault
    rate" comparison (Commit 29)."""
    return {
        level: FaultPolicy(
            fault_id=f"resilience-{level}",
            category=FaultCategory.PROVIDER,
            kind=kind.value,
            schedule=ConstantProbability(level),
            target_provider=target_provider,
            expected_recoverability=ExpectedRecoverability.RECOVERABLE,
        )
        for level in levels
    }


def apply_context_fault(text: str, kind: ContextFaultKind, rng: Random, **params: object) -> str:
    """Benchmark-**construction**-time text perturbation — applied when
    *building* a task prompt, never at dispatch time (there is no
    "context fault injector" in the request pipeline). The benchmark's own
    `ground_truth`/`ExpectedTrajectory` stays authoritative regardless of
    what this injects into the prompt text — see `ContextFaultKind`'s
    docstring for why that matters.
    """
    if kind == ContextFaultKind.DISTRACTOR_INJECTION:
        distractor = params.get(
            "distractor", "Note: the following is unrelated background information."
        )
        return f"{text}\n\n{distractor}"
    if kind == ContextFaultKind.MISSING_EVIDENCE:
        lines = text.splitlines()
        if len(lines) > 1:
            lines.pop(rng.randrange(len(lines)))
        return "\n".join(lines)
    if kind == ContextFaultKind.STALE_RECORD:
        stale_note = params.get("stale_note", "(Note: the following record may be outdated.)")
        return f"{stale_note}\n{text}"
    if kind == ContextFaultKind.CONTRADICTORY_STALE_RECORD:
        contradictory_value = params.get("contradictory_value")
        authoritative_note = params.get(
            "authoritative_note", "The record above is the authoritative, current one."
        )
        if contradictory_value is None:
            return text
        return (
            f"{text}\n\n(Superseded outdated record: {contradictory_value})\n{authoritative_note}"
        )
    raise ValueError(f"unhandled ContextFaultKind: {kind}")
