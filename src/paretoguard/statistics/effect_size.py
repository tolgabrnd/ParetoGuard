"""Effect sizes: the *magnitude* of a difference, reported alongside — never
instead of — a p-value or confidence interval. This repo never reduces an
analysis to "significant / not significant" (see CLAUDE.md and the Phase F
spec: "do not reduce analysis to significant/not significant").
"""

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class RiskDifference:
    """Effect size for two proportions (e.g. success rates)."""

    baseline_rate: float
    candidate_rate: float
    absolute_difference: float
    """`candidate_rate - baseline_rate` — in percentage points when rates
    are proportions (e.g. `+0.05` = +5 percentage points)."""
    relative_difference: float | None
    """`(candidate_rate - baseline_rate) / baseline_rate`. `None` when
    `baseline_rate == 0` — a relative change off a zero base is undefined,
    never reported as infinite or fabricated as some large number."""


def risk_difference(baseline_rate: float, candidate_rate: float) -> RiskDifference:
    absolute = candidate_rate - baseline_rate
    relative = (absolute / baseline_rate) if baseline_rate != 0 else None
    return RiskDifference(baseline_rate, candidate_rate, absolute, relative)


@dataclass(frozen=True)
class OddsRatio:
    value: float | None
    """`None` whenever any cell of the underlying 2x2 table is zero — the
    raw odds ratio is then 0, infinite, or undefined, and this module never
    substitutes a continuity-corrected estimate in its place without the
    caller asking for one explicitly (not currently offered)."""


def odds_ratio(
    baseline_successes: int, baseline_n: int, candidate_successes: int, candidate_n: int
) -> OddsRatio:
    baseline_failures = baseline_n - baseline_successes
    candidate_failures = candidate_n - candidate_successes
    if 0 in (baseline_successes, baseline_failures, candidate_successes, candidate_failures):
        return OddsRatio(None)
    value = (candidate_successes / candidate_failures) / (baseline_successes / baseline_failures)
    return OddsRatio(value)


@dataclass(frozen=True)
class ContinuousDelta:
    """Effect size for two continuous samples (cost, latency)."""

    baseline_mean: float
    candidate_mean: float
    absolute_delta: float
    relative_delta: float | None


def continuous_delta(baseline: Sequence[float], candidate: Sequence[float]) -> ContinuousDelta:
    if not baseline or not candidate:
        raise ValueError("cannot compute a delta over an empty sample")
    baseline_mean = mean(baseline)
    candidate_mean = mean(candidate)
    absolute = candidate_mean - baseline_mean
    relative = (absolute / baseline_mean) if baseline_mean != 0 else None
    return ContinuousDelta(baseline_mean, candidate_mean, absolute, relative)
