"""Bootstrap confidence intervals.

Percentile-method bootstrap: for a statistic (mean success indicator, mean
cost, ...), resample the observed data with replacement `n_resamples`
times, compute the statistic on each resample, and report the
`alpha/2`/`1 - alpha/2` percentiles of that resampled distribution as the
interval bounds.

**Determinism**: every resample draw comes from a `numpy.random.Generator`
seeded explicitly (`seed`, default 0) — never global numpy random state —
so a bootstrap CI computed twice with the same data and seed is bit-for-bit
identical, matching this repo's determinism discipline throughout (see
`docs/LIMITATIONS.md`'s Phase E/E.5 entries on exactly this point).

**Limitations, stated rather than hidden**:
- Below `MIN_RELIABLE_N` observations, the percentile bootstrap is known to
  be biased/unstable; `BootstrapResult.reliable` is `False` in that case.
  The interval is still computed and returned — never silently withheld —
  but callers must not present it as a precise bound.
- This is the plain percentile method, not bias-corrected-and-accelerated
  (BCa). BCa is more accurate for skewed statistics at small-to-moderate n
  but meaningfully more complex to implement correctly; the percentile
  method is used here for simplicity, and its known small-n bias is exactly
  why `reliable` exists.
- `n_resamples=2000` (the default) is a reasonable general-purpose choice,
  not tuned per metric.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import mean

import numpy as np

MIN_RELIABLE_N = 30
DEFAULT_N_RESAMPLES = 2000
DEFAULT_CONFIDENCE_LEVEL = 0.95


@dataclass(frozen=True)
class BootstrapResult:
    point_estimate: float
    lower: float
    upper: float
    confidence_level: float
    n_resamples: int
    n: int
    seed: int
    reliable: bool
    """`False` when `n < MIN_RELIABLE_N` — see this module's docstring."""


def bootstrap_ci(
    data: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] = mean,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = 0,
) -> BootstrapResult:
    """Bootstrap CI for `statistic(data)`. Raises `ValueError` on an empty
    sample — there is no meaningful interval over nothing."""
    if not data:
        raise ValueError("cannot bootstrap an empty sample")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")

    n = len(data)
    array = np.asarray(data, dtype=float)
    point_estimate = float(statistic(data))

    rng = np.random.default_rng(seed)
    resample_stats = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        resample_stats[i] = statistic(array[indices].tolist())

    alpha = 1.0 - confidence_level
    lower = float(np.percentile(resample_stats, 100 * (alpha / 2)))
    upper = float(np.percentile(resample_stats, 100 * (1 - alpha / 2)))

    return BootstrapResult(
        point_estimate=point_estimate,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        n=n,
        seed=seed,
        reliable=n >= MIN_RELIABLE_N,
    )


def bootstrap_proportion_ci(
    successes: Sequence[bool],
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = 0,
) -> BootstrapResult:
    """Convenience wrapper: bootstrap CI for a success-rate proportion."""
    return bootstrap_ci(
        [1.0 if s else 0.0 for s in successes],
        statistic=mean,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
    )
