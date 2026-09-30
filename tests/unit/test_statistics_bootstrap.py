"""Unit tests for statistics.bootstrap: percentile-method bootstrap CIs."""

import pytest

from paretoguard.statistics.bootstrap import (
    MIN_RELIABLE_N,
    bootstrap_ci,
    bootstrap_proportion_ci,
)


def test_bootstrap_ci_rejects_empty_sample() -> None:
    with pytest.raises(ValueError, match="empty"):
        bootstrap_ci([])


def test_bootstrap_ci_rejects_invalid_confidence_level() -> None:
    with pytest.raises(ValueError, match="confidence_level"):
        bootstrap_ci([1.0, 2.0], confidence_level=1.5)


def test_bootstrap_ci_is_deterministic_given_the_same_seed() -> None:
    data = [1.0, 2.0, 3.0, 4.0, 5.0] * 10
    a = bootstrap_ci(data, seed=42)
    b = bootstrap_ci(data, seed=42)
    assert a == b


def test_bootstrap_ci_differs_across_seeds() -> None:
    data = [1.0, 2.0, 3.0, 4.0, 5.0] * 10
    a = bootstrap_ci(data, seed=1)
    b = bootstrap_ci(data, seed=2)
    assert (a.lower, a.upper) != (b.lower, b.upper)


def test_bootstrap_ci_contains_the_true_proportion_for_a_fair_coin() -> None:
    """Coverage sanity check on a known synthetic distribution: a fair
    coin's true success rate (0.5) should fall inside the bootstrap CI of a
    reasonably large fair-coin sample the overwhelming majority of the
    time. Not a formal coverage proof, but catches a badly broken
    implementation."""
    import random

    rng = random.Random(0)
    data = [rng.random() < 0.5 for _ in range(500)]
    result = bootstrap_proportion_ci(data, seed=0)
    assert result.lower <= 0.5 <= result.upper


def test_reliable_flag_reflects_min_sample_size() -> None:
    small = bootstrap_ci([1.0, 2.0, 3.0], seed=0)
    assert small.n < MIN_RELIABLE_N
    assert not small.reliable

    large = bootstrap_ci(list(range(50)), seed=0)
    assert large.n >= MIN_RELIABLE_N
    assert large.reliable


def test_point_estimate_matches_the_statistic_over_the_full_sample() -> None:
    data = [2.0, 4.0, 6.0, 8.0]
    result = bootstrap_ci(data, seed=0)
    assert result.point_estimate == 5.0


def test_bootstrap_proportion_ci_of_all_successes_is_a_point_at_one() -> None:
    result = bootstrap_proportion_ci([True] * 40, seed=0)
    assert result.point_estimate == 1.0
    assert result.lower == 1.0
    assert result.upper == 1.0


def test_narrower_interval_for_larger_sample_size() -> None:
    """A bootstrap CI should generally narrow as n grows, for the same
    underlying variance — checked on a fixed, reproducible synthetic
    sequence, not a flaky random draw."""
    import random

    rng = random.Random(1)
    small_sample = [rng.random() < 0.7 for _ in range(30)]
    large_sample = small_sample + [rng.random() < 0.7 for _ in range(470)]

    small_result = bootstrap_proportion_ci(small_sample, seed=0)
    large_result = bootstrap_proportion_ci(large_sample, seed=0)
    assert (large_result.upper - large_result.lower) < (small_result.upper - small_result.lower)


def test_rejects_invalid_n_resamples() -> None:
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_ci([1.0, 2.0], n_resamples=0)
