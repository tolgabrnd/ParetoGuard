"""Unit tests for ProfiledMockProvider. All profiles here are SIMULATED
fixtures for testing the provider's determinism/branching, not claims about
any real model.
"""

import pytest

from paretoguard.core.models import InferenceRequest, Message, Role
from paretoguard.providers.profiled_mock import (
    ProfiledMockProvider,
    SimulatedModelProfile,
    simulated_profile_suite,
)
from paretoguard.routing.types import TASK_FAMILY_METADATA_KEY


def _request(model: str = "profile-a", *, task_family: str | None = None) -> InferenceRequest:
    metadata = {TASK_FAMILY_METADATA_KEY: task_family} if task_family else {}
    return InferenceRequest(
        provider="sim",
        model=model,
        messages=[Message(role=Role.USER, content="simulated probe")],
        metadata=metadata,
    )


def test_metadata_key_matches_core_features_convention() -> None:
    """ProfiledMockProvider reads the same metadata key
    `extract_task_features` writes (both sourced from
    `paretoguard.core.features`, never `paretoguard.routing` — providers
    must not depend on routing, see docs/ARCHITECTURE.md)."""
    from paretoguard.providers.profiled_mock import _TASK_FAMILY_METADATA_KEY

    assert _TASK_FAMILY_METADATA_KEY == TASK_FAMILY_METADATA_KEY


async def test_always_succeeds_with_probability_one() -> None:
    provider = ProfiledMockProvider(
        {"profile-a": SimulatedModelProfile(default_success_probability=1.0)}
    )
    for _ in range(20):
        response = await provider.complete(_request())
        assert response.succeeded


async def test_always_fails_with_probability_zero() -> None:
    provider = ProfiledMockProvider(
        {"profile-a": SimulatedModelProfile(default_success_probability=0.0)}
    )
    response = await provider.complete(_request())
    assert not response.succeeded
    assert response.error is not None
    assert "SIMULATED" in response.error.message


async def test_success_probability_varies_by_task_family() -> None:
    profile = SimulatedModelProfile(
        success_probability_by_task_family={"strong_family": 1.0, "weak_family": 0.0},
        default_success_probability=0.5,
    )
    provider = ProfiledMockProvider({"profile-a": profile})
    strong = await provider.complete(_request(task_family="strong_family"))
    weak = await provider.complete(_request(task_family="weak_family"))
    assert strong.succeeded
    assert not weak.succeeded


async def test_deterministic_given_same_seed_and_request_id() -> None:
    profile = SimulatedModelProfile(default_success_probability=0.5)
    provider_a = ProfiledMockProvider({"profile-a": profile}, seed=7)
    provider_b = ProfiledMockProvider({"profile-a": profile}, seed=7)
    request = _request()
    result_a = await provider_a.complete(request)
    result_b = await provider_b.complete(request)
    assert result_a.succeeded == result_b.succeeded


async def test_unknown_model_raises() -> None:
    provider = ProfiledMockProvider({"profile-a": SimulatedModelProfile()})
    with pytest.raises(ValueError, match="no profile"):
        await provider.complete(_request(model="not-configured"))


def test_simulated_profile_suite_has_no_globally_dominant_profile() -> None:
    """No single profile should beat every other profile on every task
    family — otherwise routing between them is trivial, not a genuine
    decision (see the "critical scientific requirements" in BUILD_PLAN)."""
    suite = simulated_profile_suite()
    task_families = [
        "numeric_reasoning_v1",
        "structured_extraction_v1",
        "long_context_retrieval_v1",
        "tool_use_v1",
    ]
    wins = dict.fromkeys(suite, 0)
    for family in task_families:
        best = max(suite, key=lambda name: suite[name].success_probability(family))
        wins[best] += 1
    assert max(wins.values()) < len(task_families)  # no profile wins every family


async def test_latency_varies_by_task_family() -> None:
    profile = SimulatedModelProfile(
        latency_ms_by_task_family={"slow_family": 5000.0}, default_latency_ms=100.0
    )
    provider = ProfiledMockProvider({"profile-a": profile}, seed=1)
    fast = await provider.complete(_request(task_family=None))
    slow = await provider.complete(_request(task_family="slow_family"))
    assert slow.latency.total_latency_ms > fast.latency.total_latency_ms
