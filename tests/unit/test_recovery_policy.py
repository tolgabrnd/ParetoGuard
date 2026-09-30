"""Unit tests for RecoveryRetryPolicy, select_fallback, and RecoveryPolicy."""

import pytest

from paretoguard.core.features import extract_task_features
from paretoguard.core.models import (
    FailureCategory,
    InferenceRequest,
    Message,
    ModelSpec,
    Role,
)
from paretoguard.recovery.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from paretoguard.recovery.context import AttemptRecord, RecoveryAction, RecoveryContext
from paretoguard.recovery.fallback import FallbackPolicy, select_fallback
from paretoguard.recovery.policy import RecoveryPolicy
from paretoguard.recovery.retry import RecoveryRetryPolicy


def _features():
    request = InferenceRequest(
        provider="mock", model="m", messages=[Message(role=Role.USER, content="hi there friend")]
    )
    return extract_task_features(request)


def _context(**overrides) -> RecoveryContext:
    defaults = {
        "current_provider": "mock",
        "current_model": "model-a",
        "failure_category": FailureCategory.TIMEOUT,
        "task_features": _features(),
    }
    defaults.update(overrides)
    return RecoveryContext(**defaults)


def _candidates() -> list[ModelSpec]:
    return [
        ModelSpec(name="model-a", provider="mock", context_window=100_000),
        ModelSpec(name="model-b", provider="mock", context_window=100_000),
        ModelSpec(name="model-c", provider="mock", context_window=100_000),
    ]


# -- RecoveryRetryPolicy ------------------------------------------------------


def test_retry_policy_permanent_failures_are_not_retryable() -> None:
    policy = RecoveryRetryPolicy()
    assert not policy.is_retryable(FailureCategory.SCHEMA_FAILURE)
    assert not policy.is_retryable(FailureCategory.TOOL_ARGUMENT_FAILURE)
    assert not policy.is_retryable(None)


def test_retry_policy_transient_failures_are_retryable() -> None:
    policy = RecoveryRetryPolicy()
    assert policy.is_retryable(FailureCategory.TIMEOUT)
    assert policy.is_retryable(FailureCategory.RATE_LIMIT)


def test_retry_policy_respects_attempt_budget() -> None:
    policy = RecoveryRetryPolicy(max_same_candidate_attempts=2)
    assert policy.has_retry_budget(0)
    assert policy.has_retry_budget(1)
    assert not policy.has_retry_budget(2)


def test_retry_policy_rejects_negative_budget() -> None:
    with pytest.raises(ValueError):
        RecoveryRetryPolicy(max_same_candidate_attempts=-1)


# -- select_fallback ----------------------------------------------------------


def test_select_fallback_skips_already_attempted() -> None:
    context = _context(
        attempted=(AttemptRecord("mock", "model-a", FailureCategory.TIMEOUT, False),)
    )
    selection = select_fallback(context, _candidates(), FallbackPolicy())
    assert selection.candidate is not None
    assert selection.candidate.name != "model-a"


def test_select_fallback_respects_hard_eligibility() -> None:
    candidates = [
        ModelSpec(name="too-small", provider="mock", context_window=1),
        ModelSpec(name="fits", provider="mock", context_window=100_000),
    ]
    context = _context()
    selection = select_fallback(context, candidates, FallbackPolicy())
    assert selection.candidate is not None
    assert selection.candidate.name == "fits"
    assert "too-small" in selection.excluded


def test_select_fallback_returns_none_when_depth_exhausted() -> None:
    context = _context(
        attempted=(
            AttemptRecord("mock", "model-a", FailureCategory.TIMEOUT, False),
            AttemptRecord("mock", "model-b", FailureCategory.TIMEOUT, False),
        )
    )
    selection = select_fallback(context, _candidates(), FallbackPolicy(max_fallback_depth=2))
    assert selection.candidate is None


def test_select_fallback_skips_open_circuit() -> None:
    breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1))
    breaker.record_failure("mock:model-b")
    context = _context(
        attempted=(AttemptRecord("mock", "model-a", FailureCategory.TIMEOUT, False),)
    )
    selection = select_fallback(context, _candidates(), FallbackPolicy(), circuit_breaker=breaker)
    assert selection.candidate is not None
    assert selection.candidate.name == "model-c"


def test_select_fallback_marks_half_open_as_probe() -> None:
    clock_state = {"now": 0.0}
    breaker = CircuitBreaker(
        CircuitBreakerConfig(failure_threshold=1, cooldown_s=1.0), clock=lambda: clock_state["now"]
    )
    breaker.record_failure("mock:model-b")
    clock_state["now"] = 2.0  # cooldown elapsed -> HALF_OPEN
    context = _context(
        attempted=(AttemptRecord("mock", "model-a", FailureCategory.TIMEOUT, False),)
    )
    selection = select_fallback(context, _candidates(), FallbackPolicy(), circuit_breaker=breaker)
    assert selection.candidate is not None
    assert selection.candidate.name == "model-b"
    assert selection.is_probe


def test_select_fallback_never_cycles_back_to_current_candidate() -> None:
    candidates = [ModelSpec(name="model-a", provider="mock", context_window=100_000)]
    context = _context()  # current_model="model-a", no other candidates
    selection = select_fallback(context, candidates, FallbackPolicy())
    assert selection.candidate is None


# -- RecoveryPolicy (end to end decisions) ------------------------------------


def test_recovery_policy_retries_same_on_first_transient_failure() -> None:
    policy = RecoveryPolicy(retry_policy=RecoveryRetryPolicy(max_same_candidate_attempts=1))
    decision = policy.decide(_context(), _candidates())
    assert decision.action == RecoveryAction.RETRY_SAME
    assert decision.target_model == "model-a"


def test_recovery_policy_falls_back_after_retry_budget_exhausted() -> None:
    policy = RecoveryPolicy(retry_policy=RecoveryRetryPolicy(max_same_candidate_attempts=1))
    context = _context(
        attempted=(AttemptRecord("mock", "model-a", FailureCategory.TIMEOUT, False),)
    )
    decision = policy.decide(context, _candidates())
    assert decision.action == RecoveryAction.FALLBACK_MODEL
    assert decision.target_model in {"model-b", "model-c"}


def test_recovery_policy_never_retries_permanent_failure() -> None:
    policy = RecoveryPolicy()
    context = _context(failure_category=FailureCategory.SCHEMA_FAILURE)
    decision = policy.decide(context, _candidates())
    assert decision.action != RecoveryAction.RETRY_SAME


def test_recovery_policy_fails_on_permanent_failure_with_no_fallback() -> None:
    policy = RecoveryPolicy()
    context = _context(
        failure_category=FailureCategory.TOOL_ARGUMENT_FAILURE,
        current_provider="mock",
        current_model="only-model",
    )
    decision = policy.decide(
        context, [ModelSpec(name="only-model", provider="mock", context_window=100_000)]
    )
    assert decision.action == RecoveryAction.FAIL


def test_recovery_policy_abstains_when_budget_exhausted() -> None:
    policy = RecoveryPolicy()
    context = _context(remaining_usd_budget=0.0)
    decision = policy.decide(context, _candidates())
    assert decision.action == RecoveryAction.ABSTAIN
    assert "budget" in decision.reason


def test_recovery_policy_abstains_when_latency_budget_exhausted() -> None:
    policy = RecoveryPolicy()
    context = _context(remaining_latency_budget_ms=0.0)
    decision = policy.decide(context, _candidates())
    assert decision.action == RecoveryAction.ABSTAIN


def test_recovery_policy_escalates_on_quality_failure() -> None:
    policy = RecoveryPolicy(retry_policy=RecoveryRetryPolicy(max_same_candidate_attempts=0))
    context = _context(failure_category=FailureCategory.REASONING_FAILURE)
    decision = policy.decide(context, _candidates())
    assert decision.action == RecoveryAction.ESCALATE


def test_recovery_policy_does_not_escalate_when_disabled() -> None:
    policy = RecoveryPolicy(
        retry_policy=RecoveryRetryPolicy(max_same_candidate_attempts=0), allow_escalation=False
    )
    context = _context(failure_category=FailureCategory.REASONING_FAILURE)
    decision = policy.decide(context, _candidates())
    assert decision.action == RecoveryAction.FALLBACK_MODEL


def test_recovery_policy_respects_open_circuit_for_retry_same() -> None:
    breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1))
    breaker.record_failure("mock:model-a")
    policy = RecoveryPolicy(circuit_breaker=breaker)
    decision = policy.decide(_context(), _candidates())
    assert decision.action != RecoveryAction.RETRY_SAME


def test_recovery_policy_probes_half_open_target() -> None:
    clock_state = {"now": 0.0}
    breaker = CircuitBreaker(
        CircuitBreakerConfig(failure_threshold=1, cooldown_s=1.0), clock=lambda: clock_state["now"]
    )
    breaker.record_failure("mock:model-b")
    clock_state["now"] = 5.0
    policy = RecoveryPolicy(
        retry_policy=RecoveryRetryPolicy(max_same_candidate_attempts=0), circuit_breaker=breaker
    )
    context = _context(
        attempted=(AttemptRecord("mock", "model-a", FailureCategory.TIMEOUT, False),)
    )
    decision = policy.decide(context, _candidates())
    assert decision.action == RecoveryAction.PROBE
    assert decision.target_model == "model-b"


def test_recovery_policy_never_exceeds_max_fallback_depth_across_repeated_decisions() -> None:
    """Property-style: simulate a chain of always-failing attempts and
    verify the number of distinct candidates tried never exceeds
    max_fallback_depth, and the chain always terminates in ABSTAIN/FAIL."""
    policy = RecoveryPolicy(
        retry_policy=RecoveryRetryPolicy(max_same_candidate_attempts=0),
        fallback_policy=FallbackPolicy(max_fallback_depth=2),
    )
    attempted: list[AttemptRecord] = []
    current_provider, current_model = "mock", "model-a"
    decisions = []
    for _ in range(10):
        context = _context(
            current_provider=current_provider,
            current_model=current_model,
            attempted=tuple(attempted),
        )
        decision = policy.decide(context, _candidates())
        decisions.append(decision.action)
        if decision.action in (RecoveryAction.ABSTAIN, RecoveryAction.FAIL):
            break
        attempted.append(
            AttemptRecord(current_provider, current_model, FailureCategory.TIMEOUT, False)
        )
        current_provider = decision.target_provider or current_provider
        current_model = decision.target_model or current_model

    assert decisions[-1] in (RecoveryAction.ABSTAIN, RecoveryAction.FAIL)
    distinct_candidates = {(a.provider, a.model) for a in attempted}
    assert len(distinct_candidates) <= 2
