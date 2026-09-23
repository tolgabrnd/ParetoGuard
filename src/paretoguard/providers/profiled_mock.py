"""ProfiledMockProvider: deterministic, per-task-family SIMULATED model profiles.

Distinct from `paretoguard.providers.mock.MockProvider` (which offers
hand-picked deterministic *scenarios* for provider/runtime unit tests): this
provider offers *statistical* profiles — a per-(model, task_family) success
probability resolved through a seeded RNG — for generating an offline
training matrix (Commit 21) and running Phase D's routing experiments. Every
value produced here is SIMULATED and must never be presented as, or mixed
into the same table as, real-provider numbers without that label (see
CLAUDE.md: "never fabricate benchmark results... or reliability numbers").

Model names/labels here are always synthetic (`profile-a`, etc., or whatever
the caller names them) — never a real commercial provider or model name.
"""

from dataclasses import dataclass, field
from random import Random

from paretoguard.core.features import TASK_FAMILY_METADATA_KEY as _TASK_FAMILY_METADATA_KEY
from paretoguard.core.models import (
    ErrorInfo,
    FailureCategory,
    FinishReason,
    InferenceRequest,
    InferenceResponse,
    LatencyRecord,
    TokenUsage,
)
from paretoguard.providers.base import Provider


@dataclass(frozen=True)
class SimulatedModelProfile:
    """A synthetic model's behavior: success probability and latency, each
    optionally varying by task family. `default_*` applies to any task
    family not present in the `_by_task_family` mapping (including
    `task_family is None`).
    """

    success_probability_by_task_family: dict[str, float] = field(default_factory=dict)
    default_success_probability: float = 0.9
    latency_ms_by_task_family: dict[str, float] = field(default_factory=dict)
    default_latency_ms: float = 500.0

    def success_probability(self, task_family: str | None) -> float:
        if task_family is not None and task_family in self.success_probability_by_task_family:
            return self.success_probability_by_task_family[task_family]
        return self.default_success_probability

    def latency_ms(self, task_family: str | None) -> float:
        if task_family is not None and task_family in self.latency_ms_by_task_family:
            return self.latency_ms_by_task_family[task_family]
        return self.default_latency_ms


PROFILE_A = "profile-a"
PROFILE_B = "profile-b"
PROFILE_C = "profile-c"


def simulated_profile_suite() -> dict[str, SimulatedModelProfile]:
    """Three SIMULATED model profiles with genuine, task-family-dependent
    trade-offs — deliberately no globally superior profile, so routing (rule,
    Pareto, reliability, or learned) has something real to choose between.
    Task family names match this repo's actual eval suite names
    (`paretoguard.evals.suites`), but every probability here is invented for
    testing/demonstration, never measured from a real provider.

      PROFILE_A: high success, high cost, medium latency (a "premium
        generalist" — best on most families, but not on structured
        extraction, and expensive everywhere).
      PROFILE_B: medium-high success, low cost, low latency (cheap and
        fast, weakest on long-context retrieval).
      PROFILE_C: strong structured-output reliability specifically, medium
        cost/latency, but comparatively weak elsewhere.
    """
    return {
        PROFILE_A: SimulatedModelProfile(
            success_probability_by_task_family={
                "numeric_reasoning_v1": 0.95,
                "structured_extraction_v1": 0.85,
                "long_context_retrieval_v1": 0.90,
                "tool_use_v1": 0.80,
            },
            default_success_probability=0.85,
            latency_ms_by_task_family={},
            default_latency_ms=600.0,
        ),
        PROFILE_B: SimulatedModelProfile(
            success_probability_by_task_family={
                "numeric_reasoning_v1": 0.80,
                "structured_extraction_v1": 0.75,
                "long_context_retrieval_v1": 0.65,
                "tool_use_v1": 0.70,
            },
            default_success_probability=0.75,
            latency_ms_by_task_family={},
            default_latency_ms=150.0,
        ),
        PROFILE_C: SimulatedModelProfile(
            success_probability_by_task_family={
                "numeric_reasoning_v1": 0.60,
                "structured_extraction_v1": 0.95,
                "long_context_retrieval_v1": 0.55,
                "tool_use_v1": 0.65,
            },
            default_success_probability=0.60,
            latency_ms_by_task_family={},
            default_latency_ms=400.0,
        ),
    }


class ProfiledMockProvider(Provider):
    """Serves any number of named synthetic model profiles under one
    provider. `complete()` never makes a network call; the outcome is a
    deterministic function of (profile, task_family, request_id, seed).
    """

    def __init__(
        self, profiles: dict[str, SimulatedModelProfile], *, name: str = "sim", seed: int = 0
    ) -> None:
        self.name = name
        self._profiles = profiles
        self._seed = seed

    def _rng_for(self, request: InferenceRequest) -> Random:
        return Random(f"{self._seed}:{request.request_id}")

    @staticmethod
    def _count_tokens(text: str) -> int:
        return max(1, len(text.split()))

    async def complete(self, request: InferenceRequest) -> InferenceResponse:
        if request.model not in self._profiles:
            raise ValueError(
                f"ProfiledMockProvider {self.name!r} has no profile for model "
                f"{request.model!r}; configured profiles: {sorted(self._profiles)}"
            )
        profile = self._profiles[request.model]
        task_family = request.metadata.get(_TASK_FAMILY_METADATA_KEY)
        rng = self._rng_for(request)

        input_text = " ".join(m.content for m in request.messages)
        input_tokens = self._count_tokens(input_text)
        latency_ms = profile.latency_ms(task_family) * rng.uniform(0.9, 1.1)
        succeeded = rng.random() < profile.success_probability(task_family)

        if not succeeded:
            return InferenceResponse(
                request_id=request.request_id,
                provider=self.name,
                model=request.model,
                finish_reason=FinishReason.ERROR,
                token_usage=TokenUsage(input_tokens=input_tokens, output_tokens=0),
                latency=LatencyRecord(total_latency_ms=latency_ms),
                error=ErrorInfo(
                    category=FailureCategory.REASONING_FAILURE,
                    message=(
                        f"ProfiledMockProvider {self.name!r}/{request.model!r}: SIMULATED "
                        f"failure (task_family={task_family!r})"
                    ),
                    retryable=False,
                ),
            )

        output_text = request.metadata.get(
            "mock_answer_text", f"[SIMULATED:{request.model}] response to: {input_text[:50]}"
        )
        output_tokens = self._count_tokens(output_text)
        return InferenceResponse(
            request_id=request.request_id,
            provider=self.name,
            model=request.model,
            output_text=output_text,
            finish_reason=FinishReason.STOP,
            token_usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            latency=LatencyRecord(total_latency_ms=latency_ms),
        )
