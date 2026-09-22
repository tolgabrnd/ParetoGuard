"""The provider protocol: the normalized boundary to any LLM backend."""

from abc import ABC, abstractmethod

from paretoguard.core.models import InferenceRequest, InferenceResponse


class Provider(ABC):
    """A provider adapter.

    Implementations normalize a specific backend's API into typed
    `InferenceRequest`/`InferenceResponse` objects. They must not raise on
    ordinary provider-side failures — timeouts, rate limits, malformed output,
    5xx errors — those are reported as `InferenceResponse.error` (see `ErrorInfo`
    / `FailureCategory`) so `paretoguard.runtime` can classify retryability from
    structured data instead of parsing exceptions. Only genuine misconfiguration
    (e.g. a missing API key) should raise, and only at construction/first use.
    """

    name: str

    @abstractmethod
    async def complete(self, request: InferenceRequest) -> InferenceResponse:
        """Execute one inference attempt and return a normalized response."""
        raise NotImplementedError
