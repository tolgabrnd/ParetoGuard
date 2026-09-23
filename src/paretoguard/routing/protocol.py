"""The `Router` protocol every routing strategy implements."""

from abc import ABC, abstractmethod

from paretoguard.core.models import RoutingDecision
from paretoguard.routing.types import RoutingRequest


class Router(ABC):
    """A routing strategy: `RoutingRequest` in, typed `RoutingDecision` out.

    Mirrors `paretoguard.providers.base.Provider`'s shape (an ABC with a
    `name` and one entry point) so both extension points in the system look
    the same to a new contributor.
    """

    name: str

    @abstractmethod
    def route(self, routing_request: RoutingRequest) -> RoutingDecision:
        """Select a candidate and explain the decision. Must not call a
        provider directly (see docs/ARCHITECTURE.md's module boundaries) —
        only decide; `paretoguard.routing.execution` (Commit 18's benchmark
        integration point) dispatches the selected model to `Runtime`."""
        raise NotImplementedError
