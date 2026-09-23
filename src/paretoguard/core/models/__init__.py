"""Typed domain models shared across ParetoGuard subsystems.

Import from here (`from paretoguard.core.models import InferenceRequest`) rather
than from the submodules directly.
"""

from paretoguard.core.models.common import (
    FailureCategory,
    FinishReason,
    Message,
    ProviderKind,
    Role,
)
from paretoguard.core.models.cost import (
    CostBasis,
    CostRecord,
    LatencyRecord,
    PricingEntry,
    TokenUsage,
)
from paretoguard.core.models.inference import (
    ErrorInfo,
    InferenceRequest,
    InferenceResponse,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from paretoguard.core.models.manifest import RunManifest
from paretoguard.core.models.provider import ModelSpec, ProviderSpec
from paretoguard.core.models.routing import RoutingConstraints, RoutingDecision
from paretoguard.core.models.telemetry import TraceEvent, TraceEventType

__all__ = [
    "CostBasis",
    "CostRecord",
    "ErrorInfo",
    "FailureCategory",
    "FinishReason",
    "InferenceRequest",
    "InferenceResponse",
    "LatencyRecord",
    "Message",
    "ModelSpec",
    "PricingEntry",
    "ProviderKind",
    "ProviderSpec",
    "Role",
    "RoutingConstraints",
    "RoutingDecision",
    "RunManifest",
    "TokenUsage",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "TraceEvent",
    "TraceEventType",
]
