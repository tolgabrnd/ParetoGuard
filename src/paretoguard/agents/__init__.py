"""Typed tool-using agent simulator and deterministic tool implementations.

A deliberately small, hard-bounded, reproducible environment for measuring
tool-use correctness and trajectory correctness — not a general-purpose
autonomous-agent framework. See `AgentExecutor`'s module docstring for the
runtime-retry-vs-agent-loop-limits boundary.
"""

from paretoguard.agents.executor import AgentExecutor, AgentLimits
from paretoguard.agents.protocol import Tool, ToolArgumentError
from paretoguard.agents.simulator import (
    AgentGradeOutcome,
    AgentSimulator,
    AgentTask,
    AgentTaskResult,
    ExpectedTrajectory,
    finalize_trajectory,
    grade_trajectory,
)
from paretoguard.agents.state import (
    AgentStep,
    AgentTrajectory,
    TerminationReason,
    failure_category_for,
)
from paretoguard.agents.tools import (
    CalculatorTool,
    DocumentRetrievalTool,
    InventoryLookupTool,
    OrderLookupTool,
    ShipmentLookupTool,
    default_tools,
)

__all__ = [
    "AgentExecutor",
    "AgentGradeOutcome",
    "AgentLimits",
    "AgentSimulator",
    "AgentStep",
    "AgentTask",
    "AgentTaskResult",
    "AgentTrajectory",
    "CalculatorTool",
    "DocumentRetrievalTool",
    "ExpectedTrajectory",
    "InventoryLookupTool",
    "OrderLookupTool",
    "ShipmentLookupTool",
    "TerminationReason",
    "Tool",
    "ToolArgumentError",
    "default_tools",
    "failure_category_for",
    "finalize_trajectory",
    "grade_trajectory",
]
