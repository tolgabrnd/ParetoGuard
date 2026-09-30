"""Built-in evaluation suites. Each suite module exposes `build_suite(seed, ...)`,
generating its cases deterministically rather than from static fixture files.

`structured_agent_v1` is the one exception: it exposes `build_tasks(seed, ...)
-> list[paretoguard.agents.simulator.AgentTask]` instead of an `EvalSuite` —
agent tasks are graded by `agents.simulator.grade_trajectory`, not
`evals.graders.grade_case`, so `EvalCase`'s single-turn shape doesn't fit.
"""

from paretoguard.evals.suites import (
    long_context_retrieval_v1,
    numeric_reasoning_v1,
    resilience_v1,
    structured_agent_v1,
    structured_extraction_v1,
    tool_use_v1,
)

__all__ = [
    "long_context_retrieval_v1",
    "numeric_reasoning_v1",
    "resilience_v1",
    "structured_agent_v1",
    "structured_extraction_v1",
    "tool_use_v1",
]
