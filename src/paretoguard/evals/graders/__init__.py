"""Deterministic graders. Import this package to register every built-in grader,
then look one up by kind with `get_grader`.

LLM-as-judge is intentionally not implemented here: deterministic grading is the
priority (see CLAUDE.md), and a judge-based path, if it lands, must stay clearly
separate and optional rather than another interchangeable GraderKind.
"""

from paretoguard.evals.graders import numeric, retrieval, structured, text, tool_use
from paretoguard.evals.graders.base import GradeOutcome, GraderFn, get_grader, grade_case

__all__ = ["GradeOutcome", "GraderFn", "get_grader", "grade_case"]

# Referenced only to guarantee their @register(...) decorators execute on import;
# unused-import warnings for these are expected and intentional.
_ = (numeric, retrieval, structured, text, tool_use)
