"""Built-in evaluation suites. Each suite module exposes `build_suite(seed, ...)`,
generating its cases deterministically rather than from static fixture files."""

from paretoguard.evals.suites import numeric_reasoning_v1, structured_extraction_v1

__all__ = ["numeric_reasoning_v1", "structured_extraction_v1"]
