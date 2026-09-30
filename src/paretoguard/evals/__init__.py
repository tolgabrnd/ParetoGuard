"""Evaluation suites, deterministic graders, and the benchmark runner."""

from paretoguard.evals.matrix import MatrixConfig, MatrixRow, MatrixRunner, MatrixRunResult
from paretoguard.evals.metrics import (
    MetricsSummary,
    ResilienceMetricsSummary,
    compute_metrics,
    compute_resilience_metrics,
    mean_tool_call_count,
    pass_at_k,
)
from paretoguard.evals.models import (
    EvalCase,
    EvalResult,
    EvalSuite,
    ExpectedToolCall,
    GraderConfig,
    GraderKind,
    GroundTruth,
)
from paretoguard.evals.runner import BenchmarkConfig, BenchmarkRunner, BenchmarkRunResult

__all__ = [
    "BenchmarkConfig",
    "BenchmarkRunResult",
    "BenchmarkRunner",
    "EvalCase",
    "EvalResult",
    "EvalSuite",
    "ExpectedToolCall",
    "GraderConfig",
    "GraderKind",
    "GroundTruth",
    "MatrixConfig",
    "MatrixRow",
    "MatrixRunResult",
    "MatrixRunner",
    "MetricsSummary",
    "ResilienceMetricsSummary",
    "compute_metrics",
    "compute_resilience_metrics",
    "mean_tool_call_count",
    "pass_at_k",
]
