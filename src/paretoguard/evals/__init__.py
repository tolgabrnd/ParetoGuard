"""Evaluation suites, deterministic graders, and the benchmark runner."""

from paretoguard.evals.matrix import MatrixConfig, MatrixRow, MatrixRunner, MatrixRunResult
from paretoguard.evals.metrics import MetricsSummary, compute_metrics, pass_at_k
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
    "compute_metrics",
    "pass_at_k",
]
