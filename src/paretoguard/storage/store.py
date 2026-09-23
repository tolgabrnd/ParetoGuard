"""Versioned DuckDB-backed experiment store.

This is the single place run/request/response/trace/routing data is written and
read from. Reports (`paretoguard.reports`) must only read from here — never
recompute metrics from live provider calls.
"""

import json
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any
from uuid import UUID

import duckdb
import polars as pl

from paretoguard.core.models import (
    FailureCategory,
    InferenceRequest,
    InferenceResponse,
    Message,
    RoutingDecision,
    RunManifest,
    ToolSpec,
    TraceEvent,
)
from paretoguard.storage.schema import KNOWN_TABLES, apply_pending_migrations

if TYPE_CHECKING:
    # Deferred: paretoguard.evals imports paretoguard.storage at runtime (the
    # benchmark runner persists results), so a module-level import here would
    # be circular. Only type checkers need this; runtime code imports EvalResult
    # locally inside get_eval_results, where the actual class is needed.
    from paretoguard.evals.models import EvalResult


def _dumps(value: Any) -> str | None:
    return None if value is None else json.dumps(value)


def _loads(value: str | None, default: Any) -> Any:
    return default if not value else json.loads(value)


class ExperimentStore:
    """A versioned DuckDB store for runs, requests, responses, traces, and routing
    decisions. Defaults to an in-memory database; pass a file path to persist."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._conn = duckdb.connect(str(db_path))
        apply_pending_migrations(self._conn)

    def __enter__(self) -> "ExperimentStore":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    # -- writes ---------------------------------------------------------

    def record_run(self, manifest: RunManifest) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO runs VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                manifest.run_id,
                manifest.paretoguard_version,
                manifest.git_sha,
                manifest.created_at,
                manifest.os,
                manifest.python_version,
                manifest.seed,
                manifest.suite_name,
                manifest.suite_version,
                manifest.router_name,
                _dumps(manifest.router_config),
                manifest.pricing_config_version,
                manifest.task_count,
                manifest.repetitions,
                _dumps(manifest.environment),
            ],
        )

    def record_request(self, request: InferenceRequest, run_id: str | None = None) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO requests VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                str(request.request_id),
                run_id,
                request.task_id,
                request.provider,
                request.model,
                _dumps([m.model_dump(mode="json") for m in request.messages]),
                _dumps([t.model_dump(mode="json") for t in request.tools]),
                request.max_output_tokens,
                request.temperature,
                _dumps(request.structured_output_schema),
                _dumps(request.metadata),
                request.created_at,
            ],
        )

    def record_response(self, response: InferenceResponse, run_id: str | None = None) -> None:
        usage, cost, latency, error = (
            response.token_usage,
            response.cost,
            response.latency,
            response.error,
        )
        # Named columns (rather than positional VALUES) because `cost_basis` was
        # added to this table by a later ALTER TABLE migration and so is not in
        # the same physical column position as the other cost fields it's
        # logically grouped with here.
        self._conn.execute(
            """
            INSERT OR REPLACE INTO responses (
                request_id, run_id, provider, model, output_text, structured_output,
                finish_reason, tool_calls, input_tokens, output_tokens, cached_input_tokens,
                input_cost_usd, output_cost_usd, pricing_version, cost_basis,
                total_latency_ms, time_to_first_token_ms, queued_ms,
                error_category, error_message, error_retryable,
                raw_provider_metadata, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                str(response.request_id),
                run_id,
                response.provider,
                response.model,
                response.output_text,
                _dumps(response.structured_output),
                response.finish_reason.value,
                _dumps([t.model_dump(mode="json") for t in response.tool_calls]),
                usage.input_tokens,
                usage.output_tokens,
                usage.cached_input_tokens,
                cost.input_cost_usd if cost else None,
                cost.output_cost_usd if cost else None,
                cost.pricing_version if cost else None,
                cost.basis.value if cost else None,
                latency.total_latency_ms,
                latency.time_to_first_token_ms,
                latency.queued_ms,
                error.category.value if error else None,
                error.message if error else None,
                error.retryable if error else None,
                _dumps(response.raw_provider_metadata),
                response.created_at,
            ],
        )

    def record_trace_event(self, event: TraceEvent, run_id: str | None = None) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO trace_events VALUES (?, ?, ?, ?, ?, ?)",
            [
                str(event.event_id),
                run_id if run_id is not None else event.run_id,
                str(event.request_id) if event.request_id is not None else None,
                event.event_type.value,
                event.timestamp,
                _dumps(event.payload),
            ],
        )

    def record_routing_decision(
        self, request_id: UUID, decision: RoutingDecision, run_id: str | None = None
    ) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO routing_decisions VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                str(request_id),
                run_id,
                decision.selected_model,
                _dumps(decision.candidate_scores),
                decision.predicted_success,
                decision.expected_cost_usd,
                decision.expected_latency_ms,
                decision.confidence,
                decision.explanation,
                _dumps(decision.fallback_order),
                _dumps(decision.excluded_candidates),
            ],
        )

    def record_eval_result(self, result: "EvalResult", run_id: str | None = None) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO eval_results VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                str(result.result_id),
                run_id,
                result.case_id,
                str(result.request_id),
                result.repetition,
                result.sequence,
                result.succeeded,
                result.score,
                result.grader_kind.value,
                result.explanation,
                _dumps(result.details),
                result.response_error_category.value if result.response_error_category else None,
                result.latency_ms,
                result.cost_usd,
                result.total_tokens,
            ],
        )

    # -- reads ------------------------------------------------------------

    def get_run(self, run_id: str) -> RunManifest | None:
        df = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", [run_id]).pl()
        if df.height == 0:
            return None
        row = df.row(0, named=True)
        return RunManifest(
            run_id=row["run_id"],
            paretoguard_version=row["paretoguard_version"],
            git_sha=row["git_sha"],
            created_at=row["created_at"],
            os=row["os"],
            python_version=row["python_version"],
            seed=row["seed"],
            suite_name=row["suite_name"],
            suite_version=row["suite_version"],
            router_name=row["router_name"],
            router_config=_loads(row["router_config"], {}),
            pricing_config_version=row["pricing_config_version"],
            task_count=row["task_count"],
            repetitions=row["repetitions"],
            environment=_loads(row["environment"], {}),
        )

    def list_runs(self) -> list[RunManifest]:
        df = self._conn.execute("SELECT run_id FROM runs ORDER BY created_at").pl()
        manifests = (self.get_run(run_id) for run_id in df["run_id"].to_list())
        return [m for m in manifests if m is not None]

    def get_request(self, request_id: UUID) -> InferenceRequest | None:
        df = self._conn.execute(
            "SELECT * FROM requests WHERE request_id = ?", [str(request_id)]
        ).pl()
        if df.height == 0:
            return None
        row = df.row(0, named=True)
        return InferenceRequest(
            request_id=UUID(row["request_id"]),
            task_id=row["task_id"],
            provider=row["provider"],
            model=row["model"],
            messages=[Message.model_validate(m) for m in _loads(row["messages"], [])],
            tools=[ToolSpec.model_validate(t) for t in _loads(row["tools"], [])],
            max_output_tokens=row["max_output_tokens"],
            temperature=row["temperature"],
            structured_output_schema=_loads(row["structured_output_schema"], None),
            metadata=_loads(row["metadata"], {}),
            created_at=row["created_at"],
        )

    def requests_df(self, run_id: str | None = None) -> pl.DataFrame:
        if run_id is None:
            return self._conn.execute("SELECT * FROM requests").pl()
        return self._conn.execute("SELECT * FROM requests WHERE run_id = ?", [run_id]).pl()

    def responses_df(self, run_id: str | None = None) -> pl.DataFrame:
        if run_id is None:
            return self._conn.execute("SELECT * FROM responses").pl()
        return self._conn.execute("SELECT * FROM responses WHERE run_id = ?", [run_id]).pl()

    def trace_events_df(self, run_id: str | None = None) -> pl.DataFrame:
        if run_id is None:
            return self._conn.execute("SELECT * FROM trace_events").pl()
        return self._conn.execute("SELECT * FROM trace_events WHERE run_id = ?", [run_id]).pl()

    def routing_decisions_df(self, run_id: str | None = None) -> pl.DataFrame:
        if run_id is None:
            return self._conn.execute("SELECT * FROM routing_decisions").pl()
        return self._conn.execute("SELECT * FROM routing_decisions WHERE run_id = ?", [run_id]).pl()

    def eval_results_df(self, run_id: str | None = None) -> pl.DataFrame:
        if run_id is None:
            return self._conn.execute("SELECT * FROM eval_results ORDER BY sequence").pl()
        return self._conn.execute(
            "SELECT * FROM eval_results WHERE run_id = ? ORDER BY sequence", [run_id]
        ).pl()

    def get_eval_results(self, run_id: str) -> list["EvalResult"]:
        """Reads back EvalResults for a run in canonical (case, repetition) order,
        via the `sequence` field — independent of any storage read-back order,
        which plain SQL does not otherwise guarantee."""
        from paretoguard.evals.models import (  # local: see TYPE_CHECKING note above
            EvalResult,
            GraderKind,
        )

        df = self.eval_results_df(run_id=run_id)
        results = []
        for row in df.iter_rows(named=True):
            results.append(
                EvalResult(
                    result_id=UUID(row["result_id"]),
                    case_id=row["case_id"],
                    request_id=UUID(row["request_id"]),
                    repetition=row["repetition"],
                    sequence=row["sequence"],
                    succeeded=row["succeeded"],
                    score=row["score"],
                    grader_kind=GraderKind(row["grader_kind"]),
                    explanation=row["explanation"],
                    details=_loads(row["details"], {}),
                    response_error_category=(
                        FailureCategory(row["response_error_category"])
                        if row["response_error_category"]
                        else None
                    ),
                    latency_ms=row["latency_ms"],
                    cost_usd=row["cost_usd"],
                    total_tokens=row["total_tokens"],
                )
            )
        return results

    # -- export -------------------------------------------------------------

    def export_parquet(self, table: str, path: Path) -> None:
        self._require_known_table(table)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn.execute(f"COPY {table} TO '{path.as_posix()}' (FORMAT PARQUET)")

    def export_jsonl(self, table: str, path: Path) -> None:
        self._require_known_table(table)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn.execute(f"COPY {table} TO '{path.as_posix()}' (FORMAT JSON, ARRAY false)")

    @staticmethod
    def _require_known_table(table: str) -> None:
        if table not in KNOWN_TABLES:
            raise ValueError(f"Unknown table: {table!r}")
