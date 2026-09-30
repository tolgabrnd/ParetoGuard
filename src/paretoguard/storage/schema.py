"""DuckDB schema migrations.

Each migration is applied in order, once, tracked by `_paretoguard_schema_version`.
Complex/nested fields (messages, metadata, JSON schemas, etc.) are stored as VARCHAR
JSON text rather than DuckDB's native JSON column type, keeping parameterized writes
simple and portable; `json.dumps`/`json.loads` at the Python boundary handles encoding.
"""

from collections.abc import Callable

import duckdb

SCHEMA_VERSION_TABLE = "_paretoguard_schema_version"

KNOWN_TABLES = frozenset(
    {
        "runs",
        "requests",
        "responses",
        "trace_events",
        "routing_decisions",
        "eval_results",
        "outcome_events",
    }
)


def _migration_001_initial(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE runs (
            run_id VARCHAR PRIMARY KEY,
            paretoguard_version VARCHAR,
            git_sha VARCHAR,
            created_at TIMESTAMPTZ,
            os VARCHAR,
            python_version VARCHAR,
            seed INTEGER,
            suite_name VARCHAR,
            suite_version VARCHAR,
            router_name VARCHAR,
            router_config VARCHAR,
            pricing_config_version VARCHAR,
            task_count INTEGER,
            repetitions INTEGER,
            environment VARCHAR
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE requests (
            request_id VARCHAR PRIMARY KEY,
            run_id VARCHAR,
            task_id VARCHAR,
            provider VARCHAR,
            model VARCHAR,
            messages VARCHAR,
            tools VARCHAR,
            max_output_tokens INTEGER,
            temperature DOUBLE,
            structured_output_schema VARCHAR,
            metadata VARCHAR,
            created_at TIMESTAMPTZ
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE responses (
            request_id VARCHAR PRIMARY KEY,
            run_id VARCHAR,
            provider VARCHAR,
            model VARCHAR,
            output_text VARCHAR,
            structured_output VARCHAR,
            finish_reason VARCHAR,
            tool_calls VARCHAR,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cached_input_tokens INTEGER,
            input_cost_usd DOUBLE,
            output_cost_usd DOUBLE,
            pricing_version VARCHAR,
            total_latency_ms DOUBLE,
            time_to_first_token_ms DOUBLE,
            queued_ms DOUBLE,
            error_category VARCHAR,
            error_message VARCHAR,
            error_retryable BOOLEAN,
            raw_provider_metadata VARCHAR,
            created_at TIMESTAMPTZ
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE trace_events (
            event_id VARCHAR PRIMARY KEY,
            run_id VARCHAR,
            request_id VARCHAR,
            event_type VARCHAR,
            timestamp TIMESTAMPTZ,
            payload VARCHAR
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE routing_decisions (
            request_id VARCHAR PRIMARY KEY,
            run_id VARCHAR,
            selected_model VARCHAR,
            candidate_scores VARCHAR,
            predicted_success DOUBLE,
            expected_cost_usd DOUBLE,
            expected_latency_ms DOUBLE,
            confidence DOUBLE,
            explanation VARCHAR,
            fallback_order VARCHAR,
            excluded_candidates VARCHAR
        )
        """
    )


def _migration_002_eval_results(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE eval_results (
            result_id VARCHAR PRIMARY KEY,
            run_id VARCHAR,
            case_id VARCHAR,
            request_id VARCHAR,
            repetition INTEGER,
            sequence INTEGER,
            succeeded BOOLEAN,
            score DOUBLE,
            grader_kind VARCHAR,
            explanation VARCHAR,
            details VARCHAR,
            response_error_category VARCHAR,
            latency_ms DOUBLE,
            cost_usd DOUBLE,
            total_tokens INTEGER
        )
        """
    )


def _migration_003_cost_basis(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("ALTER TABLE responses ADD COLUMN cost_basis VARCHAR")


def _migration_004_outcome_events(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE outcome_events (
            outcome_id VARCHAR PRIMARY KEY,
            run_id VARCHAR,
            task_id VARCHAR,
            initial_routing_decision_id VARCHAR,
            final_provider VARCHAR,
            final_model VARCHAR,
            attempt_count INTEGER,
            succeeded BOOLEAN,
            failure_category VARCHAR,
            total_cost_usd DOUBLE,
            total_latency_ms DOUBLE,
            recovery_actions VARCHAR,
            terminal BOOLEAN
        )
        """
    )


def _migration_005_run_label(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("ALTER TABLE runs ADD COLUMN label VARCHAR")


# (version, description, apply). Append new entries here for future schema changes;
# never edit an already-released migration in place.
MIGRATIONS: list[tuple[int, str, Callable[[duckdb.DuckDBPyConnection], None]]] = [
    (
        1,
        "initial schema: runs, requests, responses, trace_events, routing_decisions",
        _migration_001_initial,
    ),
    (2, "add eval_results table", _migration_002_eval_results),
    (3, "add cost_basis column to responses (ESTIMATED/SIMULATED)", _migration_003_cost_basis),
    (
        4,
        "add outcome_events table (ClosedLoopExecutor's one-row-per-task summary, Commit 29)",
        _migration_004_outcome_events,
    ),
    (
        5,
        "add label column to runs (SIMULATION/LIVE, Commit 31 run-compatibility checks)",
        _migration_005_run_label,
    ),
]


def current_schema_version(conn: duckdb.DuckDBPyConnection) -> int:
    conn.execute(f"CREATE TABLE IF NOT EXISTS {SCHEMA_VERSION_TABLE} (version INTEGER)")
    row = conn.execute(f"SELECT version FROM {SCHEMA_VERSION_TABLE}").fetchone()
    return int(row[0]) if row else 0


def apply_pending_migrations(conn: duckdb.DuckDBPyConnection) -> None:
    current = current_schema_version(conn)
    for version, _description, apply in MIGRATIONS:
        if version > current:
            apply(conn)
            conn.execute(f"DELETE FROM {SCHEMA_VERSION_TABLE}")
            conn.execute(f"INSERT INTO {SCHEMA_VERSION_TABLE} VALUES (?)", [version])
