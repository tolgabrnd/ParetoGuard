"""DuckDB schema migrations.

Each migration is applied in order, once, tracked by `_paretoguard_schema_version`.
Complex/nested fields (messages, metadata, JSON schemas, etc.) are stored as VARCHAR
JSON text rather than DuckDB's native JSON column type, keeping parameterized writes
simple and portable; `json.dumps`/`json.loads` at the Python boundary handles encoding.
"""

from collections.abc import Callable

import duckdb

SCHEMA_VERSION_TABLE = "_paretoguard_schema_version"

KNOWN_TABLES = frozenset({"runs", "requests", "responses", "trace_events", "routing_decisions"})


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


# (version, description, apply). Append new entries here for future schema changes;
# never edit an already-released migration in place.
MIGRATIONS: list[tuple[int, str, Callable[[duckdb.DuckDBPyConnection], None]]] = [
    (
        1,
        "initial schema: runs, requests, responses, trace_events, routing_decisions",
        _migration_001_initial,
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
