"""Unit tests for the DuckDB experiment store."""

from pathlib import Path
from uuid import uuid4

import pytest

from paretoguard.core.models import (
    CostBasis,
    CostRecord,
    ErrorInfo,
    FailureCategory,
    FinishReason,
    InferenceRequest,
    LatencyRecord,
    Message,
    OutcomeEvent,
    Role,
    RoutingDecision,
    RunManifest,
    TokenUsage,
    TraceEvent,
    TraceEventType,
)
from paretoguard.core.models.inference import InferenceResponse
from paretoguard.evals.models import EvalResult, GraderKind
from paretoguard.storage import ExperimentStore


@pytest.fixture
def store():
    with ExperimentStore(":memory:") as s:
        yield s


def _manifest(run_id: str = "run-1") -> RunManifest:
    return RunManifest(
        run_id=run_id,
        paretoguard_version="0.1.0",
        os="Windows",
        python_version="3.12.14",
        seed=42,
        suite_name="structured_extraction_v1",
        router_config={"strategy": "static"},
        environment={"CI": "false"},
    )


def _request() -> InferenceRequest:
    return InferenceRequest(
        provider="mock",
        model="mock-strong",
        messages=[Message(role=Role.USER, content="hello")],
    )


def _response(request_id) -> InferenceResponse:
    return InferenceResponse(
        request_id=request_id,
        provider="mock",
        model="mock-strong",
        output_text="hi",
        finish_reason=FinishReason.STOP,
        token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        latency=LatencyRecord(total_latency_ms=42.0),
    )


def test_fresh_store_applies_schema(store: ExperimentStore) -> None:
    # No exception on construction means migrations ran; sanity-check a query works.
    assert store.list_runs() == []


def test_record_and_get_run_roundtrips(store: ExperimentStore) -> None:
    manifest = _manifest()
    store.record_run(manifest)
    fetched = store.get_run("run-1")
    assert fetched is not None
    assert fetched.run_id == "run-1"
    assert fetched.seed == 42
    assert fetched.router_config == {"strategy": "static"}
    assert fetched.environment == {"CI": "false"}


def test_get_run_returns_none_when_missing(store: ExperimentStore) -> None:
    assert store.get_run("does-not-exist") is None


def test_list_runs_returns_all_recorded_runs(store: ExperimentStore) -> None:
    store.record_run(_manifest("run-1"))
    store.record_run(_manifest("run-2"))
    run_ids = {m.run_id for m in store.list_runs()}
    assert run_ids == {"run-1", "run-2"}


def test_record_run_upserts_by_run_id(store: ExperimentStore) -> None:
    store.record_run(_manifest("run-1"))
    updated = _manifest("run-1").model_copy(update={"seed": 99})
    store.record_run(updated)
    fetched = store.get_run("run-1")
    assert fetched is not None
    assert fetched.seed == 99
    assert len(store.list_runs()) == 1


def test_record_and_get_request_roundtrips(store: ExperimentStore) -> None:
    request = _request()
    store.record_request(request, run_id="run-1")
    fetched = store.get_request(request.request_id)
    assert fetched is not None
    assert fetched.request_id == request.request_id
    assert fetched.model == "mock-strong"
    assert fetched.messages[0].content == "hello"


def test_requests_df_filters_by_run_id(store: ExperimentStore) -> None:
    r1, r2 = _request(), _request()
    store.record_request(r1, run_id="run-1")
    store.record_request(r2, run_id="run-2")
    df = store.requests_df(run_id="run-1")
    assert df.height == 1
    assert df["request_id"][0] == str(r1.request_id)


def test_record_response(store: ExperimentStore) -> None:
    request_id = uuid4()
    store.record_response(_response(request_id), run_id="run-1")
    df = store.responses_df(run_id="run-1")
    assert df.height == 1
    assert df["output_text"][0] == "hi"
    assert df["input_tokens"][0] == 10


def test_record_response_roundtrips_cost_basis(store: ExperimentStore) -> None:
    request_id = uuid4()
    response = _response(request_id).model_copy(
        update={
            "cost": CostRecord(
                input_cost_usd=0.001,
                output_cost_usd=0.002,
                pricing_version="test-v1",
                basis=CostBasis.SIMULATED,
            )
        }
    )
    store.record_response(response, run_id="run-1")
    df = store.responses_df(run_id="run-1")
    assert df["cost_basis"][0] == "simulated"
    assert df["pricing_version"][0] == "test-v1"


def test_record_response_with_error(store: ExperimentStore) -> None:
    request_id = uuid4()
    response = _response(request_id).model_copy(
        update={
            "finish_reason": FinishReason.ERROR,
            "error": ErrorInfo(
                category=FailureCategory.TIMEOUT, message="timed out", retryable=True
            ),
        }
    )
    store.record_response(response, run_id="run-1")
    df = store.responses_df(run_id="run-1")
    assert df["error_category"][0] == "timeout"
    assert df["error_retryable"][0] is True


def test_record_trace_event(store: ExperimentStore) -> None:
    event = TraceEvent(event_type=TraceEventType.REQUEST_STARTED, payload={"note": "ok"})
    store.record_trace_event(event, run_id="run-1")
    df = store.trace_events_df(run_id="run-1")
    assert df.height == 1
    assert df["event_type"][0] == "request_started"


def test_record_routing_decision(store: ExperimentStore) -> None:
    request_id = uuid4()
    decision = RoutingDecision(
        selected_model="mock-cheap",
        candidate_scores={"mock-cheap": 0.9, "mock-strong": 0.95},
        explanation="within budget",
        fallback_order=["mock-strong"],
    )
    store.record_routing_decision(request_id, decision, run_id="run-1")
    df = store.routing_decisions_df(run_id="run-1")
    assert df.height == 1
    assert df["selected_model"][0] == "mock-cheap"


def test_record_and_get_outcome_events_roundtrip(store: ExperimentStore) -> None:
    decision_id = uuid4()
    event = OutcomeEvent(
        run_id="run-1",
        task_id="task-1",
        initial_routing_decision_id=decision_id,
        final_provider="mock",
        final_model="model-b",
        attempt_count=2,
        succeeded=True,
        failure_category=None,
        total_cost_usd=0.01,
        total_latency_ms=15.0,
        recovery_actions=["fallback_model"],
    )
    store.record_outcome_event(event)

    fetched = store.get_outcome_events("run-1")
    assert len(fetched) == 1
    assert fetched[0].task_id == "task-1"
    assert fetched[0].initial_routing_decision_id == decision_id
    assert fetched[0].attempt_count == 2
    assert fetched[0].succeeded is True
    assert fetched[0].recovery_actions == ["fallback_model"]
    assert fetched[0].terminal is True


def test_outcome_events_df_filters_by_run_id(store: ExperimentStore) -> None:
    store.record_outcome_event(
        OutcomeEvent(
            run_id="run-1",
            final_provider="mock",
            final_model="a",
            attempt_count=1,
            succeeded=True,
            total_latency_ms=1.0,
        )
    )
    store.record_outcome_event(
        OutcomeEvent(
            run_id="run-2",
            final_provider="mock",
            final_model="b",
            attempt_count=1,
            succeeded=False,
            total_latency_ms=1.0,
        )
    )
    assert store.outcome_events_df(run_id="run-1").height == 1
    assert store.outcome_events_df().height == 2


def test_record_and_get_eval_results_roundtrip_in_sequence_order(store: ExperimentStore) -> None:
    request_id = uuid4()
    result_a = EvalResult(
        case_id="case-1",
        request_id=request_id,
        repetition=0,
        sequence=1,
        succeeded=True,
        score=1.0,
        grader_kind=GraderKind.EXACT_MATCH,
        explanation="ok",
        latency_ms=5.0,
        total_tokens=3,
    )
    result_b = result_a.model_copy(
        update={
            "result_id": uuid4(),
            "sequence": 0,
            "case_id": "case-0",
            "score": 0.5,
            "succeeded": False,
        }
    )
    # Recorded out of sequence order to prove read-back sorts by `sequence`, not
    # insertion order (which plain SQL does not otherwise guarantee).
    store.record_eval_result(result_a, run_id="run-1")
    store.record_eval_result(result_b, run_id="run-1")

    fetched = store.get_eval_results("run-1")
    assert [r.sequence for r in fetched] == [0, 1]
    assert fetched[0].case_id == "case-0"
    assert fetched[0].succeeded is False
    assert fetched[1].case_id == "case-1"


def test_eval_results_df_filters_by_run_id(store: ExperimentStore) -> None:
    r1 = EvalResult(
        case_id="c1",
        request_id=uuid4(),
        repetition=0,
        sequence=0,
        succeeded=True,
        score=1.0,
        grader_kind=GraderKind.NUMERIC,
        explanation="ok",
        latency_ms=1.0,
        total_tokens=1,
    )
    store.record_eval_result(r1, run_id="run-1")
    store.record_eval_result(r1.model_copy(update={"result_id": uuid4()}), run_id="run-2")
    df = store.eval_results_df(run_id="run-1")
    assert df.height == 1


def test_export_parquet_and_jsonl(store: ExperimentStore, tmp_path: Path) -> None:
    store.record_run(_manifest())
    parquet_path = tmp_path / "runs.parquet"
    jsonl_path = tmp_path / "runs.jsonl"
    store.export_parquet("runs", parquet_path)
    store.export_jsonl("runs", jsonl_path)
    assert parquet_path.exists()
    assert jsonl_path.exists()
    assert jsonl_path.read_text(encoding="utf-8").strip() != ""


def test_export_rejects_unknown_table(store: ExperimentStore, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown table"):
        store.export_parquet("drop_all_the_things", tmp_path / "x.parquet")


def test_file_backed_store_persists_across_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "experiments.duckdb"
    with ExperimentStore(db_path) as s:
        s.record_run(_manifest("run-1"))

    with ExperimentStore(db_path) as s:
        fetched = s.get_run("run-1")
        assert fetched is not None
        assert fetched.run_id == "run-1"
