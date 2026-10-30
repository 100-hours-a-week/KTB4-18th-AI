"""외부 호출 없이 환경 분리·SSE 종료·실제 사용량 계측을 확인한다."""

import asyncio
import json
import os
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from langsmith import Client, RunTree, trace, tracing_context
from langsmith.schemas import LangSmithInfo
from langchain_core.tracers.langchain import LangChainTracer

from backend.observability import RequestTracingMiddleware, configure_tracing, record_usage
from backend.chat.streaming import stream_fast_answer
from db.types import Track


@pytest.fixture
def traced_runs(monkeypatch):
    runs = []
    monkeypatch.setenv("APP_ENV", "staging")
    monkeypatch.setenv("APP_REVISION", "test-revision")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_AGENT_ID", "old-agent")
    monkeypatch.setenv("LANGSMITH_AGENT_ENVIRONMENT", "production")
    configure_tracing()
    monkeypatch.setattr(RunTree, "post", lambda run, *a, **k: runs.append(run))
    monkeypatch.setattr(RunTree, "patch", lambda *a, **k: None)
    return runs


def test_agent_environment_and_langgraph_use_one_destination(traced_runs):
    assert os.environ["LANGSMITH_PROJECT"] == "meomuneum-staging"
    with tracing_context(enabled=True):
        with trace("request", project_name="meomuneum-staging"):
            tracer = LangChainTracer(client=MagicMock())
            run = tracer.on_chain_start({"name": "LangGraph"}, {}, run_id=uuid4())
            assert run.session_name == "meomuneum-staging"
            assert run.agent_id is None and run.agent_environment is None
            tracer.on_chain_end({}, run_id=run.id)


def test_fast_response_is_a_request_child_without_model_usage(traced_runs):
    tracks = [Track(track_id="1", title="song", artist="artist", artwork_url=None,
                    preview_url=None)]
    client = Client(api_url="http://localhost:1984", auto_batch_tracing=False, info=LangSmithInfo())
    try:
        with tracing_context(enabled=True):
            with trace("request", project_name="meomuneum-staging", client=client) as root:
                events = list(stream_fast_answer("고속 모드로 추천한 음악이에요.", tracks))
    finally:
        client.close()

    run = next(run for run in traced_runs if run.name == "고속 응답 전송")
    assert run.parent_run_id == root.id
    assert run.run_type == "chain"
    assert run.inputs["track_count"] == 1
    assert run.outputs == {"text": "고속 모드로 추천한 음악이에요."}
    assert "usage_metadata" not in run.metadata
    assert [event.split("\n", 1)[0] for event in events] == [
        "event: text", "event: tracks", "event: done",
    ]


@pytest.mark.parametrize("events,expected", [
    ([("text", {"delta": "안내"}), ("tracks", {"tracks": []}), ("done", {})], "completed"),
    ([("error", {"reason": "MODEL_UNAVAILABLE"})], "error"),
    ([("verdict", {"usable": False}), ("rejected", {"reason": "scene_unreadable"}), ("done", {})], "rejected"),
    ([("rejected", {"reason": "provider_blocked"}), ("done", {})], "provider_blocked"),
    ([("text", {"delta": "중단"})], "aborted"),
])
def test_request_trace_contains_stream_outcome(traced_runs, events, expected):
    async def app(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 200})
        for event, data in events:
            body = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()
            await send({"type": "http.response.body", "body": body, "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def receive():
        return {"type": "http.request", "body": json.dumps({"message": "음악", "thread_id": "thread", "request_id": "request"}).encode()}

    sent = []
    async def send(message):
        sent.append(message)

    asyncio.run(RequestTracingMiddleware(app)({"type": "http", "path": "/v1/chat/messages"}, receive, send))
    root = traced_runs[0]
    assert root.session_name == "meomuneum-staging"
    assert root.outputs["outcome"] == expected
    assert root.inputs["message"] == "음악"
    assert root.metadata["revision_id"] == "test-revision"
    assert root.end_time is not None
    assert sent[-1]["more_body"] is False
    assert ("cards_ms" in root.outputs) == any(e == "tracks" for e, _ in events)
    assert ("rejection_ms" in root.outputs) == any(e == "rejected" for e, _ in events)


def test_usage_records_actual_cost_and_marks_missing_cost(traced_runs):
    with tracing_context(enabled=True):
        with trace("LLM", project_name="meomuneum-staging", run_type="llm") as run:
            record_usage({"prompt_tokens": 20, "completion_tokens": 5, "cost": .002,
                          "completion_tokens_details": {"reasoning_tokens": 2}}, "model")
            usage = run.metadata["usage_metadata"]
            assert usage["input_tokens"] == 20 and usage["output_tokens"] == 5
            assert usage["total_cost"] == .002
            assert usage["output_token_details"]["reasoning"] == 2
        with trace("LLM", project_name="meomuneum-staging", run_type="llm") as run:
            record_usage({"prompt_tokens": 4}, "model")
            assert "total_cost" not in run.metadata["usage_metadata"]
            assert run.metadata["cost_available"] is False


def test_disabled_request_does_not_create_trace(monkeypatch, traced_runs):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    async def app(scope, receive, send):
        with trace("must remain local"):
            pass
    asyncio.run(RequestTracingMiddleware(app)({"type": "http", "path": "/v1/chat/messages"}, None, None))
    assert traced_runs == []


def test_invalid_environment_fails_before_logging(monkeypatch):
    monkeypatch.setenv("APP_ENV", "prodution")
    with pytest.raises(ValueError, match="APP_ENV"):
        configure_tracing()
