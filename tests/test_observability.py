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


@pytest.mark.parametrize("events,expected,http_status", [
    ([("text", {"delta": "안내"}), ("tracks", {"tracks": []}), ("done", {})], "completed", 200),
    ([("error", {"reason": "MODEL_UNAVAILABLE"})], "error", 200),
    ([("verdict", {"usable": False}), ("rejected", {"reason": "scene_unreadable"}), ("done", {})], "rejected", 200),
    ([("rejected", {"reason": "provider_blocked"}), ("done", {})], "provider_blocked", 200),
    ([("text", {"delta": "중단"})], "aborted", 200),
    ([], "error", 503),
    ([], "error", 422),
])
def test_request_trace_contains_stream_outcome(traced_runs, events, expected, http_status):
    async def app(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": http_status})
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
    assert bool(root.error) == (expected == "error")
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


@pytest.mark.parametrize("fallback,provider", [(False, "openrouter"), (True, "self-hosted")])
def test_photo_usage_records_the_actual_provider(monkeypatch, traced_runs, fallback, provider):
    import httpx
    from backend.chat.photo import vlm

    for key in ("PHOTO_VLM_API_KEY", "OPENROUTER_LLM_API_KEY", "PHOTO_FALLBACK_API_KEY"):
        monkeypatch.setenv(key, "test-only")
    monkeypatch.setenv("PHOTO_VLM_MODEL", "primary-model")
    monkeypatch.setenv("PHOTO_FALLBACK_MODEL", "fallback-model")
    monkeypatch.setenv("PHOTO_FALLBACK_URL", "https://test.invalid/v1")
    scene = {"usable": True, "reject_reason": None, "place_type": "park",
             "visual_elements": ["trees"], "lighting": "daylight", "mood": ["calm"],
             "music_query": "차분한 음악", "recommendation_context": "나무가 보여요."}

    async def post(client, url, **kwargs):
        return httpx.Response(200, request=httpx.Request("POST", url), json={
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(scene)}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        })

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    client = Client(api_url="http://localhost:1984", auto_batch_tracing=False, info=LangSmithInfo())
    try:
        with tracing_context(enabled=True):
            with trace("photo-request", client=client):
                result = asyncio.run(vlm.call_vlm(b"image", fallback=fallback))
    finally:
        client.close()
    run = next(r for r in traced_runs if r.name == "사진 분석")
    assert result.usable
    assert run.metadata["ls_provider"] == provider
    assert run.metadata["ls_model_name"] == ("fallback-model" if fallback else "primary-model")
    assert run.metadata["usage_metadata"]["input_tokens"] == 5
    assert run.metadata["cost_available"] is False
    assert "total_cost" not in run.metadata["usage_metadata"]
    assert run.inputs == {"image_bytes": 5, "fallback": fallback}


def test_staging_http_request_is_traced_without_revision(monkeypatch, traced_runs):
    from fastapi.testclient import TestClient
    from backend.main import app
    from backend.chat import router
    from backend.chat.service import ChatOutcome
    from backend import observability

    monkeypatch.delenv("APP_REVISION", raising=False)
    monkeypatch.setattr(router, "prepare_recommendation", lambda *args: ChatOutcome(
        kind="static", text="어떤 분위기의 음악을 찾으세요?"))
    client = Client(api_url="http://localhost:1984", auto_batch_tracing=False, info=LangSmithInfo())
    monkeypatch.setattr(observability, "trace", lambda *args, **kwargs: trace(*args, client=client, **kwargs))
    thread_id, request_id = str(uuid4()), str(uuid4())
    try:
        response = TestClient(app).post("/v1/chat/messages", json={
            "thread_id": thread_id, "request_id": request_id, "message": "음악 추천해줘",
        })
    finally:
        client.close()
    root = next(r for r in traced_runs if r.name == "머문음 요청")
    assert response.status_code == 200
    assert [part.splitlines()[0] for part in response.text.strip().split("\n\n")] == [
        "event: text", "event: tracks", "event: done",
    ]
    assert root.session_name == "meomuneum-staging"
    assert root.inputs["message"] == "음악 추천해줘"
    assert root.metadata["request_id"] == request_id
    assert root.metadata["thread_id"] == thread_id
    assert root.metadata["revision_id"] == "unknown"
    assert root.outputs["text"] == "어떤 분위기의 음악을 찾으세요?"
    assert root.outputs["outcome"] == "completed"
    assert root.end_time is not None
