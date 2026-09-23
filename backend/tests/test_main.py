import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend import main, recommendation
from backend.graph import nodes
from backend.schemas import Track

THREAD_ID = "11111111-1111-4111-8111-111111111111"
REQUEST_ID = "22222222-2222-4222-8222-222222222222"


def request_body(message: str = "퇴근길 음악") -> dict[str, str]:
    """메시지를 요청 JSON으로 바꿔준다."""
    return {
        "thread_id": THREAD_ID,
        "request_id": REQUEST_ID,
        "message": message,
    }


def make_track(track_id: str = "123", reason: str = "폴백 이유") -> Track:
    """pgvector가 반환한 가짜 Track이다."""
    return Track(
        track_id=track_id,
        title="TOMBOY",
        artist="HYUKOH",
        artwork_url="https://example.com/art.jpg",
        preview_url="https://example.com/preview.m4a",
        store_url="https://music.apple.com/track/123",
        reason=reason,
    )


class FakeStream:
    """openai Stream이 지원하는 컨텍스트 매니저 프로토콜을 흉내낸다."""

    def __init__(self, events: list[object]) -> None:
        self._events = events

    """리스트로 Stream을 흉내내기 위해선 enter와 exit을 정의한다"""
    def __enter__(self) -> "FakeStream":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def __iter__(self):
        return iter(self._events)


class FakeResponses:
    """가짜 openai 생성. 'stream' 여부에 따라 스트리밍 아니면 추천 이유 생성이다."""
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return FakeStream(
                [
                    SimpleNamespace(type="response.output_text.delta", delta="퇴근길에 "),
                    SimpleNamespace(type="response.output_text.delta", delta="어울리는 곡이에요."),
                    SimpleNamespace(type="response.completed"),
                ]
            )
        # reason_node가 호출하는 곡별 추천 이유 생성(assign_reasons)의 유일한 비스트리밍 호출
        return SimpleNamespace(output_text=json.dumps({"123": "퇴근길에 잘 어울리는 곡이에요."}))


class FakeOpenAI:
    def __init__(self) -> None:
        self.responses = FakeResponses()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, FakeOpenAI]:
    """파이프라인의 일부만 바꾸게 하는 애가 얘다."""
    fake_openai = FakeOpenAI()
    monkeypatch.setenv("OPENROUTER_LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "gpt-5.6-luna")
    monkeypatch.setattr(recommendation, "OpenAI", lambda **kwargs: fake_openai)

    # NOTE: classify_node는 OpenAI 구조화 출력으로 intent를 뽑는데, 여기 fixture의
    # FakeOpenAI는 이유 생성용 JSON만 흉내내므로 intent 분류는 목으로 고정해
    # recommend 경로를 계속 타게 한다.
    monkeypatch.setattr(nodes, "classify_intent", lambda client, message: "recommend")

    # NOTE: embed_node/search_node는 각각 Gemini 임베딩 API와 PostgreSQL(pgvector) 조회라는
    # 외부 I/O 경계이므로, 그 경계에서 가짜 값을 주입해 그래프 배선(embed → search →
    # reason)은 실제 코드로 검증한다.
    monkeypatch.setattr(nodes, "get_embedding_client", lambda: object())
    monkeypatch.setattr(nodes, "embed_query", lambda embedding_client, message: [0.1, 0.2, 0.3])
    monkeypatch.setattr(
        nodes,
        "vector_recommendation",
        lambda query_vector, message: (
            [make_track()],
            {"123": {"lofi": 0.9, "chill": 0.7}},
        ),
    )

    return TestClient(main.app), fake_openai


def test_chat_streams_text_before_completed_track_cards(
    client: tuple[TestClient, FakeOpenAI],
) -> None:
    test_client, fake_openai = client
    response = test_client.post(
        "/v1/chat/messages",
        json=request_body(" 퇴근길 음악 "),
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert 'event: text\ndata: {"delta": "퇴근길에 "}' in response.text
    assert "event: tracks" in response.text
    assert '"preview_url": "https://example.com/preview.m4a"' in response.text
    """최종 답변 스트리밍이 곡별 추천 이유 생성보다 먼저 오는가?"""
    assert response.text.index("event: text") < response.text.index("event: tracks")
    assert "event: done" in response.text
    # 1) reason_node의 곡별 추천 이유 생성  2) 최종 답변 스트리밍
    assert len(fake_openai.responses.calls) == 2
    assert test_client.post("/chat/stream", json=request_body()).status_code == 404


def test_chat_no_match_returns_empty_tracks(
    client: tuple[TestClient, FakeOpenAI], monkeypatch: pytest.MonkeyPatch
) -> None:
    """매치되는 음악이 없을 때를 보여준다."""
    test_client, fake_openai = client
    monkeypatch.setattr(
        nodes, "vector_recommendation", lambda query_vector, message: ([], {})
    )

    response = test_client.post("/v1/chat/messages", json=request_body())

    assert response.status_code == 200
    assert "event: text" in response.text
    assert 'event: tracks\ndata: {"tracks": []}' in response.text
    assert "event: done" in response.text
    # 추천곡이 없으면 이유 생성·최종 답변 생성 모두 LLM을 호출하지 않는다.
    assert fake_openai.responses.calls == []


def test_chat_music_catalog_unavailable_returns_503(
    client: tuple[TestClient, FakeOpenAI], monkeypatch: pytest.MonkeyPatch
) -> None:
    """DB 장애가 있을 때 오류가 잘 나오는지 확인한다."""
    test_client, _ = client

    def failing_vector_recommendation(query_vector: list[float], message: str):
        raise recommendation._catalog_unavailable()

    monkeypatch.setattr(nodes, "vector_recommendation", failing_vector_recommendation)
    response = test_client.post("/v1/chat/messages", json=request_body())

    assert response.status_code == 503
    assert response.json()["details"]["reason"] == "MUSIC_CATALOG_UNAVAILABLE"


def test_chat_rejects_blank_message(client: tuple[TestClient, FakeOpenAI]) -> None:
    """빈 메시지를 받으면 오류가 잘 나오는지 확인한다."""
    test_client, _ = client
    response = test_client.post("/v1/chat/messages", json=request_body("   "))

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_REQUEST"


def test_chat_requires_openrouter_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """openrouter api key가 없으면 오류가 잘 나오는지 확인한다."""
    monkeypatch.delenv("OPENROUTER_LLM_API_KEY", raising=False)
    response = TestClient(main.app).post("/v1/chat/messages", json=request_body("안녕"))

    assert response.status_code == 503


def test_chat_requires_openrouter_embedding_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """embedding용 openrouter api key가 없으면 오류가 잘 나오는지 확인한다."""
    monkeypatch.setenv("OPENROUTER_LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "openai/test-model")
    monkeypatch.setattr(nodes, "classify_intent", lambda client, messages: "recommend")
    monkeypatch.delenv("OPENROUTER_EMBEDDING_API_KEY", raising=False)
    monkeypatch.setattr(recommendation, "OpenAI", lambda **kwargs: FakeOpenAI())
    response = TestClient(main.app).post("/v1/chat/messages", json=request_body("안녕"))

    assert response.status_code == 503


def test_health_does_not_call_external_services() -> None:
    """health 엔드포인트가 외부 호출 없이도 status가 잘 나오는지 확인한다."""
    response = TestClient(main.app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


class FakeSession:
    """SessionLocal()이 반환하는 세션의 context manager 프로토콜을 흉내낸다."""

    def __init__(self, on_execute) -> None:
        self._on_execute = on_execute

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def execute(self, *args: object, **kwargs: object) -> None:
        self._on_execute()


def test_readiness_returns_ok_when_db_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """DB에 SELECT 1이 성공하면 readiness가 ok를 반환하는지 확인한다."""
    from db import models as db_models

    monkeypatch.setattr(db_models, "SessionLocal", lambda: FakeSession(lambda: None))

    response = TestClient(main.app).get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_returns_503_when_db_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """DB 연결이 실패하면 readiness가 503을 반환하는지 확인한다."""
    from db import models as db_models

    def failing_execute() -> None:
        raise RuntimeError("DB connection failed")

    monkeypatch.setattr(db_models, "SessionLocal", lambda: FakeSession(failing_execute))

    response = TestClient(main.app).get("/health/ready")

    assert response.status_code == 503
    assert response.json()["details"]["reason"] == "MUSIC_CATALOG_UNAVAILABLE"


@pytest.mark.parametrize("intent", ["guide", "clarify", "lookup", "out_of_scope"])
def test_non_recommendation_skips_search(client, monkeypatch, intent):
    test_client, fake_openai = client
    monkeypatch.setattr(nodes, "classify_intent", lambda client, messages: intent)

    def unexpected(*args, **kwargs):
        pytest.fail("비추천 요청에서 임베딩 또는 DB 검색 실행")

    monkeypatch.setattr(nodes, "get_embedding_client", unexpected)
    monkeypatch.setattr(nodes, "vector_recommendation", unexpected)
    response = test_client.post("/v1/chat/messages", json=request_body("안녕"))

    assert response.status_code == 200
    assert "아직 지원하지 않는 요청" in response.text
    assert 'event: tracks\ndata: {"tracks": []}' in response.text
    assert "event: done" in response.text
    assert fake_openai.responses.calls == []


def test_local_chat_page_uses_v1_fields() -> None:
    """테스트용 HTML 페이지가 v1에 맞게 나오는지 확인한다."""
    response = TestClient(main.app).get("/app/")

    assert response.status_code == 200
    assert 'const CHAT_URL = "/v1/chat/messages"' in response.text
    assert "thread_id" in response.text
    assert "request_id" in response.text
    assert "response.body.getReader()" in response.text
    assert "addTrackCards" in response.text
    assert "store_url" in response.text
