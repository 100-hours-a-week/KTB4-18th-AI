import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend import main, recommendation
from backend.schemas import Track

THREAD_ID = "11111111-1111-4111-8111-111111111111"
REQUEST_ID = "22222222-2222-4222-8222-222222222222"


def fake_stream(events):
    stream = MagicMock()
    stream.__enter__.return_value = stream
    stream.__iter__.return_value = iter(events)
    return stream


def request_body(message: str = "퇴근길 음악") -> dict[str, str]:
    return {
        "thread_id": THREAD_ID,
        "request_id": REQUEST_ID,
        "message": message,
    }


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, Mock, Mock]:
    """DB 연결 이후에도 유지할 API 계약을 가짜 검색 결과로 검증한다."""
    track = Track(
        track_id="123",
        title="TOMBOY",
        artist="HYUKOH",
        artwork_url=None,
        preview_url="https://example.com/preview.m4a",
        store_url="https://music.apple.com/track/123",
        reason="차분한 퇴근길 분위기와 어울리는 곡입니다.",
    )
    search = Mock(return_value=("차분한 기타 중심 음악", [track]))
    create = Mock(
        return_value=fake_stream([
            SimpleNamespace(type="response.output_text.delta", delta="퇴근길에 "),
            SimpleNamespace(type="response.output_text.delta", delta="어울리는 곡이에요."),
            SimpleNamespace(type="response.completed"),
        ])
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(recommendation, "search_tracks", search)
    monkeypatch.setattr(
        recommendation, "OpenAI",
        lambda api_key: SimpleNamespace(responses=SimpleNamespace(create=create)),
    )
    return TestClient(main.app), search, create


def test_chat_streams_text_before_completed_track_cards(client) -> None:
    test_client, search, create = client
    response = test_client.post("/v1/chat/messages", json=request_body(" 퇴근길 음악 "))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert 'event: text\ndata: {"delta": "퇴근길에 "}' in response.text
    assert '"preview_url": "https://example.com/preview.m4a"' in response.text
    assert response.text.index("event: text") < response.text.index("event: tracks")
    assert response.text.index("event: tracks") < response.text.index("event: done")
    search.assert_called_once_with("퇴근길 음악")
    create.assert_called_once()
    assert test_client.post("/chat/stream", json=request_body()).status_code == 404


def test_empty_search_returns_empty_tracks_without_model_call(client) -> None:
    test_client, search, create = client
    search.return_value = ("차분한 기타 중심 음악", [])
    response = test_client.post("/v1/chat/messages", json=request_body())

    assert response.status_code == 200
    assert "event: text" in response.text
    assert 'event: tracks\ndata: {"tracks": []}' in response.text
    assert "event: done" in response.text
    create.assert_not_called()


def test_chat_rejects_blank_message(client) -> None:
    test_client, search, create = client
    response = test_client.post("/v1/chat/messages", json=request_body("   "))

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_REQUEST"
    search.assert_not_called()
    create.assert_not_called()


@pytest.mark.parametrize("has_api_key", [False, True])
def test_unconnected_search_returns_503_before_creating_model_client(
    monkeypatch: pytest.MonkeyPatch, has_api_key: bool
) -> None:
    if has_api_key:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    else:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    create_client = Mock(side_effect=AssertionError("모델 클라이언트를 생성하면 안 됩니다."))
    monkeypatch.setattr(recommendation, "OpenAI", create_client)
    response = TestClient(main.app).post("/v1/chat/messages", json=request_body())

    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {
        "code": "SERVICE_UNAVAILABLE",
        "message": "음악 DB 검색 연결을 준비 중입니다.",
        "details": {"reason": "MUSIC_CATALOG_UNAVAILABLE"},
    }
    create_client.assert_not_called()


def test_chat_requires_openai_api_key_after_search(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_client, _, create = client
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = test_client.post("/v1/chat/messages", json=request_body())

    assert response.status_code == 503
    assert response.json()["details"]["reason"] == "MODEL_UNAVAILABLE"
    create.assert_not_called()


def test_health_does_not_call_external_services() -> None:
    response = TestClient(main.app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_local_chat_page_uses_v1_fields() -> None:
    response = TestClient(main.app).get("/app/")

    assert response.status_code == 200
    assert 'const CHAT_URL = "/v1/chat/messages"' in response.text
    assert "thread_id" in response.text
    assert "request_id" in response.text
    assert "response.body.getReader()" in response.text
    assert "addTrackCards" in response.text
    assert "store_url" in response.text
