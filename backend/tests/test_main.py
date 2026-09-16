import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend import main, music_search, recommendation

THREAD_ID = "11111111-1111-4111-8111-111111111111"
REQUEST_ID = "22222222-2222-4222-8222-222222222222"


def request_body(message: str = "퇴근길 음악") -> dict[str, str]:
    return {
        "thread_id": THREAD_ID,
        "request_id": REQUEST_ID,
        "message": message,
    }


class FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter(
                [
                    SimpleNamespace(type="response.output_text.delta", delta="퇴근길에 "),
                    SimpleNamespace(type="response.output_text.delta", delta="어울리는 곡이에요."),
                ]
            )
        if len(self.calls) == 1:
            return SimpleNamespace(output_text="indie")
        return SimpleNamespace(output_text="퇴근길에 어울리는 곡이에요.")


class FakeOpenAI:
    def __init__(self) -> None:
        self.responses = FakeResponses()


class FakeHttpResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, object]:
        return self.payload


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, FakeOpenAI]:
    fake_openai = FakeOpenAI()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LASTFM_API_KEY", "lastfm-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-luna")
    monkeypatch.setattr(recommendation, "OpenAI", lambda api_key: fake_openai)

    def fake_get(url: str, **kwargs: object) -> FakeHttpResponse:
        if "audioscrobbler" in url:
            return FakeHttpResponse(
                {
                    "tracks": {
                        "track": [
                            {
                                "name": "TOMBOY",
                                "artist": {"name": "HYUKOH"},
                            }
                        ]
                    }
                }
            )
        return FakeHttpResponse(
            {
                "results": [
                    {
                        "trackId": 123,
                        "trackName": "TOMBOY",
                        "artistName": "HYUKOH",
                        "artworkUrl100": "https://example.com/art.jpg",
                        "previewUrl": "https://example.com/preview.m4a",
                        "trackViewUrl": "https://music.apple.com/track/123",
                    }
                ]
            }
        )

    monkeypatch.setattr(music_search.httpx, "get", fake_get)
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
    assert response.text.index("event: text") < response.text.index("event: tracks")
    assert "event: done" in response.text
    assert len(fake_openai.responses.calls) == 2
    assert test_client.post("/chat/stream", json=request_body()).status_code == 404


def test_itunes_mismatch_returns_empty_tracks(
    client: tuple[TestClient, FakeOpenAI], monkeypatch: pytest.MonkeyPatch
) -> None:
    test_client, _ = client
    original_get = music_search.httpx.get

    def mismatched_get(url: str, **kwargs: object) -> FakeHttpResponse:
        if "audioscrobbler" in url:
            return original_get(url, **kwargs)
        return FakeHttpResponse(
            {
                "results": [
                    {
                        "trackId": 999,
                        "trackName": "TOMBOY",
                        "artistName": "Another Artist",
                        "previewUrl": "https://example.com/wrong.m4a",
                        "trackViewUrl": "https://music.apple.com/track/999",
                    }
                ]
            }
        )

    monkeypatch.setattr(music_search.httpx, "get", mismatched_get)
    response = test_client.post("/v1/chat/messages", json=request_body())

    assert response.status_code == 200
    assert "event: text" in response.text
    assert 'event: tracks\ndata: {"tracks": []}' in response.text
    assert "event: done" in response.text


def test_chat_rejects_blank_message(client: tuple[TestClient, FakeOpenAI]) -> None:
    test_client, _ = client
    response = test_client.post("/v1/chat/messages", json=request_body("   "))

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_REQUEST"


def test_chat_requires_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = TestClient(main.app).post("/v1/chat/messages", json=request_body("안녕"))

    assert response.status_code == 503


def test_chat_requires_lastfm_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("LASTFM_API_KEY", raising=False)
    monkeypatch.setattr(recommendation, "OpenAI", lambda api_key: FakeOpenAI())
    response = TestClient(main.app).post("/v1/chat/messages", json=request_body("안녕"))

    assert response.status_code == 503


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
