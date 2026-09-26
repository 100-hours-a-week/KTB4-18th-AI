"""인증 실패가 채팅·전사 실행을 차단하는지 검증한다."""

import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.recommendation import ChatOutcome

ERROR = {"code": "UNAUTHORIZED", "message": "서버 인증에 실패했습니다.", "details": None}


@pytest.mark.parametrize("path", ["/v1/chat/messages", "/v1/transcriptions"])
@pytest.mark.parametrize("authorization", [
    None, "", "test-service-key", "Basic test-service-key", "Bearer",
    "Bearer ", "Bearer wrong-key", "Bearer test-service-key extra", "bearer wrong-key",
])
def test_invalid_key_blocks_processing(monkeypatch, path, authorization):
    def unexpected(*args, **kwargs):
        pytest.fail("인증 실패 후 AI 처리 실행")

    monkeypatch.setattr(main, "prepare_recommendation", unexpected)
    monkeypatch.setattr(main, "transcribe_audio", unexpected)
    headers = {} if authorization is None else {"Authorization": authorization}
    with TestClient(main.app) as client:
        response = client.post(path, headers=headers)
    assert response.status_code == 401
    assert response.json() == ERROR


def test_valid_key_preserves_chat_and_transcription_contract(monkeypatch):
    monkeypatch.setattr(main, "prepare_recommendation", lambda *args: ChatOutcome(kind="static", text="안내"))
    monkeypatch.setattr(main, "transcribe_audio", lambda audio: "전사문")
    with TestClient(main.app, headers={"Authorization": "Bearer test-service-key"}) as client:
        chat = client.post("/v1/chat/messages", json={
            "thread_id": "11111111-1111-4111-8111-111111111111",
            "request_id": "22222222-2222-4222-8222-222222222222",
            "message": "사용법",
        })
        stt = client.post("/v1/transcriptions", files={"audio": ("recording.webm", b"audio", "audio/webm")})
    assert chat.status_code == 200
    assert "event: done" in chat.text
    assert stt.json() == {"transcript": "전사문"}


def test_bearer_scheme_is_case_insensitive(monkeypatch):
    monkeypatch.setattr(main, "transcribe_audio", lambda audio: "전사문")
    with TestClient(main.app) as client:
        response = client.post("/v1/transcriptions",
                               headers={"Authorization": "bearer test-service-key"},
                               files={"audio": ("recording.webm", b"audio", "audio/webm")})
    assert response.status_code == 200


@pytest.mark.parametrize("key", [None, "", "   "])
def test_missing_server_key_prevents_startup(monkeypatch, key):
    if key is None:
        monkeypatch.delenv("AI_SERVICE_API_KEY", raising=False)
    else:
        monkeypatch.setenv("AI_SERVICE_API_KEY", key)
    with pytest.raises(RuntimeError, match="AI_SERVICE_API_KEY"):
        with TestClient(main.app):
            pass


def test_health_endpoints_remain_unauthenticated(monkeypatch):
    monkeypatch.setattr(main, "database_is_ready", lambda: True)
    with TestClient(main.app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/readiness").status_code == 200
        schema = client.get("/openapi.json").json()
    for path in ("/v1/chat/messages", "/v1/transcriptions"):
        operation = schema["paths"][path]["post"]
        assert "401" in operation["responses"]
        assert any(p["name"] == "Authorization" for p in operation["parameters"])
