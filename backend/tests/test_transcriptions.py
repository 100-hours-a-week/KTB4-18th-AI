"""실제 FFmpeg 전처리와 모의 STT 응답으로 전사 계약을 검증한다."""

import base64
import io
import math
import struct
import subprocess
import sys
import wave
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from imageio_ffmpeg import get_ffmpeg_exe

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend import main, transcriptions as stt


def audio_wav(seconds=0.1, silent=False):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"".join(struct.pack("<h", 0 if silent else int(5000 * math.sin(i / 10)))
                                 for i in range(int(seconds * 16000))))
    return stream.getvalue()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("OPENROUTER_STT_API_KEY", "test-key")
    monkeypatch.delenv("STT_MODEL", raising=False)
    # 실패 경로가 의도치 않게 외부 API를 호출하면 테스트를 실패시킨다.
    def unexpected(*args, **kwargs):
        pytest.fail("unexpected paid API request")
    monkeypatch.setattr(stt.httpx, "post", unexpected)
    return TestClient(main.app)


def upload(client, data, mime="audio/wav"):
    return client.post("/v1/transcriptions", files={"audio": ("recording", data, mime)})


def mock_response(monkeypatch, payload, status=200):
    monkeypatch.setattr(stt.httpx, "post", lambda *a, **k: httpx.Response(
        status, json=payload, request=httpx.Request("POST", stt.STT_URL)))


@pytest.mark.parametrize("model,expected", [
    (None, "openai/whisper-large-v3-turbo"),
    ("", "openai/whisper-large-v3-turbo"),
    ("   ", "openai/whisper-large-v3-turbo"),
    (" test/custom-stt-model ", "test/custom-stt-model"),
])
def test_success_contract_and_provider_request(client, monkeypatch, model, expected):
    if model is not None:
        monkeypatch.setenv("STT_MODEL", model)
    def post(url, **kwargs):
        assert url == stt.STT_URL
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        body = kwargs["json"]
        assert body["model"] == expected
        assert body["language"] == "ko"
        assert body["input_audio"]["format"] == "wav"
        with wave.open(io.BytesIO(base64.b64decode(body["input_audio"]["data"]))) as wav:
            assert (wav.getframerate(), wav.getnchannels()) == (16000, 1)
        return httpx.Response(200, json={"text": "  퇴근길 음악 추천해 줘  "},
                              request=httpx.Request("POST", url))
    monkeypatch.setattr(stt.httpx, "post", post)
    assert upload(client, audio_wav()).json() == {"transcript": "퇴근길 음악 추천해 줘"}


@pytest.mark.parametrize("data,mime,status,reason", [
    (b"", "audio/wav", 400, "EMPTY_AUDIO"),
    (b"x" * (stt.MAX_AUDIO_BYTES + 1), "audio/wav", 413, "AUDIO_TOO_LARGE"),
    (b"bad", "audio/wav", 415, "UNSUPPORTED_AUDIO_FORMAT"),
    (b"RIFFxxxxWAVEbad", "audio/wav", 415, "UNSUPPORTED_AUDIO_FORMAT"),
    (audio_wav(), "audio/mp4", 415, "MIME_TYPE_MISMATCH"),
    (audio_wav(), "text/plain", 415, "UNSUPPORTED_AUDIO_FORMAT"),
    (audio_wav(silent=True), "audio/wav", 400, "EMPTY_AUDIO"),
    (audio_wav(60.1), "audio/wav", 400, "AUDIO_TOO_LONG"),
], ids=["empty", "oversize", "unknown", "corrupt", "mime", "unsupported", "silent", "too-long"])
def test_invalid_audio(client, data, mime, status, reason):
    response = upload(client, data, mime)
    assert response.status_code == status
    assert response.json()["details"]["reason"] == reason


@pytest.mark.parametrize("codec,container,mime", [
    ("libopus", "webm", "audio/webm;codecs=opus"),
    ("libopus", "ogg", "audio/ogg;codecs=opus"),
    ("aac", "mp4", "audio/mp4"),
])
def test_browser_formats(client, monkeypatch, codec, container, mime):
    command = [get_ffmpeg_exe(), "-loglevel", "error", "-i", "pipe:0", "-c:a", codec]
    if container == "mp4":
        command += ["-movflags", "frag_keyframe+empty_moov"]
    data = subprocess.run(command + ["-f", container, "pipe:1"], input=audio_wav(),
                          capture_output=True, check=True).stdout
    mock_response(monkeypatch, {"text": "음악 추천"})
    assert upload(client, data, mime).status_code == 200


@pytest.mark.parametrize("status,payload,expected", [
    (401, {"error": "private upstream detail"}, 503),
    (402, {}, 503), (403, {}, 503), (429, {}, 503), (500, {}, 503),
    (200, {}, 503), (200, [], 503), (200, {"text": 3}, 503),
    (200, {"text": "   "}, 400),
])
def test_provider_failures(client, monkeypatch, status, payload, expected):
    mock_response(monkeypatch, payload, status)
    response = upload(client, audio_wav())
    assert response.status_code == expected
    assert "private upstream detail" not in response.text


def test_timeout(client, monkeypatch):
    def timeout(*args, **kwargs):
        raise httpx.ReadTimeout("secret upstream detail")
    monkeypatch.setattr(stt.httpx, "post", timeout)
    response = upload(client, audio_wav())
    assert response.status_code == 503
    assert "secret" not in response.text


def test_missing_key(client, monkeypatch):
    monkeypatch.delenv("OPENROUTER_STT_API_KEY")
    assert upload(client, audio_wav()).status_code == 503


def test_missing_file_and_openapi(client):
    assert client.post("/v1/transcriptions").status_code == 422
    responses = client.get("/openapi.json").json()["paths"]["/v1/transcriptions"]["post"]["responses"]
    assert set(responses) == {"200", "400", "413", "415", "422", "503"}


def test_exact_duration_limit(client, monkeypatch):
    mock_response(monkeypatch, {"text": "음악 추천"})
    assert upload(client, audio_wav(60)).status_code == 200


def test_seekable_m4a(client, monkeypatch, tmp_path):
    target = tmp_path / "recording.m4a"
    subprocess.run([get_ffmpeg_exe(), "-loglevel", "error", "-i", "pipe:0",
                    "-c:a", "aac", str(target)], input=audio_wav(), check=True)
    mock_response(monkeypatch, {"text": "음악 추천"})
    assert upload(client, target.read_bytes(), "audio/mp4").status_code == 200


def test_decoder_timeout(client, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("ffmpeg", 15)
    monkeypatch.setattr(stt.subprocess, "run", timeout)
    assert upload(client, audio_wav()).status_code == 415


def test_invalid_provider_json(client, monkeypatch):
    monkeypatch.setattr(stt.httpx, "post", lambda *a, **k: httpx.Response(
        200, text="not json", request=httpx.Request("POST", stt.STT_URL)))
    assert upload(client, audio_wav()).status_code == 503
