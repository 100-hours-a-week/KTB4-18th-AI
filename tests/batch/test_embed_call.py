"""일간 임베딩의 OpenRouter 호출 재시도 규칙. 실제 요청 없이 urlopen을 바꿔 끼워 검사한다.

openrouter_collect.py로 15,000곡을 임베딩할 때 쓴 규칙과 같아야 한다.
"""

import io
import json
import urllib.error

import pytest


@pytest.fixture
def embed(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://test:test@localhost/test")
    from batch import embed
    monkeypatch.setattr(embed.time, "sleep", lambda s: None)
    return embed


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(code):
    return urllib.error.HTTPError("url", code, "msg", {}, io.BytesIO(b"error body"))


def fake_urlopen(monkeypatch, embed, outcomes):
    """outcomes를 순서대로 돌려준다. 예외면 raise, dict면 JSON 응답."""
    calls = []

    def urlopen(req, timeout):
        calls.append(req)
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return Response(json.dumps(outcome).encode())

    monkeypatch.setattr(embed.urllib.request, "urlopen", urlopen)
    return calls


def test_network_error_is_retried(embed, monkeypatch):
    calls = fake_urlopen(monkeypatch, embed, [TimeoutError(), {"data": []}])
    assert embed.call(b"{}", "key") == {"data": []}
    assert len(calls) == 2


def test_server_error_is_retried(embed, monkeypatch):
    calls = fake_urlopen(monkeypatch, embed, [http_error(503), http_error(429), {"data": []}])
    assert embed.call(b"{}", "key") == {"data": []}
    assert len(calls) == 3


@pytest.mark.parametrize("code", [401, 402, 403])
def test_auth_and_billing_errors_stop_immediately(embed, monkeypatch, code):
    calls = fake_urlopen(monkeypatch, embed, [http_error(code)])
    with pytest.raises(embed.Fatal):
        embed.call(b"{}", "key")
    assert len(calls) == 1


def test_gives_up_after_three_network_errors(embed, monkeypatch):
    fake_urlopen(monkeypatch, embed, [TimeoutError()] * 3)
    with pytest.raises(TimeoutError):
        embed.call(b"{}", "key")


def test_unexpected_response_names_the_body(embed, monkeypatch):
    fake_urlopen(monkeypatch, embed, [{"error": {"message": "no route"}}])
    monkeypatch.setattr(embed, "to_mp3_b64", lambda url, exe: "")
    with pytest.raises(RuntimeError, match="no route"):
        embed.embed_audio("url", "ffmpeg", "key")
