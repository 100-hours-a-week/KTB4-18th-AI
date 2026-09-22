"""실제 SDK와 모의 HTTP로 OpenRouter 임베딩 계약을 검증한다."""

import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from openai import OpenAI

from backend import recommendation


@pytest.mark.parametrize("size", [3072, 512])
def test_embedding_request_and_dimensions(monkeypatch, size):
    monkeypatch.setenv("EMBEDDING_MODEL", "google/gemini-embedding-2")

    def handle(request):
        assert str(request.url) == "https://openrouter.ai/api/v1/embeddings"
        assert request.headers["authorization"] == "Bearer embedding-test-key"
        assert json.loads(request.content) == {
            "model": "google/gemini-embedding-2", "input": "퇴근길 음악",
            "dimensions": 3072, "encoding_format": "float",
        }
        return httpx.Response(200, json={
            "data": [{"index": 0, "object": "embedding", "embedding": [0.1] * size}],
            "model": "google/gemini-embedding-2", "object": "list",
        })

    with OpenAI(api_key="embedding-test-key", base_url="https://openrouter.ai/api/v1",
                http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
        if size == 3072:
            assert len(recommendation.embed_query(client, "퇴근길 음악")) == 3072
        else:
            with pytest.raises(HTTPException) as error:
                recommendation.embed_query(client, "퇴근길 음악")
            assert error.value.status_code == 503


def test_legacy_keys_do_not_enable_recommendation(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "legacy")
    monkeypatch.setenv("GEMINI_API_KEY", "legacy")
    monkeypatch.delenv("OPENROUTER_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_EMBEDDING_API_KEY", raising=False)
    with pytest.raises(HTTPException) as error:
        recommendation.prepare_recommendation("음악")
    assert error.value.status_code == 503


def test_llm_model_and_invalid_reason_json(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "openai/test-model")

    def create(**kwargs):
        assert kwargs["model"] == "openai/test-model"
        return SimpleNamespace(output_text="[]")

    track = SimpleNamespace(track_id="1", title="곡", artist="가수", reason="기존 이유")
    recommendation.assign_reasons(SimpleNamespace(responses=SimpleNamespace(create=create)),
                                  "음악", [track], {})
    assert track.reason == "기존 이유"
