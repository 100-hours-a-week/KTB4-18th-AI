"""VLM 호출의 폴백과 공급자 차단·응답 실패 분류를 외부 호출 없이 검증한다."""

import asyncio

import httpx
import pytest
from fastapi import HTTPException

from backend.chat.photo import vlm
from backend.chat.photo.schemas import Scene


def scene():
    return Scene(usable=True, reject_reason=None, place_type="park",
                       visual_elements=["trees"], lighting="daylight", mood=["calm"],
                       music_query="나무 아래서 듣기 좋은 잔잔한 음악",
                       recommendation_context="사진에 보이는 나무에서 차분한 분위기가 느껴져요. 잔잔한 음악 위주로 찾아볼게요.")


@pytest.mark.parametrize("status,should_fallback", [(429, True), (503, True), (400, False), (401, False)])
def test_fallback_only_for_transient_errors(monkeypatch, status, should_fallback):
    for name in ("PHOTO_FALLBACK_URL", "PHOTO_FALLBACK_MODEL", "PHOTO_FALLBACK_API_KEY"):
        monkeypatch.setenv(name, "test")
    calls = []

    async def call(_image, *, fallback=False):
        calls.append(fallback)
        if not fallback:
            response = httpx.Response(status, request=httpx.Request("POST", "https://test.invalid"))
            raise httpx.HTTPStatusError("failure", request=response.request, response=response)
        return scene()

    monkeypatch.setattr(vlm, "call_vlm", call)
    if should_fallback:
        assert asyncio.run(vlm.analyze_image(b"image"))[1] == "fallback"
        assert calls == [False, True]
    else:
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(vlm.analyze_image(b"image"))
        assert calls == [False]


def test_transport_failure_and_invalid_output(monkeypatch):
    for name in ("PHOTO_FALLBACK_URL", "PHOTO_FALLBACK_MODEL", "PHOTO_FALLBACK_API_KEY"):
        monkeypatch.setenv(name, "test")
    async def call(_image, *, fallback=False):
        if not fallback:
            raise httpx.ConnectError("offline")
        return scene()

    monkeypatch.setattr(vlm, "call_vlm", call)
    assert asyncio.run(vlm.analyze_image(b"image"))[1] == "fallback"
    with pytest.raises(ValueError):
        Scene(**{**scene().model_dump(), "reject_reason": "not_a_place"})


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("model", [None, "", "  ", "<YOUR_MODEL>"])
def test_unselected_model_never_calls_provider(monkeypatch, fallback, model):
    name = "PHOTO_FALLBACK_MODEL" if fallback else "PHOTO_VLM_MODEL"
    if model is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, model)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("모델 미지정 시 HTTP 클라이언트를 생성하면 안 된다.")

    monkeypatch.setattr(httpx, "AsyncClient", forbidden)
    with pytest.raises(HTTPException) as error:
        asyncio.run(vlm.call_vlm(b"image", fallback=fallback))
    assert error.value.detail["details"]["reason"] == "PHOTO_MODEL_NOT_CONFIGURED"


REAL_ASYNC_CLIENT = httpx.AsyncClient


def vlm_client(monkeypatch, response):
    monkeypatch.setenv("PHOTO_VLM_MODEL", "test/model")
    monkeypatch.setenv("OPENROUTER_LLM_API_KEY", "test-key")
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: REAL_ASYNC_CLIENT(
        transport=httpx.MockTransport(lambda _request: response), **kwargs))


def completion(content, finish_reason="stop", **message):
    return httpx.Response(200, json={"choices": [{"finish_reason": finish_reason,
                                                  "message": {"content": content, **message}}]})


@pytest.mark.parametrize("response,expected", [
    (httpx.Response(403, json={"error": {"code": 403, "message": "flagged",
                                         "metadata": {"reasons": ["sexual"], "flagged_input": "..."}}}),
     vlm.ProviderBlocked),
    (httpx.Response(403, json={"error": {"code": "content_policy_violation", "message": "blocked"}}),
     vlm.ProviderBlocked),
    (httpx.Response(403, json={"error": {"code": 403, "message": "Key limit exceeded"}}), httpx.HTTPStatusError),
    (completion(None, "content_filter", refusal="I can't help with that."), vlm.ProviderBlocked),
    (completion('{"usable": true, "reject_', "length"), vlm.ModelResponseInvalid),
    (completion(None), vlm.ModelResponseInvalid),
    (completion("not json"), vlm.ModelResponseInvalid),
])
def test_provider_blocks_are_separated_from_response_and_service_failures(monkeypatch, response, expected):
    vlm_client(monkeypatch, response)
    with pytest.raises(expected):
        asyncio.run(vlm.call_vlm(b"image"))

