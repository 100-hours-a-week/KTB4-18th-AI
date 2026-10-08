"""사진 추천 API의 SSE 순서·요청 제어·대화 연결을 외부 호출 없이 검증한다."""

import asyncio
import io
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.chat.graph import nodes, workflow
from backend.chat.photo import image, service, vlm
from backend.chat.photo.schemas import Scene
from backend.main import app
from db.types import Track


def jpeg():
    output = io.BytesIO()
    photo = Image.new("RGB", (40, 20), "green")
    exif = Image.Exif()
    exif[274] = 6
    exif[270] = "private metadata"
    photo.save(output, "JPEG", exif=exif)
    return output.getvalue()


def scene():
    return Scene(usable=True, reject_reason=None, place_type="park",
                       visual_elements=["trees"], lighting="daylight", mood=["calm"],
                       music_query="나무 아래서 듣기 좋은 잔잔한 음악",
                       recommendation_context="사진에 보이는 나무에서 차분한 분위기가 느껴져요. 잔잔한 음악 위주로 찾아볼게요.")


def png():
    output = io.BytesIO()
    Image.new("RGB", (40, 20), "green").save(output, "PNG")
    return output.getvalue()


@pytest.mark.parametrize("failure", [httpx.ReadTimeout("timeout"), httpx.HTTPStatusError(
    "limited", request=httpx.Request("POST", "https://test.invalid"), response=httpx.Response(429))])
def test_unconfigured_fallback_preserves_primary_failure(monkeypatch, failure):
    monkeypatch.delenv("PHOTO_FALLBACK_MODEL", raising=False)
    calls = []

    async def call(_image, *, fallback=False):
        calls.append(fallback)
        raise failure

    monkeypatch.setattr(vlm, "call_vlm", call)
    with pytest.raises(type(failure)) as error:
        asyncio.run(vlm.analyze_image(b"image"))
    assert error.value is failure
    assert calls == [False]
    response = TestClient(app).post("/v1/chat/images", data={
        "thread_id": str(uuid4()), "request_id": str(uuid4()),
    }, files={"image": ("photo.jpg", jpeg(), "image/jpeg")})
    assert "HTTP 429" in response.text if isinstance(failure, httpx.HTTPStatusError) else "대기 시간이 초과" in response.text
    assert "PHOTO_FALLBACK_MODEL" not in response.text


def test_missing_model_is_reported_in_stream(monkeypatch):
    monkeypatch.delenv("PHOTO_VLM_MODEL", raising=False)
    response = TestClient(app).post("/v1/chat/images", data={
        "thread_id": str(uuid4()), "request_id": str(uuid4()),
    }, files={"image": ("photo.jpg", jpeg(), "image/jpeg")})
    assert "event: error" in response.text
    assert "PHOTO_VLM_MODEL" in response.text
    assert "event: tracks" not in response.text


def test_rejection_and_duplicate(monkeypatch):
    rejected = Scene(**{**scene().model_dump(), "usable": False,
                              "reject_reason": "scene_unreadable", "music_query": ""})
    calls = []

    async def call(_image, *, fallback=False):
        calls.append(fallback)
        return rejected

    monkeypatch.setattr(vlm, "call_vlm", call)
    with TestClient(app) as client:
        body = {"thread_id": str(uuid4()), "request_id": str(uuid4())}
        response = client.post("/v1/chat/images", data=body, files={"image": ("photo.jpg", jpeg(), "image/jpeg")})
        assert response.status_code == 200
        assert "event: done" in response.text
        assert "event: rejected" in response.text
        assert "event: scene" in response.text
        assert response.text.index("event: verdict") < response.text.index("event: rejected")
        assert '"usable": false' in response.text
        assert "event: tracks" not in response.text
        assert calls == [False]
        duplicate = client.post("/v1/chat/images", data=body, files={"image": ("photo.jpg", jpeg(), "image/jpeg")})
        assert duplicate.status_code == 409


def test_real_stream_searches_after_scene_and_reports_failure(monkeypatch):

    async def analyze(_image):
        return scene(), "openrouter"

    monkeypatch.setattr(vlm, "analyze_image", analyze)
    monkeypatch.setattr(service, "recommend_scene", lambda *_args: [Track(
        track_id="1", title="test", artist="test", artwork_url=None,
        preview_url=None, store_url=None, reason="test").model_dump()])
    with TestClient(app) as client:
        body = {"thread_id": str(uuid4()), "request_id": str(uuid4())}
        response = client.post("/v1/chat/images", data=body, files={"image": ("x.jpg", jpeg(), "image/jpeg")})
        events = [block.split("\n", 1)[0][7:] for block in response.text.strip().split("\n\n")]
        # NOTE: 설명 문장은 검색 시작 전에 먼저 오고, 곡 카드는 검색이 끝난 뒤에 온다.
        assert events == ["status", "verdict", "scene", "text", "status", "tracks", "done"]
        assert f'{{"delta": "{scene().recommendation_context}"}}' in response.text
        assert "골랐어요" not in response.text

        monkeypatch.setattr(service, "recommend_scene", lambda *_args: [])
        body["request_id"] = str(uuid4())
        empty = client.post("/v1/chat/images", data=body, files={"image": ("x.jpg", jpeg(), "image/jpeg")})
        assert empty.text.index(scene().recommendation_context) < empty.text.index("조건에 맞는 곡을 찾지 못했어요.")
        assert 'event: tracks\ndata: {"tracks": []}' in empty.text

        def broken(*_args):
            raise RuntimeError("private database details")

        monkeypatch.setattr(service, "recommend_scene", broken)
        body["request_id"] = str(uuid4())
        response = client.post("/v1/chat/images", data=body, files={"image": ("x.jpg", jpeg(), "image/jpeg")})
        assert "event: error" in response.text
        assert "event: done" not in response.text
        assert "private database details" not in response.text


def test_photo_graph_reuses_search_and_context_for_next_text(monkeypatch):
    calls = []
    track = Track(track_id="1", title="test", artist="test", artwork_url=None,
                  preview_url=None, store_url=None, reason="test")
    monkeypatch.setattr(nodes, "get_embedding_client", lambda: object())
    monkeypatch.setattr(nodes, "embed_query", lambda *_args: [1.0])

    def search(*_args, **kwargs):
        calls.append(kwargs["exclude_ids"])
        return [track], {}

    monkeypatch.setattr(nodes, "vector_recommendation", search)
    monkeypatch.setattr(nodes, "assign_reasons", lambda *_args: None)
    thread = str(uuid4())
    config = {"configurable": {"thread_id": thread, "client": object()}}
    initial = {"messages": ["사진: 강변"], "recommend_query": "잔잔한 음악",
               "recommend_genres": None, "recommend_min_year": None, "recommend_max_year": None}
    assert workflow.build_graph(photo=True).invoke(initial, config=config)["tracks"] == [track]
    seen = []

    def classify(_client, messages):
        seen.extend(messages)
        return {"intent": "guide"}

    monkeypatch.setattr(nodes, "classify", classify)
    workflow.build_graph().invoke({"message": "더 신나게", "messages": ["더 신나게"]}, config=config)
    assert seen == ["사진: 강변", "더 신나게"]
    workflow.build_graph(photo=True).invoke(initial, config=config)
    assert calls == [set(), {1}]


def post_photo(client, body, data=None, mime="image/jpeg"):
    return client.post("/v1/chat/images", data=body, files={"image": ("x.jpg", data or jpeg(), mime)})


def assert_slots_released():
    assert service._slots._value == service.MAX_CONCURRENT
    assert not service._active_threads


def test_mislabeled_jpeg_is_judged_by_bytes(monkeypatch):

    async def analyze(_image):
        return Scene(**{**scene().model_dump(), "usable": False,
                              "reject_reason": "scene_unreadable", "music_query": ""}), "openrouter"

    monkeypatch.setattr(vlm, "analyze_image", analyze)
    with TestClient(app) as client:
        for mime in ("application/octet-stream", "image/jpg"):
            body = {"thread_id": str(uuid4()), "request_id": str(uuid4())}
            assert "event: verdict" in post_photo(client, body, mime=mime).text
        bad = post_photo(client, {"thread_id": str(uuid4()), "request_id": str(uuid4())}, png())
        assert bad.status_code == 415
    assert_slots_released()


def test_same_request_sent_twice_at_once_is_accepted_once(monkeypatch):
    original = image.normalize_image

    def slow(data):
        time.sleep(0.2)
        return original(data)

    async def analyze(_image):
        return Scene(**{**scene().model_dump(), "usable": False,
                              "reject_reason": "scene_unreadable", "music_query": ""}), "openrouter"

    monkeypatch.setattr(image, "normalize_image", slow)
    monkeypatch.setattr(vlm, "analyze_image", analyze)
    body = {"thread_id": str(uuid4()), "request_id": str(uuid4())}
    with TestClient(app) as client, ThreadPoolExecutor(2) as pool:
        responses = list(pool.map(lambda _: post_photo(client, body), range(2)))
    codes = sorted(response.status_code for response in responses)
    assert codes == [200, 409]
    conflict = next(response for response in responses if response.status_code == 409)
    assert conflict.json()["details"]["reason"] in ("DUPLICATE_REQUEST", "PHOTO_BUSY")
    assert_slots_released()


def test_busy_reasons_distinguish_same_chat_and_capacity(monkeypatch):
    thread = str(uuid4())
    with TestClient(app) as client:
        service._active_threads.add(thread)
        try:
            response = post_photo(client, {"thread_id": thread, "request_id": str(uuid4())})
        finally:
            service._active_threads.discard(thread)
        assert response.status_code == 409
        assert response.json() == {"code": "CONFLICT", "message": "사진을 처리 중입니다. 완료 후 다시 시도하세요.",
                                   "details": {"reason": "PHOTO_BUSY"}}
        for _ in range(service.MAX_CONCURRENT):
            service._slots.acquire()
        try:
            response = post_photo(client, {"thread_id": str(uuid4()), "request_id": str(uuid4())})
        finally:
            for _ in range(service.MAX_CONCURRENT):
                service._slots.release()
        assert response.json()["details"]["reason"] == "PHOTO_BUSY"
    assert_slots_released()


def test_retry_with_same_id_only_before_analysis_result(monkeypatch):
    outcomes = iter([httpx.ConnectError("offline"), (scene(), "openrouter")])

    async def analyze(_image):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def broken(*_args):
        raise RuntimeError("search failed")

    monkeypatch.setattr(vlm, "analyze_image", analyze)
    monkeypatch.setattr(service, "recommend_scene", broken)
    body = {"thread_id": str(uuid4()), "request_id": str(uuid4())}
    with TestClient(app) as client:
        first = post_photo(client, body)
        assert '"reason": "MODEL_UNAVAILABLE"' in first.text
        # NOTE: 분석 결과를 받기 전 실패라 같은 ID의 재시도를 새 실행으로 받는다.
        second = post_photo(client, body)
        assert "event: verdict" in second.text and "event: error" in second.text
        # NOTE: 검색까지 시작한 뒤의 실패는 대화 상태가 바뀌었을 수 있어 같은 ID를 다시 받지 않는다.
        third = post_photo(client, body)
        assert third.status_code == 409
        assert third.json()["details"]["reason"] == "DUPLICATE_REQUEST"
    assert_slots_released()



def test_provider_block_is_a_neutral_rejection_not_an_error(monkeypatch):
    monkeypatch.delenv("PHOTO_FALLBACK_MODEL", raising=False)

    async def blocked(_image, *, fallback=False):
        raise vlm.ProviderBlocked()

    async def invalid(_image, *, fallback=False):
        raise vlm.ModelResponseInvalid("length")

    monkeypatch.setattr(vlm, "call_vlm", blocked)
    body = {"thread_id": str(uuid4()), "request_id": str(uuid4())}
    with TestClient(app) as client:
        response = post_photo(client, body)
        assert '"reject_reason": "provider_blocked"' in response.text
        assert service.REJECT_MESSAGES["provider_blocked"] in response.text
        assert "event: done" in response.text and "event: error" not in response.text
        assert post_photo(client, body).json()["details"]["reason"] == "DUPLICATE_REQUEST"
        monkeypatch.setattr(vlm, "call_vlm", invalid)
        broken = post_photo(client, {"thread_id": str(uuid4()), "request_id": str(uuid4())})
        assert '"reason": "MODEL_RESPONSE_INVALID"' in broken.text
        assert "event: rejected" not in broken.text
    assert_slots_released()
