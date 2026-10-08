"""사진 추천 흐름: 요청 예약·처리 슬롯, VLM 판정, 기존 추천 그래프 실행, SSE 생성.

NOTE: 채팅은 그래프 실행(service)과 SSE 생성(streaming)을 나누지만, 사진은 요청 예약부터 검색 종료 후
슬롯 반환까지가 하나의 생명주기로 묶여 있어 이 모듈에서 함께 관리한다.
"""

import asyncio
import logging
import os
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable

import httpx
from fastapi import HTTPException, UploadFile
from openai import OpenAI

from backend.chat.graph.workflow import build_graph
from backend.chat.photo import image, vlm
from backend.chat.photo.schemas import Scene
from backend.chat.streaming import sse
from backend.core.errors import _error, _model_unavailable
from backend.providers.openrouter import OPENROUTER_BASE_URL

logger = logging.getLogger(__name__)
# TODO: 처리 슬롯·중복 기록·채팅별 진행 여부는 프로세스 단위다. 다중 워커 배포 전 BE·공유 저장소로 옮긴다.
# NOTE: 슬롯은 이미지 디코딩 전에 잡고, 시간 초과 뒤 계속되는 검색 작업이 끝날 때 반환한다. 대기열은 두지 않는다.
MAX_CONCURRENT = 2
_slots = threading.BoundedSemaphore(MAX_CONCURRENT)
_active_threads: set[str] = set()
_requests: OrderedDict[tuple[str, str], float] = OrderedDict()
REJECT_MESSAGES = {
    "scene_unreadable": "풍경의 특징을 확인하기 어려워요. 주변이 보이도록 다시 촬영해 주세요.",
    "not_a_place": "풍경의 특징을 확인하기 어려워요. 주변이 보이도록 다시 촬영해 주세요.",
    # NOTE: 공급자 차단은 서비스가 사진을 부적절하다고 판정한 것이 아니므로 중립 문구를 쓴다. 최종 문구는 FE·디자인과 합의한다.
    "provider_blocked": "이 사진으로는 추천을 진행하기 어려워요. 다른 장면을 촬영해 주세요.",
}


def recommend_scene(scene: Scene, thread_id: str) -> list[dict]:
    """사진 맥락을 메모리에 남기고 기존 임베딩·검색·추천 이유를 재사용한다."""
    key, model = os.getenv("OPENROUTER_LLM_API_KEY", ""), os.getenv("LLM_MODEL", "")
    if not key.strip() or not model.strip():
        raise _model_unavailable()
    client = OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL, timeout=15, max_retries=0)
    result = build_graph(photo=True).invoke({
        "message": scene.music_query,
        "messages": ["사진의 장소 맥락: " + scene.model_dump_json()],
        "intent": "recommend", "recommend_query": scene.music_query,
        "recommend_genres": None, "recommend_min_year": None, "recommend_max_year": None,
    }, config={"configurable": {"thread_id": thread_id, "client": client}})
    return [track.model_dump() for track in result["tracks"]]


async def open_photo_stream(
    upload: UploadFile, thread: str, request_id: str,
) -> tuple[AsyncIterator[str], Callable[[], Awaitable[None]]]:
    """요청을 예약·검증하고 SSE 이벤트 생성기와 스트림 미시작 시 정리 함수를 반환한다."""
    request_key = (thread, request_id)
    now = time.monotonic()
    while _requests and next(iter(_requests.values())) < now - 600:
        _requests.popitem(last=False)
    # NOTE: await 전에 검사와 예약을 끝내야 같은 요청 두 건이 함께 통과하지 못한다.
    if request_key in _requests:
        raise _error(409, "DUPLICATE_REQUEST", "이미 접수한 사진 요청입니다. 새 request_id로 요청하세요.")
    # NOTE: 같은 채팅의 사진 요청은 대화 상태가 섞이지 않게 하나씩 받는다. 사진·텍스트 동시 요청은 아직 막지 않는다.
    if thread in _active_threads or not _slots.acquire(blocking=False):
        raise _error(409, "PHOTO_BUSY", "사진을 처리 중입니다. 완료 후 다시 시도하세요.")
    _requests[request_key] = now
    # TODO: 최근 256건을 10분간만 보관한다. 공유·영속화는 정식 BE 계약 후 적용한다.
    if len(_requests) > 256:
        _requests.popitem(last=False)
    _active_threads.add(thread)
    job = {"released": False, "started": False}

    def release(*, allow_retry: bool) -> None:
        """슬롯·채팅 점유를 한 번만 반환한다. 새 실행으로 인정할 실패면 같은 ID 재시도를 허용한다."""
        if job["released"]:
            return
        job["released"] = True
        _active_threads.discard(thread)
        if allow_retry:
            _requests.pop(request_key, None)
        _slots.release()

    try:
        try:
            raw = await upload.read(image.MAX_BYTES + 1)
            normalized = await asyncio.to_thread(image.normalize_image, raw)
        finally:
            await upload.close()
    except BaseException:
        release(allow_retry=True)
        raise

    async def release_if_never_started() -> None:
        """연결이 먼저 끊겨 스트림이 시작되지 않으면 생성기의 finally 대신 여기서 반환한다."""
        if not job["started"]:
            release(allow_retry=True)

    async def events() -> AsyncIterator[str]:
        """원본·분석 내용을 로그에 남기지 않고 단계별 상태를 전달한다."""
        job["started"] = True
        worker = None
        # NOTE: 분석 결과를 받기 전 실패만 새 실행으로 인정한다. 이후 실패는 과금·대화 상태 변경 가능성이 있어 새 ID를 요구한다.
        retryable = True
        started = time.monotonic()
        try:
            async with asyncio.timeout(45):
                yield sse("status", {"stage": "analyzing"})
                try:
                    scene, provider = await vlm.analyze_image(normalized)
                except vlm.ProviderBlocked:
                    retryable = False
                    yield sse("verdict", {"usable": False, "reject_reason": "provider_blocked"})
                    yield sse("rejected", {"reason": "provider_blocked", "message": REJECT_MESSAGES["provider_blocked"]})
                    yield sse("done", {})
                    return
                retryable = False
                # NOTE: FE는 검색 결과보다 먼저 판정을 받아 사진 화면의 이동 여부를 결정한다.
                yield sse("verdict", {"usable": scene.usable, "reject_reason": scene.reject_reason})
                # NOTE: 거절도 분석 결과를 표시해 모델 판정과 API 실패를 구분할 수 있게 한다.
                yield sse("scene", {"scene": scene.model_dump(), "provider": provider})
                if not scene.usable:
                    yield sse("rejected", {"reason": scene.reject_reason, "message": REJECT_MESSAGES[scene.reject_reason]})
                    yield sse("done", {})
                    return
                # NOTE: VLM 설명은 검색 전에 이미 있으므로 먼저 보내 채팅 이동 직후 빈 말풍선을 없앤다.
                # 검색 결과를 모르는 시점의 문장이라 곡을 골랐다고 단정하는 문구를 붙이지 않는다.
                yield sse("text", {"delta": scene.recommendation_context.strip()})
                yield sse("status", {"stage": "searching"})
                # NOTE: 시간 초과 후에도 SDK·DB 처리가 계속될 수 있어, 작업이 끝날 때 슬롯을 반환한다.
                worker = asyncio.create_task(asyncio.to_thread(recommend_scene, scene, thread))
                worker.add_done_callback(lambda task: (task.exception() if not task.cancelled() else None,
                                                       release(allow_retry=False)))
                tracks = await asyncio.shield(worker)
                if not tracks:
                    yield sse("text", {"delta": "\n\n조건에 맞는 곡을 찾지 못했어요."})
                yield sse("tracks", {"tracks": tracks})
                yield sse("done", {"elapsed_ms": round((time.monotonic() - started) * 1000)})
        except Exception as error:
            status = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
            reason, message = "PHOTO_PROCESSING_FAILED", "사진 처리에 실패했습니다. 모델·DB 설정 또는 연결을 확인하세요."
            if isinstance(error, vlm.ModelResponseInvalid):
                reason, message = "MODEL_RESPONSE_INVALID", "모델 응답을 해석하지 못했습니다. 잠시 후 다시 시도하세요."
            elif status is not None:
                reason = "MODEL_UNAVAILABLE"
                message = f"모델 API가 HTTP {status}로 요청을 거절했습니다. 모델 지원·키·사용 한도·공급자 상태를 확인하세요."
            elif isinstance(error, httpx.TimeoutException):
                reason, message = "MODEL_UNAVAILABLE", "모델 API 응답 대기 시간이 초과되었습니다. 잠시 후 다시 시도하세요."
            elif isinstance(error, httpx.TransportError):
                reason, message = "MODEL_UNAVAILABLE", "모델 API에 연결하지 못했습니다. 네트워크·공급자 상태를 확인하세요."
            elif isinstance(error, TimeoutError):
                reason, message = "PHOTO_TIMEOUT", "사진 처리 시간이 초과되었습니다. 잠시 후 다시 시도하세요."
            elif isinstance(error, HTTPException) and isinstance(error.detail, dict):
                reason = error.detail.get("details", {}).get("reason", reason)
                if reason in ("PHOTO_MODEL_NOT_CONFIGURED", "PHOTO_PROVIDER_NOT_CONFIGURED"):
                    message = error.detail["message"]
            # NOTE: 공급자 응답 본문·헤더에는 민감한 정보가 있을 수 있어 오류 유형·상태 코드만 기록한다.
            logger.warning("Photo recommendation failed: %s status=%s reason=%s", type(error).__name__, status, reason)
            yield sse("error", {"detail": message, "reason": reason})
        finally:
            if worker is None:
                release(allow_retry=retryable)

    return events(), release_if_never_started
