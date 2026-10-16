"""사진 추천 HTTP 진입점."""

from uuid import UUID

from fastapi import APIRouter, Form, UploadFile
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from backend.chat.photo import service
from backend.http.schemas import ErrorResponse

router = APIRouter()


@router.post(
    "/v1/chat/images",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"text/event-stream": {}}},
        409: {"model": ErrorResponse, "description": "중복 요청 또는 다른 사진 처리 중"},
        413: {"model": ErrorResponse, "description": "사진 용량·픽셀 수 초과"},
        415: {"model": ErrorResponse, "description": "JPEG가 아니거나 손상된 사진"},
    },
    tags=["chat"],
)
async def photo_chat(
    image: UploadFile, thread_id: UUID = Form(), request_id: UUID = Form(),
) -> StreamingResponse:
    """사진을 검증하고 적합 판정과 추천 결과를 순서대로 반환한다."""
    events, release_if_never_started = await service.open_photo_stream(image, str(thread_id), str(request_id))
    # NOTE: 연결이 먼저 끊겨 스트림이 시작되지 않으면 생성기 대신 응답 종료 작업이 슬롯을 반환한다.
    return StreamingResponse(events, media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                             background=BackgroundTask(release_if_never_started))
