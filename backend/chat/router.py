from collections.abc import Iterator
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from backend.chat.schemas import ChatRequest
from backend.chat.service import prepare_recommendation
from backend.chat.streaming import sse, stream_answer, stream_lookup_answer
from backend.http.schemas import ErrorResponse

router = APIRouter()


@router.post(
    "/v1/chat/messages",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"text/event-stream": {}}},
        400: {
            "model": ErrorResponse,
            "description": "요청값 검증 실패",
        },
        503: {
            "model": ErrorResponse,
            "description": "모델 또는 음악 검색 서비스 사용 불가",
        },
    },
    tags=["chat"],
)
def chat(body: ChatRequest) -> StreamingResponse:
    """추천 문장과 완성된 곡 목록을 Spring Backend에 SSE로 반환한다."""

    # TODO: request_id 중복 처리는 책임 범위 확정 후 연결한다.
    # thread_id는 LangGraph checkpointer의 대화 세션 키로 흘려보낸다.
    result = prepare_recommendation(body.message, str(body.thread_id))

    # NOTE: guide/clarify/out_of_scope, recommend·lookup의 예외 분기는 고정 문구라 이렇게 처리한다.
    if result.kind == "static":
        def static_answer() -> Iterator[str]:
            yield sse("text", {"delta": result.text})
            yield sse("tracks", {"tracks": []})
            yield sse("done", {})

        return StreamingResponse(
            static_answer(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if result.kind == "lookup":
        return StreamingResponse(
            stream_lookup_answer(result.client, result.query, result.tracks, body.user_context, result.hints),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return StreamingResponse(
        stream_answer(result.client, result.query, result.tracks, body.user_context, result.hints),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
