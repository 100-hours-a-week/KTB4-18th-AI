from collections.abc import Iterator
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from backend.chat.schemas import ChatRequest
from backend.chat.service import prepare_recommendation
from backend.chat.streaming import sse, stream_answer, stream_fast_answer, stream_lookup_answer
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
    import os
    if os.getenv("AGENTIC_CHAT") == "1":
        return StreamingResponse(
            _live_agent_response(body), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return _chat_response(body)


async def _live_agent_response(body):
    """검색을 별도 스레드에서 실행하고 진행 이벤트를 발생 순서대로 전달한다."""
    import asyncio
    import os
    import threading
    import time
    from backend.chat.graph.agent import AgentCancelled

    started = time.monotonic()
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    stopped = threading.Event()
    inspect = os.getenv("LOCAL_INSPECTOR") == "1"

    def on_event(event):
        if stopped.is_set():
            raise AgentCancelled()
        event = {**event, "elapsed_ms": round((time.monotonic() - started) * 1000),
                 "request_id": str(body.request_id)}
        loop.call_soon_threadsafe(queue.put_nowait, event)

    async def prepare():
        try:
            return await asyncio.to_thread(
                prepare_recommendation, body.message, str(body.thread_id), on_event=on_event)
        finally:
            queue.put_nowait(None)

    yield sse("status", {"stage": "accepted", "elapsed_ms": 0,
                         "request_id": str(body.request_id)})
    task = asyncio.create_task(prepare())
    try:
        while True:
            event = await queue.get()
            if event is None:
                break
            # 상세 검색어와 모델의 결정 요약은 로컬 인스펙터에만 전달한다.
            public = {k: v for k, v in event.items()
                      if k in ("stage", "elapsed_ms", "request_id", "attempt", "count")}
            yield sse("status", public)
            if inspect:
                yield sse("agent_step", event)
        result = await task
        yield sse("status", {"stage": "answer_start",
                             "elapsed_ms": round((time.monotonic()-started)*1000)})
        response = _outcome_response(body, result)
        async for chunk in response.body_iterator:
            yield chunk
    except Exception:
        yield sse("error", {"detail": "챗봇 처리 중 오류가 발생했습니다. 다시 요청해 주세요."})
    finally:
        stopped.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def _chat_response(body: ChatRequest) -> StreamingResponse:
    """추천 문장과 완성된 곡 목록을 Spring Backend에 SSE로 반환한다."""

    # TODO: request_id 중복 처리는 책임 범위 확정 후 연결한다.
    # thread_id는 LangGraph checkpointer의 대화 세션 키로 흘려보낸다.
    result = prepare_recommendation(body.message, str(body.thread_id))
    return _outcome_response(body, result)


def _outcome_response(body, result):

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

    if result.kind == "fast":
        return StreamingResponse(
            stream_fast_answer(result.text, result.tracks, result.hints, thread_id=str(body.thread_id)),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if result.kind == "lookup":
        return StreamingResponse(
            stream_lookup_answer(result.client, result.query, result.tracks, body.user_context, result.hints,
                                 thread_id=str(body.thread_id)),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return StreamingResponse(
        stream_answer(result.client, result.query, result.tracks, body.user_context, result.hints,
                      thread_id=str(body.thread_id)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
