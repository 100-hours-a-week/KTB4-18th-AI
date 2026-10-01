"""추천·조회 SSE 응답."""

import json
import os
from collections.abc import Iterator
from typing import Any
from openai import OpenAI
from db.types import Track
from backend.chat.schemas import UserContext
from backend.chat.answer import answer_input, answer_instructions, lookup_answer_input, lookup_answer_instructions

NO_TRACKS_MESSAGE = "조건에 맞는 곡을 찾지 못했어요. 조금 더 구체적인 질문과 함께 다시 요청해 주세요."


def sse(event: str, data: Any) -> str:
    """Spring Backend에 반환할 SSE 이벤트를 직렬화한다."""

    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def stream_answer(
    client: OpenAI,
    message: str,
    tracks: list[Track],
    user_context: UserContext | None,
) -> Iterator[str]:
    """추천 문장 조각과 완성된 추천곡을 순서대로 전송한다."""

    if not tracks:
        yield sse("text", {"delta": NO_TRACKS_MESSAGE})
        yield sse("tracks", {"tracks": []})
        yield sse("done", {})
        return

    error_message = "챗봇 응답 스트리밍에 실패했습니다."

    try:
        stream = client.responses.create(
            model=os.environ["LLM_MODEL"],
            instructions=answer_instructions(),
            input=answer_input(message, tracks, user_context),
            stream=True,
        )

        with stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    yield sse("text", {"delta": event.delta})

                elif event.type == "response.completed":
                    yield sse(
                        "tracks",
                        {"tracks": [track.model_dump() for track in tracks]},
                    )
                    yield sse("done", {})
                    return

                elif event.type in (
                    "response.failed",
                    "response.incomplete",
                    "error",
                ):
                    yield sse("error", {"detail": error_message})
                    return

        # 완료 이벤트 없이 연결이 끝나면 성공으로 처리하지 않는다.
        yield sse("error", {"detail": error_message})

    except Exception:
        yield sse("error", {"detail": error_message})


def stream_lookup_answer(
    client: OpenAI,
    message: str,
    tracks: list[Track],
    user_context: UserContext | None,
) -> Iterator[str]:
    """조회 결과 문장과 조회된 곡 카드를 순서대로 전송한다.

    NOTE: stream_answer와 스트리밍 루프가 거의 동일하지만, 나중에 트랙을
    한 번에 안 보내고 진짜로 스트리밍하는 식으로 바뀔 수 있다는 얘기가 있어
    지금은 공통 함수로 묶지 않고 각자 독립적으로 뒀다.
    """

    error_message = "챗봇 응답 스트리밍에 실패했습니다."

    try:
        stream = client.responses.create(
            model=os.environ["LLM_MODEL"],
            instructions=lookup_answer_instructions(),
            input=lookup_answer_input(message, tracks, user_context),
            stream=True,
        )

        with stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    yield sse("text", {"delta": event.delta})

                elif event.type == "response.completed":
                    yield sse(
                        "tracks",
                        {"tracks": [track.model_dump() for track in tracks]},
                    )
                    yield sse("done", {})
                    return

                elif event.type in (
                    "response.failed",
                    "response.incomplete",
                    "error",
                ):
                    yield sse("error", {"detail": error_message})
                    return

        yield sse("error", {"detail": error_message})

    except Exception:
        yield sse("error", {"detail": error_message})
