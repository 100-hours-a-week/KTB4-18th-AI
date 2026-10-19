"""음악 검색 결과를 사용자 응답으로 만드는 V1 추천 흐름."""

import os
from dataclasses import dataclass, field
from typing import Literal
from openai import OpenAI
from db.types import Track
from backend.chat.answer import AnswerHints
from backend.core.errors import _model_unavailable
from backend.providers.openrouter import OPENROUTER_BASE_URL

# NOTE: 안내 문구는 일단은 고정 문구로 하기로 선택했다.
GUIDE_TEXT = (
    "저는 상황이나 기분에 어울리는 음악을 추천해드리고, 곡·아티스트 정보도 찾아드리는 "
    "챗봇이에요. '퇴근길에 듣기 좋은 잔잔한 노래'처럼 분위기·장르·연도를 알려주시면 "
    "그에 맞는 곡을 찾아드려요. 다만 주관적인 기준의 곡만 추천하거나 가사·차트 순위로 "
    "찾는 건 아직 지원하지 않아요."
)
OUT_OF_SCOPE_TEXT = "죄송해요, 저는 음악 추천과 곡 정보 조회만 도와드릴 수 있어요."
NOT_FOUND_TEXT = "요청하신 곡 정보를 찾지 못했어요. 곡 제목이나 아티스트명을 다시 확인해 주세요."
CLARIFY_TEXT = {
    # NOTE: 채팅방을 나가면 대화 기억이 사라지고 추천곡만 히스토리에 남는 구조라,
    # 대화에 없는 내용을 가리키면 히스토리를 안내한다.
    "missing_reference": (
        "말씀하신 내용을 이 대화에서 찾지 못했어요. 채팅방을 나가면 이전 대화는 "
        "이어지지 않고, 추천받았던 곡은 히스토리에서 다시 확인하실 수 있어요. "
        "찾으시는 곡이나 분위기를 다시 알려주시면 도와드릴게요."
    ),
    "lookup_ambiguous": (
        "어떤 곡이나 아티스트를 찾으시는지 조금 더 구체적으로 알려주세요. 예: '아이유 밤편지'"
    ),
    "recommend_insufficient_info": (
        "어떤 분위기나 상황에 어울리는 음악을 찾으시나요? 예: '퇴근길에 듣기 좋은 잔잔한 노래'"
    ),
    "recommend_unsupported_condition": (
        "가사 내용이나 노래의 주제, 차트 순위, 정확한 BPM 같은 조건으로는 아직 곡을 "
        "찾을 수 없어요. 대신 '비 오는 날 듣기 좋은 잔잔한 노래'처럼 분위기·상황·장르·"
        "연도로 말씀해 주시면 찾아드릴게요."
    ),
}
# NOTE: 고속 모드는 끄는 경로 없이, 채팅방을 새로 열면(thread_id가 바뀌면) 풀린다.
FAST_MODE_ON_TEXT = (
    "고속 모드를 켰어요. 이제부터 설명 없이 곡만 빠르게 보여드려요. "
    "설명이 있는 일반 대화로 돌아가려면 채팅방을 새로 열어주세요."
)
FAST_RECOMMEND_TEXT = "고속 모드로 추천한 음악이에요."
FAST_LOOKUP_TEXT = "고속 모드로 찾은 곡이에요."


@dataclass
class ChatOutcome:
    """prepare_recommendation의 결과. main.py가 이 kind를 보고 SSE 조립 방식을 고른다."""

    kind: Literal["recommend", "lookup", "static", "fast"]
    client: OpenAI | None = None
    tracks: list[Track] | None = None
    text: str | None = None
    query: str | None = None
    hints: AnswerHints = field(default_factory=AnswerHints)


def prepare_recommendation(message: str, thread_id: str) -> ChatOutcome:
    """OpenAI client를 준비하고 추천 그래프를 실행해, intent·조건에 맞는
    ChatOutcome(무엇을 어떻게 스트리밍할지)을 만든다."""

    api_key = os.getenv("OPENROUTER_LLM_API_KEY", "").strip()
    if not api_key or api_key.startswith("<") or not os.getenv("LLM_MODEL", "").strip():
        raise _model_unavailable()
    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL,
                    timeout=30.0, max_retries=0)

    # NOTE: recommendation → graph.graph → graph.nodes → recommendation로 이어지는
    # 순환 import를 피하려고 그래프 모듈은 호출 시점에 import한다.
    from backend.chat.graph.workflow import build_graph

    result = build_graph().invoke(
        {"message": message, "messages": [message]},
        config={"configurable": {"thread_id": thread_id, "client": client}},
    )

    intent = result["intent"]

    if intent == "fast_mode":
        return ChatOutcome(kind="static", text=FAST_MODE_ON_TEXT)

    # NOTE: 고속 모드도 분기는 일반 모드와 같고, 곡을 보여주는 경우에만 LLM 답변
    # 생성 없이 고정 문구 + 곡 카드(kind="fast")로 내려준다.
    fast_mode = result.get("fast_mode", False)

    if intent == "recommend":
        if result["recommend_unsupported_condition"]:
            return ChatOutcome(kind="static", text=CLARIFY_TEXT["recommend_unsupported_condition"])
        if not result["recommend_has_enough_info"]:
            return ChatOutcome(kind="static", text=CLARIFY_TEXT["recommend_insufficient_info"])
        if fast_mode:
            return ChatOutcome(kind="fast", tracks=result["tracks"], text=FAST_RECOMMEND_TEXT)
        return ChatOutcome(
            kind="recommend", client=client, tracks=result["tracks"],
            query=result["recommend_query"],
            hints=AnswerHints.from_classification(result),
        )

    if intent == "guide":
        return ChatOutcome(kind="static", text=GUIDE_TEXT)

    if intent == "clarify":
        return ChatOutcome(kind="static", text=CLARIFY_TEXT["missing_reference"])

    if intent == "lookup":
        status = result["lookup_status"]
        if status == "found":
            if fast_mode:
                return ChatOutcome(kind="fast", tracks=result["lookup_tracks"], text=FAST_LOOKUP_TEXT)
            return ChatOutcome(
                kind="lookup", client=client, tracks=result["lookup_tracks"],
                query=result["recommend_query"],
                hints=AnswerHints.from_classification(result),
            )
        if status == "ambiguous":
            return ChatOutcome(kind="static", text=CLARIFY_TEXT["lookup_ambiguous"])
        return ChatOutcome(kind="static", text=NOT_FOUND_TEXT)

    return ChatOutcome(kind="static", text=OUT_OF_SCOPE_TEXT)
