"""음악 검색 결과를 사용자 응답으로 만드는 V1 추천 흐름."""

import os
from dataclasses import dataclass
from typing import Literal
from openai import OpenAI
from db.types import Track
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
    "intent_unclear": (
        "어떤 걸 도와드릴까요? 음악을 추천받고 싶으시면 지금 기분이나 상황을, 곡 정보가 "
        "궁금하시면 곡 제목이나 아티스트명을 알려주세요."
    ),
    "recommend_insufficient_info": (
        "어떤 분위기나 상황에 어울리는 음악을 찾으시나요? 예: '퇴근길에 듣기 좋은 잔잔한 노래'"
    ),
}


@dataclass
class ChatOutcome:
    """prepare_recommendation의 결과. main.py가 이 kind를 보고 SSE 조립 방식을 고른다."""

    kind: Literal["recommend", "lookup", "static"]
    client: OpenAI | None = None
    tracks: list[Track] | None = None
    text: str | None = None
    query: str | None = None


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

    if intent == "recommend":
        if result["recommend_unsupported_condition"]:
            return ChatOutcome(kind="static", text=GUIDE_TEXT)
        if not result["recommend_has_enough_info"]:
            return ChatOutcome(kind="static", text=CLARIFY_TEXT["recommend_insufficient_info"])
        return ChatOutcome(
            kind="recommend", client=client, tracks=result["tracks"],
            query=result["recommend_query"],
        )

    if intent == "guide":
        return ChatOutcome(kind="static", text=GUIDE_TEXT)

    if intent == "clarify":
        return ChatOutcome(kind="static", text=CLARIFY_TEXT["intent_unclear"])

    if intent == "lookup":
        status = result["lookup_status"]
        if status == "found":
            return ChatOutcome(kind="lookup", client=client, tracks=result["lookup_tracks"])
        if status == "ambiguous":
            return ChatOutcome(kind="static", text=CLARIFY_TEXT["intent_unclear"])
        return ChatOutcome(kind="static", text=NOT_FOUND_TEXT)

    return ChatOutcome(kind="static", text=OUT_OF_SCOPE_TEXT)
