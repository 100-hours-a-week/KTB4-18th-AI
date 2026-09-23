"""RecommendationState를 받아 부분 상태(dict)를 반환하는 LangGraph 노드.
각 노드는 기존 로직을 그대로 호출하는 얇은 wrapper다.
"""

from langchain_core.runnables import RunnableConfig

from backend.graph.state import RecommendationState
from backend.recommendation import (
    assign_reasons,
    classify_intent,
    embed_query,
    get_embedding_client,
    vector_recommendation,
)


def classify_node(state: RecommendationState, config: RunnableConfig) -> dict:
    """대화 기록을 참고해 사용자 메시지의 의도를 분류한다."""

    intent = classify_intent(config["configurable"]["client"], state["messages"])
    return {"intent": intent}


def embed_node(state: RecommendationState) -> dict:
    """사용자 메시지를 검색용 벡터로 변환한다."""

    embedding_client = get_embedding_client()
    query_vector = embed_query(embedding_client, state["message"])
    return {"query_vector": query_vector}


def search_node(state: RecommendationState) -> dict:
    """query_vector로 pgvector 검색을 수행해 후보 곡과 무드 태그를 가져온다."""
    exclude_ids = {int(tid) for tid in state.get("shown_track_ids", [])}
    tracks, mood_tags = vector_recommendation(
        state["query_vector"], state["message"], exclude_ids=exclude_ids,
    )
    return {"tracks": tracks, "mood_tags": mood_tags, "shown_track_ids": [t.track_id for t in tracks]}


def reason_node(state: RecommendationState, config: RunnableConfig) -> dict:
    """검색된 곡마다 추천 이유를 채운다."""

    tracks = state["tracks"]
    assign_reasons(config["configurable"]["client"], state["message"], tracks, state["mood_tags"])
    return {"tracks": tracks}


# NOTE: 최종 답변 생성(stream_answer)은 SSE 스트리밍이라 dict를 반환하는 일반
# 노드 계약과 맞지 않아 그래프 밖(main.py)에서 그대로 처리한다. 그래프는
# reason_node에서 끝난다.
