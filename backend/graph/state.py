"""추천 그래프의 노드 간에 주고받는 상태 정의."""

from typing import TypedDict

from openai import OpenAI

from backend.schemas import Track, UserContext


class RecommendationState(TypedDict, total=False):
    """음악 추천 파이프라인의 상태."""

    # ── 입력 ──
    message: str
    user_context: UserContext | None
    client: OpenAI  # reason_node가 사용. genai client는 embed_node 안에서만 쓰여 state에 안 둠.

    # ── embed_node가 채움 ──
    query_vector: list[float]

    # ── search_node가 채움 ──
    tracks: list[Track]
    mood_tags: dict[str, dict]
