"""추천 그래프의 노드 간에 주고받는 상태 정의.

TODO: 노드를 채워나가며 실제로 필요한 필드만 남기고 정리할 것.
"""
from google import genai


from typing import TypedDict

from backend.schemas import Track, UserContext


class RecommendationState(TypedDict, total=False):
    """음악 추천 파이프라인의 상태."""
    genai_client: genai.Client
    message: str
    query_vector: list[float]

    user_context: UserContext | None

    

    # ── search_node가 채움 ──
    tracks: list[Track]
    mood_tags_by_id: dict[str, dict]

    # ── reason_node가 채움 ──
    # tracks[i].reason을 갱신하는 방식이라 별도 필드가 필요 없을 수도 있음

    # ── answer_node가 채움 ──
    answer: str
