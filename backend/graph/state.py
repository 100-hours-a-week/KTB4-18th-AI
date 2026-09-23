"""추천 그래프의 노드 간에 주고받는 상태 정의."""

import operator
from typing import Annotated, TypedDict

from backend.schemas import Track, UserContext


class RecommendationState(TypedDict, total=False):
    """음악 추천 파이프라인의 상태."""

    # NOTE: client 여러 군데 쓰여서 여기 놨었다가 터지더라.
    # 생각해보니 config라는 방법을 배웠어서 그걸로 사용했다.

    # ── 입력 ──
    message: str
    user_context: UserContext | None

    # ── 대화 세션 동안 누적되는 기록. checkpointer가 thread_id별로 이어붙여준다 ──
    messages: Annotated[list[str], operator.add]

    # ── classify_node가 채움 ──
    intent: str

    # ── embed_node가 채움 ──
    query_vector: list[float]

    # ── search_node가 채움 ──
    tracks: list[Track]
    mood_tags: dict[str, dict]
