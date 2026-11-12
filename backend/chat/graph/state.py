"""추천 그래프의 노드 간에 주고받는 상태 정의."""

import operator
from typing import Annotated, TypedDict

from backend.chat.schemas import UserContext
from db.types import Track

def _cap_shown_ids(old: list[str], new: list[str]) -> list[str]:
    return (old + new)[-100:]

class RecommendationState(TypedDict, total=False):
    """음악 추천 파이프라인의 상태."""

    # NOTE: client 여러 군데 쓰여서 여기 놨었다가 터지더라.
    # 생각해보니 config라는 방법을 배웠어서 그걸로 사용했다.

    # ── 입력 ──
    message: str
    user_context: UserContext | None

    # ── 대화 세션 동안 누적되는 기록. checkpointer가 thread_id별로 이어붙여준다 ──
    messages: Annotated[list[str], operator.add]

    # ── 고속 모드. classify_node가 켜고, checkpointer가 thread_id별로 기억한다 ──
    # NOTE: 끄는 경로는 없다. 채팅방을 나가면 thread_id가 바뀌어 일반 모드로 돌아온다.
    fast_mode: bool
    # jev_node가 Jev 호출에 실패하면 True. route_by_jev가 보고 그 턴만 classify로 보낸다.
    jev_failed: bool
    # Jev가 미지원 조건(가사·순위·정확한 수치·주제 기준)을 감지하면 True. 그 턴만 classify로 보낸다.
    jev_unsupported: bool
    # 고속 모드 검색 문장(recommend_query)을 이루는 메시지들. 추천 턴에서만 jev_node가 갱신한다.
    search_parts: list[str]

    # ── classify_node(고속 모드에선 jev_node)가 채움 ──
    intent: str
    recommend_has_enough_info: bool
    recommend_unsupported_condition: bool
    recommend_genres: list[str] | None
    recommend_min_year: int | None
    recommend_max_year: int | None
    recommend_query: str
    lookup_song: str | None
    lookup_song_alt: str | None
    lookup_artist: str | None
    requested_count: int | None
    has_non_music_request: bool
    response_style: str | None

    # ── 대화형 추천. classify_node가 부족한 정보·요청 관계를, dialogue_node가 에피소드를 채운다 ──
    # NOTE: 질문 횟수 상한은 LLM이 아니라 코드가 지키도록 에피소드(같은 추천 요청) 단위로 센다.
    missing_info: list[str]  # place·activity·mood 중 아직 모르는 정보
    episode_relation: str  # "new" | "continue"
    episode: int
    question_count: int  # 이번 에피소드에서 실제로 전송된 추가 질문 수
    followup_info: list[str]  # 이번 턴 답변에서 물을 정보. 비면 묻지 않는다
    asked_info: list[str]  # 실제로 전송된 마지막 질문의 정보. 분류 LLM이 짧은 답을 해석할 때 쓴다
    known_info: list[str]  # 이번 턴까지 확보한 정보(missing_info의 나머지)

    # ── embed_node가 채움 ──
    query_vector: list[float]

    # ── search_node가 채움 ──
    tracks: list[Track]

    # ── SSE 곡 카드 전송 시 갱신 ──
    shown_track_ids: Annotated[list[str], _cap_shown_ids]
    recent_tracks: list[dict]

    # ── lookup_node가 채움 ──
    lookup_status: str  # "found" | "ambiguous" | "not_found"
    lookup_tracks: list[Track]
