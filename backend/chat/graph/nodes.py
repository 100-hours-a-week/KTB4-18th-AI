"""RecommendationState를 받아 부분 상태(dict)를 반환하는 LangGraph 노드.
각 노드는 기존 로직을 그대로 호출하는 얇은 wrapper다.
"""

from langchain_core.runnables import RunnableConfig

from backend.chat.graph.state import RecommendationState
from db.types import Track
from backend.recommendation.reasons import assign_reasons
from backend.recommendation.intent import classify
from backend.recommendation.embedding import embed_query
from backend.providers.openrouter import get_embedding_client
from backend.recommendation.retrieval import lookup_tracks_db, vector_recommendation


def classify_node(state: RecommendationState, config: RunnableConfig) -> dict:
    """대화 기록을 참고해 사용자 메시지의 의도·대상·조건을 분류한다."""

    return classify(config["configurable"]["client"], state["messages"])


def embed_node(state: RecommendationState) -> dict:
    """classify_node가 재구성한 recommend_query를 검색용 벡터로 변환한다.

    NOTE: state["message"](이번 턴 원문)가 아니라 recommend_query를 쓴다. 후속
    요청("신나게 다른 곡")은 이번 턴 메시지만으론 상황(원래 분위기)이 안 담겨
    있어서, classify_node가 대화 맥락을 합쳐 재구성해준 문장을 써야 한다.
    """

    embedding_client = get_embedding_client()
    query_vector = embed_query(embedding_client, state["recommend_query"])
    return {"query_vector": query_vector}


def search_node(state: RecommendationState) -> dict:
    """query_vector로 pgvector 검색을 수행해 후보 곡과 무드 태그를 가져온다.
    classify_node가 뽑은 genres/min_year/max_year를 메타데이터 조건으로 같이 넘긴다."""
    exclude_ids = {int(tid) for tid in state.get("shown_track_ids", [])}
    tracks, mood_tags = vector_recommendation(
        state["query_vector"], state["recommend_query"], exclude_ids=exclude_ids,
        genres=state.get("recommend_genres"), min_year=state.get("recommend_min_year"),
        max_year=state.get("recommend_max_year"),
    )
    return {"tracks": tracks, "mood_tags": mood_tags, "shown_track_ids": [t.track_id for t in tracks]}


def reason_node(state: RecommendationState, config: RunnableConfig) -> dict:
    """검색된 곡마다 추천 이유를 채운다."""

    tracks = state["tracks"]
    assign_reasons(config["configurable"]["client"], state["recommend_query"], tracks, state["mood_tags"])
    return {"tracks": tracks}


LOOKUP_FALLBACK_REASON = "조회하신 곡 정보예요."


def lookup_node(state: RecommendationState) -> dict:
    """곡 제목·아티스트로 DB를 조회해 lookup_status(found/ambiguous/not_found)를 정한다.

    - 제목·아티스트 둘 다 없으면 조회 없이 바로 모호 처리.
    - 그 외 결과가 있으면(동명이곡으로 여러 아티스트가 걸려도) found로 카드를 다 보여주고,
      없으면 not_found.
    """

    song, artist = state.get("lookup_song"), state.get("lookup_artist")
    if not song and not artist:
        return {"lookup_status": "ambiguous"}

    # 제목+아티스트를 둘 다 콕 집어 물어본 "정확한 지정" 조회는 이전에 보여준 적이
    # 있어도 다시 답해야 하므로 exclude하지 않는다. 하나만 준 "폭넓은" 조회일 때만
    # (예: "이 아티스트 곡 더 보여줘") 턴 간 중복을 뺀다.
    exclude_ids = None
    if not (song and artist):
        exclude_ids = {int(tid) for tid in state.get("shown_track_ids", [])}

    rows = lookup_tracks_db(song_title=song, artist=artist, exclude_ids=exclude_ids)

    # NOTE: 한국 곡 제목은 카탈로그에 한글("심술")로도, 공식 영문 제목("How People
    # Move")으로도 저장돼 있어서, 원제로 못 찾으면 classify가 준 영문 제목으로 다시 찾는다.
    song_alt = state.get("lookup_song_alt")
    if not rows and song and song_alt:
        rows = lookup_tracks_db(song_title=song_alt, artist=artist, exclude_ids=exclude_ids)

    # NOTE: "coin 노래"처럼 이름 하나만 주면 classify가 곡 제목(IU의 Coin)을 동명의
    # 아티스트(COIN)로 읽는 등 어느 쪽인지 갈리는 경우가 있어, 하나만 채워졌는데
    # 결과가 없으면 같은 이름을 반대 필드로 한 번 더 조회한다.
    if not rows and not (song and artist):
        rows = lookup_tracks_db(song_title=artist, artist=song, exclude_ids=exclude_ids)

    if not rows:
        return {"lookup_status": "not_found"}

    tracks = [
        Track(
            track_id=str(row.track_id),
            title=row.title,
            artist=row.artist,
            artwork_url=row.artwork_url,
            preview_url=row.preview_url,
            store_url=row.store_url,
            reason=LOOKUP_FALLBACK_REASON,
        )
        for row in rows[:5]
    ]
    return {
        "lookup_status": "found",
        "lookup_tracks": tracks,
        "shown_track_ids": [t.track_id for t in tracks],
    }


# NOTE: 최종 답변 생성(stream_answer/stream_lookup_answer)은 SSE 스트리밍이라
# dict를 반환하는 일반 노드 계약과 맞지 않아 그래프 밖(main.py)에서 그대로
# 처리한다. 그래프는 reason_node 또는 lookup_node에서 끝난다.
