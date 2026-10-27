"""RecommendationState를 받아 부분 상태(dict)를 반환하는 LangGraph 노드.
각 노드는 기존 로직을 그대로 호출하는 얇은 wrapper다.
"""

from langchain_core.runnables import RunnableConfig

from backend.chat.graph.state import RecommendationState
from db.types import Track
from backend.recommendation.intent import classify
from backend.recommendation.embedding import embed_query
from backend.providers.jev import JevUnavailable, decide
from backend.recommendation.fast import (
    artist_question, find_lookup_rows, genre_filter, genre_question, jev_state, next_search_parts, resolve_intent,
    resolve_years, year_questions,
)
from backend.providers.openrouter import get_embedding_client
from backend.recommendation.retrieval import lookup_tracks_db, vector_recommendation


def classify_node(state: RecommendationState, config: RunnableConfig) -> dict:
    """대화 기록을 참고해 사용자 메시지의 의도·대상·조건을 분류한다.
    고속 모드 요청이면 fast_mode를 켜서 다음 턴부터 jev_node로 들어가게 한다."""

    result = classify(config["configurable"]["client"], state["messages"])
    if result["intent"] == "fast_mode":
        result["fast_mode"] = True
    return result


# NOTE: Jev는 정해진 선택지 중 하나를 확률과 함께 고르는 결정 전용 모델이라, 고속 모드의
# 첫 번째 Jev 호출에서 의도와 "제목을 말했는가"를 묻는다. 아티스트·장르·발매연도 질문은
# 선택지(seed 아티스트, DB 장르 목록, 올해까지의 연도)가 실행 시점에 정해지므로 jev_node에서
# fast의 함수로 합친다.
# NOTE: "아이유가 좋아하는 음식 알려주고 데이식스 노래"에서 아티스트를 IU(0.53)로 고르는 것처럼
# 음악 외 요청의 말이 답에 새어 들어가서, 모든 질문에 음악 요청 부분만 보라고 지시한다.
JEV_QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "음악 챗봇 사용자가 이번 메시지로 원하는 것. 음악 외 요청이 섞여 있으면 음악 요청 "
                        "부분을 기준으로 고르고, 음악 요청이 전혀 없을 때만 out_of_scope",
        "criteria": {
            "recommend": "특정 아티스트 한 명으로 한정하지 않고 분위기·상황·장르·연도로 음악 추천을 원함",
            "lookup": "특정 곡이나 아티스트의 정보를 찾거나, 특정 아티스트 한 명의 곡을 원함",
            "guide": "챗봇 사용법이나 지원 범위를 물음",
            "out_of_scope": "음악 추천·조회와 무관한 요청(인사, 잡담, 다른 주제, 의미 없는 입력)",
        },
    },
    "has_title": {
        "type": "noul",
        "instructions": "음악 외 요청 부분(드라마·영화·책 제목 등)은 무시하고, 이번 메시지에 특정 곡 제목이 "
                        "들어 있는가",
        "criteria": {"true": "곡 제목을 말함", "false": "아티스트만 말했거나 곡 제목이 없음"},
    },
    # NOTE: classify는 "좀 더 신나게"를 이전 대화와 합쳐 "비 오는 날 듣기 좋은 신나는 노래"로 다시
    # 써주지만 Jev는 문장을 만들 수 없다. 대신 지난 추천 요청과의 관계를 고르게 하고, 지난 검색
    # 문장에 이번 메시지를 이어 붙여 임베딩한다(fast.next_search_parts).
    # NOTE: 처음엔 "몇 개 앞 메시지부터"(0/1/2)를 골랐는데, 두 개 앞 판단은 확신이 0.27~0.53으로
    # 낮았고 중간에 낀 잡담까지 같이 붙었다. 지난 검색 문장에 누적하면 이 판단이 필요 없다.
    "continues": {
        "type": "choice",
        "instructions": "이번 메시지가 이전 대화의 노래 추천 요청과 어떤 관계인가. 이전 메시지가 모드 전환·"
                        "잡담처럼 노래 조건과 무관하면 그 메시지는 관계 판단에서 빼고 본다",
        "criteria": {
            "new": "이번 메시지만으로 원하는 노래를 알 수 있는 새 요청(주제가 바뀐 경우 포함)",
            "continue": "이전 추천 요청의 분위기·상황을 이어받아 조건을 더하거나 바꿈(예: '좀 더 신나게', '힙합으로')",
            "repeat": "새 조건 없이 같은 조건으로 다시 달라는 요청(예: '다른 곡으로 더 줘', '하나 더')",
        },
    },
}


def jev_node(state: RecommendationState) -> dict:
    """고속 모드에서 classify_node 대신 Jev로 의도를 판단하고, 조회면 LLM 없이 곡을 찾는다.

    조회 곡을 못 찾으면(lookup_status=None) route_by_jev가 그 턴만 classify로 보낸다.

    NOTE: LLM이 대화 맥락을 합쳐 재구성해주던 recommend_query 대신, 지난 검색 문장에 이번
    메시지를 이어 붙인 문장을 검색에 쓴다(추천 턴에서만 갱신해서 잡담·조회 턴은 맥락에 안 섞인다).
    장르·연도 필터도 Jev 답으로 채우고, 추천 조건 판정과 조회 상태는 이전 턴 값이 checkpointer에
    남아 있으면 새어 들어가므로 매번 덮어쓴다.
    """

    messages = state["messages"]
    shown_ids = {int(tid) for tid in state.get("shown_track_ids", [])}

    # NOTE: Jev가 실패해도 사용자에게 오류를 보이지 않고, 그 턴만 classify(LLM)로 처리한다.
    try:
        questions = {**JEV_QUESTIONS, "artist": artist_question(), "genre": genre_question(),
                     **year_questions()}
        answers = decide(jev_state(messages), questions)
        intent = resolve_intent(answers)
        rows = find_lookup_rows(messages, answers, shown_ids) if intent == "lookup" else None
    except JevUnavailable:
        return {"jev_failed": True}

    min_year, max_year = resolve_years(state["message"], answers)
    result = {
        "jev_failed": False,
        "intent": intent,
        "lookup_status": None,
        "recommend_has_enough_info": True,
        "recommend_unsupported_condition": False,
        "recommend_genres": genre_filter(answers["genre"]),
        "recommend_min_year": min_year,
        "recommend_max_year": max_year,
    }
    if intent == "recommend":
        # NOTE: 지난 턴을 classify가 처리했으면(고속 모드 전, Jev 폴백) search_parts가 없거나 낡았다.
        # 그럴 땐 classify가 쓴 검색 문장(이 시점엔 아직 덮어쓰기 전인 recommend_query)을 출발점으로 삼는다.
        previous_parts = state.get("search_parts") or []
        previous_query = state.get("recommend_query")
        if previous_query and " ".join(previous_parts) != previous_query:
            previous_parts = [previous_query]
        parts = next_search_parts(previous_parts, state["message"], answers)
        result |= {"search_parts": parts, "recommend_query": " ".join(parts)}
    if rows:
        tracks = [
            Track(
                track_id=str(row.track_id),
                title=row.title,
                artist=row.artist,
                artwork_url=row.artwork_url,
                preview_url=row.preview_url,
                store_url=row.store_url,
            )
            for row in rows[:5]
        ]
        result |= {"lookup_status": "found", "lookup_tracks": tracks,
                   "shown_track_ids": [t.track_id for t in tracks]}
    return result


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
    """query_vector로 pgvector 검색을 수행해 후보 곡을 가져온다.
    classify_node가 뽑은 genres/min_year/max_year를 메타데이터 조건으로 같이 넘긴다."""
    exclude_ids = {int(tid) for tid in state.get("shown_track_ids", [])}
    tracks = vector_recommendation(
        state["query_vector"], exclude_ids=exclude_ids,
        genres=state.get("recommend_genres"), min_year=state.get("recommend_min_year"),
        max_year=state.get("recommend_max_year"),
    )
    return {"tracks": tracks, "shown_track_ids": [t.track_id for t in tracks]}


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
# 처리한다. 그래프는 search_node 또는 lookup_node에서 끝난다.
