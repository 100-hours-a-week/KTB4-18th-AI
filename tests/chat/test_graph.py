"""The graph retains one conversation across compiled invocations."""

from types import SimpleNamespace
from uuid import uuid4

from backend.chat.graph import nodes, workflow
from backend.providers.jev import JevUnavailable
from backend.recommendation import fast


def test_thread_messages_survive_graph_rebuild(monkeypatch):
    seen = []

    def classify(_client, messages):
        seen.append(messages)
        return {"intent": "guide"}

    monkeypatch.setattr(nodes, "classify", classify)
    config = {"configurable": {"thread_id": str(uuid4()), "client": object()}}
    workflow.build_graph().invoke({"message": "first", "messages": ["first"]}, config=config)
    workflow.build_graph().invoke({"message": "second", "messages": ["second"]}, config=config)
    assert seen == [["first"], ["first", "second"]]


def test_fast_mode_switches_entry_to_jev(monkeypatch):
    classify_calls, jev_calls, embedded, searched = [], [], [], []

    # 메시지별 Jev 1단계 답: (intent, artist, artist 확신, has_title, genre, year_from, year_to, continues)
    jev_answers = {
        "90년대 잔잔한 힙합": ("recommend", "none", 0.9, 0.0, "Hip-Hop", "1990", "1999", "new"),
        "오늘 점심 뭐 먹지": ("out_of_scope", "none", 0.9, 0.0, "none", "none", "none", "new"),
        "좀 더 신나게": ("recommend", "none", 0.9, 0.0, "none", "none", "none", "continue"),
        "악뮤 노래": ("lookup", "AKMU", 0.99, 0.05, "K-Pop", "none", "none", "new"),
        "그 노래 찾아줘": ("lookup", "none", 0.6, 0.5, "none", "none", "none", "new"),
    }

    def classify(_client, messages):
        classify_calls.append(messages[-1])
        if messages[-1] == "그 노래 찾아줘":
            return {"intent": "lookup", "lookup_song": None, "lookup_artist": None}
        return {"intent": "fast_mode", "recommend_genres": ["Pop"]}

    def decide(state, _questions):
        # jev_state가 마지막 줄을 "이번 메시지(판단 대상): ..."로 만든다.
        message = state[-1].split(": ", 1)[1]
        jev_calls.append(message)
        intent, artist, confidence, has_title, genre, year_from, year_to, continues = jev_answers[message]
        return {"intent": {"choice": intent}, "artist": {"choice": artist, "confidence": confidence},
                "has_title": {"noul": has_title}, "genre": {"choice": genre, "confidence": 0.9},
                "year_from": {"choice": year_from, "confidence": 0.9},
                "year_to": {"choice": year_to, "confidence": 0.9},
                "continues": {"choice": continues, "confidence": 0.9}}

    akmu_row = SimpleNamespace(track_id=1, title="How People Move", artist="AKMU",
                               artwork_url="a", preview_url="p", store_url="s")

    # 가짜 벡터에 문장을 그대로 담아, 검색에 어떤 후보 문장의 벡터가 쓰였는지 확인한다.
    def embed_query(_client, text):
        embedded.append(text)
        return [text]

    def vector_recommendation(vector, **kwargs):
        searched.append((vector[0], kwargs["genres"], kwargs["min_year"], kwargs["max_year"]))
        return []

    monkeypatch.setattr(nodes, "classify", classify)
    monkeypatch.setattr(nodes, "decide", decide)
    monkeypatch.setattr(fast, "get_embedding_client", lambda: object())
    monkeypatch.setattr(fast, "embed_query", embed_query)
    monkeypatch.setattr(nodes, "vector_recommendation", vector_recommendation)
    monkeypatch.setattr(nodes, "lookup_tracks_db", lambda **_kwargs: [])
    monkeypatch.setattr(nodes, "artist_question", lambda: {})
    monkeypatch.setattr(nodes, "genre_question", lambda: {})
    monkeypatch.setattr(nodes, "year_questions", lambda: {})
    monkeypatch.setattr(fast, "lookup_tracks_db", lambda **_kwargs: [akmu_row])
    config = {"configurable": {"thread_id": str(uuid4()), "client": object()}}

    def send(message):
        return workflow.build_graph().invoke({"message": message, "messages": [message]}, config=config)

    assert send("고속 모드 켜줘")["fast_mode"] is True
    assert send("90년대 잔잔한 힙합")["intent"] == "recommend"
    assert send("오늘 점심 뭐 먹지")["intent"] == "out_of_scope"
    # 이어받기는 지난 검색 문장에 누적한다. 사이에 낀 잡담 턴은 검색 문장을 바꾸지 않는다.
    assert send("좀 더 신나게")["recommend_query"] == "90년대 잔잔한 힙합 좀 더 신나게"
    # 아티스트만 말한 조회는 classify 없이 Jev가 고른 아티스트로 바로 찾는다.
    artist_only = send("악뮤 노래")
    assert artist_only["lookup_status"] == "found"
    assert [t.title for t in artist_only["lookup_tracks"]] == ["How People Move"]
    # 아티스트를 못 고르면 그 턴만 classify를 거쳐 lookup_node로 간다.
    looked_up = send("그 노래 찾아줘")
    assert looked_up["intent"] == "lookup"
    assert looked_up["lookup_status"] == "ambiguous"
    assert looked_up["fast_mode"] is True

    assert classify_calls == ["고속 모드 켜줘", "그 노래 찾아줘"]
    assert jev_calls == ["90년대 잔잔한 힙합", "오늘 점심 뭐 먹지", "좀 더 신나게", "악뮤 노래", "그 노래 찾아줘"]
    # 임베딩은 Jev 답을 기다리지 않고 후보마다 미리 출발한다("좀 더 신나게" 턴은 new/continue/repeat 후보 3개).
    assert {"좀 더 신나게", "90년대 잔잔한 힙합 좀 더 신나게", "90년대 잔잔한 힙합"} <= set(embedded)
    # 검색에는 Jev 답에 맞는 후보의 벡터가 쓰인다. 장르·연도는 고속 모드 전 classify가 남긴
    # 장르(Pop) 대신 Jev가 고른 값이다.
    assert searched == [
        ("90년대 잔잔한 힙합", ["Hip-Hop"], 1990, 1999),
        ("90년대 잔잔한 힙합 좀 더 신나게", None, None, None),
    ]


def test_fast_mode_falls_back_to_classify_when_jev_fails(monkeypatch):
    classify_calls = []

    def classify(_client, messages):
        classify_calls.append(messages[-1])
        if messages[-1] == "고속 모드 켜줘":
            return {"intent": "fast_mode"}
        return {"intent": "guide"}

    def decide(_state, _questions):
        raise JevUnavailable("timeout")

    monkeypatch.setattr(nodes, "classify", classify)
    monkeypatch.setattr(nodes, "decide", decide)
    monkeypatch.setattr(nodes, "artist_question", lambda: {})
    monkeypatch.setattr(nodes, "genre_question", lambda: {})
    monkeypatch.setattr(nodes, "year_questions", lambda: {})
    monkeypatch.setattr(fast, "get_embedding_client", lambda: object())
    monkeypatch.setattr(fast, "embed_query", lambda _client, text: [text])
    config = {"configurable": {"thread_id": str(uuid4()), "client": object()}}

    def send(message):
        return workflow.build_graph().invoke({"message": message, "messages": [message]}, config=config)

    send("고속 모드 켜줘")
    result = send("사용법 알려줘")

    # Jev가 실패해도 오류 없이 그 턴만 classify가 처리하고, 고속 모드는 유지된다.
    assert classify_calls == ["고속 모드 켜줘", "사용법 알려줘"]
    assert result["intent"] == "guide"
    assert result["fast_mode"] is True
