"""The graph retains one conversation across compiled invocations."""

from uuid import uuid4

from backend.chat.graph import nodes, workflow


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

    jev_intents = {"잔잔한 노래": "recommend", "오늘 점심 뭐 먹지": "out_of_scope", "아이유 밤편지": "lookup"}

    def classify(_client, messages):
        classify_calls.append(messages[-1])
        if messages[-1] == "아이유 밤편지":
            return {"intent": "lookup", "lookup_song": "밤편지", "lookup_artist": "IU"}
        return {"intent": "fast_mode", "recommend_genres": ["Pop"]}

    def decide(state, _questions):
        jev_calls.append(state)
        return {"intent": {"choice": jev_intents[state]}}

    def embed_query(_client, text):
        embedded.append(text)
        return [0.0]

    def vector_recommendation(_vector, **kwargs):
        searched.append(kwargs["genres"])
        return []

    monkeypatch.setattr(nodes, "classify", classify)
    monkeypatch.setattr(nodes, "decide", decide)
    monkeypatch.setattr(nodes, "get_embedding_client", lambda: object())
    monkeypatch.setattr(nodes, "embed_query", embed_query)
    monkeypatch.setattr(nodes, "vector_recommendation", vector_recommendation)
    monkeypatch.setattr(nodes, "lookup_tracks_db", lambda **_kwargs: [])
    config = {"configurable": {"thread_id": str(uuid4()), "client": object()}}

    def send(message):
        return workflow.build_graph().invoke({"message": message, "messages": [message]}, config=config)

    assert send("고속 모드 켜줘")["fast_mode"] is True
    assert send("잔잔한 노래")["intent"] == "recommend"
    assert send("오늘 점심 뭐 먹지")["intent"] == "out_of_scope"
    # 조회는 제목·아티스트 추출이 필요해 그 턴만 classify를 거쳐 lookup으로 간다.
    looked_up = send("아이유 밤편지")
    assert looked_up["intent"] == "lookup"
    assert looked_up["lookup_status"] == "not_found"
    assert looked_up["fast_mode"] is True

    assert classify_calls == ["고속 모드 켜줘", "아이유 밤편지"]
    assert jev_calls == ["잔잔한 노래", "오늘 점심 뭐 먹지", "아이유 밤편지"]
    # 고속 모드 전 classify가 남긴 장르 필터가 검색에 새어 들어가면 안 된다.
    assert embedded == ["잔잔한 노래"]
    assert searched == [None]
