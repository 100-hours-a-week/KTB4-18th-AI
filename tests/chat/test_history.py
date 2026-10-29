"""검색·생성 실패를 표시 이력에 넣지 않고 실제 SSE 카드 순서로 후속 요청을 해석한다."""

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from backend.chat import streaming
from backend.chat.graph import nodes, workflow
from db.types import Track


def track(number):
    return Track(track_id=str(number), title=f"song {number}", artist=f"artist {number}",
                 artwork_url=None, preview_url=None)


class ModelStream:
    def __init__(self, terminal):
        self.terminal = terminal

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        return iter([SimpleNamespace(type=self.terminal)])


@pytest.mark.parametrize("stream_answer", [streaming.stream_answer, streaming.stream_lookup_answer])
@pytest.mark.parametrize("completed", [True, False])
def test_only_emitted_cards_update_history(monkeypatch, stream_answer, completed):
    monkeypatch.setenv("LLM_MODEL", "test-model")
    terminal = "response.completed" if completed else "response.failed"
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_kwargs: ModelStream(terminal)))
    thread = str(uuid4())
    config = {"configurable": {"thread_id": thread}}
    workflow.record_shown_tracks(thread, [track(1).model_dump()])
    events = list(stream_answer(client, "request", [track(2)], None, thread_id=thread))
    state = workflow.build_graph().get_state(config).values
    assert state["shown_track_ids"] == (["1", "2"] if completed else ["1"])
    assert state["recent_tracks"] == [{
        "rank": 1, "track_id": "2" if completed else "1",
        "title": "song 2" if completed else "song 1",
        "artist": "artist 2" if completed else "artist 1",
    }]
    assert any(event.startswith("event: tracks\n") for event in events) is completed


def test_search_does_not_mark_unsent_cards_as_shown(monkeypatch):
    calls = []
    monkeypatch.setattr(nodes, "get_embedding_client", lambda: object())
    monkeypatch.setattr(nodes, "embed_query", lambda *_args: [1.0])
    monkeypatch.setattr(nodes, "vector_recommendation",
                        lambda *_args, **kwargs: calls.append(kwargs["exclude_ids"]) or [track(1)])
    config = {"configurable": {"thread_id": str(uuid4())}}
    initial = {"recommend_query": "calm"}
    for _ in range(2):
        workflow.build_graph(photo=True).invoke(initial, config=config)
    assert calls == [set(), set()]
    assert not workflow.build_graph().get_state(config).values.get("shown_track_ids")


def test_recent_card_order_reaches_classifier_only_in_same_thread(monkeypatch):
    seen = []
    monkeypatch.setattr(nodes, "classify", lambda _client, messages: seen.append(messages) or {"intent": "guide"})
    thread = str(uuid4())
    workflow.record_shown_tracks(thread, [track(2).model_dump(), track(1).model_dump()])
    for current_thread in (thread, str(uuid4())):
        workflow.build_graph().invoke({"messages": ["두 번째 곡 가수 노래 더"]}, config={
            "configurable": {"thread_id": current_thread, "client": object()}})
    snapshot = json.loads(seen[0][-2].split(": ", 1)[1])
    assert [(t["rank"], t["artist"]) for t in snapshot] == [(1, "artist 2"), (2, "artist 1")]
    assert seen[0][-1] == "두 번째 곡 가수 노래 더"
    assert seen[1] == ["두 번째 곡 가수 노래 더"]


def test_artist_more_excludes_shown_but_exact_song_can_repeat(monkeypatch):
    seen = []
    monkeypatch.setattr(nodes, "lookup_tracks_db", lambda **kwargs: seen.append(kwargs["exclude_ids"]) or [])
    common = {"shown_track_ids": ["1"], "lookup_artist": "artist 1"}
    nodes.lookup_node(common)
    nodes.lookup_node({**common, "lookup_song": "song 1"})
    assert seen == [{1}, {1}, None]


def test_history_caps_ids_and_replaces_latest_batch():
    thread = str(uuid4())
    for start in range(0, 105, 5):
        workflow.record_shown_tracks(thread, [track(i).model_dump() for i in range(start, start + 5)])
    state = workflow.build_graph().get_state({"configurable": {"thread_id": thread}}).values
    assert state["shown_track_ids"] == [str(i) for i in range(5, 105)]
    assert [t["track_id"] for t in state["recent_tracks"]] == [str(i) for i in range(100, 105)]
    assert workflow.build_graph().get_state({"configurable": {"thread_id": thread}}).next == ()
