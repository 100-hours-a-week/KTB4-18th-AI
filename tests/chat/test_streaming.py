"""Streaming failures keep the original SSE completion contract."""

from types import SimpleNamespace

from backend.chat.streaming import stream_answer, stream_lookup_answer
from db.types import Track


class FakeStream:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        return iter([SimpleNamespace(type="response.output_text.delta", delta="hello"),
                     SimpleNamespace(type="response.failed")])


def test_failed_streams_emit_error_without_done(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "test-model")
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_kwargs: FakeStream()))
    track = Track(track_id="1", title="song", artist="artist", artwork_url=None,
                  preview_url=None, reason="reason")
    for stream in (stream_answer(client, "request", [track], None),
                   stream_lookup_answer(client, "request", [track])):
        events = list(stream)
        assert [event.split("\n", 1)[0] for event in events] == ["event: text", "event: error"]
