"""Search preserves candidate order, duplicate removal, filters, and response fields."""

from types import SimpleNamespace


def test_search_result_and_filters_without_database(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://test:test@localhost/test")
    from db.search import search

    def row(track_id, artist):
        return SimpleNamespace(track_id=track_id, title="Song", artist=artist,
                               artwork_url=None, preview_url=None,
                               store_url="https://example.com/song", mood_tags={"calm": 0.8})

    class Session:
        def execute(self, stmt):
            self.statement = stmt
            return SimpleNamespace(all=lambda: [(row(1, "Artist"), 0.1),
                                                (row(2, "ARTIST"), 0.2),
                                                (row(3, "Other"), 0.3)])

    session = Session()
    tracks, moods = search(session, [1.0] * 3072, "quiet", exclude_ids={99},
                           min_year=2020, genres=["K-Pop"])
    assert [track.track_id for track in tracks] == ["1", "3"]
    assert [track.reason for track in tracks] == ["'quiet' 상황에 어울리는 곡입니다."] * 2
    assert moods == {"1": {"calm": 0.8}, "3": {"calm": 0.8}}
    sql = str(session.statement)
    assert "release_date" in sql and "genre" in sql and "NOT IN" in sql
