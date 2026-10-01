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


def test_lookup_orders_exact_title_match_before_release_date(monkeypatch):
    """제목 부분 일치 결과("butterflies")가 최신이어도 정확히 같은 제목("Butter")이 먼저 온다."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://test:test@localhost/test")
    from db.search import lookup

    class Session:
        def execute(self, stmt):
            self.statement = stmt
            return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: []))

    session = Session()
    lookup(session, song_title="Butter")
    sql = str(session.statement).lower()
    assert sql.index("case") < sql.index("release_date desc")


def test_lookup_spreads_results_across_albums(monkeypatch):
    """최신 앨범 하나가 결과를 독차지하지 않도록 앨범당 1곡씩 먼저 고르고,
    모자라면 남은 곡을 최신순으로 채우며 최종 순서는 최신순을 유지한다."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://test:test@localhost/test")
    from db.search import lookup

    def row(track_id, title, album):
        return SimpleNamespace(track_id=track_id, title=title, artist="IU", album=album)

    # DB가 이미 release_date desc로 정렬해 돌려준 상태라고 가정한다.
    rows = [row(1, "A1", "New Album"), row(2, "A2", "New Album"), row(3, "A3", "New Album"),
            row(4, "B1", "Single B"), row(5, "C1", "Old Album"), row(6, "C2", "Old Album")]

    class Session:
        def execute(self, stmt):
            return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: rows))

    # 앨범당 1곡(1, 4, 5)을 먼저 고르고, 남은 2자리는 나머지 중 최신순(2, 3)으로 채운다.
    picked = lookup(Session(), artist="IU", limit=5)
    assert [r.track_id for r in picked] == [1, 2, 3, 4, 5]

    picked = lookup(Session(), artist="IU", limit=3)
    assert [r.track_id for r in picked] == [1, 4, 5]
