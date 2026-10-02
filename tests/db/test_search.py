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
        def __init__(self):
            self.statements = []

        def execute(self, stmt):
            self.statements.append(stmt)
            return SimpleNamespace(all=lambda: [(row(1, "Artist"), 0.1),
                                                (row(2, "ARTIST"), 0.2),
                                                (row(3, "Other"), 0.3)])

    session = Session()
    tracks, moods = search(session, [1.0] * 3072, "quiet", exclude_ids={99},
                           min_year=2020, genres=["K-Pop"])
    assert [track.track_id for track in tracks] == ["1", "3"]
    assert [track.reason for track in tracks] == ["'quiet' 상황에 어울리는 곡입니다."] * 2
    assert moods == {"1": {"calm": 0.8}, "3": {"calm": 0.8}}
    sql = str(session.statements[0])
    assert "release_date" in sql and "genre" in sql and "NOT IN" in sql


def _row(track_id):
    return SimpleNamespace(track_id=track_id, title=f"Song {track_id}", artist="Artist",
                           artwork_url=None, preview_url=None,
                           store_url="https://example.com/song", mood_tags={})


def test_search_fills_without_genre_when_genre_results_are_short(monkeypatch):
    """장르 필터 결과가 k곡보다 적으면, 장르만 빼고(연도는 유지) 다시 찾아 뒤에 채운다."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://test:test@localhost/test")
    from db.search import search

    class Session:
        def __init__(self):
            self.statements = []

        def execute(self, stmt):
            self.statements.append(stmt)
            if len(self.statements) == 1:  # 장르 필터: 1곡뿐
                rows = [(_row(1), 0.1)]
            else:  # 장르 없이: 장르 결과와 겹치는 1번 포함
                rows = [(_row(i), 0.1 * i) for i in (2, 1, 3, 4, 5, 6)]
            return SimpleNamespace(all=lambda: rows)

    session = Session()
    tracks, _ = search(session, [1.0] * 3072, "잔잔한 발라드", genres=["Singer/Songwriter"], max_year=2014)

    assert [track.track_id for track in tracks] == ["1", "2", "3", "4", "5"]
    first, second = (str(stmt) for stmt in session.statements)
    assert "genre IN" in first and "genre IN" not in second
    assert "release_date <=" in first and "release_date <=" in second


def test_search_does_not_requery_when_genre_results_are_enough(monkeypatch):
    """장르 필터만으로 k곡이 채워지면 다시 검색하지 않는다."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://test:test@localhost/test")
    from db.search import search

    class Session:
        def __init__(self):
            self.statements = []

        def execute(self, stmt):
            self.statements.append(stmt)
            return SimpleNamespace(all=lambda: [(_row(i), 0.1 * i) for i in range(1, 7)])

    session = Session()
    tracks, _ = search(session, [1.0] * 3072, "신나는 케이팝", genres=["K-Pop"])

    assert len(tracks) == 5
    assert len(session.statements) == 1


def test_search_applies_year_range(monkeypatch):
    """연도 하한·상한이 둘 다 있으면 release_date에 >=와 <= 조건이 함께 걸린다."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://test:test@localhost/test")
    from db.search import search

    class Session:
        def execute(self, stmt):
            self.statement = stmt
            return SimpleNamespace(all=lambda: [])

    session = Session()
    search(session, [1.0] * 3072, "90년대 노래", min_year=1990, max_year=1999)
    sql = str(session.statement)
    assert "release_date >=" in sql and "release_date <=" in sql


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
