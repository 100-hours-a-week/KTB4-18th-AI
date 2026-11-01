"""Search preserves candidate order, duplicate removal, filters, and response fields."""

from types import SimpleNamespace
import pytest


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
    tracks = search(session, [1.0] * 3072, exclude_ids={99},
                           min_year=2020, genres=["K-Pop"])
    assert [track.track_id for track in tracks] == ["1", "3"]
    assert all("reason" not in track.model_dump() for track in tracks)
    sql = str(session.statements[0])
    assert "release_date" in sql and "genre" in sql and "NOT IN" in sql


def _row(track_id):
    return SimpleNamespace(track_id=track_id, title=f"Song {track_id}", artist="Artist",
                           artwork_url=None, preview_url=None,
                           store_url="https://example.com/song", mood_tags={})


@pytest.mark.parametrize("space", ["gemini", "v4"])
def test_search_preserves_genre_when_results_are_short(monkeypatch, space):
    """개수가 부족해도 장르·연도 조건 밖 곡으로 채우지 않는다."""
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
    tracks = search(session, [1.0] * (3073 if space == "v4" else 3072),
                    genres=["Singer/Songwriter"], max_year=2014, space=space)

    assert [track.track_id for track in tracks] == ["1"]
    assert len(session.statements) == 1
    sql = str(session.statements[0])
    assert "genre IN" in sql and "release_date <=" in sql


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
    tracks = search(session, [1.0] * 3072, genres=["K-Pop"])

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
    search(session, [1.0] * 3072, min_year=1990, max_year=1999)
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


class _RecordingSession:
    """search()가 실행한 SQL을 모으고, 지정한 (행, 거리) 목록을 돌려준다."""

    def __init__(self, rows=()):
        self.statements, self.rows = [], list(rows)

    def execute(self, stmt):
        self.statements.append(stmt)
        return SimpleNamespace(all=lambda: self.rows)


def test_search_gemini_space_keeps_cosine_on_emb_gemini(monkeypatch):
    """기본(gemini)은 기존과 같은 emb_gemini 코사인 검색이고, deferred인 emb_v4는 SELECT에 없다."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://test:test@localhost/test")
    from db.search import search

    session = _RecordingSession()
    search(session, [1.0] * 3072)
    sql = str(session.statements[0])
    assert "emb_gemini <=>" in sql and "emb_gemini IS NOT NULL" in sql
    assert "emb_v4" not in sql


def test_search_v4_space_uses_inner_product_on_emb_v4(monkeypatch):
    """v4는 emb_v4 내적(<#>, 음의 내적)으로 정렬하고, 점수는 부호를 뒤집은 내적이다."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://test:test@localhost/test")
    from db.search import search

    session = _RecordingSession([(_row(1), -0.42), (_row(2), -0.30)])
    tracks = search(session, [0.1] * 3072 + [1.0], space="v4")
    sql = str(session.statements[0])
    assert "emb_v4 <#>" in sql and "emb_v4 IS NOT NULL" in sql
    # NOTE: select(TrackRow)가 emb_gemini를 SELECT 목록에는 넣는다(기존 동작). 정렬·필터에만 안 쓰이면 된다
    assert "emb_gemini <=>" not in sql and "emb_gemini IS NOT NULL" not in sql
    assert [track.track_id for track in tracks] == ["1", "2"]


def test_known_genres_counts_the_same_search_column(monkeypatch):
    """장르 목록도 검색과 같은 칸으로 세야 '목록엔 있는데 검색하면 0곡'이 안 생긴다."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://test:test@localhost/test")
    from db.search import known_genres

    class Session:
        def execute(self, stmt):
            self.statement = stmt
            return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: []))

    for space, column in (("gemini", "emb_gemini"), ("v4", "emb_v4")):
        session = Session()
        known_genres(session, space=space)
        assert f"{column} IS NOT NULL" in str(session.statement)
