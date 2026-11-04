from types import SimpleNamespace

import pytest

from backend.recommendation import retrieval


class Session:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


@pytest.mark.parametrize("space", ["gemini", "v4"])
@pytest.mark.parametrize("count", [1, 2, 5])
def test_recommendation_forwards_count_to_database(monkeypatch, space, count):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://test:test@localhost/test")
    from db import models, search

    monkeypatch.setattr(models, "SessionLocal", Session)
    monkeypatch.setattr(retrieval, "search_space", lambda: space)
    monkeypatch.setattr(retrieval, "to_v4_query", lambda _vector: [0.5, 1.0])
    calls = []
    monkeypatch.setattr(search, "search", lambda session, vector, **kwargs: calls.append((vector, kwargs)) or [])
    assert retrieval.vector_recommendation([0.5], k=count, genres=["Jazz"], min_year=1990, max_year=1999) == []
    vector, args = calls[0]
    assert vector == ([0.5, 1.0] if space == "v4" else [0.5])
    assert args == {"k": count, "space": space, "genres": ["Jazz"], "min_year": 1990,
                    "max_year": 1999, "exclude_ids": None}


def test_lookup_forwards_count_to_database(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://test:test@localhost/test")
    from db import models, search

    monkeypatch.setattr(models, "SessionLocal", Session)
    calls = []
    row = SimpleNamespace(track_id=1)
    monkeypatch.setattr(search, "lookup", lambda session, **kwargs: calls.append(kwargs) or [row])
    assert retrieval.lookup_tracks_db(artist="IU", limit=5, exclude_ids={2}) == [row]
    assert calls == [{"song_title": None, "artist": "IU", "limit": 5, "exclude_ids": {2}, "seed_artist": None}]


def test_db_time_limit_uses_time_left_at_each_query(monkeypatch):
    from types import SimpleNamespace

    applied = []
    session = SimpleNamespace(execute=lambda stmt, params: applied.append(params["limit"]))
    clock = iter([100.0, 102.0])
    monkeypatch.setattr(retrieval.time, "monotonic", lambda: next(clock))
    with retrieval.db_time_limit(103.0):
        retrieval.apply_time_limit(session)
        retrieval.apply_time_limit(session)
    retrieval.apply_time_limit(session)
    assert applied == ["3000", "1000"]


def test_expired_db_time_limit_starts_no_query(monkeypatch):
    from types import SimpleNamespace

    session = SimpleNamespace(execute=lambda *a: pytest.fail("query started"))
    monkeypatch.setattr(retrieval.time, "monotonic", lambda: 10.0)
    with retrieval.db_time_limit(10.0), pytest.raises(TimeoutError):
        retrieval.apply_time_limit(session)
