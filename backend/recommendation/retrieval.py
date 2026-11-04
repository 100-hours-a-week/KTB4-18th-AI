"""추천·조회 DB 연결."""

import time
from contextlib import contextmanager
from contextvars import ContextVar

from sqlalchemy import text

from db.types import Track
from backend.core.errors import _catalog_unavailable
from backend.recommendation.adapter import search_space, to_v4_query
from backend.recommendation.counts import DEFAULT_TRACK_COUNT


_db_deadline: ContextVar[float | None] = ContextVar("db_deadline", default=None)


@contextmanager
def db_time_limit(deadline: float):
    """이 범위의 SQL이 턴 마감을 넘지 않게 쿼리마다 남은 시간을 다시 계산한다."""

    token = _db_deadline.set(deadline)
    try:
        yield
    finally:
        _db_deadline.reset(token)


def apply_time_limit(session) -> None:
    deadline = _db_deadline.get()
    if deadline is None:
        return
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError("turn deadline passed")
    session.execute(text("select set_config('statement_timeout', :limit, true)"),
                    {"limit": str(max(1, round(left * 1000)))})


def vector_recommendation(
    qvec: list[float],
    exclude_ids: set[int] = None,
    genres: list[str] = None,
    min_year: int = None,
    max_year: int = None,
    k: int = DEFAULT_TRACK_COUNT,
) -> list[Track]:
    """pgvector 검색 결과의 추천곡을 가져온다. SEARCH_EMBEDDING=v4면 검색어를 v4 공간으로 바꿔 찾는다."""

    try:
        # NOTE: main.py의 load_dotenv()보다 먼저 실행되면 db.models가 읽는
        # DATABASE_URL이 아직 없을 수 있어, 요청 처리 시점까지 import를 늦춘다.
        from db.models import SessionLocal
        from db.search import search as vector_search

        space = search_space()
        if space == "v4":
            qvec = to_v4_query(qvec)

        with SessionLocal() as session:
            apply_time_limit(session)
            tracks = vector_search(
                session, qvec, exclude_ids=exclude_ids, genres=genres,
                min_year=min_year, max_year=max_year, space=space, k=k,
            )
    except Exception as error:
        raise _catalog_unavailable() from error

    return tracks


def lookup_tracks_db(
    song_title: str = None, artist: str = None, exclude_ids: set[int] = None,
    limit: int = DEFAULT_TRACK_COUNT, seed_artist: str = None,
) -> list:
    """제목/아티스트로 DB에서 곡을 조회한다."""

    try:
        from db.models import SessionLocal
        from db.search import lookup as db_lookup

        with SessionLocal() as session:
            apply_time_limit(session)
            return db_lookup(session, song_title=song_title, artist=artist, exclude_ids=exclude_ids, limit=limit,
                             seed_artist=seed_artist)
    except Exception as error:
        raise _catalog_unavailable() from error


def resolve_title_db(song_title: str, artist: str):
    """가수가 정해진 제목 오타를 실제 카탈로그 후보로 해석한다."""

    try:
        from db.models import SessionLocal
        from db.search import resolve_title

        with SessionLocal() as session:
            apply_time_limit(session)
            return resolve_title(session, song_title, artist)
    except Exception as error:
        raise _catalog_unavailable() from error


def seed_artists_db() -> list[str]:
    """고속 모드 Jev가 고를 seed 아티스트 목록을 가져온다."""

    try:
        from db.models import SessionLocal
        from db.search import seed_artists

        with SessionLocal() as session:
            return seed_artists(session)
    except Exception as error:
        raise _catalog_unavailable() from error


def seed_artist_titles_db(seed_artist: str) -> list[str]:
    """seed 아티스트의 곡 제목 목록을 가져온다."""

    try:
        from db.models import SessionLocal
        from db.search import seed_artist_titles

        with SessionLocal() as session:
            return seed_artist_titles(session, seed_artist)
    except Exception as error:
        raise _catalog_unavailable() from error
