"""추천·조회 DB 연결."""

from db.types import Track
from backend.core.errors import _catalog_unavailable


def vector_recommendation(
    qvec: list[float],
    message: str,
    exclude_ids: set[int] = None,
    genres: list[str] = None,
    min_year: int = None,
) -> tuple[list[Track], dict[str, dict]]:
    """pgvector 검색 결과의 추천곡과 무드 태그를 가져온다."""

    try:
        # NOTE: main.py의 load_dotenv()보다 먼저 실행되면 db.models가 읽는
        # DATABASE_URL이 아직 없을 수 있어, 요청 처리 시점까지 import를 늦춘다.
        from db.models import SessionLocal
        from db.search import search as vector_search

        with SessionLocal() as session:
            tracks, mood_tags_by_id = vector_search(
                session, qvec, message, exclude_ids=exclude_ids, genres=genres, min_year=min_year,
            )
    except Exception as error:
        raise _catalog_unavailable() from error

    return tracks, mood_tags_by_id


def lookup_tracks_db(song_title: str = None, artist: str = None, exclude_ids: set[int] = None) -> list:
    """제목/아티스트로 DB에서 곡을 조회한다."""

    try:
        from db.models import SessionLocal
        from db.search import lookup as db_lookup

        with SessionLocal() as session:
            return db_lookup(session, song_title=song_title, artist=artist, exclude_ids=exclude_ids)
    except Exception as error:
        raise _catalog_unavailable() from error
