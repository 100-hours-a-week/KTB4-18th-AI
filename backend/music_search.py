"""PostgreSQL·pgvector 음악 검색 연결 지점."""

from fastapi import HTTPException

from backend.schemas import Track


def search_tracks(message: str) -> tuple[str, list[Track]]:
    """DB 검색 연결 전에는 빈 검색 결과 대신 사용 불가 오류를 반환한다.

    연결 후 음악 검색 문맥과 검증된 추천곡을 반환한다.
    """

    # TODO: DB 스키마와 임베딩 규격 확인 후 음악 문맥 생성 → 텍스트 임베딩 → 벡터 검색을 연결한다.
    raise HTTPException(
        status_code=503,
        detail={
            "code": "SERVICE_UNAVAILABLE",
            "message": "음악 DB 검색 연결을 준비 중입니다.",
            "details": {"reason": "MUSIC_CATALOG_UNAVAILABLE"},
        },
    )
