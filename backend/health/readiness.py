"""배포 상태 확인용 PostgreSQL 연결·추천 데이터 점검."""

import os

import psycopg
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError


def database_is_ready() -> bool:
    """추천에 필요한 tracks 테이블에 곡이 하나 이상 있는지 확인한다."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return False

    try:
        # NOTE: SQLAlchemy의 +psycopg 접두사를 libpq가 받는 PostgreSQL URL로 변환한다.
        url = make_url(database_url)
        if url.get_backend_name() != "postgresql":
            return False
        conninfo = url.set(drivername="postgresql").render_as_string(hide_password=False)
        # NOTE: 추천 요청의 커넥션 풀과 분리하고 연결·쿼리 대기를 제한한다.
        # 이 연결은 상태 확인 직후 닫으며 API 키나 모델 호출을 사용하지 않는다.
        with psycopg.connect(
            conninfo,
            connect_timeout=2,
            options="-c statement_timeout=1000 -c default_transaction_read_only=on",
            autocommit=True,
        ) as connection:
            row = connection.execute(
                "SELECT EXISTS (SELECT 1 FROM tracks "
                "WHERE emb_gemini IS NOT NULL AND store_url IS NOT NULL)"
            ).fetchone()
            return bool(row and row[0])
    except (psycopg.Error, ArgumentError, ValueError):
        # 접속 정보와 DB 오류 상세는 외부 응답에 노출하지 않는다.
        return False
