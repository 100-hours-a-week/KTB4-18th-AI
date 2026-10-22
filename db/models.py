import os
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import (
    JSON, BigInteger, Date, DateTime, String, Text, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, deferred, mapped_column, sessionmaker

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

DATABASE_URL = os.environ["DATABASE_URL"]

EMB_DIM = 512
MODEL_VERSION = "laion-clap-htsat-fused"

GEMINI_EMB_DIM = 3072
GEMINI_MODEL_VERSION = "gemini-embedding-2"

# NOTE: v4 검색 공간 = [s_mix (3072), −β·r(s) (1)]. backend/recommendation/adapter.py 참고
V4_EMB_DIM = GEMINI_EMB_DIM + 1

from pgvector.sqlalchemy import HALFVEC, Vector
VectorType = Vector(EMB_DIM)

class Base(DeclarativeBase):
    """이 안에 들어오면 전부 DB로 만들어주겠다 + 이런 방식으로 쓰면 타입 힌트랑(아래의 Mapped) 자동 완성 잘 된다"""
    pass

class TrackRow(Base):
    """Table의 Column을 정의"""
    __tablename__ = "tracks"

    track_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    artist: Mapped[str] = mapped_column(Text)
    album: Mapped[str] = mapped_column(Text, nullable=True)
    genre: Mapped[str] = mapped_column(String(64), nullable=True)
    release_date: Mapped[date] = mapped_column(Date, nullable=True)

    preview_url: Mapped[str] = mapped_column(Text, nullable=True)
    artwork_url: Mapped[str] = mapped_column(Text, nullable=True)
    store_url: Mapped[str] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[int] = mapped_column(nullable=True)

    seed_artist: Mapped[str] = mapped_column(Text, nullable=True)
    bucket: Mapped[str] = mapped_column(String(32), nullable=True)

    # NOTE: mood head가 뽑은 태그 상위 N개. 필터와 추천 이유 생성에 사용
    mood_tags: Mapped[dict] = mapped_column(JSON, nullable=True)

    # NOTE: NULL = 아직 임베딩되지 않음 = 야간 배치 대상. 곧 이것이 DB에 없는 곡 queue 역할
    emb_clap = mapped_column(VectorType, nullable=True)

    # NOTE: gemini-embedding-2 오디오·텍스트 동일 공간 임베딩. emb_clap과 같은 NULL=대기열 패턴
    emb_gemini = mapped_column(Vector(GEMINI_EMB_DIM), nullable=True)

    # NOTE: 곡 어댑터 + 허브 보정을 미리 계산한 v4 검색용 벡터 (scripts/backfill_emb_v4.py가 채운다).
    # deferred: select(TrackRow)가 이 칸을 읽지 않아서, 칸을 추가하기 전의 DB에서도 기존 검색이 그대로 돈다.
    emb_v4 = deferred(mapped_column(Vector(V4_EMB_DIM), nullable=True))

    # NOTE: 임베딩 모델이 다르면 전부 다시 해야 하므로 추가
    model_version: Mapped[str] = mapped_column(String(128), nullable=True)
    embedded_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    # NOTE: 임베딩 실패 이력. 반복 실패하는 곡을 거를 수 있음
    fail_count: Mapped[int] = mapped_column(default=0)
    fail_reason: Mapped[str] = mapped_column(Text, nullable=True)

class V4RefQuery(Base):
    """v4 허브 점수 r(s)의 기준 = 학습에 쓴 train 검색어 17,802개 (v4 검색어 공간으로 바꾼 벡터).

    크론잡(batch/embed.py)이 새 곡의 emb_v4를 계산할 때 전부 읽는다. 서버 검색은 읽지 않는다.
    NOTE: 110MB라 git 대신 DB에 둔다. halfvec(fp16)이어도 추천 순위는 float32와 같다 (2026-10-20 확인)
    """

    __tablename__ = "v4_ref_queries"

    id: Mapped[int] = mapped_column(primary_key=True)
    emb = mapped_column(HALFVEC(GEMINI_EMB_DIM), nullable=False)


class CorpusStat(Base):
    """임베딩 벡터값을 평균을 뺀 수치로 하면 결과 다양성 증가. 같은 PostgreSQL에 저장"""

    __tablename__ = "corpus_stats"

    key: Mapped[str] = mapped_column(String(32), primary_key=True)
    value = mapped_column(VectorType)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)



def make_engine(url: str = None, echo: bool = False):
    """커넥션 풀 생성: 연결을 미리 만들어 놓고, 필요할 때 쓰면 반납한다"""
    return create_engine(url or DATABASE_URL, echo=echo, pool_pre_ping=True)

engine = make_engine()
# NOTE: DB 세션 만들어 준다. 커밋 후 객체 속성 재접근 시 이미 실행했던 쿼리 재시도 방지
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

from sqlalchemy import text

def init_db(engine_=None):
    """테이블 생성. 커넥션 당 pgvector 확장도 확인한다."""
    # NOTE: Short-circuit-evaluation을 통한 커넥션 풀 지정 혹은 생성
    eng = engine_ or engine

    # NOTE: begin으로 커넥션 풀에서 커넥션 하나만 가져오자. 그 커넥션에 pgvector 없으면 추가해라
    with eng.begin() as con:
        con.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    # NOTE: DB에 TrackRow에 저장된(위에 Base로 정의해서 가능) 칼럼 없으면 다 채워넣는다
    Base.metadata.create_all(eng)
