import os
from datetime import date, datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sqlalchemy import (
    JSON, BigInteger, Date, DateTime, String, Text, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

DATABASE_URL = os.environ["DATABASE_URL"]

EMB_DIM = 512
MODEL_VERSION = "laion-clap-htsat-fused"

from pgvector.sqlalchemy import Vector
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

    # NOTE: 임베딩 모델이 다르면 전부 다시 해야 하므로 추가
    model_version: Mapped[str] = mapped_column(String(128), nullable=True)
    embedded_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    # NOTE: 임베딩 실패 이력. 반복 실패하는 곡을 거를 수 있음
    fail_count: Mapped[int] = mapped_column(default=0)
    fail_reason: Mapped[str] = mapped_column(Text, nullable=True)

class CorpusStat(Base):
    """임베딩 벡터값을 평균을 뺀 수치로 하면 결과 다양성 증가. 같은 PostgreSQL에 저장"""

    __tablename__ = "corpus_stats"

    key: Mapped[str] = mapped_column(String(32), primary_key=True)
    value = mapped_column(VectorType)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)

def decode_vector(raw) -> np.ndarray:
    """DB 저장 형식 → numpy"""
    if raw is None:
        return None
    return np.asarray(raw, dtype=np.float32)

def l2norm(v: np.ndarray) -> np.ndarray:
    """바로 코사인 유사도 계산되게 미리 정규화"""
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-10)

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