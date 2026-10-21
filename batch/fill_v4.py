"""
v4 채우기 — emb_gemini는 있는데 emb_v4가 비어 있는 곡을 v4 검색 공간으로 바꿔 저장한다. API 호출 없음

대기열은 batch/embed.py와 같은 패턴으로 `emb_v4 IS NULL` 자체다. 크론잡(batch/embed.py)이 임베딩 뒤에
부르고, 기존 곡을 한꺼번에 채울 때는 scripts/backfill_emb_v4.py가 부른다. 둘이 같은 함수를 써야
새 곡과 기존 곡의 emb_v4가 같은 계산으로 만들어진다.

계산은 backend/recommendation/adapter.to_v4_songs, r(s)의 기준 train 검색어는 v4_ref_queries 테이블.
검색은 SEARCH_EMBEDDING과 상관없이 두 칸을 다 채워 둔다 — 스위치를 언제 바꿔도 빠진 곡이 없게.
"""
import sys
import time
from pathlib import Path

import numpy as np
from sqlalchemy import func, select, update

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.recommendation.adapter import to_v4_songs  # noqa: E402
from db.models import SessionLocal, TrackRow, V4RefQuery  # noqa: E402

BATCH = 500


def load_ref_queries(session) -> np.ndarray:
    """v4_ref_queries 전체 (17,802, 3072) float32. 한 번 읽어 여러 배치에 쓴다 (약 220MB)."""

    rows = session.scalars(select(V4RefQuery.emb).order_by(V4RefQuery.id)).all()
    if not rows:
        raise RuntimeError("v4_ref_queries가 비어 있다. scripts/load_v4_ref_queries.py로 먼저 넣는다")
    return np.asarray(rows, dtype=np.float32)


def missing():
    """대기열: emb_gemini는 있는데 emb_v4가 없는 곡."""

    return select(TrackRow.track_id, TrackRow.emb_gemini).where(
        TrackRow.emb_gemini.isnot(None), TrackRow.emb_v4.is_(None))


def count_missing(session) -> int:
    return session.scalar(select(func.count()).select_from(missing().subquery()))


def fill_missing() -> int:
    """대기열을 BATCH곡씩 채우고 매번 커밋한다 → 끊겨도 다음 실행이 이어서 한다. 채운 곡 수를 돌려준다."""

    with SessionLocal() as session:
        todo = count_missing(session)
        if not todo:
            return 0
        ref = load_ref_queries(session)
    print(f"v4 채울 곡: {todo:,}곡")

    done, start = 0, time.time()
    while True:
        with SessionLocal() as session:
            rows = session.execute(missing().order_by(TrackRow.track_id).limit(BATCH)).all()
            if not rows:
                break
            emb, _ = to_v4_songs(np.stack([np.asarray(e, dtype=np.float32) for _, e in rows]), ref)
            session.execute(update(TrackRow), [
                {"track_id": track_id, "emb_v4": vec} for (track_id, _), vec in zip(rows, emb)])
            session.commit()
        done += len(rows)
        # NOTE: 쓴 곡은 emb_v4가 채워져 다음 조회에서 빠진다. 그래도 처음 센 수를 넘으면 뭔가 잘못된 것이라 멈춘다
        if done > todo:
            raise RuntimeError(f"처음 센 {todo}곡보다 많이 썼다 ({done}곡). 중단")
        print(f"  {done:,}/{todo:,}곡 ({time.time() - start:.0f}초)")
    return done
