from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Track, l2norm


@dataclass
class Hit:
    track: Track
    score: float


def _dedup(hits: list[Hit], k: int) -> list[Hit]:
    """검색된 음악 중 소문자로 해서 같은 음악이 존재한다면 추천에서 제거. 아래에 overfetch가 있으니 k개 되면 stop"""
    seen, out = set(), []
    for h in hits:
        key = (h.track.artist.lower(), h.track.title.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
        if len(out) >= k:
            break
    return out

def search(
    session: Session,
    qvec: np.ndarray,
    k: int = 5,
    exclude_ids: set[int] = None,
    min_year: int = None,
    genres: list[str] = None,
    overfetch: int = 6,
) -> list[Hit]:
    """sqlalchemy를 이용해 임베딩이 들어있는 DB에서 유사도가 높은 순서를 가져옴"""
    
    # NOTE: overfetch 방식은 일리는 있으나, 더 많이 가져오는데 문제가 있으면 수정해도 괜찮다
    qvec = l2norm(np.asarray(qvec, dtype=np.float32))
    n = k * overfetch
    exclude_ids = exclude_ids or set()

    dist = Track.emb_clap.cosine_distance(qvec)
    stmt = (
        select(Track, dist.label("dist"))
        .where(Track.emb_clap.isnot(None))
        .order_by(dist)
        .limit(n)
    )
    if exclude_ids:
        stmt = stmt.where(Track.track_id.notin_(exclude_ids))
    if min_year:
        stmt = stmt.where(Track.release_date >= f"{min_year}-01-01")
    if genres:
        stmt = stmt.where(Track.genre.in_(genres))
    rows = session.execute(stmt).all()

    # NOTE: cosine distance랑 similarity가 반대되는 개념임을 인지
    hits = [Hit(track=r[0], score=1.0 - float(r[1])) for r in rows]

    return _dedup(hits, k)
