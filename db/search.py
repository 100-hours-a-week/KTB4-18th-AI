import re
from dataclasses import dataclass
from datetime import date

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.schemas import Track
from .models import TrackRow, l2norm


@dataclass
class Hit:
    track: TrackRow
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
    sound_description: str,
    k: int = 5,
    exclude_ids: set[int] = None,
    min_year: int = None,
    genres: list[str] = None,
    overfetch: int = 6,
) -> tuple[list[Track], dict[str, dict]]:
    """sqlalchemy를 이용해 임베딩이 들어있는 DB에서 유사도가 높은 순서로 API Track을 가져옴.
    두 번째 반환값은 Effnet을 통해 구한 track_id별 mood_tags다. 호출 측이 곡별ㅜ추천 이유를 생성할 때 근거로 쓴다.
    """

    # NOTE: overfetch 방식은 일리는 있으나, 더 많이 가져오는데 문제가 있으면 수정해도 괜찮다
    qvec = l2norm(np.asarray(qvec, dtype=np.float32))
    n = k * overfetch
    exclude_ids = exclude_ids or set()

    dist = TrackRow.emb_gemini.cosine_distance(qvec)
    stmt = (
        select(TrackRow, dist.label("dist"))
        # NOTE: store_url은 backfill 배치가 채우기 전까지 비어 있을 수 있다.
        # 스키마상으로는 선택 필드이지만, 구매 링크 없는 곡을 추천하지 않도록
        # 채워질 때까지 검색 대상에서 뺀다.
        # emb_gemini가 NULL인 곡(아직 gemini로 임베딩 안 된 대기열)은 검색 대상에서 뺀다.
        .where(TrackRow.emb_gemini.isnot(None), TrackRow.store_url.isnot(None))
        .order_by(dist)
        .limit(n)
    )
    if exclude_ids:
        stmt = stmt.where(TrackRow.track_id.notin_(exclude_ids))
    if min_year:
        stmt = stmt.where(TrackRow.release_date >= date(min_year, 1, 1))
    if genres:
        stmt = stmt.where(TrackRow.genre.in_(genres))
    rows = session.execute(stmt).all()

    # NOTE: cosine distance랑 similarity가 반대되는 개념임을 인지
    hits = [Hit(track=r[0], score=1.0 - float(r[1])) for r in rows]
    hits = _dedup(hits, k)

    # NOTE: LLM 호출이 실패용 이유 공통 폴백 문구다.
    reason = f"'{sound_description}' 상황에 어울리는 곡입니다."
    tracks = [
        Track(
            track_id=str(hit.track.track_id),
            title=hit.track.title,
            artist=hit.track.artist,
            artwork_url=hit.track.artwork_url,
            preview_url=hit.track.preview_url,
            store_url=hit.track.store_url,
            reason=reason,
        )
        for hit in hits
    ]
    mood_tags_by_id = {str(hit.track.track_id): hit.track.mood_tags or {} for hit in hits}
    return tracks, mood_tags_by_id


def lookup(
    session: Session,
    song_title: str = None,
    artist: str = None,
    limit: int = 5,
    exclude_ids: set[int] = None,
) -> list[TrackRow]:
    """제목/아티스트 텍스트로 곡을 조회한다."""

    stmt = select(TrackRow)
    if song_title:
        stmt = stmt.where(TrackRow.title.ilike(f"%{song_title}%"))
    if artist:
        # NOTE: ilike(%artist%)는 짧은 아티스트명(예: "IU")이 "XIUMIN", "genius" 같은
        # 무관한 단어 안에 우연히 포함돼 걸리는 문제가 있어, 단어 경계(\y) 정규식으로 매칭한다.
        stmt = stmt.where(TrackRow.artist.op("~*")(rf"\y{re.escape(artist)}\y"))
    if exclude_ids:
        stmt = stmt.where(TrackRow.track_id.notin_(exclude_ids))
    stmt = stmt.order_by(TrackRow.release_date.desc()).limit(limit * 6)
    rows = session.execute(stmt).scalars().all()

    seen, out = set(), []
    for row in rows:
        key = (row.artist.lower(), row.title.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if len(out) >= limit:
            break
    return out


def known_genres(session: Session) -> list[str]:
    """DB에 실제 존재하는 장르 문자열 목록(중복 제거, 정렬). classify 프롬프트가
    장르를 예시로 참고할 때 쓴다 — 카탈로그가 늘어나면 자동으로 반영된다."""

    rows = session.execute(
        select(TrackRow.genre).distinct().where(TrackRow.genre.isnot(None))
    ).scalars().all()
    return sorted(rows)
