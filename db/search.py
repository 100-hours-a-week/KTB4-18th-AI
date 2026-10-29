import re
from dataclasses import dataclass
from datetime import date

import numpy as np
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from db.types import Track
from .models import TrackRow
from .vectors import l2norm


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

def search_column(space: str):
    """검색 공간(adapter.search_space())에 해당하는 tracks 칸."""

    return TrackRow.emb_v4 if space == "v4" else TrackRow.emb_gemini


def search(
    session: Session,
    qvec: np.ndarray,
    k: int = 5,
    exclude_ids: set[int] = None,
    min_year: int = None,
    genres: list[str] = None,
    max_year: int = None,
    overfetch: int = 6,
    space: str = "gemini",
) -> list[Track]:
    """임베딩 유사도와 검색 조건으로 추천곡을 선정한다.

    space="gemini": qvec은 검색어 임베딩(3072), emb_gemini와 코사인 거리
    space="v4": qvec은 to_v4_query()의 [q′, 1](3073), emb_v4와 내적
    """

    # NOTE: overfetch 방식은 일리는 있으나, 더 많이 가져오는데 문제가 있으면 수정해도 괜찮다
    n = k * overfetch
    exclude_ids = exclude_ids or set()
    column = search_column(space)

    if space == "v4":
        # NOTE: [q′, 1]·[s_mix, −β·r(s)] = q′·s_mix − β·r(s). 코사인은 곡 벡터 길이(r(s)마다 다름)로
        # 나눠 허브 패널티를 비트므로 내적을 쓴다. pgvector <#>는 음의 내적이라 오름차순 정렬이 그대로 맞다.
        dist = column.max_inner_product(np.asarray(qvec, dtype=np.float32))
    else:
        dist = column.cosine_distance(l2norm(np.asarray(qvec, dtype=np.float32)))

    stmt = (
        select(TrackRow, dist.label("dist"))
        # NOTE: store_url은 backfill 배치가 채우기 전까지 비어 있을 수 있다.
        # 스키마상으로는 선택 필드이지만, 구매 링크 없는 곡을 추천하지 않도록
        # 채워질 때까지 검색 대상에서 뺀다.
        # 검색 칸이 NULL인 곡(아직 임베딩·v4 계산이 안 된 대기열)은 검색 대상에서 뺀다.
        .where(column.isnot(None), TrackRow.store_url.isnot(None))
        .order_by(dist)
        .limit(n)
    )
    if exclude_ids:
        stmt = stmt.where(TrackRow.track_id.notin_(exclude_ids))
    if min_year:
        stmt = stmt.where(TrackRow.release_date >= date(min_year, 1, 1))
    if max_year:
        stmt = stmt.where(TrackRow.release_date <= date(max_year, 12, 31))
    if genres:
        stmt = stmt.where(TrackRow.genre.in_(genres))
    # NOTE: cosine distance랑 similarity가 반대되는 개념임을 인지. v4는 음의 내적이라 부호만 뒤집는다
    rows = session.execute(stmt).all()
    if space == "v4":
        hits = [Hit(track=r[0], score=-float(r[1])) for r in rows]
    else:
        hits = [Hit(track=r[0], score=1.0 - float(r[1])) for r in rows]
    hits = _dedup(hits, k)

    tracks = [
        Track(
            track_id=str(hit.track.track_id),
            title=hit.track.title,
            artist=hit.track.artist,
            artwork_url=hit.track.artwork_url,
            preview_url=hit.track.preview_url,
            store_url=hit.track.store_url,
        )
        for hit in hits
    ]
    return tracks


def lookup(
    session: Session,
    song_title: str = None,
    artist: str = None,
    limit: int = 5,
    exclude_ids: set[int] = None,
    seed_artist: str = None,
) -> list[TrackRow]:
    """제목/아티스트 텍스트로 곡을 조회한다.

    seed_artist는 고속 모드에서 Jev가 seed_artists() 목록 중 고른 값이다. 실제 artist
    표기와 다를 수 있어서(seed "Baek Yerin" / 곡 "Yerin Baek") 정확히 같은 값으로 찾는다.
    """

    stmt = select(TrackRow)
    if song_title:
        stmt = stmt.where(TrackRow.title.ilike(f"%{song_title}%"))
    if seed_artist:
        stmt = stmt.where(TrackRow.seed_artist == seed_artist)
    if artist:
        # NOTE: ilike(%artist%)는 짧은 아티스트명(예: "IU")이 "XIUMIN", "genius" 같은
        # 무관한 단어 안에 우연히 포함돼 걸리는 문제가 있어, 단어 경계(\y) 정규식으로 매칭한다.
        stmt = stmt.where(TrackRow.artist.op("~*")(rf"\y{re.escape(artist)}\y"))
    if exclude_ids:
        stmt = stmt.where(TrackRow.track_id.notin_(exclude_ids))
    # NOTE: 제목은 부분 일치라 "butter"가 "butterflies" 같은 최신곡에 밀려 BTS의
    # "Butter"가 잘리는 문제가 있어, 제목이 정확히 같은 곡을 최신순보다 먼저 둔다.
    order = [TrackRow.release_date.desc()]
    if song_title:
        exact = case((func.lower(TrackRow.title) == song_title.lower(), 0), else_=1)
        order.insert(0, exact)
    stmt = stmt.order_by(*order).limit(limit * 6)
    rows = session.execute(stmt).scalars().all()

    seen, unique = set(), []
    for row in rows:
        key = (row.artist.lower(), row.title.lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)

    # NOTE: 최신순 그대로 자르면 최근 정규 앨범 하나가 결과를 다 차지해서, 앨범당
    # 1곡씩 먼저 고르고 모자라면 남은 곡을 최신순으로 채운다. 앨범이 적은
    # 아티스트도 limit만큼은 채워지도록 앨범당 상한을 두지 않았다.
    picked_albums, first, rest = set(), [], []
    for row in unique:
        album = (row.artist.lower(), (row.album or "").lower())
        if row.album and album in picked_albums:
            rest.append(row)
            continue
        picked_albums.add(album)
        first.append(row)
    chosen = {id(row) for row in first[:limit] + rest[:max(0, limit - len(first))]}
    return [row for row in unique if id(row) in chosen]


# NOTE: 장르는 검색에서 하드 필터라, 검색 가능한 곡이 1~2곡뿐인 장르("Adult
# Contemporary" 1곡)가 뽑히면 카드가 1장 이하로 나온다. 이보다 작은 장르는 필터
# 후보에서 빼고, 그 단어는 recommend_query에 남겨 임베딩 검색에 맡긴다.
MIN_GENRE_TRACKS = 10


def known_genres(session: Session, space: str = "gemini") -> list[str]:
    """검색 가능한 곡이 MIN_GENRE_TRACKS곡 이상인 장르 목록(정렬). classify 프롬프트가
    장르 필터 후보로 쓴다 — 카탈로그가 늘어나면 자동으로 반영된다.

    search()와 같은 조건(같은 검색 칸·store_url 있음)으로 세야, 목록에 있는데 검색하면
    0곡인 장르가 생기지 않는다.
    """

    rows = session.execute(
        select(TrackRow.genre)
        .where(
            TrackRow.genre.isnot(None),
            search_column(space).isnot(None),
            TrackRow.store_url.isnot(None),
        )
        .group_by(TrackRow.genre)
        .having(func.count() >= MIN_GENRE_TRACKS)
    ).scalars().all()
    return sorted(rows)


def seed_artists(session: Session) -> list[str]:
    """카탈로그를 모을 때 기준으로 삼은 아티스트 목록(정렬).

    NOTE: 피처링 표기까지 합친 실제 artist 값은 2,700개가 넘어 Jev Choice(최대 255개)에
    안 들어가지만, seed는 250명 정도라 한 번에 고를 수 있다.
    """

    rows = session.execute(
        select(TrackRow.seed_artist).where(TrackRow.seed_artist.isnot(None)).distinct()
    ).scalars().all()
    return sorted(rows)


def seed_artist_titles(session: Session, seed_artist: str) -> list[str]:
    """seed 아티스트의 곡 제목 목록(중복 제거, 정렬)."""

    rows = session.execute(
        select(TrackRow.title).where(TrackRow.seed_artist == seed_artist).distinct()
    ).scalars().all()
    return sorted(rows)
