"""
주간 수집 배치 — KR 인기 차트 곡을 카탈로그에 추가하고 임베딩한다

신곡 전체가 아니라 "사람들이 모를 수 없는" 곡을 노린다. Apple KR
most-played 차트(신곡 + 스테디셀러)가 출처이고, 차트에서 처음 본
아티스트는 대표곡 몇 곡을 함께 가져온다.

흐름
  1. KR 차트 받기               (Apple RSS, iTunes ID 포함 → 매칭 불필요)
  2. DB에 없는 곡만 남기기       (track_id + (artist, title) 키 중복 제거)
  3. lookup으로 메타데이터       (US 스토어. 기존 DB와 이름 표기를 맞춘다)
  4. 처음 본 아티스트 확장        (검색 관련도순 = 대표곡 근사, 커버 필터 적용)
     노래가 아닌 트랙(DJ 믹스, 짧은 인트로 등)은 차트 곡 포함 전부 거른다
  5. 메타데이터 INSERT           (emb_gemini = NULL)
  6. emb_gemini IS NULL 임베딩   (실패해도 다음 실행에서 재시도)

5와 6을 나눈 이유: 실패하는 단계는 임베딩이다. 둘을 묶으면 임베딩에
실패한 곡은 DB에 남지 않고, 다음 주 차트에서 빠지면 다시 시도되지 않는다.
검색은 emb_gemini IS NOT NULL만 보므로 벡터 없는 행은 추천에 섞이지 않는다.

임베딩은 emb_test/openrouter_collect.py와 같은 입력·요청으로 만든다.
(30초 · 모노 · 24kHz · 64kbps mp3 → OpenRouter, 원본 벡터 저장)
입력 조건이 다르면 기존 벡터와 같은 공간이라는 보장이 없다.

사용:
  python -m batch.weekly --dry-run       # 추가될 곡만 확인 (쓰기 없음)
  python -m batch.weekly --limit 5       # 임베딩만 5곡. INSERT는 전부 한다
  python -m batch.weekly

운영: AI EC2(고정 1대, ASG 미적용)의 crontab에서 주 1회 실행한다.
서버 코드는 이 모듈을 import하지 않으므로 crontab 등록 전까지는 아무 일도 하지 않는다.
  # 한국 시간 월요일 06시 = UTC 일요일 21시
  0 21 * * 0  cd <compose 위치> && /usr/bin/flock -n /tmp/weekly.lock \\
    /usr/bin/docker compose run --rm --no-deps -T <ai 서비스> \\
    python -m batch.weekly >> /var/log/weekly.log 2>&1
  - flock -n: 이전 실행이 아직 돌고 있으면 건너뛴다 (인스턴스가 1대라 이걸로 충분)
  - -T: cron에는 TTY가 없다. 빠뜨리면 "the input device is not a TTY"로 실패한다
  - 필요한 환경변수: DATABASE_URL, OPENROUTER_EMBEDDING_API_KEY (서버와 동일)
등록 전 검증 순서: 서버에서 --dry-run 수동 실행 → --limit 1 → crontab을 5분 뒤로
임시 등록해 로그 확인 → 주간 스케줄로 변경.
첫 실행은 새 아티스트가 많아 약 200곡(임베딩 약 $1), 이후엔 차트 변동분만 들어온다.
"""
import argparse
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from urllib.parse import quote

import numpy as np
from sqlalchemy import func, select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.models import GEMINI_EMB_DIM, SessionLocal, TrackRow  # noqa: E402

CHART_URL = ("https://rss.marketingtools.apple.com/api/v2/kr/music/"
             "most-played/100/songs.json")
LOOKUP_URL = "https://itunes.apple.com/lookup"
SEARCH_URL = "https://itunes.apple.com/search"
EMBED_URL = "https://openrouter.ai/api/v1/embeddings"
EMBED_MODEL = "google/gemini-embedding-2"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"

ITUNES_DELAY = 4.0    # 분당 15콜. 사용자 곡 검색과 같은 출구 IP를 나눠 쓴다
ARTIST_TOP = 10       # 처음 본 아티스트당 가져올 대표곡 수
EMBED_LIMIT = 200     # 한 번에 임베딩할 최대 곡 수. 곡당 약 $0.005
MAX_FAIL = 3          # 이 횟수 이상 실패한 곡은 더 시도하지 않는다
STOP_STREAK = 10      # 연속 실패 시 중단
EMBED_SLEEP = 0.1
AUDIO_SEC = 30

# 이 배치가 넣은 곡만 임베딩한다. 기존 15,000곡의 대기분은 로컬
# openrouter_collect.py가 처리 중이라, 겹치면 같은 곡에 두 번 과금된다.
BUCKET_CHART = "chart_kr"
BUCKET_ARTIST = "chart_artist"
OWN_BUCKETS = (BUCKET_CHART, BUCKET_ARTIST)

# 다른 버전 필터. emb_test/collect.py의 EXCLUDE_TITLE과 같은 목록이다.
# 아티스트 확장 곡에만 적용한다. 차트 곡은 실제로 많이 듣는 곡이라 커버일
# 일이 거의 없고, 걸면 'Deluxe Edition' 같은 원곡이 잘못 빠질 위험만 있다.
# collect.py는 부분 문자열로 검사해 'Alive'의 live까지 걸렀다. 여기선 단어 경계(\b).
# remastered는 '(Remastered 2011)'이 붙으면 (artist, title) 중복 제거를 빠져나가서 뺀다.
COVER_RE = re.compile(
    r"\b(instrumental|inst|remix|live|acoustic|ver|version|japanese|jp|"
    r"sped up|slowed|remastered|karaoke|reprise|edit)\b", re.IGNORECASE)

# 노래가 아닌 트랙 필터. 차트 곡 포함 전부에 적용하고, 제목·앨범명을 모두 본다.
# collect.py 목록에 없어 기존 DB에 DJ 믹스 조각과 짧은 인터루드가 섞여 들어왔다.
# 아래 단어들은 K-pop에서 정식 곡 제목으로도 쓰여서 단순 매칭하면 오탐이 난다.
# (ENHYPEN 'Mixed Up', 세븐틴·윤하 'Highlight', 위너 'Teaser', 엔플라잉 'Preview',
#  앨범 "Fe's 10th : Preview - EP") 규칙은 실제 DB의 duration_ms로 검증했다.
I = re.IGNORECASE
# 1. 항상 제외: 괄호로 붙은 Mixed(DJ 믹스 버전), DJ 믹스 앨범, 메들리
MIXED_RE = re.compile(r"[(\[]\s*mixed\s*[)\]]", I)
MIX_ALBUM_RE = re.compile(r"\b(dj mix|mixed by)\b", I)
MEDLEY_RE = re.compile(r"\bmedley\b", I)
# 2. 괄호 안에 있을 때만 제외
BRACKET_RE = re.compile(r"[(\[][^)\]]*\b(highlight|preview|teaser)\b[^)\]]*[)\]]", I)
# 3. 짧을 때만 제외. 99곡 중 55곡이 90초 이상의 정식 곡이었다
#    (지민 'Rebirth (Intro)' 144s, BTS 'Outro: Tear' 284s, Slom 'SKIT (feat. LeeHi & Loco)' 151s)
SHORT_RE = re.compile(r"\b(intro|interlude|outro|skit)\b", I)
SHORT_MS = 90_000


class Fatal(Exception):
    """재시도해도 소용없는 오류. 즉시 중단한다."""


# ── iTunes ──────────────────────────────────────────────

RETRIES = 5


def get_json(url: str) -> dict:
    """Apple 쪽 일시 장애는 기다렸다 재시도한다.

    차트 피드는 Apple 게이트웨이의 30초 한도에 걸려 502/504가 몇 분씩 이어지다
    1초대로 돌아오곤 한다. curl도 똑같이 실패해 클라이언트 문제가 아니다 (2026-10 실측).
    무인 주간 배치라 대기를 30초 → 60초 → … 로 늘려 몇 분까지 버틴다.
    """
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read().decode("utf-8", "replace")
            return json.loads(body) if body.strip() else {}
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == RETRIES:
                raise
            reason = str(e.code)
        except (TimeoutError, urllib.error.URLError) as e:
            if attempt == RETRIES:
                raise
            reason = type(e).__name__
        print(f"    {reason}. {30 * attempt}초 대기 ({attempt}/{RETRIES - 1})", flush=True)
        time.sleep(30 * attempt)
    return {}


def fetch_chart() -> list[int]:
    results = get_json(CHART_URL).get("feed", {}).get("results", [])
    return [int(r["id"]) for r in results]


def tracks_of(d: dict) -> list[dict]:
    """lookup 응답에서 곡만 남긴다. 아티스트·앨범 래퍼 행이 섞여 온다."""
    return [r for r in d.get("results", [])
            if r.get("wrapperType") == "track" and r.get("kind") == "song"]


def lookup_tracks(ids: list[int]) -> list[dict]:
    """country 미지정 = US 스토어. 기존 DB가 US 표기라 중복 판정이 맞는다."""
    if not ids:
        return []
    return tracks_of(get_json(
        f"{LOOKUP_URL}?id={','.join(map(str, ids))}&entity=song"))


def artist_top(artist_id: int, name: str) -> list[dict]:
    """아티스트 이름 검색 결과 상위 곡. 관련도순이 대표곡 순서와 거의 같다.

    lookup(sort=recent)는 최근 앨범 수록곡으로 채워져 대표곡이 빠진다.
    (ILLIT: 최근순이면 2026 수록곡, 검색순이면 Magnetic·jellyous·Tick-Tack)
    이름 검색이라 동명 아티스트·커버가 섞이므로 artistId가 같은 곡만 남긴다.
    그러다 보니 흔한 이름(BESTie, Shane)은 하나도 안 남기도 해서, 그때는
    최근 발매순으로 대신한다.
    """
    d = get_json(f"{SEARCH_URL}?term={quote(name)}&entity=song"
                 f"&attribute=artistTerm&limit=50")
    top = [r for r in tracks_of(d) if r.get("artistId") == artist_id][:ARTIST_TOP]
    if top:
        return top
    time.sleep(ITUNES_DELAY)
    return tracks_of(get_json(
        f"{LOOKUP_URL}?id={artist_id}&entity=song&sort=recent&limit={ARTIST_TOP}"))


# ── 필터 · 변환 ──────────────────────────────────────────

def dedup_key(artist: str, title: str) -> tuple[str, str]:
    return (artist or "").lower(), (title or "").lower()


def is_cover(r: dict) -> bool:
    """앨범명 + 곡명의 괄호·대시 뒤 부분만 본다. 'Live Your Life' 같은 제목 자체는 통과."""
    title = r.get("trackName") or ""
    extras = re.findall(r"[(\[]([^)\]]*)[)\]]", title)
    if " - " in title:
        extras.append(title.split(" - ", 1)[1])
    return bool(COVER_RE.search(" ".join([r.get("collectionName") or "", *extras])))


def is_junk(r: dict) -> bool:
    """노래가 아닌 트랙. 'SKIT'처럼 제목 전체가 키워드인 경우가 있어 제목 전체를 본다."""
    text = f"{r.get('trackName') or ''} {r.get('collectionName') or ''}"
    if MIXED_RE.search(text) or MIX_ALBUM_RE.search(text) or MEDLEY_RE.search(text):
        return True
    if BRACKET_RE.search(text):
        return True
    return bool(SHORT_RE.search(text)) and (r.get("trackTimeMillis") or 0) < SHORT_MS


def to_row(r: dict, bucket: str) -> dict:
    released = r.get("releaseDate")
    return {
        "track_id": r["trackId"],
        "title": r.get("trackName"),
        "artist": r.get("artistName"),
        "album": r.get("collectionName"),
        "genre": r.get("primaryGenreName"),
        "release_date": date.fromisoformat(released[:10]) if released else None,
        "preview_url": r.get("previewUrl"),
        "artwork_url": r.get("artworkUrl100"),
        "store_url": r.get("trackViewUrl"),
        "duration_ms": r.get("trackTimeMillis"),
        "bucket": bucket,
        "fail_count": 0,
    }


# ── 1~4. 새 곡 모으기 ─────────────────────────────────────

def collect_new(s) -> list[dict]:
    known_ids = set(s.scalars(select(TrackRow.track_id)))
    known_keys = {dedup_key(a, t) for a, t in
                  s.execute(select(TrackRow.artist, TrackRow.title))}
    known_artists = {a for a, _ in known_keys}

    chart_ids = fetch_chart()
    print(f"차트 {len(chart_ids)}곡")
    new_ids = [i for i in chart_ids if i not in known_ids]
    time.sleep(ITUNES_DELAY)
    chart = [r for r in lookup_tracks(new_ids)
             if r.get("previewUrl") and not is_junk(r)]
    print(f"  DB에 없는 ID {len(new_ids)}곡 → US 스토어 조회 {len(chart)}곡")

    candidates = [(r, BUCKET_CHART) for r in chart]

    # 차트 곡 기준으로 처음 본 아티스트. 이름으로 판정한다(tracks에 artist_id 없음).
    new_artists = {r["artistId"]: r["artistName"] for r in chart
                   if r["artistName"].lower() not in known_artists}
    for artist_id, name in new_artists.items():
        time.sleep(ITUNES_DELAY)
        top = [r for r in artist_top(artist_id, name)
               if r.get("previewUrl") and not is_cover(r) and not is_junk(r)]
        print(f"  새 아티스트 {name}: 대표곡 {len(top)}곡")
        candidates += [(r, BUCKET_ARTIST) for r in top]

    # 같은 (artist, title)이면 가장 낮은 track_id 하나만. DB에 이미 있으면 제외.
    candidates.sort(key=lambda c: c[0]["trackId"])
    rows, seen = [], set(known_keys)
    for r, bucket in candidates:
        key = dedup_key(r.get("artistName"), r.get("trackName"))
        if r["trackId"] in known_ids or key in seen:
            continue
        seen.add(key)
        rows.append(to_row(r, bucket))
    return rows


# ── 6. 임베딩 ───────────────────────────────────────────

def to_mp3_b64(url: str, exe: str) -> str:
    """openrouter_collect.py와 동일한 변환. 입력이 다르면 벡터 공간이 어긋난다."""
    with tempfile.TemporaryDirectory() as tmp:
        m4a, mp3 = Path(tmp) / "a.m4a", Path(tmp) / "a.mp3"
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as r, open(m4a, "wb") as f:
            f.write(r.read())
        subprocess.run(
            [exe, "-y", "-loglevel", "error", "-i", str(m4a),
             "-t", str(AUDIO_SEC), "-ac", "1", "-ar", "24000",
             "-b:a", "64k", str(mp3)], check=True)
        return base64.b64encode(mp3.read_bytes()).decode()


def embed_audio(url: str, exe: str, key: str) -> list[float]:
    data_url = f"data:audio/mpeg;base64,{to_mp3_b64(url, exe)}"
    body = json.dumps({
        "model": EMBED_MODEL,
        "input": [{"content": [{"type": "image_url", "image_url": {"url": data_url}}]}],
        "encoding_format": "float",
        # 지정하지 않으면 10% 비싼 Vertex eu로 라우팅된다 (2026-10 실측)
        "provider": {"ignore": ["google-vertex/eu"]},
    }).encode()
    req = urllib.request.Request(
        EMBED_URL, data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    for attempt in (1, 2, 3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.loads(r.read().decode())
            break
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:300]
            if e.code in (401, 402, 403):
                raise Fatal(f"HTTP {e.code}: {msg}") from None
            if e.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(15 * attempt)
                continue
            raise RuntimeError(f"HTTP {e.code}: {msg}") from None

    v = np.asarray(d["data"][0]["embedding"], dtype=np.float32)
    if v.shape[0] != GEMINI_EMB_DIM:
        raise Fatal(f"차원이 {v.shape[0]}. 기존 벡터와 섞을 수 없다")
    return v.tolist()


def embed_pending(s, limit: int) -> None:
    key = os.environ.get("OPENROUTER_EMBEDDING_API_KEY", "").strip()
    if not key or key.startswith("<"):
        print("OPENROUTER_EMBEDDING_API_KEY가 없어 임베딩을 건너뜁니다.")
        return

    targets = list(s.scalars(
        select(TrackRow)
        .where(TrackRow.emb_gemini.is_(None), TrackRow.fail_count < MAX_FAIL,
               TrackRow.bucket.in_(OWN_BUCKETS), TrackRow.preview_url.isnot(None))
        .order_by(TrackRow.track_id).limit(limit)))
    print(f"\n임베딩 대상 {len(targets)}곡")
    if not targets:
        return

    from imageio_ffmpeg import get_ffmpeg_exe
    exe = get_ffmpeg_exe()
    ok = fail = streak = 0

    for t in targets:
        try:
            t.emb_gemini = embed_audio(t.preview_url, exe, key)
            t.fail_reason = None
            ok += 1
            streak = 0
        except Fatal as e:
            s.commit()
            print(f"  치명적 오류. 중단합니다: {e}")
            break
        except Exception as e:
            t.fail_count = (t.fail_count or 0) + 1
            t.fail_reason = f"{type(e).__name__}: {e}"[:300]
            fail += 1
            streak += 1
            if streak >= STOP_STREAK:
                s.commit()
                print(f"  연속 {streak}회 실패. 중단합니다. 마지막 오류: {e}")
                break
        s.commit()    # 곡당 커밋. 도중에 죽어도 다음 실행이 이어서 한다
        time.sleep(EMBED_SLEEP)

    print(f"  성공 {ok} / 실패 {fail}")


# ── 실행 ───────────────────────────────────────────────

def run(limit: int = EMBED_LIMIT, dry_run: bool = False) -> None:
    with SessionLocal() as s:
        rows = collect_new(s)
        print(f"\n추가할 곡 {len(rows)}곡")
        for r in rows[:20]:
            print(f"  [{r['bucket']}] {r['track_id']}  {r['artist']} - {r['title']}")
        if dry_run:
            return

        if rows:
            s.bulk_insert_mappings(TrackRow, rows)
            s.commit()

        embed_pending(s, limit)

        total, done = s.execute(
            select(func.count(), func.count(TrackRow.emb_gemini))
            .where(TrackRow.bucket.in_(OWN_BUCKETS))).one()
        print(f"\n주간 배치 곡 {total}곡 / 임베딩 완료 {done}곡")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=EMBED_LIMIT, help="이번 실행 최대 임베딩 곡 수")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    run(limit=a.limit, dry_run=a.dry_run)
