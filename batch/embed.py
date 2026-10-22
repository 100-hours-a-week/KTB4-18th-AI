"""
일간 임베딩 배치 — emb_gemini가 비어 있는 곡을 임베딩한다

대기열은 별도 테이블이 아니라 `emb_gemini IS NULL` 조건 자체다. 주간 수집
(batch/weekly.py)이든 다른 경로든, 메타데이터만 들어온 곡은 전부 여기서 처리한다.
검색은 emb_gemini IS NOT NULL만 보므로 대기 중인 곡은 추천에 섞이지 않는다.

수집과 나눈 이유: 처리 속도를 수집 주기와 따로 조절하기 위해서다. 둘을 묶으면
처리량이 "수집 1회당 N곡"에 묶여, 들어오는 곡이 그보다 많아지는 순간 대기열이
매주 늘어나기만 한다. --limit이 곧 하루 예산이고(곡당 약 $0.005, 약 2초),
끝에 찍히는 "남은 대기"가 며칠째 줄지 않으면 limit을 올리거나 수집을 줄인다.

임베딩은 emb_test/openrouter_collect.py와 같은 입력·요청으로 만든다.
(30초 · 모노 · 24kHz · 64kbps mp3 → OpenRouter, 원본 벡터 저장)
입력 조건이 다르면 기존 벡터와 같은 공간이라는 보장이 없다.

임베딩 뒤에는 2단계로 emb_v4(v4 검색 공간)도 채운다 (batch/fill_v4.py, API 호출 없음).
필요한 것: tracks.emb_v4 칸, v4_ref_queries 테이블, backend/recommendation/assets/song_adapter_v4.npz.

사용:
  python -m batch.embed --dry-run     # 대상과 남은 대기(임베딩·v4)만 확인 (API 호출 없음)
  python -m batch.embed --limit 1
  python -m batch.embed               # 기본 200곡

운영: AI EC2의 crontab에서 매일 실행한다. 주간 수집(토 21:00 UTC) 한 시간 뒤에
돌도록 맞춰, 새로 들어온 곡은 같은 날 임베딩된다.
  # 한국 시간 매일 07시 = UTC 22시
  0 22 * * *  cd <compose 위치> && /usr/bin/flock -n /tmp/embed.lock \\
    /usr/bin/docker compose run --rm --no-deps -T <ai 서비스> \\
    python -m batch.embed >> <로그 경로>/embed.log 2>&1
  - 잠금 파일은 주간 수집과 따로 쓴다. 수집과 겹쳐도 안전하지만(수집은 INSERT,
    임베딩은 커밋된 NULL 행만 UPDATE), 임베딩 두 개가 동시에 돌면 같은 곡을 두 번 처리한다
  - 필요한 환경변수: DATABASE_URL, OPENROUTER_EMBEDDING_API_KEY (서버와 동일)
  - 키가 없거나 401/402/403이면 0이 아닌 종료 코드로 끝난다. crontab의 `||`로 알림을 붙일 수 있다
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from sqlalchemy import func, select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from batch.fill_v4 import count_missing, fill_missing  # noqa: E402
from db.models import GEMINI_EMB_DIM, SessionLocal, TrackRow  # noqa: E402

EMBED_URL = "https://openrouter.ai/api/v1/embeddings"
EMBED_MODEL = "google/gemini-embedding-2"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"

EMBED_LIMIT = 200     # 하루 최대 임베딩 곡 수. 곡당 약 $0.005
MAX_FAIL = 3          # 이 횟수 이상 실패한 곡은 더 시도하지 않는다
STOP_STREAK = 10      # 연속 실패 시 중단
EMBED_SLEEP = 0.1
AUDIO_SEC = 30


class Fatal(Exception):
    """재시도해도 소용없는 오류. 즉시 중단한다."""


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


def call(body: bytes, key: str) -> dict:
    """openrouter_collect.call()과 같은 재시도 규칙.

    401/402/403(인증·결제)은 다시 해도 소용없어 즉시 중단한다.
    429·5xx는 15초, 30초 기다리고, 타임아웃·연결 끊김 같은 네트워크 오류는
    5초, 10초 기다렸다가 다시 시도한다.
    """
    req = urllib.request.Request(
        EMBED_URL, data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    for attempt in (1, 2, 3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:300]
            if e.code in (401, 402, 403):
                raise Fatal(f"HTTP {e.code}: {msg}") from None
            if e.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(15 * attempt)
                continue
            raise RuntimeError(f"HTTP {e.code}: {msg}") from None
        except Exception:
            if attempt < 3:
                time.sleep(5 * attempt)
                continue
            raise
    raise RuntimeError("재시도 초과")


def embed_audio(url: str, exe: str, key: str) -> list[float]:
    """openrouter_collect.embed()와 같은 요청. 단, 입력 형식 폴백은 뺐다.

    원본은 image_url이 400으로 막히면 input_audio로 바꿔 보내는데, 전역 변수로 상태를
    바꾸는 방식이라 옮기지 않았다. 막히면 매 곡이 400으로 실패해 연속 실패
    중단(종료 코드 1)으로 드러난다.
    """
    data_url = f"data:audio/mpeg;base64,{to_mp3_b64(url, exe)}"
    body = json.dumps({
        "model": EMBED_MODEL,
        "input": [{"content": [{"type": "image_url", "image_url": {"url": data_url}}]}],
        "encoding_format": "float",
        # 지정하지 않으면 10% 비싼 Vertex eu로 라우팅된다 (2026-10 실측)
        "provider": {"ignore": ["google-vertex/eu"]},
    }).encode()
    d = call(body, key)
    if "data" not in d:
        raise RuntimeError(f"예상과 다른 응답: {json.dumps(d)[:200]}")
    v = np.asarray(d["data"][0]["embedding"], dtype=np.float32)
    if v.shape[0] != GEMINI_EMB_DIM:
        raise Fatal(f"차원이 {v.shape[0]}. 기존 벡터와 섞을 수 없다")
    return v.tolist()


def pending():
    """대기열. 미리듣기가 없거나 반복 실패한 곡은 시도해도 소용없어 뺀다."""
    return (select(TrackRow)
            .where(TrackRow.emb_gemini.is_(None), TrackRow.fail_count < MAX_FAIL,
                   TrackRow.preview_url.isnot(None)))


def run(limit: int = EMBED_LIMIT, dry_run: bool = False) -> None:
    key = os.environ.get("OPENROUTER_EMBEDDING_API_KEY", "").strip()
    if not key or key.startswith("<"):
        sys.exit("OPENROUTER_EMBEDDING_API_KEY가 없습니다.")

    embed_new(key, limit, dry_run)

    # NOTE: 2단계 — emb_gemini는 있는데 emb_v4가 없는 곡(이번에 임베딩한 곡 + 예전에 v4만 못 채운 곡)을
    # 채운다. API 호출 없음. SEARCH_EMBEDDING과 상관없이 늘 채워서 스위치를 언제 바꿔도 빠진 곡이 없다.
    # 실패하면 예외로 끝나 종료 코드가 0이 아니고, 1단계에서 커밋한 emb_gemini는 그대로라 다음 실행이 이어서 채운다.
    with SessionLocal() as s:
        print(f"v4 대기 {count_missing(s)}곡")
    if dry_run:
        return
    filled = fill_missing()
    print(f"v4 채움 {filled}곡")


def embed_new(key: str, limit: int, dry_run: bool) -> None:
    """1단계: emb_gemini가 비어 있는 곡을 임베딩한다 (API 호출, 곡당 커밋)."""

    with SessionLocal() as s:
        backlog = s.scalar(select(func.count()).select_from(pending().subquery()))
        targets = list(s.scalars(pending().order_by(TrackRow.track_id).limit(limit)))
        print(f"대기 {backlog}곡 / 이번 처리 {len(targets)}곡 "
              f"(약 ${len(targets) * 0.005:.2f}, 약 {len(targets) * 2 / 60:.0f}분)")
        if dry_run or not targets:
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
                sys.exit(f"치명적 오류. 중단합니다: {e}")
            except Exception as e:
                t.fail_count = (t.fail_count or 0) + 1
                t.fail_reason = f"{type(e).__name__}: {e}"[:300]
                fail += 1
                streak += 1
                if streak >= STOP_STREAK:
                    s.commit()
                    sys.exit(f"연속 {streak}회 실패. 중단합니다. 마지막 오류: {e}")
            s.commit()    # 곡당 커밋. 도중에 죽어도 다음 실행이 이어서 한다
            time.sleep(EMBED_SLEEP)

        print(f"성공 {ok} / 실패 {fail} / 남은 대기 {backlog - ok}곡")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=EMBED_LIMIT, help="이번 실행 최대 임베딩 곡 수")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    run(limit=a.limit, dry_run=a.dry_run)
