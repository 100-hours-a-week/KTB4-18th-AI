"""
3c. 비인기곡(3b에서 찾은 곡) → Gemini Batch API 임베딩 → 서비스 DB tracks_candidates  ⚠ 유료, DB에 쓴다

입력      runs/caches/itunes_tail.db (3b 결과, --itunes_db로 바꿀 수 있다. 읽기만), 매칭(config.MAPPING),
          로컬 Postgres(컨테이너 PG_CONTAINER, 기본 music-pg)의 tracks·tracks_candidates 곡 목록
출력      runs/caches/tail_batches.db (곡이 어느 배치에 들어갔는지), runs/caches/tail_XXX.jsonl (제출 파일)
          fetch --write_db: DB tracks_candidates에 곡 추가 (서비스용 tracks는 건드리지 않는다)
안전장치   submit은 예상 비용을, fetch는 넣을 곡 수와 DB를 보여주고 y를 받아야 진행한다.
          fetch는 --write_db 없이 실행하면 DB에 쓰지 않고 넣을 곡 수만 보여준다
비용      곡당 약 $0.0024 (원래 실험 3,925곡 ≈ $9.5)

서비스 곡과 같은 벡터가 되도록 같은 조건으로 임베딩한다 (gemini_collect.embed_preview와 같음)
  미리듣기 앞 30초 → 모노 24kHz 64kbps mp3 → base64 → gemini-embedding-2, 3072차원
Batch API는 일반 호출의 반값 (곡당 약 $0.0024). 결과는 보통 24시간 안에 나온다.

단계 (곡이 어느 배치에 들어갔는지 tail_batches.db에 기록 → 같은 곡을 두 번 사지 않는다)
  build   찾은 곡(strict, roman) 중 다음 N곡을 골라 JSONL을 만든다. 곡마다 받고 → 변환 → 한 줄 추가 → 오디오 삭제
          제외: 이미 tracks/tracks_candidates에 있는 곡, 매칭 v3로 카탈로그에 있는 멜론 곡, 같은 iTunes 곡 중복
  submit  JSONL을 올리고 배치 제출
  status  배치 상태
  fetch   끝난 배치 결과 → tracks_candidates에 넣고 JSONL·올린 파일 삭제

tracks_candidates 칸: iTunes 결과 그대로 (기존 곡과 같은 형태), bucket = melon_tail,
  model_version·embedded_at은 비움 (tracks에서도 이 둘은 CLAP 기준이고 Gemini 적재 때 건드리지 않았다)

사용 (GEMINI_API_KEY는 환경 변수 또는 이 저장소의 .env):
  python -m dataset.3_tail_songs.c_embed_batch build --n 1000
  python -m dataset.3_tail_songs.c_embed_batch submit
  python -m dataset.3_tail_songs.c_embed_batch status
  python -m dataset.3_tail_songs.c_embed_batch fetch               # 넣을 곡 수만 확인
  python -m dataset.3_tail_songs.c_embed_batch fetch --write_db    # DB에 넣기
"""
import argparse
import base64
import csv
import io
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from google import genai

import config
from core.safe_io import cache_db, confirm, connect_readonly

CACHE_DIR = config.RUNS_DIR / "caches"
ITUNES_DB = CACHE_DIR / "itunes_tail.db"
MAPPING_V3 = config.MAPPING
PG_CONTAINER = os.environ.get("PG_CONTAINER", "music-pg")
COST_PER_SONG = 0.00243
MODEL = "gemini-embedding-2"
DIM = 3072
AUDIO_SEC = 30
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
BUCKET = "melon_tail"
FOUND = ("strict", "roman")
# tracks_candidates에 넣는 칸 (tracks 칸 + 실험용 칸). 나머지(emb_clap, mood_tags 등)는 NULL/기본값
COLUMNS = ["track_id", "title", "artist", "album", "genre", "release_date", "preview_url", "artwork_url",
           "store_url", "duration_ms", "bucket", "fail_count", "emb_gemini",
           "melon_id", "tier", "match_verdict", "batch"]


# ── 준비물 ─────────────────────────────────────────────────────
def gemini_client() -> genai.Client:
    load_dotenv(config.ROOT / ".env")
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"].strip().strip('"'))


def ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def psql(sql: str, stdin: str | None = None) -> str:
    """로컬 Postgres 컨테이너에서 SQL 실행 (docker exec). 오류면 예외"""
    cmd = ["docker", "exec", "-i", PG_CONTAINER, "psql", "-U", "postgres", "-d", "music",
           "-v", "ON_ERROR_STOP=1", "-tA", "-c", sql]
    return subprocess.run(cmd, input=stdin, capture_output=True, text=True, check=True).stdout


def open_state() -> sqlite3.Connection:
    con = cache_db("tail_batches.db")
    con.execute("CREATE TABLE IF NOT EXISTS members (track_id INTEGER PRIMARY KEY, melon_id INTEGER, tier TEXT, "
                "verdict TEXT, batch TEXT, status TEXT, itunes TEXT)")   # status: built / failed / done
    con.execute("CREATE TABLE IF NOT EXISTS jobs (batch TEXT PRIMARY KEY, jsonl TEXT, file_name TEXT, "
                "job_name TEXT, n INTEGER)")
    return con


def preview_b64(url: str, exe: str) -> str:
    """미리듣기 → 30초 모노 24kHz 64kbps mp3 → base64. 오디오 파일은 함수가 끝나면 지워진다
    NOTE: gemini_collect.embed_preview와 조건이 같아야 서비스 곡과 같은 벡터가 된다
    """
    with tempfile.TemporaryDirectory() as tmp:
        m4a, mp3 = Path(tmp) / "a.m4a", Path(tmp) / "a.mp3"
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
            m4a.write_bytes(r.read())
        subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(m4a), "-t", str(AUDIO_SEC), "-ac", "1",
                        "-ar", "24000", "-b:a", "64k", str(mp3)], check=True)
        return base64.b64encode(mp3.read_bytes()).decode()


# ── build ──────────────────────────────────────────────────────
def open_batch(con: sqlite3.Connection) -> str:
    """끝나지 않은 배치(곡 기록은 있고 jobs에는 없음)가 있으면 그 이름, 없으면 새 이름"""
    row = con.execute("SELECT DISTINCT batch FROM members WHERE batch NOT IN (SELECT batch FROM jobs)").fetchone()
    if row:
        return row[0]
    n_batches = con.execute("SELECT COUNT(DISTINCT batch) FROM members").fetchone()[0]
    return f"tail_{n_batches + 1:03d}"


def reconcile(con: sqlite3.Connection, batch: str, jsonl: Path) -> None:
    """중간에 멈춘 배치를 이어 만들기 전에 JSONL과 기록을 맞춘다
      쓰다 만 마지막 줄 → 버림
      파일에는 있는데 기록이 없는 줄 (쓰고 나서 기록하기 전에 멈춤) → 버림
      기록은 built인데 파일에 없는 곡 → 기록을 지워 다시 받게 함
    """
    recorded = {t for (t,) in con.execute("SELECT track_id FROM members WHERE batch = ? AND status = 'built'",
                                          (batch,))}
    kept, keys = [], set()
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            try:
                key = int(json.loads(line)["key"])
            except json.JSONDecodeError:   # 쓰다 만 줄
                continue
            if key in recorded and key not in keys:
                kept.append(line)
                keys.add(key)
    jsonl.write_text("".join(line + "\n" for line in kept))
    con.executemany("DELETE FROM members WHERE track_id = ?", [(t,) for t in recorded - keys])
    con.commit()
    if recorded - keys:
        print(f"  이어 만들기: 파일에 없던 {len(recorded - keys)}곡은 다시 받는다")


def build(n: int, itunes_db: Path) -> None:
    """배치 하나(N곡)를 만든다. 중간에 멈추면 다시 실행할 때 같은 배치를 이어서 채운다 (DB는 읽기만, 비용 없음)"""
    con = open_state()
    batch = open_batch(con)
    jsonl = CACHE_DIR / f"{batch}.jsonl"
    reconcile(con, batch, jsonl)
    already = con.execute("SELECT COUNT(*) FROM members WHERE batch = ?", (batch,)).fetchone()[0]

    # 제외할 곡: DB에 이미 있는 iTunes 곡, 매칭 v3로 카탈로그에 있는 멜론 곡, 이미 배치에 넣은 곡
    in_db = {int(x) for x in psql("SELECT track_id FROM tracks UNION SELECT track_id FROM tracks_candidates").split()}
    in_catalog_v3 = {int(k) for k in json.load(open(MAPPING_V3))}
    taken = {t for (t,) in con.execute("SELECT track_id FROM members")}

    picked = []
    rows = connect_readonly(itunes_db).execute(
        f"SELECT melon_id, tier, verdict, itunes FROM verdicts WHERE verdict IN {FOUND} ORDER BY rowid")
    for melon_id, tier, verdict, itunes in rows:
        t = json.loads(itunes)
        if t["trackId"] in in_db or t["trackId"] in taken or melon_id in in_catalog_v3:
            continue
        taken.add(t["trackId"])
        picked.append((melon_id, tier, verdict, t))
        if already + len(picked) >= n:
            break
    if already + len(picked) == 0:
        print("넣을 곡이 더 없다 (찾은 곡을 모두 배치에 넣었음). 배치를 만들지 않는다")
        return
    print(f"[{batch}] {len(picked):,}곡을 추가한다 (이미 {already:,}곡, 목표 {n:,}곡 / 제외: DB {len(in_db):,}곡, "
          f"매칭 v3 카탈로그 {len(in_catalog_v3):,}곡)")

    exe = ffmpeg()
    with open(jsonl, "a") as f:
        for i, (melon_id, tier, verdict, t) in enumerate(picked, 1):
            try:
                b64 = preview_b64(t["previewUrl"], exe)
                status = "built"
            except (OSError, subprocess.CalledProcessError) as e:   # 미리듣기 링크 만료·변환 실패
                print(f"\n  실패 {t['trackName']} / {t['artistName']}: {e}")
                status = "failed"
            if status == "built":
                f.write(json.dumps({"key": str(t["trackId"]),
                                    "request": {"content": {"parts": [{"inline_data": {"mime_type": "audio/mpeg",
                                                                                       "data": b64}}]},
                                                "output_dimensionality": DIM}}) + "\n")
                f.flush()   # 기록하기 전에 파일에 먼저 남긴다 (멈추면 reconcile이 맞춘다)
            con.execute("INSERT INTO members VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (t["trackId"], melon_id, tier, verdict, batch, status, json.dumps(t, ensure_ascii=False)))
            con.commit()
            print(f"  {i:5,d}/{len(picked):,}  {jsonl.stat().st_size / 1e6:6.1f}MB", end="\r")

    n_built = con.execute("SELECT COUNT(*) FROM members WHERE batch = ? AND status = 'built'", (batch,)).fetchone()[0]
    con.execute("INSERT INTO jobs VALUES (?, ?, NULL, NULL, ?)", (batch, str(jsonl), n_built))
    con.commit()
    print(f"\n저장: {jsonl} ({n_built:,}곡, {jsonl.stat().st_size / 1e6:.0f}MB). "
          f"예상 비용 ${n_built * COST_PER_SONG:.2f}")


# ── submit / status ────────────────────────────────────────────
def submit() -> None:
    con = open_state()
    todo = con.execute("SELECT batch, jsonl, n FROM jobs WHERE job_name IS NULL").fetchall()
    if not todo:
        print("제출할 배치가 없다")
        return
    n = sum(k for *_, k in todo)
    confirm(f"배치 {len(todo)}개, {n:,}곡을 Gemini Batch API에 제출한다. 예상 비용 ${n * COST_PER_SONG:.2f} (유료)")
    client = gemini_client()
    for batch, jsonl, _ in todo:
        up = client.files.upload(file=jsonl, config={"display_name": batch, "mime_type": "jsonl"})
        job = client.batches.create_embeddings(model=MODEL, src={"file_name": up.name}, config={"display_name": batch})
        con.execute("UPDATE jobs SET file_name = ?, job_name = ? WHERE batch = ?", (up.name, job.name, batch))
        con.commit()
        print(f"[{batch}] 제출: {job.name} ({job.state})")


def status() -> None:
    con = open_state()
    client = gemini_client()
    for batch, job_name, n in con.execute("SELECT batch, job_name, n FROM jobs").fetchall():
        state = client.batches.get(name=job_name).state.name if job_name else "아직 제출 안 함"
        done = con.execute("SELECT COUNT(*) FROM members WHERE batch = ? AND status = 'done'", (batch,)).fetchone()[0]
        print(f"  {batch}  {n:,}곡  {state}  (DB에 넣음 {done:,})")


# ── fetch ──────────────────────────────────────────────────────
def candidate_row(t: dict, vec: list[float], member: tuple) -> list:
    """COLUMNS 순서의 한 행. iTunes 결과는 기존 tracks와 같은 형태로 (artworkUrl100, trackViewUrl 그대로)"""
    melon_id, tier, verdict, batch = member
    return [t["trackId"], t["trackName"], t["artistName"], t.get("collectionName"),
            (t.get("primaryGenreName") or "")[:64] or None, (t.get("releaseDate") or "")[:10] or None,
            t["previewUrl"], t.get("artworkUrl100"), t.get("trackViewUrl"), t.get("trackTimeMillis"),
            BUCKET, 0, "[" + ",".join(map(str, vec)) + "]", melon_id, tier, verdict, batch]


def fetch(write_db: bool) -> None:
    """끝난 배치 → tracks_candidates. 결과 한 줄: {"key": track_id, "response": {"embedding": {"values": [...]}}}
    (2026-10-14 시험: DB에 있는 3곡을 배치로 다시 임베딩 → DB 벡터와 cos 1.000000)
    write_db=False면 넣을 곡 수만 보여주고 DB·기록·파일을 바꾸지 않는다
    """
    con = open_state()
    client = gemini_client()
    pending = con.execute("SELECT DISTINCT j.batch, j.jsonl, j.file_name, j.job_name FROM jobs j JOIN members m "
                          "ON m.batch = j.batch WHERE m.status = 'built' AND j.job_name IS NOT NULL").fetchall()
    for batch, jsonl, file_name, job_name in pending:
        job = client.batches.get(name=job_name)
        if job.state.name != "JOB_STATE_SUCCEEDED":
            print(f"[{batch}] 아직 {job.state.name}")
            continue

        vectors, errors = {}, {}
        for line in client.files.download(file=job.dest.file_name).decode().splitlines():
            d = json.loads(line)
            if "response" in d:
                vectors[int(d["key"])] = d["response"]["embedding"]["values"]
            else:
                errors[int(d["key"])] = json.dumps(d.get("error"))[:200]

        members = con.execute("SELECT track_id, melon_id, tier, verdict, batch, itunes FROM members "
                              "WHERE batch = ? AND status = 'built'", (batch,)).fetchall()
        # NOTE: DB에 넣은 직후 멈췄다가 다시 fetch하면 이미 들어간 곡이 있다 → 건너뛴다
        in_db = {int(x) for x in psql(f"SELECT track_id FROM tracks_candidates WHERE batch = '{batch}'").split()}
        buf = io.StringIO()
        writer = csv.writer(buf)
        n_rows = 0
        for track_id, melon_id, tier, verdict, b, itunes in members:
            if track_id in vectors and track_id not in in_db:
                writer.writerow(candidate_row(json.loads(itunes), vectors[track_id], (melon_id, tier, verdict, b)))
                n_rows += 1
        if not write_db:
            print(f"[{batch}] 넣을 곡 {n_rows:,}곡 (벡터 {len(vectors):,}, 오류 {len(errors):,}). "
                  "DB에 넣으려면 --write_db")
            continue
        confirm(f"[{batch}] {n_rows:,}곡을 컨테이너 {PG_CONTAINER}의 DB music, 표 tracks_candidates에 넣는다 "
                "(tracks는 건드리지 않는다)")
        # NOTE: \copy는 한 번에 들어가거나 하나도 안 들어간다 → 실패하면 상태를 안 바꾸고 다시 fetch하면 된다
        if buf.getvalue():
            psql(f"\\copy tracks_candidates ({', '.join(COLUMNS)}) FROM STDIN WITH (FORMAT csv)",
                 stdin=buf.getvalue())

        con.executemany("UPDATE members SET status = 'done' WHERE track_id = ?", [(t,) for t in vectors])
        con.executemany("UPDATE members SET status = 'failed' WHERE track_id = ?", [(t,) for t in errors])
        con.commit()
        n_db = int(psql(f"SELECT COUNT(*) FROM tracks_candidates WHERE batch = '{batch}'"))
        print(f"[{batch}] tracks_candidates에 {n_db:,}곡 (벡터 {len(vectors):,}, 오류 {len(errors):,})")
        if errors:
            print("  오류 예:", list(errors.items())[:3])

        # 다 들어갔으면 로컬 JSONL과 올린 파일 정리 (결과 파일은 Google이 6주 뒤 지운다)
        Path(jsonl).unlink(missing_ok=True)
        if any(f.name == file_name for f in client.files.list()):   # 지난 fetch에서 이미 지웠을 수 있다
            client.files.delete(name=file_name)
        print(f"  삭제: {jsonl}, {file_name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "submit", "status", "fetch"])
    ap.add_argument("--n", type=int, default=1000, help="build: 이번 배치에 넣을 곡 수")
    ap.add_argument("--itunes_db", type=Path, default=ITUNES_DB, help="build: 3b 결과 (읽기만)")
    ap.add_argument("--write_db", action="store_true", help="fetch: 실제로 DB tracks_candidates에 넣는다")
    args = ap.parse_args()
    {"build": lambda: build(args.n, args.itunes_db), "submit": submit, "status": status,
     "fetch": lambda: fetch(args.write_db)}[args.step]()


if __name__ == "__main__":
    main()
