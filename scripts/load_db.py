import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from sqlalchemy import func, select

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from db.models import (  # noqa: E402
    EMB_DIM, MODEL_VERSION, Base, CorpusStat, SessionLocal, TrackRow,
    engine, init_db, l2norm,
)

OUT = ROOT / "out"
MODEL_DIR = ROOT / "models"
TOP_TAGS = 10
MIN_TAG_PROB = 0.05


def top_tags(row: np.ndarray, names: list[str]) -> dict:
    """기준치 MIN_TAG_PROB 이상의 수치를 가진 음악 성질 전부 모음"""
    out = {}
    for j in np.argsort(-row)[:TOP_TAGS]:
        p = float(row[j])
        if p < MIN_TAG_PROB:
            break
        out[names[j]] = round(p, 3)
    return out


def parse_date(s: str):
    """복잡하게 들어오는 날짜 문자열에서 앞 10글자만 떼내, fromisoformat으로 날짜 포맷 고정"""
    try:
        return date.fromisoformat((s or "")[:10])
    except Exception:
        return None


def main(recreate: bool = False, batch: int = 500):

    # NOTE: 기존 로컬 메타데이터/임베딩/태그 가져옴
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    emb = np.load(OUT / "emb_clap.npy")
    tags = np.load(OUT / "tags_mood.npy")
    tag_names = json.loads(
        (MODEL_DIR / "mtg_jamendo_moodtheme-discogs-effnet-1.json")
        .read_text(encoding="utf-8"))["classes"]

    assert len(meta) == len(emb) == len(tags), "meta와 벡터 개수 불일치"
    assert emb.shape[1] == EMB_DIM, f"차원 불일치: {emb.shape[1]} != {EMB_DIM}"
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-3), "벡터가 L2 정규화되지 않음"
    print(f"{len(meta)}곡, {emb.shape[1]}차원, 정규화 확인")

    # NOTE: 기존 로컬 데이터용 중심화 정의
    mean = emb.mean(axis=0).astype(np.float32)
    centered = l2norm(emb - mean).astype(np.float32)

    if recreate:
        Base.metadata.drop_all(engine)
        print("기존 테이블 삭제")
    init_db()

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = [{
        "track_id": int(m["track_id"]),
        "title": m["title"],
        "artist": m["artist"],
        "album": m.get("album"),
        "genre": m.get("genre"),
        "release_date": parse_date(m.get("release_date")),
        "preview_url": m["preview_url"],
        "artwork_url": m.get("artwork_url"),
        "seed_artist": m.get("seed_artist"),
        "bucket": m.get("bucket"),
        "mood_tags": top_tags(t, tag_names),
        "emb_clap": v,
        "model_version": MODEL_VERSION,
        "embedded_at": now,
        "fail_count": 0,
    } for m, v, t in zip(meta, centered, tags)]

    with SessionLocal() as s:
        existing = {r[0] for r in s.execute(select(TrackRow.track_id))}
        new = [r for r in rows if r["track_id"] not in existing]
        print(f"기존 {len(existing)}곡 / 신규 {len(new)}곡")

        # NOTE: 배치 별로 나눠서 DB에 넣고 커밋한다
        for i in range(0, len(new), batch):
            s.bulk_insert_mappings(TrackRow, new[i:i + batch])
            s.commit()
            print(f"  {min(i + batch, len(new))}/{len(new)}", end="\r")
        print()

        # NOTE: 중심화 새로 반영
        stat = s.get(CorpusStat, "clap_mean")
        if stat is None:
            s.add(CorpusStat(key="clap_mean", value=mean,
                             updated_at=now))
        else:
            stat.value = mean
            stat.updated_at = now
        s.commit()

        total = s.scalar(select(func.count()).select_from(TrackRow))
        with_emb = s.scalar(select(func.count()).select_from(TrackRow)
                            .where(TrackRow.emb_clap.isnot(None)))
        print(f"\n적재 완료: 총 {total}곡, 임베딩 보유 {with_emb}곡")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--batch", type=int, default=500)
    a = ap.parse_args()
    main(recreate=a.recreate, batch=a.batch)
