"""
1. 서비스 DB(Postgres tracks) → 실험용 카탈로그 스냅샷 (DB는 읽기만)

실험이 DB 상태(크론 잡으로 곡이 늘어난다)에 흔들리지 않도록 한 시점의 카탈로그를 파일로 고정한다.
emb_gemini(서비스가 쓰는 Gemini Embedding 2 곡 벡터)가 있는 곡만, track_id 오름차순.

입력      KTB4_ROOT의 DB 설정(db.models.SessionLocal)과 tracks 표. --with_tail이면 tracks_candidates(3c가 넣은 비인기곡)도
출력      runs/{날짜}_1_export_catalog/catalog_{날짜}.npy   (N, 3072) float32 곡 벡터 (DB에 저장된 그대로, 이미 L2 정규화)
                                         catalog_{날짜}.json  [{track_id, title, artist, album, bucket[, source, melon_id, tier]}]
          --with_tail: tracks 곡을 track_id 순으로 먼저, 그 뒤에 tracks_candidates 곡 → 앞부분이 tracks만 뜬 것과 행 순서까지 같다
원래 산출물  catalog_2026-10-13 (tracks 15,102곡, 멜론 매칭 기준), catalog_v3_2026-10-15 (+ 비인기곡 = 19,027곡, 모든 v4 실험)

사용 (팀 저장소 폴더에서, 팀 가상환경으로 — sqlalchemy·DB 설정(.env)이 그쪽에 있다):
  cd <KTB4_ROOT>
  PYTHONPATH=<이 폴더> uv run python -m dataset.1_export_catalog [--with_tail]
"""
import argparse
import json
import sys
from datetime import date

import numpy as np

import config
from core.safe_io import guarded_output, new_run_dir

sys.path.insert(0, str(config.KTB4_ROOT))
from sqlalchemy import BigInteger, Column, MetaData, Text, select  # noqa: E402

from db.models import SessionLocal, TrackRow  # noqa: E402
from db.vectors import decode_vector  # noqa: E402


def fetch(table, extra=()):
    """emb_gemini가 있는 곡만 track_id 오름차순 (SELECT만)"""
    with SessionLocal() as s:
        return s.execute(
            select(table.c.track_id, table.c.title, table.c.artist, table.c.album, table.c.bucket,
                   table.c.emb_gemini, *extra)
            .where(table.c.emb_gemini.isnot(None))
            .order_by(table.c.track_id)
        ).all()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with_tail", action="store_true", help="tracks ∪ tracks_candidates (비인기곡)")
    args = ap.parse_args()

    rows = fetch(TrackRow.__table__)
    meta = [{"track_id": r.track_id, "title": r.title, "artist": r.artist, "album": r.album, "bucket": r.bucket}
            for r in rows]
    if args.with_tail:
        # NOTE: tracks_candidates는 ORM 모델이 없다. tracks 칸 정의를 복사하고 실험용 칸만 더한다
        cand = TrackRow.__table__.to_metadata(MetaData(), name="tracks_candidates")
        cand.append_column(Column("melon_id", BigInteger))
        cand.append_column(Column("tier", Text))
        cand_rows = fetch(cand, (cand.c.melon_id, cand.c.tier))
        assert not {r.track_id for r in rows} & {r.track_id for r in cand_rows}, "tracks와 tracks_candidates track_id 겹침"
        for m in meta:
            m["source"] = "tracks"
        meta += [{"track_id": r.track_id, "title": r.title, "artist": r.artist, "album": r.album, "bucket": r.bucket,
                  "source": "candidates", "melon_id": r.melon_id, "tier": r.tier} for r in cand_rows]
        rows = rows + cand_rows
        print(f"tracks {len(rows) - len(cand_rows):,}곡 + tracks_candidates {len(cand_rows):,}곡")

    vecs = np.stack([decode_vector(r.emb_gemini) for r in rows])
    run = new_run_dir("1_export_catalog")
    name = f"catalog_{'v3_' if args.with_tail else ''}{date.today().isoformat()}"
    np.save(guarded_output(run / f"{name}.npy"), vecs)
    with open(guarded_output(run / f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    norms = np.linalg.norm(vecs, axis=1)
    print(f"{len(meta):,}곡, shape {vecs.shape}, 노름 범위 {norms.min():.4f}~{norms.max():.4f}")
    print(f"저장: {run}/{name}.{{npy,json}}")


if __name__ == "__main__":
    main()
