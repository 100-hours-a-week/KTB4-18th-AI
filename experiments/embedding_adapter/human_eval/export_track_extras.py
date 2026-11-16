"""
7. 서비스 DB → 곡별 부가 정보 (사람 평가 목록을 서비스 search()와 같은 규칙으로 만들고 미리듣기를 붙이는 데 쓴다). DB는 읽기만

입력      KTB4_ROOT의 DB 설정, tracks ∪ tracks_candidates
출력      runs/{날짜}_track_extras/track_extras.json  {track_id: {store_url, preview_url, genre, release_date}}
          release_date는 "YYYY-MM-DD" 또는 null. export_lists.py --extras로 넘긴다

사용 (팀 저장소 폴더에서, 팀 가상환경으로):
  cd <KTB4_ROOT>
  PYTHONPATH=<이 폴더> uv run python -m human_eval.export_track_extras
"""
import sys

import config
from core.safe_io import new_run_dir, write_json

sys.path.insert(0, str(config.KTB4_ROOT))
from sqlalchemy import MetaData, select  # noqa: E402

from db.models import SessionLocal, TrackRow  # noqa: E402


def main() -> None:
    # NOTE: tracks_candidates는 ORM 모델이 없다 (dataset/1_export_catalog.py와 같은 방식으로 tracks 칸 정의를 빌린다)
    tables = [TrackRow.__table__, TrackRow.__table__.to_metadata(MetaData(), name="tracks_candidates")]
    extras = {}
    with SessionLocal() as s:
        for t in tables:
            rows = s.execute(select(t.c.track_id, t.c.store_url, t.c.preview_url, t.c.genre, t.c.release_date)).all()
            extras.update({r.track_id: {"store_url": r.store_url, "preview_url": r.preview_url, "genre": r.genre,
                                        "release_date": r.release_date.isoformat() if r.release_date else None}
                           for r in rows})
            n = lambda key: sum(1 for r in rows if getattr(r, key))
            print(f"{t.name}: {len(rows):,}곡 / store_url {n('store_url'):,} preview_url {n('preview_url'):,} "
                  f"genre {n('genre'):,} release_date {n('release_date'):,}")
    out = write_json(new_run_dir("track_extras") / "track_extras.json", extras, ensure_ascii=False)
    print(f"저장: {out} ({len(extras):,}곡)")


if __name__ == "__main__":
    main()
