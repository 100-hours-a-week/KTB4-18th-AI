"""
6. 서비스 형식 데이터셋: 같은 재생목록·분할·정답·cold, 검색어 문장만 서비스 형식 (로컬 계산)

입력      4의 dataset.json (기본 원래 dataset_v3.json), 5의 변환 결과 service_query.db
          (기본 원래 EMB_DATA_DIR/service_query.db, 읽기만. 새로 변환했으면 --service_db runs/caches/service_query.db)
출력      runs/{날짜}_6_service_dataset/dataset.json  = config.DATASETS["service_format"]
          record에 melon_query(원래 "제목 | 태그")를 남긴다 (비교용)
규칙      intent가 recommend(추천 분기)인 재생목록만 남긴다 (74%). 조회·대화 이탈·되묻기 분기는 서비스에서도 임베딩 검색을 안 탄다
          변환이 반복해서 실패한 입력은 뺀다 (예: "이 ㅜㅜㅜ…" → 모델이 ㅜ를 반복하다 시간 초과. 서비스에서도 추천이 안 나가는 입력).
          5개를 넘게 빠지면 변환 문제이므로 멈춘다. 변환 모델이 섞여 있어도 멈춘다
재현      --input title → 원래 dataset_v4.json과 같은 파일

사용:
  python -m dataset.6_service_dataset
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import config
from core.safe_io import connect_readonly, new_run_dir, write_json

MAX_MISSING = 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=config.EMB_DATA_DIR / "dataset_v3.json")
    ap.add_argument("--service_db", type=Path, default=config.SERVICE_QUERY_DB)
    ap.add_argument("--input", choices=["title", "full"], default="title", help="5의 --input과 같게 (full은 기록용)")
    args = ap.parse_args()
    table = {"title": "conv_title", "full": "conv"}[args.input]

    src = json.load(open(args.source, encoding="utf-8"))
    rows = connect_readonly(args.service_db).execute(f"SELECT pid, recommend_query, result, model FROM {table}").fetchall()
    conv = {pid: (q, json.loads(res)["intent"]) for pid, q, res, _ in rows}
    models = {m for *_, m in rows}

    missing = [r["pid"] for r in src["records"] if r["pid"] not in conv]
    assert len(missing) <= MAX_MISSING, f"변환 안 된 재생목록 {len(missing):,}개 (5_service_queries --all 먼저)"
    if missing:
        print(f"변환 실패로 뺀 재생목록 {len(missing)}개: {missing}")
    assert len(models) == 1, f"변환 모델이 섞여 있다: {models}"

    records, dropped = [], Counter()
    for r in src["records"]:
        query, intent = conv.get(r["pid"], ("", "변환 실패"))
        if intent != "recommend" or not query.strip():
            dropped[(r["split"], intent)] += 1
            continue
        records.append({**r, "query": query.strip(), "melon_query": r["query"]})

    # NOTE: version·source 값은 원래 실험의 이름 (데이터셋 파일에 그대로 저장된다)
    cfg = {**src["config"], "version": {"title": "v4", "full": "v3_service"}[args.input], "source": args.source.name,
           "query": f"서비스 classify() recommend_query, 입력 {args.input}, intent=recommend만 ({models.pop()})"}
    out = write_json(new_run_dir("6_service_dataset") / "dataset.json",
                     {"config": cfg, "cold_tracks": src["cold_tracks"], "records": records}, ensure_ascii=False)

    before, after = Counter(r["split"] for r in src["records"]), Counter(r["split"] for r in records)
    for s in ("train", "val", "test"):
        gone = {i: n for (sp, i), n in dropped.items() if sp == s}
        print(f"  {s:5s} {before[s]:6,} → {after[s]:6,}  (빠진 것 {gone})")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
