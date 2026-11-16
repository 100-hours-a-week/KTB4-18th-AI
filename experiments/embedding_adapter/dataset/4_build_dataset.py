"""
4. 멜론 플레이리스트 → 학습·평가 데이터셋 (정답, train/val/test 분할, cold 곡, 근접 중복 제거). 로컬 계산

입력      매칭 (기본 config.MAPPING = 2_match_melon --title_aliases)
          비인기곡이 든 카탈로그 메타 (기본 catalog_v3_2026-10-15.json = 1_export_catalog --with_tail)
          이전 분할 (기본 config.PREVIOUS_DATASET): 그 val·test에 쓴 제목은 test에 넣지 않는다
          MELON_DIR/train.json
출력      runs/{날짜}_4_build_dataset/mapping_with_tail.json   매칭 + 비인기곡 {멜론 곡 id: track_id}
                                     dataset.json             {"config", "cold_tracks", "records"}
          record = {pid, split, query, title_key, pos_warm, pos_cold}. query는 멜론 원문 "제목 | 태그" (서비스 형식은 5·6)
규칙 (문서 (2) 정답과 분할)
  1. 정답 = 플레이리스트 곡 중 카탈로그와 매칭된 곡. 3곡 이상인 플레이리스트만
  2. 분할: 정규화한 제목의 해시로 train 80 / val 10 / test 10 → 같은 제목이 train과 test에 동시에 들어가지 않는다
     해시로 정하는 이유: 곡을 추가해 카탈로그가 바뀌어도 같은 제목은 같은 split, 같은 곡은 같은 cold 여부를 유지한다
  3. cold 곡: 정답에 등장하는 곡의 15% (곡 해시). 학습 정답·오답 어디에도 쓰지 않는다. train 정답에서는 빼고,
     빼고 나서도 3곡 이상 남는 train 플레이리스트만 남긴다
  4. 근접 중복 제거: 정답 겹침 |A∩B| / min(|A|,|B|) ≥ 0.8 (둘 다 정답 5곡 이상)이면 해시 순서로 하나만 남긴다.
     같은 사용자의 연작("…보이스라떼" 65개)이 train/test로 갈라지면 어댑터가 외워서 맞힌다.
     묶지 않고 버리는 이유: 묶으면 인기곡을 타고 연쇄로 이어져 거대 그룹이 생긴다 (dataset/overlap.py)
  5. 누수 점검 (모두 0이어야 한다): split 간 같은 제목, train 정답의 cold 곡, split을 넘는 근접 중복 쌍
재현      --recipe final → 원래 melon_to_track_v3_tail.json, dataset_v3.json 과 같은 파일
          --recipe first → 원래 dataset_v2.json (첫 분할. 비인기곡 없음, 이전 분할 제외 없음)

사용:
  python -m dataset.4_build_dataset --recipe final
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

import config
from core.safe_io import new_run_dir, write_json
from dataset.overlap import load_playlists, overlap_pairs

# NOTE: 이 dict는 데이터셋 파일의 "config"에 그대로 저장된다. 원래 파일과 같게 만들려면 고치지 말 것
#   (version·mapping·test_exclude의 값은 원래 실험의 파일 이름이다)
RECIPES = {
    "first": {
        "version": "v2",
        "mapping": "melon_to_track_v2.json",
        "seed": 0,                       # 해시에 섞는 값. 바꾸면 분할이 완전히 달라진다
        "min_positives": 3,              # 매칭된 정답 곡이 이 개수 이상인 플레이리스트만
        "dedup_min_overlap": 0.8,        # |A∩B| / min(|A|,|B|)가 이 이상이면 근접 중복
        "dedup_min_positives": 5,        # 두 플레이리스트 모두 정답이 이 이상일 때만 (작으면 인기곡끼리 우연히 겹친다)
        "split_ratio": {"train": 0.8, "val": 0.1, "test": 0.1},
        "cold_ratio": 0.15,
    },
    "final": {
        "version": "v3",
        "mapping": "melon_to_track_v3_tail.json",
        # NOTE: 첫 분할의 test는 이미 열었다. seed가 같으면 test 제목이 그대로라 새 test가 못 된다.
        #   seed만 바꾸면 첫 분할의 val(α·β 규칙을 고른 검색어)·test 제목이 다시 섞인다 → seed를 바꾸고 그 제목은 train으로
        "seed": 1,
        "min_positives": 3,
        "dedup_min_overlap": 0.8,
        "dedup_min_positives": 5,
        "split_ratio": {"train": 0.8, "val": 0.1, "test": 0.1},
        "cold_ratio": 0.15,
        "test_exclude": ["dataset_v2.json"],
    },
}


def make_query(title: str, tags: list[str]) -> str:
    """멜론 원문 형식 검색어: 제목 + 태그. 태그가 없으면 제목만"""
    title = title.strip()
    return f"{title} | {', '.join(tags)}" if tags else title


def merge_tail_mapping(mapping: dict, catalog_meta: list[dict]) -> dict:
    """매칭 + 비인기곡(카탈로그 메타의 source = candidates)의 멜론 id → track_id"""
    tail = [t for t in catalog_meta if t["source"] == "candidates"]
    # 비인기곡은 매칭에서 못 찾은 멜론 곡 중에서 골랐다 → 키·값 모두 겹치면 안 된다
    assert not {str(t["melon_id"]) for t in tail} & mapping.keys(), "멜론 id가 이미 매칭에 있음"
    assert not {t["track_id"] for t in tail} & set(mapping.values()), "track_id가 이미 매칭에 있음"
    assert set(mapping.values()) <= {t["track_id"] for t in catalog_meta}, "매칭에 카탈로그 밖 곡"
    merged = dict(mapping)
    merged.update({str(t["melon_id"]): t["track_id"] for t in tail})
    print(f"매칭 {len(mapping):,} + 비인기곡 {len(tail):,} = {len(merged):,} "
          f"({', '.join(f'{k} {n:,}' for k, n in Counter(t['tier'] for t in tail).items())})")
    return merged


def build(cfg: dict, mapping_path: Path, test_exclude: set[str]) -> dict:
    def unit_hash(key) -> float:
        """seed와 key로 정해지는 [0, 1) 값. 같은 key는 언제 돌려도 같은 값"""
        return int(hashlib.sha256(f"{cfg['seed']}:{key}".encode()).hexdigest()[:12], 16) / 16 ** 12

    def split_of(title_key: str) -> str:
        h = unit_hash(f"split:{title_key}")
        train, val = cfg["split_ratio"]["train"], cfg["split_ratio"]["val"]
        split = "train" if h < train else "val" if h < train + val else "test"
        return "train" if split == "test" and title_key in test_exclude else split

    def dedup_rule(pairs: tuple) -> np.ndarray:
        _, _, overlap, smaller = pairs
        return (overlap >= cfg["dedup_min_overlap"]) & (smaller >= cfg["dedup_min_positives"])

    # 1. 정답 3곡 이상, 제목 있는 플레이리스트
    candidates = load_playlists(mapping_path, cfg["min_positives"])
    print(f"필터 통과 플레이리스트: {len(candidates):,}개")

    # 2. 제목 해시로 split, 곡 해시로 cold
    positive_tracks = sorted({t for c in candidates for t in c["positives"]})
    cold = {t for t in positive_tracks if unit_hash(f"cold:{t}") < cfg["cold_ratio"]}
    records = []
    for c in candidates:
        split = split_of(c["title_key"])
        pos_warm = [t for t in c["positives"] if t not in cold]
        pos_cold = [t for t in c["positives"] if t in cold]
        if split == "train":
            # NOTE: cold 곡을 빼고 나서도 정답이 min_positives 이상 남아야 학습에 쓴다
            if len(pos_warm) < cfg["min_positives"]:
                continue
            pos_cold = []
        records.append({"pid": c["pid"], "split": split, "query": make_query(c["title"], c["tags"]),
                        "title_key": c["title_key"], "pos_warm": pos_warm, "pos_cold": pos_cold})

    # 3. 근접 중복 제거: 해시 순서로 보면서, 이미 남긴 것과 근접 중복이면 버린다
    # NOTE: train은 cold 곡이 빠진 정답이라 겹침이 달라진다. 그래서 cold 처리가 끝난 최종 정답으로 잰다
    pairs = overlap_pairs([{"positives": r["pos_warm"] + r["pos_cold"]} for r in records], t_min=cfg["dedup_min_overlap"])
    dup = dedup_rule(pairs)
    neighbors = [[] for _ in records]
    for a, b in zip(pairs[0][dup], pairs[1][dup]):
        neighbors[a].append(b)
        neighbors[b].append(a)
    kept = np.zeros(len(records), dtype=bool)
    for k in sorted(range(len(records)), key=lambda k: unit_hash(f"dedup:{records[k]['pid']}")):
        kept[k] = not any(kept[x] for x in neighbors[k])
    print(f"근접 중복 제거: {len(records):,}개 중 {(~kept).sum():,}개 버림 → {kept.sum():,}개 남음")
    records = [r for r, keep in zip(records, kept) if keep]

    # 4. 리포트
    print(f"\n정답 등장 곡 {len(positive_tracks):,}개 중 cold {len(cold):,}개 ({len(cold) / len(positive_tracks):.1%})")
    for split in ("train", "val", "test"):
        rs = [r for r in records if r["split"] == split]
        avg_pos = sum(len(r["pos_warm"]) + len(r["pos_cold"]) for r in rs) / len(rs)
        print(f"  {split:5s} {len(rs):6,d}개 | 평균 정답 {avg_pos:5.1f}곡 | cold 정답 있는 검색어 "
              f"{sum(1 for r in rs if r['pos_cold']):5,d}개")

    # 5. 누수 점검 (모두 0이어야 한다)
    print("\n[누수 점검]")
    titles = {s: {r["title_key"] for r in records if r["split"] == s} for s in ("train", "val", "test")}
    print(f"  train∩val 제목 {len(titles['train'] & titles['val'])}개, train∩test 제목 {len(titles['train'] & titles['test'])}개")
    train_pos = {t for r in records if r["split"] == "train" for t in r["pos_warm"]}
    print(f"  train 정답에 등장한 cold 곡 {len(train_pos & cold)}개")
    if test_exclude:
        print(f"  test에 들어간 이전 분할 평가 제목 {len(titles['test'] & test_exclude)}개")
    final_pairs = overlap_pairs([{"positives": sorted(r["pos_warm"] + r["pos_cold"])} for r in records],
                                t_min=cfg["dedup_min_overlap"])
    dup = dedup_rule(final_pairs)
    cross = sum(records[a]["split"] != records[b]["split"] for a, b in zip(final_pairs[0][dup], final_pairs[1][dup]))
    print(f"  split을 넘는 근접 중복 쌍 {cross}개")
    print("\n[가장 많이 반복된 제목]")
    print("  " + ", ".join(f"{tk}({n})" for tk, n in Counter(r["title_key"] for r in records).most_common(10)))
    return {"config": cfg, "cold_tracks": sorted(cold), "records": records}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", choices=list(RECIPES), default="final")
    ap.add_argument("--mapping", type=Path, help="매칭 (기본: final → config.MAPPING, first → config.MAPPING_NO_ALIAS)")
    ap.add_argument("--catalog_meta", type=Path, default=config.EMB_DATA_DIR / "catalog_v3_2026-10-15.json",
                    help="final: 비인기곡이 든 카탈로그 메타")
    ap.add_argument("--previous", type=Path, default=config.PREVIOUS_DATASET, help="final: 이전 분할 데이터셋")
    args = ap.parse_args()
    cfg = RECIPES[args.recipe]
    run = new_run_dir("4_build_dataset")

    if args.recipe == "final":
        mapping = json.load(open(args.mapping or config.MAPPING))
        merged = merge_tail_mapping(mapping, json.load(open(args.catalog_meta, encoding="utf-8")))
        mapping_path = write_json(run / "mapping_with_tail.json", merged)
        prev = json.load(open(args.previous, encoding="utf-8"))["records"]
        test_exclude = {r["title_key"] for r in prev if r["split"] in ("val", "test")}
        print(f"test 제외 제목 {len(test_exclude):,}개 ({args.previous.name}의 val·test)")
    else:
        mapping_path, test_exclude = args.mapping or config.MAPPING_NO_ALIAS, set()

    out = write_json(run / "dataset.json", build(cfg, mapping_path, test_exclude), ensure_ascii=False)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
