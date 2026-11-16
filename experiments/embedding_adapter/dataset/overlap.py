"""
정답 집합이 많이 겹치는 플레이리스트 찾기 (근접 중복 제거용, 4_build_dataset이 쓴다)

제목 그룹만으로는 "빌보드 hot50 2월24일자"와 "3월3일자"처럼 제목은 달라도 정답이 거의 같은
플레이리스트가 train/test로 갈라진다. 어댑터는 train 쪽을 외워서 test를 맞힌다.

  겹침(A, B) = |A ∩ B| / min(|A|, |B|)     작은 쪽 정답이 큰 쪽에 얼마나 들어 있나
  간선      = 겹침 ≥ t 이고 두 플레이리스트 모두 정답 m곡 이상 (작은 플레이리스트는 인기곡끼리 우연히 겹친다)
  그룹      = 간선 + 같은 제목으로 이어진 연결 성분

직접 실행하면 기준(t, m)을 바꿔 가며 가장 큰 그룹의 크기를 본다 (분석, 파일을 쓰지 않는다).
결론: 묶으면 인기곡을 타고 연쇄로 이어져 거대 그룹이 생긴다 → 묶지 않고 근접 중복을 하나만 남기고 버린다

사용:
  python -m dataset.overlap
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components

import config
from dataset.match_rules import norm

MIN_POSITIVES = 3      # 4_build_dataset과 같은 필터
CHUNK = 1000           # 겹침 계산을 몇 행씩 나눠 할지 (메모리)


def load_playlists(mapping: Path, min_positives: int = MIN_POSITIVES) -> list[dict]:
    """정답(매칭 곡) min_positives곡 이상, 제목이 있는 멜론 train 플레이리스트: {pid, title, tags, title_key, positives}"""
    melon_to_track = {int(k): v for k, v in json.load(open(mapping)).items()}
    out = []
    for p in json.load(open(config.MELON_DIR / "train.json", encoding="utf-8")):
        positives = sorted({melon_to_track[x] for x in p["songs"] if x in melon_to_track})
        title_key = norm(p["plylst_title"])
        if len(positives) >= min_positives and title_key:
            out.append({"pid": p["id"], "title": p["plylst_title"], "tags": p["tags"], "title_key": title_key,
                        "positives": positives})
    return out


def overlap_pairs(playlists: list[dict], t_min: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """겹침 ≥ t_min인 쌍 (i < j) → (i, j, 겹침, 작은 쪽 정답 수)"""
    track_col = {t: c for c, t in enumerate(sorted({t for p in playlists for t in p["positives"]}))}
    rows = [i for i, p in enumerate(playlists) for _ in p["positives"]]
    cols = [track_col[t] for p in playlists for t in p["positives"]]
    X = csr_matrix((np.ones(len(rows), np.int32), (rows, cols)), shape=(len(playlists), len(track_col)))
    sizes = np.array([len(p["positives"]) for p in playlists])

    found = []
    for start in range(0, len(playlists), CHUNK):
        inter = (X[start:start + CHUNK] @ X.T).toarray()                 # (CHUNK, 전체) 공통 곡 수
        smaller = np.minimum(sizes[start:start + CHUNK, None], sizes[None, :])
        ii, jj = np.nonzero(inter >= t_min * smaller)
        ii += start
        keep = ii < jj                                                    # 자기 자신, 중복 쌍 제외
        ii, jj = ii[keep], jj[keep]
        found.append((ii, jj, inter[ii - start, jj] / smaller[ii - start, jj], smaller[ii - start, jj]))
    return tuple(np.concatenate(x) for x in zip(*found))


def groups(playlists: list[dict], pairs: tuple, t: float, m: int) -> np.ndarray:
    """그룹 번호 (플레이리스트 수,). 간선 = 겹침 ≥ t & 작은 쪽 정답 ≥ m, 그리고 같은 제목"""
    i, j, overlap, smaller = pairs
    keep = (overlap >= t) & (smaller >= m)
    # NOTE: 같은 제목은 그 제목의 첫 플레이리스트에 잇는다 (제목 그룹 분할과 같은 효과)
    first_of_title = {}
    ti, tj = [], []
    for k, p in enumerate(playlists):
        first = first_of_title.setdefault(p["title_key"], k)
        if first != k:
            ti.append(first)
            tj.append(k)
    src = np.concatenate([i[keep], ti]).astype(np.int64)
    dst = np.concatenate([j[keep], tj]).astype(np.int64)
    n = len(playlists)
    graph = coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def describe(label: str, group_of: np.ndarray) -> None:
    sizes = np.bincount(group_of)
    n = len(group_of)
    print(f"  {label:22s} 그룹 {len(sizes):6,d}개 | 가장 큰 그룹 {sizes.max():6,d}개 ({sizes.max() / n:5.1%}) | "
          f"2개 이상 그룹에 속한 플레이리스트 {sizes[sizes > 1].sum() / n:5.1%}")


def main() -> None:
    playlists = load_playlists(config.MAPPING_NO_ALIAS)
    print(f"쓸 수 있는 플레이리스트 {len(playlists):,}개 (매핑 {config.MAPPING_NO_ALIAS.name})")
    pairs = overlap_pairs(playlists, t_min=0.5)
    print(f"겹침 ≥ 0.5인 쌍 {len(pairs[0]):,}개\n")

    describe("제목만", groups(playlists, pairs, t=2.0, m=0))   # t > 1이면 겹침 간선 없음
    for m in (3, 5, 10):
        for t in (0.5, 0.8, 1.0):
            describe(f"겹침≥{t} & 정답≥{m}", groups(playlists, pairs, t, m))

    group_of = groups(playlists, pairs, t=0.8, m=5)
    print("\n[겹침≥0.8 & 정답≥5: 큰 그룹 상위 5개의 제목 예시]")
    for g, size in Counter(group_of).most_common(5):
        titles = [playlists[k]["title"] for k in np.nonzero(group_of == g)[0][:6]]
        print(f"  {size:5,d}개: {titles}")


if __name__ == "__main__":
    main()
