"""
평가지표 (문서 (3)). 지표마다 "막고 싶은 실패"가 하나씩 있다

  R@10, MRR          정답을 못 찾음 / 첫 정답이 너무 아래
  cold R@10          학습한 곡만 외움 (cold 정답만 채점, 같은 검색어의 warm 정답은 순위에서 뺀다)
  비인기곡 warm R@10   비인기곡 정답만 (5-7 진단)
  검색어 간 겹침       모두에게 같은 곡: 서로 다른 두 검색어의 top-10이 평균 몇 곡 공유하나 (0~10)
  허브 상위 1% 점유    top-10에 가장 자주 나온 상위 1% 곡이 top-10 자리에서 차지한 비율
  top-10 kpop        언어 편향. 정답 곡의 kpop 비율에 가까울수록 좋다 (낮을수록 좋은 게 아니다)
  재생목록 응집도      검색어 없이 곡끼리: 정답 절반의 평균 방향 → 나머지 절반 R@10 (warm / cold)
  SDD R@10           범용 음악 검색 회귀 테스트 → core/sdd.py

evaluate_all()이 위를 한 번에 낸다. 모든 비교 스크립트가 같은 묶음을 쓴다
비교는 같은 검색어 묶음에 대한 paired bootstrap 95% 구간: 하한 > 0이면 개선, 0을 포함하면 구분 불가
"""
import hashlib

import numpy as np

import config
from core.data import Bench, l2norm
from core.systems import Space

COHESION_MIN_POS = 4   # 응집도에 쓰는 재생목록의 최소 정답 수


def evaluate(Q: np.ndarray, S: np.ndarray, positives: list[list[int]],
             ignore: list[list[int]] | None = None, ks=(10, 50)) -> dict:
    """Q (검색어, D), S (곡, D) 내적으로 순위. positives[i] = 검색어 i의 정답 곡 행 번호

    등수 = 그 정답보다 점수가 높은 곡 수 + 1 (동점은 정답에 유리). 정답이 빈 검색어는 빠진다
    """
    scores = Q @ S.T
    if ignore is not None:
        for i, rows in enumerate(ignore):
            scores[i, rows] = -np.inf

    recall = {k: [] for k in ks}
    mrr, ndcg = [], []
    for i, pos in enumerate(positives):
        if not pos:
            continue
        ranks = np.sort((scores[i][None, :] > scores[i, pos][:, None]).sum(axis=1) + 1)
        for k in ks:
            recall[k].append((ranks <= k).sum() / len(pos))
        mrr.append(1 / ranks[0])
        dcg = (1 / np.log2(ranks[ranks <= 10] + 1)).sum()
        idcg = (1 / np.log2(np.arange(1, min(10, len(pos)) + 1) + 1)).sum()
        ndcg.append(dcg / idcg)

    result = {f"R@{k}": float(np.mean(v)) for k, v in recall.items()}
    result.update({"MRR": float(np.mean(mrr)), "nDCG@10": float(np.mean(ndcg)), "n": len(mrr)})
    # 부트스트랩용 검색어별 값. 같은 정답 목록이면 두 시스템에서 같은 검색어가 빠진다
    result["per_query"] = {**{f"R@{k}": np.array(v) for k, v in recall.items()},
                           "MRR": np.array(mrr), "nDCG@10": np.array(ndcg)}
    return result


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = config.N_BOOT,
                     seed: int = config.BOOT_SEED) -> tuple[float, float, float]:
    """같은 검색어들에 대한 두 시스템의 검색어별 점수 → (b − a 평균 차이, 95% 하한, 상한)"""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = (b[idx] - a[idx]).mean(axis=1)
    return float(b.mean() - a.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def kpop_share_top10(Q: np.ndarray, S: np.ndarray, is_kpop: np.ndarray) -> float:
    top10 = np.argpartition(-(Q @ S.T), 10, axis=1)[:, :10]
    return float(is_kpop[top10].mean())


def diversity(Q: np.ndarray, S: np.ndarray, melon_rows: set, train_rows: set = frozenset()) -> dict:
    """top-10이 일부 곡에 쏠리지 않았는지

    overlap    서로 다른 두 검색어의 top-10에 같은 곡이 평균 몇 곡 (0~10)
    hub_top1   상위 1% 곡이 top-10 자리 전체에서 차지한 비율
    hub_max    가장 자주 나온 한 곡이 top-10에 들어간 검색어 비율
    coverage   top-10에 한 번이라도 나온 곡 / 카탈로그
    non_melon  top-10 중 멜론에 없는(라벨 없는) 곡 비율
    seen       top-10 중 학습 정답으로 본 곡 비율 (train_rows를 줄 때만 의미)
    """
    flat = np.argpartition(-(Q @ S.T), 10, axis=1)[:, :10].ravel()
    n10 = np.sort(np.bincount(flat, minlength=S.shape[0]))[::-1]   # 곡마다 top-10 등장 횟수
    # NOTE: 곡 하나가 c개 검색어의 top-10에 있으면 그 곡이 만드는 "같은 곡 공유 쌍"은 c(c−1)개 (순서 있는 쌍)
    n_q = Q.shape[0]
    return {"overlap": float((n10.astype(np.int64) * (n10 - 1)).sum() / (n_q * (n_q - 1))),
            "coverage": len(np.unique(flat)) / S.shape[0],
            "seen": float(np.isin(flat, list(train_rows)).mean()),
            "non_melon": float(1 - np.isin(flat, list(melon_rows)).mean()),
            "hub_max": float(n10[0] / n_q),
            "hub_top1": float(n10[:max(1, S.shape[0] // 100)].sum() / len(flat))}


def split_half(rec: dict) -> tuple[list[int], list[int]]:
    """정답을 pid로 정해지는 순서로 섞어 (단서 절반, 맞힐 절반). 언제 돌려도 같다"""
    pos = sorted(rec["warm_rows"] + rec["cold_rows"])
    seed = int(hashlib.sha256(f"ceiling:{rec['pid']}".encode()).hexdigest()[:8], 16)
    order = np.random.default_rng(seed).permutation(pos).tolist()
    return order[:len(order) // 2], order[len(order) // 2:]


def cohesion(songs: np.ndarray, bench: Bench) -> dict:
    """재생목록 응집도: 정답 절반의 평균 방향으로 나머지 절반을 찾는 R@10 (맞힐 곡이 warm / cold)

    정답을 본 "반칙" 검색어라 텍스트와 무관하게 곡 배치만 본다. 단서 곡과 다른 그룹 곡은 순위에서 뺀다
    """
    recs = [r for r in bench.records if len(r["warm_rows"]) + len(r["cold_rows"]) >= COHESION_MIN_POS]
    halves = [split_half(r) for r in recs]
    Q = l2norm(np.stack([songs[c].mean(axis=0) for c, _ in halves]))
    out = {}
    for g, f in (("warm", lambda j: j not in bench.cold), ("cold", lambda j: j in bench.cold)):
        out[g] = evaluate(Q, songs, [[j for j in t if f(j)] for _, t in halves],
                          ignore=[c + [j for j in t if not f(j)] for c, t in halves])["R@10"]
    return out


def evaluate_all(space: Space, bench: Bench, with_cohesion: bool = True) -> dict:
    """공통 지표 묶음 (SDD 제외 — SDD는 다른 곡·검색어라 core/sdd.py에서 따로)"""
    Q, S = space.Q, space.S
    m = evaluate(Q, S, bench.all_pos)
    c = evaluate(Q, S, bench.cold_pos, ignore=bench.warm_ign)
    d = diversity(Q, S, bench.melon_rows)
    out = {"R@10": m["R@10"], "R@50": m["R@50"], "MRR": m["MRR"], "nDCG@10": m["nDCG@10"], "n": m["n"],
           "cold_R@10": c["R@10"], "cold_n": c["n"],
           "overlap": d["overlap"], "hub_top1": d["hub_top1"], "hub_max": d["hub_max"], "coverage": d["coverage"],
           "kpop": kpop_share_top10(Q, S, bench.is_kpop),
           "per_query": {"R@10": m["per_query"]["R@10"], "cold_R@10": c["per_query"]["R@10"]}}
    if bench.tail:
        tail_pos = [[j for j in r["warm_rows"] if j in bench.tail] for r in bench.records]
        tail_ign = [[j for j in p if j not in bench.tail] for p in bench.all_pos]
        t = evaluate(Q, S, tail_pos, ignore=tail_ign)
        out["tail_warm_R@10"] = t["R@10"]
        out["per_query"]["tail_warm_R@10"] = t["per_query"]["R@10"]
    if with_cohesion:
        h = cohesion(space.songs, bench)
        out["cohesion_warm"], out["cohesion_cold"] = h["warm"], h["cold"]
    return out


def diff(base: dict, other: dict, key: str = "R@10") -> tuple[float, float, float]:
    """evaluate_all 결과 둘의 검색어별 차이 (other − base) 부트스트랩"""
    return paired_bootstrap(base["per_query"][key], other["per_query"][key])
