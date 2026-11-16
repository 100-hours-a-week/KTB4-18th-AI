"""
5-5. 허브 보정 β와 혼합 α를 함께 고르기 (α × β 격자 + 사전 고정 규칙)

질문      어댑터를 붙이자 인기곡이 모든 검색어에 다시 나타났다 (허브 상위 1% 20% → 35%).
          누구에게나 가까운 곡에 벌점 β·r(s)를 주고(CSLS, Conneau et al., 2018), 평균 제거 검색어를 α만큼 섞으면
          쏠림을 평균 제거 수준으로 누르면서 정확도를 지킬 수 있나
입력      쿼리 어댑터 체크포인트 하나 (기본 query_adapter_r512), service_format val, train 검색어(r(s) 계산), SDD
          α ∈ {0, 0.25, 0.5, 0.75, 1} (1 = 평균 제거), β ∈ {0, 0.25, 0.5, 0.75} (0 = 벌점 없음)
출력      runs/{날짜}_5_5_alpha_beta_grid/report.md
          격자 표 7개 (R@10, 겹침, MRR, cold R@10, cold 비인기·인기 R@10, top-10 비인기곡 비율) + 규칙 지점의 공통 지표
판정 기준  (2026-10-14, 결과 보기 전 고정) 겹침이 평균 제거(α=1, β=0) 이하인 조합 중 val R@10 최대
결과      α = 0.25, β = 0.5 (r512: 허용 13개 조합 중 R@10 0.0420, 겹침 0.040).
          rank 32·128·512·3072 모두 같은 지점이 골라졌다 (5-6). β = 1이면 아무와도 먼 엉뚱한 곡이 떠올라 0.75까지만 본다
          (원본 로그: grid_stage4_r512_ds4_q3_base_e5.log)

사용:
  python -m studies.5_5_alpha_beta_grid                                  # query_adapter_r512
  python -m studies.5_5_alpha_beta_grid --checkpoint query_adapter_r32     # 다른 체크포인트에 같은 격자·규칙
"""
import argparse

import numpy as np

import config
from core.data import load_bench
from core.evaluator import Evaluator, adapter
from core.metrics import evaluate
from core.report import Report
from core.selection import grid, grid_table, rule_point

TABLES = [("R@10", "R@10", ".4f"), ("overlap", "검색어 간 겹침", ".3f"), ("MRR", "MRR", ".4f"),
          ("cold_R@10", "cold R@10", ".4f"), ("cold_tail_R@10", "cold 비인기곡 R@10", ".4f"),
          ("cold_popular_R@10", "cold 인기곡 R@10", ".4f"), ("top10_tail", "top-10 중 비인기곡 비율", ".3f")]


def tail_metrics(bench):
    """cold 정답을 비인기곡·나머지로 나눠서 (한쪽을 볼 때 warm 정답과 다른 쪽 cold 정답은 순위에서 뺀다) + top-10 비인기곡 비율"""
    is_tail = np.zeros(len(bench.S_raw), dtype=bool)
    is_tail[list(bench.tail)] = True
    splits = {}
    for key, want in (("cold_tail_R@10", True), ("cold_popular_R@10", False)):
        splits[key] = ([[j for j in r["cold_rows"] if is_tail[j] == want] for r in bench.records],
                       [r["warm_rows"] + [j for j in r["cold_rows"] if is_tail[j] != want] for r in bench.records])

    def extra(space):
        out = {k: evaluate(space.Q, space.S, pos, ignore=ign)["R@10"] for k, (pos, ign) in splits.items()}
        top10 = np.argpartition(-(space.Q @ space.S.T), 10, axis=1)[:, :10]
        out["top10_tail"] = float(is_tail[top10].mean())
        return out
    return extra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="query_adapter_r512", choices=[k for k in config.CHECKPOINTS if "query" in k])
    args = ap.parse_args()

    ev = Evaluator(load_bench("service_format", "val"))
    rep = Report("5_5_alpha_beta_grid", f"5-5. α × β 격자: {args.checkpoint}")
    rep.add(f"service_format val {len(ev.bench.records):,}개. 행 = α (1이면 평균 제거), 열 = β (0이면 허브 벌점 없음)\n")

    results = grid(ev, args.checkpoint, extra=tail_metrics(ev.bench))
    for key, title, fmt in TABLES:
        rep.add(f"### {title}")
        rep.add(grid_table(results, key, fmt) + "\n")

    limit = results[1.0, 0.0]["overlap"]
    (a, b), allowed = rule_point(results, limit)
    best = results[a, b]
    rep.add(f"## 선택 규칙\n겹침 ≤ {limit:.3f}곡(평균 제거)인 {len(allowed)}개 조합 중 R@10 최대 → **α = {a}, β = {b}**")
    rep.add(f"R@10 {best['R@10']:.4f}, MRR {best['MRR']:.4f}, cold R@10 {best['cold_R@10']:.4f}, 겹침 {best['overlap']:.3f}곡, "
            f"cold 비인기곡 {best['cold_tail_R@10']:.4f}, cold 인기곡 {best['cold_popular_R@10']:.4f}, "
            f"top-10 비인기곡 {best['top10_tail']:.4f}\n")

    rep.add("## 규칙 지점의 공통 지표")
    rep.table({f"{args.checkpoint} α{a} β{b}": ev.measure(adapter("", args.checkpoint, a, b))})
    rep.save()


if __name__ == "__main__":
    main()
