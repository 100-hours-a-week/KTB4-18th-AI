"""
5-6. 쿼리 어댑터 rank: 32 / 128 / 512 / 3072(full)

질문      Low-Rank Residual의 rank k를 키우면 나아지나. 낮은 rank는 모든 검색어를 같은 방향으로 밀어 쏠림을 키운다는 가설
입력      체크포인트 query_adapter_r{32,128,512,3072} (같은 데이터·설정·5에폭, rank만 다름), service_format val, SDD
          각 체크포인트는 5-5와 같은 격자·규칙으로 자기 지점을 고른다
출력      runs/{날짜}_5_6_rank/report.md — rank별 규칙 지점, 공통 지표 표, 이웃 rank끼리 paired bootstrap
판정 기준  (결과 보기 전 고정) 이웃 rank끼리 R@10 구간 하한 > 0이면 큰 rank. 0을 포함하면 작은 쪽에서 멈춘다
결과      r512. 32 → 128 +0.0052, 128 → 512 +0.0019 [+0.0001, +0.0035], 512 → 3072 0.
          모든 rank가 α0.25 β0.5. cold는 모든 rank에서 구분 불가 (원본 로그: v4_rank_compare.log, grid_stage4_r*_ds4_*.log)

사용:
  python -m studies.5_6_rank
"""
from core.data import load_bench
from core.evaluator import Evaluator, adapter
from core.report import Report
from core.selection import query_adapter_rule_point

RANKS = (32, 128, 512, 3072)


def main() -> None:
    ev = Evaluator(load_bench("service_format", "val"))
    rep = Report("5_6_rank", "5-6. 쿼리 어댑터 rank")
    rep.add(f"service_format val {len(ev.bench.records):,}개. 지점은 rank마다 5-5 격자·규칙으로\n")

    res = {}
    for r in RANKS:
        name = f"query_adapter_r{r}"
        a, b, n = query_adapter_rule_point(ev, name)
        rep.add(f"- r{r}: 겹침 기준 이하 {n}개 조합 중 규칙 지점 α{a} β{b}")
        res[r] = (f"r{r} α{a} β{b}", ev.measure(adapter("", name, a, b)))
    rep.add()
    rep.table(dict(res.values()))

    rep.add("## 이웃 rank끼리 (큰 rank − 작은 rank)")
    chosen = RANKS[0]
    for small, big in zip(RANKS, RANKS[1:]):
        _, lo, _ = rep.ci(f"r{big} − r{small}", res[small][1], res[big][1])
        rep.ci(f"r{big} − r{small}", res[small][1], res[big][1], "cold_R@10")
        if chosen == small and lo > 0:
            chosen = big
    rep.add(f"\n판정: **r{chosen}** (이웃보다 R@10 구간 하한 > 0인 동안 rank를 키운다)")
    rep.save()


if __name__ == "__main__":
    main()
