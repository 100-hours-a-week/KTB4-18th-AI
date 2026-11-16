"""
5-1. 평균 제거: 학습 없이 각자의 평균을 빼는 것만으로 언어 편향·허브가 줄어드나

질문      텍스트 임베딩과 오디오 임베딩은 각자 공통 성분을 가진다 (modality gap, Liang et al., 2022).
          q_c = norm(q − μ_q), s′ = norm(s − μ_s)로 빼면 한국어 검색어끼리 뭉치는 쏠림이 풀리나
입력      service_format val 2,258개, train 17,802개(μ_q 계산에만), 카탈로그 19,027곡,
          eval.txt 사용자 말투 요청 87개(멜론과 출처가 다르고 라벨 없음), SDD
출력      runs/{날짜}_5_1_mean_removal/report.md
          표: 원래 / 평균 제거(검색어만) / 평균 제거(둘 다) × 공통 지표, eval.txt 쏠림 지표, μ_q 표본 수 곡선
판정 기준  (2026-10-17, 결과 보기 전 고정) 이 실험이 풀려던 문제 = 언어 편향과 허브
          평균 제거(검색어만)가 원래보다 겹침·허브 상위 1%가 val과 eval.txt 둘 다에서 낮으면 → 채택
          R@10·cold R@10은 서술, kpop은 정답 곡의 kpop 비율과 나란히 서술
결과      채택. 겹침 0.235 → 0.048, kpop 78.9% → 42.4% (정답 48.9%). eval.txt에서도 같은 방향.
          R@10 이득은 없다 → 평균 제거는 정확도가 아니라 쏠림 해소 장치.
          곡 평균까지 빼면 검색어만보다 R@10 +0.0012 [+0.0002, +0.0022] → 서비스에는 둘 다 뺀다
          (원본 로그: v4_stage2_center.log)

사용:
  python -m studies.5_1_mean_removal
"""
import numpy as np

from core.data import l2norm, load_bench, load_evaltxt_queries
from core.evaluator import CENTER, CENTER_QUERY, RAW, Evaluator
from core.metrics import diversity, evaluate, kpop_share_top10
from core.report import Report
from core.systems import centered_space

MU_SAMPLE_SIZES = (30, 100, 300, 1000, 3000)
MU_DRAWS = 5


def main() -> None:
    ev = Evaluator(load_bench("service_format", "val"))
    b = ev.bench
    rep = Report("5_1_mean_removal", "5-1. 평균 제거")
    rep.add(f"service_format val {len(b.records):,}개 (cold 정답 있는 검색어 {sum(bool(c) for c in b.cold_pos):,}개), "
            f"train {len(b.Qt_raw):,}개로 μ_q. val 정답 곡의 kpop 비율 **{b.label_kpop:.1%}**\n")

    systems = (RAW, CENTER_QUERY, CENTER)
    res = {s.label: ev.measure(s) for s in systems}
    rep.add("## val")
    rep.table(res)

    _, Qe_raw = load_evaltxt_queries()
    rep.add(f"## eval.txt 사용자 말투 요청 {len(Qe_raw)}개 (라벨 없음 → 쏠림 지표만)")
    ev_res = {}
    for s in systems:
        sp = ev.space(s, Q_raw=Qe_raw)
        ev_res[s.label] = {**diversity(sp.Q, sp.S, b.melon_rows), "kpop": kpop_share_top10(sp.Q, sp.S, b.is_kpop)}
    rep.table(ev_res)

    rep.add("## 판정 (원래 → 평균 제거 검색어만)")
    a, c = RAW.label, CENTER_QUERY.label
    ok_val = res[c]["overlap"] < res[a]["overlap"] and res[c]["hub_top1"] < res[a]["hub_top1"]
    ok_ev = ev_res[c]["overlap"] < ev_res[a]["overlap"] and ev_res[c]["hub_top1"] < ev_res[a]["hub_top1"]
    rep.add(f"- val: 겹침 {res[a]['overlap']:.3f} → {res[c]['overlap']:.3f}, 허브 상위 1% "
            f"{res[a]['hub_top1']:.1%} → {res[c]['hub_top1']:.1%} → {'통과' if ok_val else '실패'}")
    rep.add(f"- eval.txt: 겹침 {ev_res[a]['overlap']:.3f} → {ev_res[c]['overlap']:.3f}, 허브 상위 1% "
            f"{ev_res[a]['hub_top1']:.1%} → {ev_res[c]['hub_top1']:.1%} → {'통과' if ok_ev else '실패'}")
    rep.add(f"- **{'평균 제거 채택' if ok_val and ok_ev else '평균 제거 채택 안 함'}**\n")

    rep.add("## 서술 (paired bootstrap 95%)")
    for x, y in ((RAW, CENTER_QUERY), (CENTER_QUERY, CENTER)):
        for key in ("R@10", "cold_R@10"):
            rep.ci(f"{y.label} − {x.label}", res[x.label], res[y.label], key)
    rep.ci(f"{CENTER.label} − {RAW.label}", res[RAW.label], res[CENTER.label], "sdd_R@10", digits=2)

    rep.add(f"\n## μ_q를 train 검색어 몇 개로 내야 하나 (무작위 {MU_DRAWS}번 평균 ± 표준편차, 평균 제거 검색어만)")
    rep.add("| train 검색어 수 | R@10 | 겹침 | kpop |\n|---|---|---|---|")
    rng = np.random.default_rng(0)
    S0 = l2norm(b.S_raw)
    for n in (*MU_SAMPLE_SIZES, len(b.Qt_raw)):
        vals = []
        for _ in range(MU_DRAWS if n < len(b.Qt_raw) else 1):
            mu = l2norm(b.Qt_raw[rng.choice(len(b.Qt_raw), n, replace=False)]).mean(axis=0)
            sp = centered_space(b.Q_raw, b.S_raw, mu, None)
            vals.append((evaluate(sp.Q, S0, b.all_pos)["R@10"], diversity(sp.Q, S0, b.melon_rows)["overlap"],
                         kpop_share_top10(sp.Q, S0, b.is_kpop)))
        v = np.array(vals)
        rep.add(f"| {n:,} | {v[:, 0].mean():.4f} ± {v[:, 0].std():.4f} | {v[:, 1].mean():.3f} ± {v[:, 1].std():.3f} "
                f"| {v[:, 2].mean():.1%} |")
    rep.save()


if __name__ == "__main__":
    main()
