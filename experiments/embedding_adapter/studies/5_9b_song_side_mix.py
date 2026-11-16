"""
5-9b. 곡 쪽 섞기 γ: 쿼리·곡 어댑터가 처음 보는 곡을 흐트러뜨리는 것을 재학습 없이 줄일 수 있나 (비용 0)

질문      test를 연 뒤 SDD 회귀 테스트(6_sdd)에서 쿼리·곡 어댑터가 원래보다 −4.56 [−7.91, −1.21]을 보였다.
          SDD 곡은 학습에 한 번도 안 나온 곡이라 곡 어댑터에게는 전부 처음 보는 곡이다.
            s_mix = norm(γ·s′ + (1 − γ)·s″)    γ = 0 → 곡 어댑터 100%, γ = 1 → 곡 어댑터 끔
입력      두 타워 체크포인트 (기본 two_tower_lam05, --checkpoint로 λ 변형), γ ∈ {0, 0.25, 0.5, 0.75, 1},
          service_format val, SDD, 비교 기준 query_adapter_r512 α0.25 β0.5
출력      runs/{날짜}_5_9b_song_side_mix/report.md
          γ마다 규칙 지점·val R@10·SDD·안전 여부, 판정, γ별 재생목록 응집도, λ별 응집도
판정 기준  (2026-10-20, 결과 보기 전 고정)
          1. γ마다 val α × β 격자 + 기존 규칙(겹침 ≤ 평균 제거 중 R@10 최대)으로 지점을 고른다
          2. 그 지점의 SDD valid R@10 − 원래 공간의 구간 상한 > 0 인 γ만 남긴다 ("유의하게 낮지 않음")
          3. 남은 γ 중 val R@10 최대가 쿼리 어댑터보다 R@10 구간 하한 > 0 → 쿼리·곡 어댑터(그 γ), 아니면 쿼리 어댑터
          한계: SDD를 고르는 데 썼으므로 SDD는 더 이상 독립 확인이 아니라 "처음 보는 곡 안전 조건"이다
결과      γ = 0.25 (α0.5 β0.5): val R@10 0.0459, SDD 36.19 (원래 대비 +0.27 [−2.68, +3.62] 안전),
          쿼리 어댑터 대비 +0.0038 [+0.0003, +0.0073] → 쿼리·곡 어댑터 γ0.25. λ0.25도 같은 판정, λ1은 쿼리 어댑터
          (원본 로그: v4_song_mix.log, v4_song_mix_lam025.log, v4_song_mix_lam1.log, v4_song_mix_cohesion.log,
           v4_lambda_cohesion.log)

사용:
  python -m studies.5_9b_song_side_mix
  python -m studies.5_9b_song_side_mix --checkpoint two_tower_lam025 --gammas 0 0.25
"""
import argparse

import config
from core import sdd as sddlib
from core.data import load_bench
from core.evaluator import RAW, Evaluator, adapter
from core.metrics import cohesion, diversity, paired_bootstrap
from core.report import Report
from core.selection import grid, rule_point

QUERY_ADAPTER = adapter("쿼리 어댑터 r512 α0.25 β0.5", "query_adapter_r512", 0.25, 0.5)
LAMBDA_CHECKPOINTS = (("λ0.25", "two_tower_lam025"), ("λ0.5", "two_tower_lam05"), ("λ1.0", "two_tower_lam1"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="two_tower_lam05", choices=[k for k in config.CHECKPOINTS if "tower" in k])
    ap.add_argument("--gammas", type=float, nargs="+", default=list(config.GAMMAS))
    args = ap.parse_args()

    ev = Evaluator(load_bench("service_format", "val"))
    rep = Report("5_9b_song_side_mix", f"5-9b. 곡 쪽 섞기 γ: {args.checkpoint}")
    system = ev.adapter_system(args.checkpoint)
    base_overlap = diversity(system.Q_c, system.S_c, ev.bench.melon_rows)["overlap"]   # 평균 제거 (둘 다)
    A = ev.measure(QUERY_ADAPTER, with_cohesion=False)
    sdd_raw = sddlib.score(ev.space(RAW, "sdd"), ev.sdd)
    rep.add(f"기준: 평균 제거 겹침 {base_overlap:.3f}, {QUERY_ADAPTER.label} val R@10 {A['R@10']:.4f}, "
            f"SDD 원래 {sdd_raw['R@10']:.2f}\n")

    rep.add("| γ | 규칙 지점 | val R@10 | cold R@10 | 겹침 | SDD R@10 | SDD − 원래 | 처음 보는 곡 안전 |\n|---|---|---|---|---|---|---|---|")
    chosen = {}
    for g in args.gammas:
        results = grid(ev, args.checkpoint, gamma=g)
        (a, b), _ = rule_point(results, base_overlap)
        m = results[a, b]
        s = sddlib.score(ev.space(adapter("", args.checkpoint, a, b, g), "sdd"), ev.sdd)
        d, lo, hi = paired_bootstrap(sdd_raw["per_query"], s["per_query"])
        chosen[g] = (m, hi > 0)
        rep.add(f"| {g} | α{a} β{b} | {m['R@10']:.4f} | {m['cold_R@10']:.4f} | {m['overlap']:.3f} | {s['R@10']:.2f} | "
                f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}] | {'안전' if hi > 0 else '불합격'} |")

    rep.add("\n## 판정")
    ok = [g for g, (_, safe) in chosen.items() if safe]
    if not ok:
        rep.add("SDD 안전 조건을 만족하는 γ 없음 → **쿼리 어댑터**")
    else:
        g = max(ok, key=lambda x: chosen[x][0]["R@10"])
        _, lo, _ = rep.ci(f"안전한 γ 중 val R@10 최대 γ={g} − {QUERY_ADAPTER.label}", A, chosen[g][0])
        rep.add(f"→ **{'쿼리·곡 어댑터 γ=' + str(g) if lo > 0 else '쿼리 어댑터'}**")

    rep.add(f"\n## γ별 재생목록 응집도 ({args.checkpoint}, val)")
    rep.add("곡끼리만 본다 (검색어·α·β 무관). γ1 = 곡 어댑터 끔 = 쿼리 어댑터·평균 제거와 같은 곡 공간\n")
    rep.add("| γ | 응집도 warm | 응집도 cold |\n|---|---|---|")
    for g in args.gammas:
        h = cohesion(system.songs(g), ev.bench)
        rep.add(f"| {g} | {h['warm']:.4f} | {h['cold']:.4f} |")

    rep.add("\n## λ별 재생목록 응집도 (val, 곡 어댑터 100% = γ0 / 75% = γ0.25)")
    rep.add("| λ | γ | 응집도 warm | 응집도 cold |\n|---|---|---|---|")
    for lam, name in LAMBDA_CHECKPOINTS:
        for g in (0.0, 0.25):
            h = cohesion(ev.adapter_system(name).songs(g), ev.bench)
            rep.add(f"| {lam} | {g} | {h['warm']:.4f} | {h['cold']:.4f} |")
    rep.save()


if __name__ == "__main__":
    main()
