"""
6. 최종 결과: test를 한 번만 연다

질문      val로 고른 시스템들이 설정 선택에 한 번도 쓰지 않은 test에서도 같은 순서인가
입력      service_format test 1,968개 (cold 정답 있는 검색어 1,250개), 카탈로그 19,027곡, 모든 체크포인트·지점은 val로 고정
출력      runs/{날짜}_6_final_test/report.md — 공통 지표 표(응집도 포함), 주 비교, 서술 비교, 곡 쪽 섞기 사후 확인
판정 기준  (2026-10-17, 결과 보기 전 고정)
          주 비교: 쿼리·곡 어댑터(λ0.5, α0.5 β0.5) − 쿼리 어댑터(r512, α0.25 β0.5) R@10 구간 하한 > 0 → 쿼리·곡 어댑터
          서술: 쿼리 어댑터 − 평균 제거, 원래 → 평균 제거(겹침·허브·kpop), 모든 시스템 cold·겹침·응집도
          사후 확인 (2026-10-20, 판정 아님): 5-9b에서 val로 고른 곡 쪽 섞기 γ0.25가 test에서도 같은 방향인가
결과      쿼리·곡 어댑터 R@10 0.0547 vs 0.0470, +0.0078 [+0.0036, +0.0121], 겹침 0.040 동일, cold 구분 불가.
          평균 제거 vs 원래: R@10 차이 없음, 겹침 0.246 → 0.041, kpop 79.3% → 44.8% (정답 49.5%).
          γ0.25 사후 확인: 쿼리 어댑터와 R@10 구분 불가 (−0.0001 [−0.0040, +0.0044])
          (원본 결과: final_test_v4.md, final_test_v4_posthoc.md → results/)

test는 이미 2026-10-17(주 비교)·2026-10-20(사후 확인)에 열었다. 원본 결과 파일이 있으면 실행 전에 멈춘다.
이 파일은 "test를 어떻게 평가했나"를 보여주는 기록이다.

사용:
  python -m studies.6_final_test
"""
import config
from core.data import load_bench
from core.evaluator import CENTER, CENTER_QUERY, RAW, Evaluator, adapter
from core.report import Report

PUBLISHED = (config.EMB_DATA_DIR / "final_test_v4.md", config.EMB_DATA_DIR / "final_test_v4_posthoc.md")

QUERY_ADAPTER = adapter("쿼리 어댑터 r512 α0.25 β0.5", "query_adapter_r512", 0.25, 0.5)
TWO_TOWER = adapter("쿼리·곡 어댑터 λ0.5 α0.5 β0.5", "two_tower_lam05", 0.5, 0.5)
REFERENCE = (adapter("(참고) 두 타워 사용자 설계 λ0 α0 β0.75", "two_tower_lam0", 0.0, 0.75),
             adapter("(참고) 멜론 형식으로 학습한 쿼리 어댑터 α0.25 β0.5", "query_adapter_r512_melon_trained", 0.25, 0.5))
SONG_MIX = (TWO_TOWER.at(0.5, 0.5, 0.0, "쿼리·곡 어댑터 γ0 (곡 어댑터 100%)"),
            TWO_TOWER.at(0.5, 0.5, 0.25, "쿼리·곡 어댑터 γ0.25 (곡 어댑터 75%)"))


def main() -> None:
    opened = [p.name for p in PUBLISHED if p.exists()]
    if opened:
        raise SystemExit(f"test는 이미 열었다 ({', '.join(opened)}). 다시 실행하지 않는다 — 결과는 results/")

    ev = Evaluator(load_bench("service_format", "test", allow_test=True), with_sdd=False)
    b = ev.bench
    rep = Report("6_final_test", "6. 최종 결과 (test, 한 번만)")
    rep.add(f"service_format test {len(b.records):,}개 / cold 정답 있는 검색어 {sum(bool(c) for c in b.cold_pos):,}개. "
            f"test 정답 곡 kpop {b.label_kpop:.1%}\n")

    systems = (RAW, CENTER, CENTER_QUERY, QUERY_ADAPTER, TWO_TOWER, *REFERENCE)
    R = {s.label: ev.measure(s) for s in systems}
    rep.table(R)

    rep.add("## 주 비교 (사전 고정)")
    _, lo, _ = rep.ci(f"{TWO_TOWER.label} − {QUERY_ADAPTER.label}", R[QUERY_ADAPTER.label], R[TWO_TOWER.label])
    rep.add(f"→ **{'쿼리·곡 어댑터' if lo > 0 else '쿼리 어댑터'}가 최종 후보**")
    rep.ci(f"{TWO_TOWER.label} − {QUERY_ADAPTER.label}", R[QUERY_ADAPTER.label], R[TWO_TOWER.label], "cold_R@10")

    rep.add("\n## 서술")
    for x, y in ((CENTER, QUERY_ADAPTER), (RAW, CENTER), *((QUERY_ADAPTER, r) for r in REFERENCE)):
        for key in ("R@10", "cold_R@10"):
            rep.ci(f"{y.label} − {x.label}", R[x.label], R[y.label], key)
    o, c = R[RAW.label], R[CENTER.label]
    rep.add(f"- 원래 → 평균 제거: 겹침 {o['overlap']:.3f} → {c['overlap']:.3f}, 허브 상위 1% {o['hub_top1']:.1%} → "
            f"{c['hub_top1']:.1%}, kpop {o['kpop']:.1%} → {c['kpop']:.1%}")

    rep.add("\n## 사후 확인: 곡 쪽 섞기 γ (판정 아님, γ0.25는 test를 연 뒤 val + SDD 안전 조건으로 골랐다)")
    P = {QUERY_ADAPTER.label: R[QUERY_ADAPTER.label], **{s.label: ev.measure(s, with_cohesion=False) for s in SONG_MIX}}
    rep.table(P)
    qa, g0, g25 = P.values()
    for x, y, lx, ly in ((qa, g25, "쿼리 어댑터", "γ0.25"), (g0, g25, "γ0", "γ0.25"), (qa, g0, "쿼리 어댑터", "γ0")):
        for key in ("R@10", "cold_R@10"):
            rep.ci(f"{ly} − {lx}", x, y, key)
    rep.save()


if __name__ == "__main__":
    main()
