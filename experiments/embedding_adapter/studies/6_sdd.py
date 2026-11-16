"""
6. SDD 회귀 테스트: 최종 후보들이 범용 음악 검색(영어 캡션 → 곡)을 망가뜨리지 않았나 (비용 0, 저장된 임베딩)

질문      어댑터는 멜론 재생목록에 맞춰 학습했다. 멜론과 무관한 곡·문장에서도 원래 공간보다 나빠지지 않았나
입력      SDD 캡션 1,106개 → 곡 706개 (valid: 746개 → 547곡), 시스템은 서비스에 넣을 그대로
            μ_q = service_format train 평균, μ_s = 카탈로그 평균, 허브 벌점 r(s) = SDD 곡마다 service_format train 검색어로
          β0 행: 허브 벌점 없이 (영어 캡션은 멜론 train 검색어와 언어가 달라 r(s)가 엉뚱할 수 있어 참고로)
출력      runs/{날짜}_6_sdd/report.md — valid R@10·R@1·MedR, full R@10·MedR, 원래 공간 대비 paired bootstrap
판정 기준  서술 (원래 공간의 v1 기록 valid 35.92와 같은지로 채점 방식 확인)
결과      평균 제거 38.47, 쿼리 어댑터 36.33, 쿼리·곡 어댑터(곡 어댑터 100%) 31.37 = 원래 대비 −4.56 [−7.91, −1.21]
          → 처음 보는 곡을 흐트러뜨린다. 5-9b의 곡 쪽 섞기 γ0.25로 36.19 (원본 로그: v4_sdd.log, v4_song_mix.log)

사용:
  python -m studies.6_sdd
"""
from core import sdd as sddlib
from core.data import load_bench
from core.evaluator import CENTER, RAW, Evaluator, adapter
from core.metrics import paired_bootstrap
from core.report import Report

SYSTEMS = (RAW, CENTER,
           adapter("쿼리 어댑터 r512 α0.25 β0.5", "query_adapter_r512", 0.25, 0.5),
           adapter("(참고) 쿼리 어댑터 r512 α0.25 β0", "query_adapter_r512", 0.25, 0.0),
           adapter("쿼리·곡 어댑터 λ0.5 α0.5 β0.5 (곡 어댑터 100%)", "two_tower_lam05", 0.5, 0.5),
           adapter("(참고) 쿼리·곡 어댑터 λ0.5 α0.5 β0", "two_tower_lam05", 0.5, 0.0),
           adapter("쿼리·곡 어댑터 λ0.5 α0.5 β0.5 γ0.25 (최종)", "two_tower_lam05", 0.5, 0.5, 0.25))


def main() -> None:
    ev = Evaluator(load_bench("service_format", "val"))   # train 검색어(μ_q, r(s))만 쓴다. val은 채점하지 않는다
    rep = Report("6_sdd", "6. SDD 회귀 테스트 (영어 캡션 → 곡, 멜론·학습과 무관)")
    rep.add("v1 기록: 원래 35.92 / 평균 제거 39.01 / 어댑터 38.87 (valid R@10)\n")
    rep.add("| 시스템 | valid R@10 | R@1 | MedR | full R@10 | full MedR | valid R@10 − 원래 |\n|---|---|---|---|---|---|---|")
    base = None
    for spec in SYSTEMS:
        space = ev.space(spec, "sdd")
        v, f = sddlib.score(space, ev.sdd, "valid"), sddlib.score(space, ev.sdd, "full")
        base = base or v
        d, lo, hi = paired_bootstrap(base["per_query"], v["per_query"])
        ci = "—" if spec is RAW else f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"
        rep.add(f"| {spec.label} | {v['R@10']:.2f} | {v['R@1']:.2f} | {v['MedR']:.0f} | {f['R@10']:.2f} | {f['MedR']:.0f} | {ci} |")
    rep.save()


if __name__ == "__main__":
    main()
