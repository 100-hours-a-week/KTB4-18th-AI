"""
5-9a. 두 타워: 곡 어댑터를 같이 학습하면 쿼리 어댑터만보다 나은가 (곡-곡 손실 λ)

질문      쿼리 어댑터는 곡끼리의 거리를 바꾸지 못한다. 같은 재생목록 곡들이 공간에 흩어져 있으면(응집도 0.01)
          검색어 하나로는 다 모을 수 없다 → 곡 쪽에도 어댑터 s″ = norm(s′ + s′·A_s·B_s)를 붙여 같이 학습 (문서 4-6)
            L = L_text(검색어 → 곡) + λ · L_song(같은 재생목록 곡끼리: 정답 절반의 평균 → 나머지 절반)
입력      쿼리 어댑터 query_adapter_r512 (5-5 규칙 지점)
          두 타워 4개 변형 (train/train_two_tower.py): λ ∈ {0, 0.5} × 쿼리 어댑터 시작 {0, r512 학습된 것}
            각 변형의 지점 = 학습 스크립트가 끝에 같은 격자·규칙으로 고른 지점 (아래 TOWERS, 원본 stage14_v4_*.log)
            NOTE: 학습 스크립트의 격자는 r(s)를 학습에 쓴 train 검색어(16,939개)로 계산해 여기 값과 끝자리가 조금 다르다
          service_format val, SDD
출력      runs/{날짜}_5_9a_two_tower/report.md
          공통 지표 표 (각자 지점 + 쿼리 어댑터와 같은 지점 α0.25 β0.5), 각 − 쿼리 어댑터 paired bootstrap
판정 기준  (2026-10-17, 결과 보기 전 고정) 사용자 설계(λ0, 0에서 시작) − 쿼리 어댑터 R@10 구간 하한 > 0 → 두 타워 채택
          λ0.5·r512에서 시작은 참고 변형. cold·겹침·응집도는 서술
결과      사용자 설계는 손해 −0.0051 [−0.0092, −0.0015]: λ = 0이면 곡 어댑터가 "같이 듣는 곡 모으기"가 아니라
          "인기곡 끌어내기"를 배운다. 참고 변형 λ0.5는 +0.0102 [+0.0061, +0.0142], 같은 겹침, 응집도 0.0097 → 0.134.
          cold는 어떤 변형도 개선 없음 → test 주 비교로 넘겼다 (원본 로그: v4_twotower_compare.log)

사용:
  python -m studies.5_9a_two_tower
"""
from core.data import load_bench
from core.evaluator import Evaluator, adapter
from core.report import Report
from core.selection import query_adapter_rule_point

# (설명, 체크포인트, 학습 스크립트가 고른 α, β)
TOWERS = (("λ0, 0에서 시작 (사용자 설계)", "two_tower_lam0", 0.0, 0.75),
          ("λ0, 쿼리 어댑터 r512에서 시작", "two_tower_lam0_init", 0.0, 0.75),
          ("λ0.5, 0에서 시작", "two_tower_lam05", 0.5, 0.5),
          ("λ0.5, 쿼리 어댑터 r512에서 시작", "two_tower_lam05_init", 0.5, 0.5))


def main() -> None:
    ev = Evaluator(load_bench("service_format", "val"))
    rep = Report("5_9a_two_tower", "5-9a. 두 타워 (쿼리 어댑터 + 곡 어댑터)")
    rep.add(f"service_format val {len(ev.bench.records):,}개. 곡 어댑터 100% 적용 (곡 쪽 섞기 γ는 5-9b)\n")

    a0, b0, _ = query_adapter_rule_point(ev, "query_adapter_r512")
    base_label = f"쿼리 어댑터 r512 α{a0} β{b0}"
    rows = {base_label: ev.measure(adapter("", "query_adapter_r512", a0, b0))}
    for label, name, a, b in TOWERS:
        rows[f"{label} α{a} β{b}"] = ev.measure(adapter("", name, a, b))
        if (a, b) != (a0, b0):
            rows[f"  └ {label}, 쿼리 어댑터 지점 α{a0} β{b0}"] = ev.measure(adapter("", name, a0, b0))
    rep.table(rows)

    rep.add(f"## 각 − {base_label}")
    base = rows[base_label]
    verdict = None
    for label, m in list(rows.items())[1:]:
        _, lo, _ = rep.ci(label.strip(), base, m, "R@10")
        rep.ci(label.strip(), base, m, "cold_R@10")
        if verdict is None:
            verdict = lo > 0   # 첫 변형 = 사용자 설계의 자기 지점
    rep.add(f"\n판정 (사전 고정, 사용자 설계 기준): **{'두 타워 채택' if verdict else '사용자 설계는 쿼리 어댑터보다 낫지 않다'}**. "
            "나머지 변형은 참고 — λ0.5의 결과는 test 주 비교(6_final_test)로 넘긴다")
    rep.save()


if __name__ == "__main__":
    main()
