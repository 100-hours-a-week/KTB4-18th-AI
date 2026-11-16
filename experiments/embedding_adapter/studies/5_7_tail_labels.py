"""
5-7. 비인기곡 라벨을 어떻게 쓸까: A (정답이자 오답) vs C (정답으로만)

질문      정답의 95%가 멜론 인기 상위 10%라, 비인기곡 3,925곡을 추가 임베딩했다 (Google Batch API, 곡당 약 $0.0024).
          비인기곡은 곡당 학습 정답이 약 6번(기존 곡 평균 43번)이라, 나머지 모든 검색어에서 오답으로 쓰이면
          통째로 밀려난다 (원래 실험 7단계 진단). 오답으로는 쓰지 않으면(C) 비인기곡을 더 잘 찾나
입력      query_adapter_r512 (A: train/train_query_adapter.py 기본)
          query_adapter_r512_tail_pos_only (C: 같은 설정 + --tail_pos_only), 각자 5-5 규칙 지점, service_format val, SDD
출력      runs/{날짜}_5_7_tail_labels/report.md — 공통 지표 표, C − A paired bootstrap (비인기곡 warm, R@10, cold)
판정 기준  (결과 보기 전 고정) C − A 비인기곡 warm R@10 구간 하한 > 0 이고 R@10 유의한 손해가 없으면 C, 아니면 A
결과      A 유지. 비인기곡 warm +0.0022 [−0.0033, +0.0083] 구분 불가, R@10 −0.0051 [−0.0070, −0.0035] 손해.
          재생목록 수가 고정이라 곡을 더 사도 곡당 라벨은 늘지 않는다 → 추가 구매 중단
          (원본 로그: v4_tail_ac.log, grid_stage4_r512_ds4_q3_tailpos_e5.log)
          라벨 없음(B, 대조군)과 A의 비교는 예전 형식 검색어에서 했다 (원본 폴더 stage7_* 기록)

사용:
  python -m studies.5_7_tail_labels
"""
from core.data import load_bench
from core.evaluator import Evaluator, adapter
from core.report import Report
from core.selection import query_adapter_rule_point

SYSTEMS = (("A: 비인기곡도 다른 곡과 똑같이", "query_adapter_r512"),
           ("C: 비인기곡 = 정답 전용", "query_adapter_r512_tail_pos_only"))


def main() -> None:
    ev = Evaluator(load_bench("service_format", "val"))
    rep = Report("5_7_tail_labels", "5-7. 비인기곡 라벨: A vs C")
    rep.add(f"service_format val {len(ev.bench.records):,}개, 카탈로그 중 비인기곡 {len(ev.bench.tail):,}곡\n")

    res = {}
    for label, name in SYSTEMS:
        a, b, n = query_adapter_rule_point(ev, name)
        rep.add(f"- {label} ({name}): 규칙 지점 α{a} β{b} (허용 {n}개 조합)")
        res[label] = ev.measure(adapter("", name, a, b))
    rep.add()
    rep.table(res)

    (la, A), (lc, C) = res.items()
    rep.add("## C − A")
    _, tail_lo, _ = rep.ci("C − A", A, C, "tail_warm_R@10")
    _, _, r_hi = rep.ci("C − A", A, C, "R@10")
    rep.ci("C − A", A, C, "cold_R@10")
    use_c = tail_lo > 0 and r_hi >= 0
    rep.add(f"\nC 채택 조건(비인기곡 warm 하한 > 0, R@10 유의한 손해 없음): {'충족' if use_c else '충족 안 됨'} "
            f"→ 판정: **{lc if use_c else la}**")
    rep.save()


if __name__ == "__main__":
    main()
