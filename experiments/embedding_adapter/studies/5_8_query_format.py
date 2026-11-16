"""
5-8. 학습 검색어 형식: 서비스 형식으로 학습 vs 멜론 원문 형식으로 학습

질문      초기 실험의 검색어는 멜론 원문 "제목 | 태그 나열"이었다. 서비스가 실제로 임베딩하는 문장(classify()의
          recommend_query)과 모양이 다르다. 같은 재생목록·정답으로 검색어 형식만 바꿔 학습하면 서비스 형식에서 나아지나
입력      query_adapter_r512 (service_format train으로 학습)
          query_adapter_r512_melon_trained (melon_format train으로 학습한 A·B + service_format의 μ_q)
            NOTE: μ_q가 다르면 어댑터 차이인지 평균 차이인지 섞인다 → 평가 형식(서비스)의 μ_q로 맞춘 사본을 쓴다.
                  사본은 --make_checkpoint로 다시 만들 수 있고, 실행할 때마다 기존 사본과 같은지 확인한다
          각자 5-5 규칙 지점, service_format val, SDD
출력      runs/{날짜}_5_8_query_format/report.md — 공통 지표 표, 멜론 형식 학습 − 서비스 형식 학습 paired bootstrap
판정 기준  서술 (참고 비교. 서비스 형식 학습은 이미 다른 실험들에서 쓰고 있었다)
결과      멜론 형식 학습 R@10 −0.0042 [−0.0074, −0.0010] → 학습 검색어 형식을 서비스에 맞추는 게 유의하게 낫다
          (test에서도 −0.0044 [−0.0086, −0.0004], 6_final_test). 원본 로그: v4_melon_ref.log

사용:
  python -m studies.5_8_query_format
  python -m studies.5_8_query_format --make_checkpoint   # μ_q 맞춘 사본을 runs/에 새로 만든다
"""
import argparse

import torch

import config
from core.data import load_bench
from core.evaluator import Evaluator, adapter
from core.report import Report
from core.safe_io import guarded_output, new_run_dir
from core.selection import query_adapter_rule_point

SERVICE, MELON, MELON_MU = "query_adapter_r512", "query_adapter_r512_melon_format", "query_adapter_r512_melon_trained"


def melon_trained_with_service_mu() -> dict:
    """멜론 형식 학습 체크포인트의 A·B는 그대로, μ_q만 서비스 형식 학습 체크포인트의 것으로"""
    melon = torch.load(config.checkpoint_path(MELON))
    svc = torch.load(config.checkpoint_path(SERVICE))
    assert torch.equal(melon["mu_s"], svc["mu_s"]), "μ_s가 다르다 (같은 카탈로그여야 한다)"
    return {**melon, "mu_q": svc["mu_q"]}


def check_existing_copy() -> None:
    """실험에 쓴 사본이 위 규칙으로 만든 것과 같은지 (A·B·μ 모두)"""
    made, used = melon_trained_with_service_mu(), torch.load(config.checkpoint_path(MELON_MU))
    for k in ("A", "B", "mu_q", "mu_s"):
        assert torch.equal(made[k], used[k]), f"{MELON_MU}의 {k}가 만드는 규칙과 다르다"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make_checkpoint", action="store_true", help="μ_q 맞춘 사본을 runs/에 만든다")
    args = ap.parse_args()
    if args.make_checkpoint:
        out = guarded_output(new_run_dir("5_8_checkpoint") / f"{config.CHECKPOINTS[MELON_MU]}.pt")
        torch.save({**melon_trained_with_service_mu(), "note": f"{MELON}의 A·B + {SERVICE}의 μ_q"}, out)
        print(f"저장: {out}")
        return

    check_existing_copy()
    ev = Evaluator(load_bench("service_format", "val"))
    rep = Report("5_8_query_format", "5-8. 학습 검색어 형식")
    rep.add(f"평가: service_format val {len(ev.bench.records):,}개. 두 어댑터 모두 같은 μ_q(서비스 형식 train 평균)\n")

    res = {}
    for label, name in (("서비스 형식으로 학습", SERVICE), ("멜론 원문 형식으로 학습", MELON_MU)):
        a, b, n = query_adapter_rule_point(ev, name)
        rep.add(f"- {label} ({name}): 규칙 지점 α{a} β{b} (허용 {n}개 조합)")
        res[label] = ev.measure(adapter("", name, a, b))
    rep.add()
    rep.table(res)

    (_, svc), (_, mel) = res.items()
    rep.add("## 멜론 형식 학습 − 서비스 형식 학습")
    rep.ci("멜론 − 서비스", svc, mel, "R@10")
    rep.ci("멜론 − 서비스", svc, mel, "cold_R@10")
    rep.save()


if __name__ == "__main__":
    main()
