"""
α × β 격자와 선택 규칙 (문서 5-5. 2026-10-14, 결과를 보기 전에 고정)

  α와 β는 둘 다 인기곡 쏠림을 누르므로 독립이 아니다. 손잡이가 둘이면 val로 고를 때 선택 편향이 커진다
  → 규칙: 검색어 간 겹침이 평균 제거 이하인 조합 중 val R@10 최대
          = "평균 제거만큼 검색어마다 다른 곡을 주면서 정확도는 최대로"
"""
import config
from core.evaluator import Evaluator
from core.metrics import evaluate_all


def grid(ev: Evaluator, checkpoint: str, gamma: float | None = None,
         alphas=config.ALPHAS, betas=config.BETAS, extra=None) -> dict:
    """{(α, β): 지표}. r(s)는 α마다 그 공간의 train 검색어로 다시 계산한다 (k = 10)

    extra(space) → dict: 칸마다 더 볼 지표 (예: 비인기곡 지표)
    """
    system = ev.adapter_system(checkpoint)
    out = {}
    for a in alphas:
        for b in betas:
            space = system.space(a, b, gamma)
            out[a, b] = evaluate_all(space, ev.bench, with_cohesion=False)
            if extra is not None:
                out[a, b].update(extra(space))
    return out


def rule_point(results: dict, limit: float) -> tuple[tuple[float, float], list]:
    """(겹침 ≤ limit인 조합 중 R@10 최대 지점, 허용된 조합들). 동점이면 격자 순서(α, β 오름차순)에서 먼저 나온 것"""
    allowed = [k for k, v in results.items() if v["overlap"] <= limit]
    return max(allowed, key=lambda k: results[k]["R@10"]), allowed


def query_adapter_rule_point(ev: Evaluator, checkpoint: str) -> tuple[float, float, int]:
    """쿼리 어댑터의 규칙 지점 (α, β, 허용 조합 수). 기준 겹침 = 같은 격자의 α=1, β=0 칸 (= 평균 제거)"""
    results = grid(ev, checkpoint)
    (a, b), allowed = rule_point(results, results[1.0, 0.0]["overlap"])
    return a, b, len(allowed)


def grid_table(results: dict, key: str, fmt: str, alphas=config.ALPHAS, betas=config.BETAS) -> str:
    lines = ["| α \\ β | " + " | ".join(str(b) for b in betas) + " |", "|---|" + "---|" * len(betas)]
    lines += [f"| {a} | " + " | ".join(format(results[a, b][key], fmt) for b in betas) + " |" for a in alphas]
    return "\n".join(lines)
