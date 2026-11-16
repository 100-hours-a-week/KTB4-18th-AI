"""
학습하는 것: Low-Rank Residual 어댑터와 multi-positive InfoNCE 손실 (문서 4-2, 4-4)

체크포인트 형식 (원래 실험 그대로)
  쿼리 어댑터    {"A", "B", "mu_q", "mu_s", "config"}                 config["rank"], config.get("act")
  쿼리·곡 어댑터  {"q_A", "q_B", "s_A", "s_B", "mu_q", "mu_s", "config"}  config["q_rank"], ["s_rank"], ["song_act"]
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

import config


class LowRankAdapter(torch.nn.Module):
    """x → normalize(x + (x·A)·B)   (= normalize(x·W), W = I + A·B 를 d×d 행렬 없이)

    A는 작은 무작위, B는 0으로 시작 → 처음엔 W = I (아무 변환도 안 함). LoRA와 같은 초기화.
    act="gelu": x → normalize(x + GELU(x·A)·B). 파라미터 수는 같고 비선형만 다르다
      NOTE: x·A가 ±0.02 수준이면 GELU가 사실상 직선이다. A를 표준편차 1로 시작해 휘는 구간에 둔다
    """

    def __init__(self, dim: int, rank: int, act: str | None = None):
        super().__init__()
        self.act = act
        scale = 1.0 if act == "gelu" else dim ** -0.5
        self.A = torch.nn.Parameter(torch.randn(dim, rank) * scale)
        self.B = torch.nn.Parameter(torch.zeros(rank, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x @ self.A
        if self.act == "gelu":
            h = F.gelu(h)
        return F.normalize(x + h @ self.B, dim=-1)

    def apply_np(self, x: np.ndarray) -> np.ndarray:
        """numpy 행렬에 씌운다 (평가용, 기울기 없음, CPU float32)"""
        with torch.no_grad():
            return self(torch.from_numpy(x)).numpy()


def multi_positive_infonce(scores: torch.Tensor, pos_mask: torch.Tensor, tau: float,
                           bias: torch.Tensor | None = None, pos_weight: torch.Tensor | None = None,
                           neg_exclude: torch.Tensor | None = None) -> torch.Tensor:
    """scores (검색어, 곡) 코사인, pos_mask (검색어, 곡) 정답이면 True → 평균 손실

    정답 곡 하나마다 "이 곡을 고르시오" 문제를 따로 낸다.
      보기 = 그 정답 곡 + 모든 오답 곡 (같은 검색어의 다른 정답은 보기에서 뺀다)
      손실 = −log(그 정답을 고를 softmax 확률), 정답 곡들에 대해 평균
    다른 정답을 보기에 넣으면 "A를 고르려면 C를 낮춰라"가 되어 정답끼리 서로 밀어낸다.

    bias (곡,): 학습할 때만 logit에 더하는 곡별 보너스 (인기도 보정 logQ, 5-3)
    pos_weight (곡,): 정답마다 손실에 곱할 가중치 (1/빈도^γ). 주면 배치 전체 정답의 가중 평균
    neg_exclude (곡,) bool: 오답으로는 쓰지 않을 곡 (5-7 C: 비인기곡 = 정답 전용)
    """
    logits = scores / tau
    if bias is not None:
        logits = logits + bias

    # 오답들만의 logsumexp: 정답(과 오답 제외 곡)을 −∞로 가린다
    not_neg = pos_mask if neg_exclude is None else pos_mask | neg_exclude[None, :]
    neg_lse = torch.logsumexp(logits.masked_fill(not_neg, float("-inf")), dim=1, keepdim=True)

    # 곡마다 "자기 자신 + 오답들" 중 그 곡을 고를 log 확률. 분모에 정답은 자기 하나뿐 → 다른 정답은 가려진 셈
    log_prob = logits - torch.logaddexp(logits, neg_lse)

    if pos_weight is not None:
        weights = pos_mask * pos_weight
        return -(log_prob * weights).sum() / weights.sum()

    per_query = -(log_prob * pos_mask).sum(dim=1) / pos_mask.sum(dim=1)
    return per_query.mean()


@dataclass
class Checkpoint:
    name: str
    query: LowRankAdapter
    song: LowRankAdapter | None    # 쿼리 어댑터만이면 None
    mu_q: np.ndarray
    mu_s: np.ndarray
    config: dict


def load_checkpoint(name: str) -> Checkpoint:
    """config.CHECKPOINTS의 이름으로 불러온다 (원래 .pt는 읽기만)"""
    ck = torch.load(config.checkpoint_path(name))
    c = ck["config"]
    if "q_A" in ck:
        qa = LowRankAdapter(config.DIM, c["q_rank"])
        sa = LowRankAdapter(config.DIM, c["s_rank"], c["song_act"])
        qa.load_state_dict({"A": ck["q_A"], "B": ck["q_B"]})
        sa.load_state_dict({"A": ck["s_A"], "B": ck["s_B"]})
    else:
        qa, sa = LowRankAdapter(config.DIM, c["rank"], c.get("act")), None   # act 없음 = 선형 (이전 체크포인트)
        qa.load_state_dict({"A": ck["A"], "B": ck["B"]})
    return Checkpoint(name, qa, sa, ck["mu_q"].numpy(), ck["mu_s"].numpy(), c)
