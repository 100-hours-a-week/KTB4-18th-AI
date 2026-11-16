"""
"시스템" = 검색어 행렬과 곡 행렬을 어떤 공간에 놓고 내적으로 순위를 매기는가 (문서 (4))

  원래 공간         q·s                                          지금 서비스
  평균 제거         q_c = norm(norm(q) − μ_q),  s′ = norm(norm(s) − μ_s)            5-1
  쿼리 어댑터       q′ = norm(α·q_c + (1−α)·q_a),  q_a = 쿼리어댑터(q_c)            4-2, 5-2
  쿼리·곡 어댑터    + 곡 쪽 s″ = 곡어댑터(s′),  s_mix = norm(γ·s′ + (1−γ)·s″)       4-6, 5-9
  허브 벌점         점수 = q′·s − β·r(s),  r(s) = s와 가장 가까운 train 검색어 k개의 평균 코사인   5-5
                   r(s)의 train 검색어도 같은 공간(q′와 같은 α)으로 바꿔서 잰다. 라벨은 쓰지 않는다

벌점은 곡 벡터 끝에 −β·r(s), 검색어 끝에 1을 붙여 내적 하나로 계산한다 (서비스 DB에 그대로 넣는 형태)
"""
from dataclasses import dataclass

import numpy as np

import config
from core.adapters import Checkpoint
from core.data import l2norm

HUB_CHUNK = 2000   # 곡 × train 검색어 유사도를 몇 곡씩 나눠 계산할지 (메모리)


@dataclass
class Space:
    """채점할 공간. Q·S.T가 점수. songs = 벌점 칸을 붙이기 전 곡 벡터 (응집도처럼 곡끼리 볼 때)"""
    Q: np.ndarray
    S: np.ndarray
    songs: np.ndarray


def hub_scores(S: np.ndarray, Q_ref: np.ndarray, k: int = config.HUB_K) -> np.ndarray:
    """r(s) = 곡 s와 가장 가까운 기준 검색어 k개의 평균 코사인 (곡 수,)"""
    r = np.empty(len(S), dtype=np.float32)
    for start in range(0, len(S), HUB_CHUNK):
        sims = S[start:start + HUB_CHUNK] @ Q_ref.T
        r[start:start + HUB_CHUNK] = -np.partition(-sims, k, axis=1)[:, :k].mean(axis=1)
    return r


def with_hub_penalty(S: np.ndarray, Q: np.ndarray, r: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """곡 끝에 −β·r(s), 검색어 끝에 1 → 내적 = q·s − β·r(s)

    NOTE: 검색어는 L2 정규화해서 넘길 것. 정규화 안 된 α 혼합이면 q·s의 배율이 검색어마다 달라
    β·r(s)와의 비율이 검색어마다 바뀐다
    """
    S_aug = np.hstack([S, -beta * r[:, None]]).astype(np.float32)
    Q_aug = np.hstack([Q, np.ones((len(Q), 1), dtype=np.float32)])
    return S_aug, Q_aug


def means(Qt_raw: np.ndarray, S_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(μ_q, μ_s): train 검색어 평균, 카탈로그 곡 평균 (각각 L2 정규화 후 평균). 평가 검색어는 쓰지 않는다"""
    return l2norm(Qt_raw).mean(axis=0), l2norm(S_raw).mean(axis=0)


def center(X_raw: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return l2norm(l2norm(X_raw) - mu)


def raw_space(Q_raw: np.ndarray, S_raw: np.ndarray) -> Space:
    """원래 공간 (지금 서비스): 코사인 그대로"""
    S = l2norm(S_raw)
    return Space(l2norm(Q_raw), S, S)


def centered_space(Q_raw: np.ndarray, S_raw: np.ndarray, mu_q: np.ndarray, mu_s: np.ndarray | None) -> Space:
    """평균 제거. mu_s=None이면 검색어만 뺀다"""
    S = l2norm(S_raw) if mu_s is None else center(S_raw, mu_s)
    return Space(center(Q_raw, mu_q), S, S)


class AdapterSystem:
    """체크포인트 하나를 검색어·곡에 씌워 두고 α, β, γ를 바꿔 가며 공간을 만든다

      Q_raw   채점할 검색어 (val, test, SDD 캡션 …)
      S_raw   채점할 곡 (카탈로그, SDD 곡 …)
      Qt_raw  허브 벌점 기준 train 검색어 (언제나 데이터셋의 train 전부)
    μ_q, μ_s는 체크포인트에 저장된 값 (= train 검색어 평균, 카탈로그 평균)
    """

    def __init__(self, ck: Checkpoint, Q_raw: np.ndarray, S_raw: np.ndarray, Qt_raw: np.ndarray):
        self.ck = ck
        self.S_c = center(S_raw, ck.mu_s)
        self.S_tw = ck.song.apply_np(self.S_c) if ck.song is not None else None
        self.Q_c, self.Qt_c = center(Q_raw, ck.mu_q), center(Qt_raw, ck.mu_q)
        self.Q_a, self.Qt_a = ck.query.apply_np(self.Q_c), ck.query.apply_np(self.Qt_c)
        self._hub = {}

    def songs(self, gamma: float | None) -> np.ndarray:
        """곡 벡터. 곡 어댑터가 없으면 s′. 있으면 γ=None → s″ 그대로, γ 숫자 → norm(γ·s′ + (1−γ)·s″)"""
        if self.S_tw is None:
            return self.S_c
        if gamma is None:
            return self.S_tw
        return l2norm(gamma * self.S_c + (1 - gamma) * self.S_tw)

    def space(self, alpha: float, beta: float, gamma: float | None = None) -> Space:
        S = self.songs(gamma)
        key = (alpha, gamma)
        if key not in self._hub:   # r(s)는 α·γ에만 달렸다 → β 격자에서 다시 계산하지 않는다
            self._hub[key] = hub_scores(S, l2norm(alpha * self.Qt_c + (1 - alpha) * self.Qt_a))
        S_aug, Q_aug = with_hub_penalty(S, l2norm(alpha * self.Q_c + (1 - alpha) * self.Q_a), self._hub[key], beta)
        return Space(Q_aug, S_aug, S)
