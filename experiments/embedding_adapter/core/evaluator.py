"""
시스템을 이름으로 정의하고(Spec), 같은 지표 묶음으로 채점한다(Evaluator)

  Spec("원래 공간", "raw")
  Spec("평균 제거", "center")                    검색어·곡 둘 다 평균 제거
  Spec("평균 제거 쿼리만", "center_query")
  Spec("쿼리 어댑터 r512", "adapter", "query_adapter_r512", alpha=0.25, beta=0.5)
  Spec("쿼리·곡 어댑터", "adapter", "two_tower_lam05", alpha=0.5, beta=0.5, gamma=0.25)

Evaluator.measure(spec) = evaluate_all(카탈로그) + SDD valid R@10. 모든 비교 스크립트가 이것을 쓴다
"""
from dataclasses import dataclass

import numpy as np

from core import sdd as sddlib
from core.adapters import load_checkpoint
from core.data import Bench
from core.metrics import evaluate_all
from core.systems import AdapterSystem, Space, centered_space, means, raw_space


@dataclass(frozen=True)
class Spec:
    label: str
    kind: str                      # raw | center | center_query | adapter
    checkpoint: str | None = None  # config.CHECKPOINTS의 이름
    alpha: float | None = None
    beta: float | None = None
    gamma: float | None = None     # 곡 어댑터가 있을 때만. None = 곡 어댑터 100%

    def at(self, alpha: float, beta: float, gamma: float | None = None, label: str | None = None) -> "Spec":
        """같은 체크포인트, 다른 지점"""
        return Spec(label or self.label, self.kind, self.checkpoint, alpha, beta, gamma)


def adapter(label: str, checkpoint: str, alpha: float, beta: float, gamma: float | None = None) -> Spec:
    return Spec(label, "adapter", checkpoint, alpha, beta, gamma)


RAW = Spec("원래 공간 (지금 서비스)", "raw")
CENTER = Spec("평균 제거 (검색어·곡 둘 다)", "center")
CENTER_QUERY = Spec("평균 제거 (검색어만)", "center_query")


class Evaluator:
    """bench(카탈로그·검색어)와 SDD에 시스템을 만들어 채점한다. 어댑터 출력은 체크포인트마다 한 번만 계산"""

    def __init__(self, bench: Bench, with_sdd: bool = True):
        self.bench = bench
        self.mu_q, self.mu_s = means(bench.Qt_raw, bench.S_raw)
        self.sdd = sddlib.load_sdd() if with_sdd else None
        self._systems = {}

    def adapter_system(self, checkpoint: str, on: str = "catalog") -> AdapterSystem:
        key = (checkpoint, on)
        if key not in self._systems:
            ck = load_checkpoint(checkpoint)
            Q_raw, S_raw = (self.bench.Q_raw, self.bench.S_raw) if on == "catalog" else (self.sdd.text, self.sdd.audio)
            self._systems[key] = AdapterSystem(ck, Q_raw, S_raw, self.bench.Qt_raw)
        return self._systems[key]

    def space(self, spec: Spec, on: str = "catalog", Q_raw: np.ndarray | None = None) -> Space:
        """on = catalog (평가 split 검색어 × 카탈로그) | sdd (캡션 × SDD 곡). Q_raw를 주면 그 검색어로 (라벨 없는 문장 등)"""
        if Q_raw is None:
            Q_raw = self.bench.Q_raw if on == "catalog" else self.sdd.text
        S_raw = self.bench.S_raw if on == "catalog" else self.sdd.audio
        if spec.kind == "raw":
            return raw_space(Q_raw, S_raw)
        if spec.kind == "center":
            return centered_space(Q_raw, S_raw, self.mu_q, self.mu_s)
        if spec.kind == "center_query":
            return centered_space(Q_raw, S_raw, self.mu_q, None)
        assert Q_raw is self.bench.Q_raw or on == "sdd", "어댑터 시스템에 다른 검색어를 쓰려면 AdapterSystem을 직접 만든다"
        return self.adapter_system(spec.checkpoint, on).space(spec.alpha, spec.beta, spec.gamma)

    def measure(self, spec: Spec, with_cohesion: bool = True) -> dict:
        m = evaluate_all(self.space(spec), self.bench, with_cohesion)
        if self.sdd is not None:
            s = sddlib.score(self.space(spec, "sdd"), self.sdd)
            m["sdd_R@10"] = s["R@10"]
            m["per_query"]["sdd_R@10"] = s["per_query"]
        return m
