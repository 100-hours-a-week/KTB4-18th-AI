"""검색어 임베딩을 v4 검색 공간(쿼리 어댑터 + 허브 보정)으로 바꾼다.

  q_c = normalize(normalize(q) − μ_q)
  q_a = normalize(q_c + q_c·A·B)            (쿼리 어댑터)
  q′  = normalize(α·q_c + (1 − α)·q_a)
  반환 [q′, 1]  (3073차원)

곡 쪽 tracks.emb_v4 = [s_mix, −β·r(s)]와 내적하면 q′·s_mix − β·r(s)가 된다.
끝의 1은 곡 벡터 마지막 칸의 허브 패널티를 내적에 그대로 통과시키는 자리다.
가중치 출처는 assets/query_adapter_v4.npz의 meta (실험 폴더 emb_test/adapter, scripts/build_adapter_bundle.py).
"""

import os
from functools import cache
from pathlib import Path

import numpy as np

ASSET = Path(__file__).parent / "assets" / "query_adapter_v4.npz"
SPACES = ("gemini", "v4")


def search_space() -> str:
    """SEARCH_EMBEDDING 설정: gemini(기본, 기존 emb_gemini 코사인) 또는 v4(emb_v4 내적)."""

    space = os.getenv("SEARCH_EMBEDDING", "").strip().lower() or "gemini"
    # NOTE: 오타("v 4" 등)를 조용히 gemini로 돌리면 켠 줄 알고 모르는 채 지나가므로 실패시킨다.
    if space not in SPACES:
        raise ValueError(f"SEARCH_EMBEDDING은 {SPACES} 중 하나여야 한다: {space!r}")
    return space


def _l2norm(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


@cache
def _weights() -> dict:
    """가중치를 프로세스당 한 번만 읽는다 (약 12.6MB)."""

    with np.load(ASSET) as bundle:
        return {
            "A": bundle["q_A"].astype(np.float32),
            "B": bundle["q_B"].astype(np.float32),
            "mu": bundle["mu_q"].astype(np.float32),
            "alpha": float(bundle["alpha"]),
        }


def to_v4_query(query_vector: list[float]) -> np.ndarray:
    """Gemini 검색어 임베딩(3072) → v4 검색어 벡터 [q′, 1] (3073)."""

    w = _weights()
    q = np.asarray(query_vector, dtype=np.float32)
    q_c = _l2norm(_l2norm(q) - w["mu"])
    q_a = _l2norm(q_c + (q_c @ w["A"]) @ w["B"])
    q_mix = _l2norm(w["alpha"] * q_c + (1 - w["alpha"]) * q_a)
    return np.append(q_mix, np.float32(1.0))
