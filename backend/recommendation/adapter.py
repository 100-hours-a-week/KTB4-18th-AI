"""검색어·곡 임베딩을 v4 검색 공간(쿼리·곡 어댑터 + 허브 보정)으로 바꾼다.

검색어 (서버, 요청마다)
  q_c = normalize(normalize(q) − μ_q)
  q_a = normalize(q_c + q_c·A_q·B_q)        (쿼리 어댑터)
  q′  = normalize(α·q_c + (1 − α)·q_a)
  반환 [q′, 1]  (3073차원)

곡 (배치, tracks.emb_v4를 채울 때)
  s′    = normalize(normalize(s) − μ_s)
  s″    = normalize(s′ + s′·A_s·B_s)        (곡 어댑터)
  s_mix = normalize(γ·s′ + (1 − γ)·s″)
  r(s)  = s_mix와 가장 가까운 train 검색어 TOP_K개의 평균 cos
  반환 [s_mix, −β·r(s)]  (3073차원)

둘을 내적하면 q′·s_mix − β·r(s)가 된다. 검색어 끝의 1은 곡 벡터 마지막 칸의 허브 패널티를
내적에 그대로 통과시키는 자리다.
가중치 출처는 assets/*.npz의 meta (실험 폴더 emb_test/adapter, scripts/build_adapter_bundle.py).
"""

import os
from functools import cache
from pathlib import Path

import numpy as np

ASSET = Path(__file__).parent / "assets" / "query_adapter_v4.npz"
SONG_ASSET = Path(__file__).parent / "assets" / "song_adapter_v4.npz"
SPACES = ("gemini", "v4")
TOP_K = 10   # r(s): 가까운 train 검색어 몇 개의 평균인지 (실험 hub_scores와 같게)


def search_space() -> str:
    """SEARCH_EMBEDDING 설정: gemini(기본, 기존 emb_gemini 코사인) 또는 v4(emb_v4 내적)."""

    space = os.getenv("SEARCH_EMBEDDING", "").strip().lower() or "gemini"
    # NOTE: 오타("v 4" 등)를 조용히 gemini로 돌리면 켠 줄 알고 모르는 채 지나가므로 실패시킨다.
    if space not in SPACES:
        raise ValueError(f"SEARCH_EMBEDDING은 {SPACES} 중 하나여야 한다: {space!r}")
    return space


def _l2norm(x: np.ndarray) -> np.ndarray:
    """벡터 하나(3072,)든 여러 개(n, 3072)든 마지막 축 기준으로 정규화한다."""
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


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


@cache
def _song_weights() -> dict:
    """곡 어댑터 가중치. 서버는 쓰지 않고 emb_v4를 채우는 배치만 읽는다 (약 12.6MB)."""

    with np.load(SONG_ASSET) as bundle:
        return {
            "A": bundle["s_A"].astype(np.float32),
            "B": bundle["s_B"].astype(np.float32),
            "mu": bundle["mu_s"].astype(np.float32),
            "gamma": float(bundle["gamma"]),
            "beta": float(bundle["beta"]),
        }


def to_v4_songs(song_vectors: np.ndarray, ref_queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """곡 임베딩 (n, 3072) → (tracks.emb_v4 (n, 3073), r(s) (n,)).

    ref_queries: r(s)의 기준인 train 검색어 (m, 3072), 검색어와 같은 v4 공간 = normalize(α·q_c + (1−α)·q_a).
    NOTE: float32로 넘길 것. 매번 변환하면 17,802개 기준 약 220MB를 호출마다 새로 만든다.
    """

    w = _song_weights()
    s = np.asarray(song_vectors, dtype=np.float32)
    s_c = _l2norm(_l2norm(s) - w["mu"])
    s_a = _l2norm(s_c + (s_c @ w["A"]) @ w["B"])
    s_mix = _l2norm(w["gamma"] * s_c + (1 - w["gamma"]) * s_a)
    sims = s_mix @ ref_queries.T                                     # (곡, 기준 검색어)
    r = -np.partition(-sims, TOP_K, axis=1)[:, :TOP_K].mean(axis=1)
    return np.hstack([s_mix, -w["beta"] * r[:, None]]).astype(np.float32), r
