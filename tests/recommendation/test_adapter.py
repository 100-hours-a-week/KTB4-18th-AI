"""v4 검색어 변환이 실험 코드(torch LowRankAdapter)와 같은 값을 내고, SEARCH_EMBEDDING 설정을 지키는지 검증한다."""

from pathlib import Path

import numpy as np
import pytest

from backend.recommendation import adapter

# NOTE: 실험 폴더(emb_test/adapter)에서 torch로 계산한 test 검색어 3개의 기대 q′
CASES = Path(__file__).parent / "samples" / "v4_query_cases.npz"
# NOTE: 같은 방식(pairwise_export.py, γ0)으로 계산한 곡 3개의 기대 emb_v4. 기준 검색어는 20개만 잘라 넣었다
SONG_CASES = Path(__file__).parent / "samples" / "v4_song_cases.npz"


def test_to_v4_query_matches_experiment():
    cases = np.load(CASES)
    for raw, expected in zip(cases["raw"], cases["expected"]):
        out = adapter.to_v4_query(raw.tolist())

        assert out.shape == (3073,)
        assert out[-1] == 1.0  # 곡 벡터 끝의 −β·r(s)를 통과시키는 자리
        np.testing.assert_allclose(out[:3072], expected, atol=1e-6)


def test_to_v4_songs_matches_experiment():
    cases = np.load(SONG_CASES)
    emb, r = adapter.to_v4_songs(cases["raw"], cases["ref"])

    assert emb.shape == (3, 3073)
    np.testing.assert_allclose(r, cases["r"], atol=1e-6)
    np.testing.assert_allclose(emb, cases["expected"], atol=1e-6)


def test_v4_query_and_song_dot_is_score_minus_hub_penalty():
    """[q′, 1]·[s_mix, −β·r(s)] = q′·s_mix − β·r(s): 끝의 1이 패널티를 그대로 통과시킨다."""
    q = adapter.to_v4_query(np.load(CASES)["raw"][0].tolist())
    cases = np.load(SONG_CASES)
    emb, r = adapter.to_v4_songs(cases["raw"], cases["ref"])
    np.testing.assert_allclose(emb @ q, emb[:, :-1] @ q[:-1] - 0.5 * r, atol=1e-6)


@pytest.mark.parametrize("value, space",[(None, "gemini"), ("", "gemini"), ("gemini", "gemini"),
                                          ("v4", "v4"), (" V4 ", "v4")])
def test_search_space_default_is_gemini(monkeypatch, value, space):
    if value is None:
        monkeypatch.delenv("SEARCH_EMBEDDING", raising=False)
    else:
        monkeypatch.setenv("SEARCH_EMBEDDING", value)
    assert adapter.search_space() == space


def test_search_space_rejects_typo(monkeypatch):
    monkeypatch.setenv("SEARCH_EMBEDDING", "v 4")
    with pytest.raises(ValueError):
        adapter.search_space()
