"""v4 검색어 변환이 실험 코드(torch LowRankAdapter)와 같은 값을 내고, SEARCH_EMBEDDING 설정을 지키는지 검증한다."""

from pathlib import Path

import numpy as np
import pytest

from backend.recommendation import adapter

# NOTE: 실험 폴더(emb_test/adapter)에서 torch로 계산한 test 검색어 3개의 기대 q′
CASES = Path(__file__).parent / "samples" / "v4_query_cases.npz"


def test_to_v4_query_matches_experiment():
    cases = np.load(CASES)
    for raw, expected in zip(cases["raw"], cases["expected"]):
        out = adapter.to_v4_query(raw.tolist())

        assert out.shape == (3073,)
        assert out[-1] == 1.0  # 곡 벡터 끝의 −β·r(s)를 통과시키는 자리
        np.testing.assert_allclose(out[:3072], expected, atol=1e-6)


@pytest.mark.parametrize("value, space", [(None, "gemini"), ("", "gemini"), ("gemini", "gemini"),
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
