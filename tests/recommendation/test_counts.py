import pytest

from backend.recommendation.counts import track_limit


@pytest.mark.parametrize("count,expected", [(None, 5), (1, 1), (2, 2), (5, 5), (6, 5), (10, 5), (10000, 5)])
def test_track_limit(count, expected):
    assert track_limit(count) == expected


@pytest.mark.parametrize("count", [0, -2, True, "2", 2.5])
def test_track_limit_rejects_invalid_counts(count):
    with pytest.raises(ValueError):
        track_limit(count)
