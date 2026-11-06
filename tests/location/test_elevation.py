"""고도 파일(assets/korea_9s.npy)에서 주변 기복을 읽는 규칙을 검증한다."""

from backend.location import elevation


def test_relief_matches_open_meteo_judgement():
    """만들 때 Open-Meteo 값과 비교한 곳: 남한산성은 산(기복 250m 이상), 잠실대교는 평지"""
    mountain = elevation.relief(37.47904, 127.18126)
    flat = elevation.relief(37.528, 127.089)
    assert mountain["relief_m"] >= 250
    assert flat["relief_m"] < 50
    assert mountain["relief_m"] == mountain["max_m"] - mountain["min_m"]


def test_outside_grid_is_none():
    """격자 밖(도쿄)은 음수·넘친 칸을 읽지 않고 None이다"""
    assert elevation.relief(35.68, 139.69) is None
    assert elevation.relief(39.0, 127.0) is None
