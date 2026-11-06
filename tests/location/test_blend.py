"""장소·날씨·시간 점수를 섞어 곡을 고르는 규칙을 검증한다."""

from types import SimpleNamespace

import numpy as np
import pytest

from backend.location.blend import choose_tracks, ingredient_phrase, ladder_weights, mix, slot_phrases


@pytest.mark.parametrize(("weather", "temperature", "month", "expected"), [
    ("rain", 15.0, 6, "비 오는 날"),
    ("storm", 25.0, 7, "비 오는 날"),
    ("snow", -3.0, 12, "눈 오는 날"),
    ("snow", -3.0, 1, "함박눈 내리는 날"),
    ("clear", -5.0, 1, "추운 날"),
    ("cloudy", 9.9, 3, "추운 날"),
    ("clear", 33.0, 8, "더운 날"),
    ("cloudy", 24.0, 5, "따뜻한 날"),
    ("clear", 15.0, 10, "맑은 날"),
    ("cloudy", 15.0, 10, "흐린 날"),
    ("fog", 15.0, 10, "흐린 날"),
    (None, None, 10, "맑은 날"),  # 날씨를 모를 때
])
def test_ingredient_phrase(weather, temperature, month, expected):
    assert ingredient_phrase(weather, temperature, month) == f"{expected}에 듣기 좋은 노래"


@pytest.mark.parametrize(("weather", "time_of_day", "expected"), [
    ("clear", "afternoon", []),
    ("clear", "evening", []),
    ("clear", "night", ["밤"]),
    ("rain", "afternoon", ["비 오는 날"]),
    ("rain", "dawn", ["새벽", "비 오는 날"]),
    ("snow", "morning", ["아침", "함박눈 내리는 날"]),
    (None, "night", ["밤"]),
])
def test_slot_phrases(weather, time_of_day, expected):
    assert slot_phrases(weather, time_of_day, month=2) == [f"{w}에 듣기 좋은 노래" for w in expected]


def test_ladder_weights():
    assert ladder_weights(5) == [1.0, 0.9, 0.8, 0.7, 0.6]
    assert ladder_weights(3) == [1.0, 0.8, 0.6]
    assert ladder_weights(1) == [1.0]


def test_v4_mix_keeps_hub_term():
    """[q′, 1]의 마지막 1은 섞지 않는다. 섞은 벡터의 내적 = 두 점수를 섞은 값"""
    a, b = np.array([1.0, 0.0, 1.0]), np.array([0.0, 1.0, 1.0])
    song = np.array([0.3, 0.8, -0.2])
    mixed = mix("v4", a, b, 0.7)
    assert mixed[-1] == 1.0
    assert mixed @ song == pytest.approx(0.7 * (a @ song) + 0.3 * (b @ song))


def track(i, title=None):
    return SimpleNamespace(track_id=str(i), title=title or f"t{i}", artist="a")


def test_choose_tracks_fills_ladder_then_slots():
    """장소 칸은 장소·날씨말 후보에서, 특별 칸은 자기 후보에서 고르고 같은 곡은 두 번 고르지 않는다"""
    place, ingredient, night = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    vectors = {"1": place, "2": 0.9 * place + 0.3 * ingredient, "3": 0.5 * place + 0.5 * ingredient,
               "4": ingredient, "5": night, "6": 0.8 * place + 0.4 * night}
    tracks = {i: track(i) for i in range(1, 7)}
    pools = {"place": [tracks[1], tracks[2], tracks[6]], "ingredient": [tracks[3], tracks[4]],
             "slots": [[tracks[5], tracks[6]]]}
    chosen = choose_tracks("gemini", place, ingredient, [night], pools, vectors, k=5)
    assert [t.track_id for t in chosen[:4]] == ["1", "2", "6", "3"]  # 장소 몫 1.0 → 0.6
    assert chosen[4].track_id == "5"  # 6은 이미 골랐다


def test_same_song_is_not_chosen_twice():
    a = np.array([1.0, 0.0])
    vectors = {"1": a, "2": a, "3": 0.5 * a}
    pools = {"place": [track(1, "Rain"), track(2, "rain"), track(3)], "ingredient": [], "slots": []}
    chosen = choose_tracks("gemini", a, a, [], pools, vectors, k=2)
    assert [t.track_id for t in chosen] == ["1", "3"]
