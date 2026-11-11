"""장소·날씨·시간 점수를 섞어 곡을 고르는 규칙을 검증한다."""

import random
from types import SimpleNamespace

import numpy as np
import pytest

from backend.location.blend import (
    choose_tracks, ingredient_phrase, ladder_weights, mix, place_vector, slot_phrases,
)


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
    chosen = choose_tracks("gemini", place, ingredient, [night], pools, vectors, k=5, top_n=1)
    assert [t.track_id for t in chosen[:4]] == ["1", "2", "6", "3"]  # 장소 몫 1.0 → 0.6
    assert chosen[4].track_id == "5"  # 6은 이미 골랐다


def test_same_song_is_not_chosen_twice():
    a = np.array([1.0, 0.0])
    vectors = {"1": a, "2": a, "3": 0.5 * a}
    pools = {"place": [track(1, "Rain"), track(2, "rain"), track(3)], "ingredient": [], "slots": []}
    chosen = choose_tracks("gemini", a, a, [], pools, vectors, k=2, top_n=1)
    assert [t.track_id for t in chosen] == ["1", "3"]


def test_pick_stays_within_top_n():
    """칸마다 점수 상위 3곡 안에서만 뽑고, 실행마다 그 안에서 바뀐다"""
    a = np.array([1.0, 0.0])
    vectors = {str(i): (1 - i / 10) * a for i in range(1, 7)}  # 1이 가장 높고 6이 가장 낮다
    pools = {"place": [track(i) for i in range(1, 7)], "ingredient": [], "slots": []}
    firsts = {choose_tracks("gemini", a, a, [], pools, vectors, k=1, rng=random.Random(seed))[0].track_id
              for seed in range(30)}
    assert firsts == {"1", "2", "3"}


@pytest.mark.parametrize("space", ["gemini", "v4"])
def test_place_vector_weights_main_word(space):
    """주 단어 0.6, 보조 단어 둘이 0.2씩. v4는 마지막 1(허브 보정 자리)을 그대로 둔다"""
    main, b, c = np.array([1.0, 0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0, 1.0])
    song = np.array([0.3, 0.8, -0.2, 0.5])
    vector = place_vector(space, main, [b, c])
    expected = 0.6 * (main @ song) + 0.2 * (b @ song) + 0.2 * (c @ song)
    assert vector @ song == pytest.approx(expected)
    if space == "v4":
        assert vector[-1] == 1.0
    assert place_vector(space, main, []) is main
