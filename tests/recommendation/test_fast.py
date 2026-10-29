"""고속 모드 판단 로직 중 Jev 호출 없이 코드로 정하는 부분."""

from datetime import date

import pytest

from backend.recommendation.fast import (
    asks_unsupported, next_search_parts, pattern_year_range, requested_count, resolve_years, year_range,
)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("90년대 후반 노래 들려줘", (1997, 1999)),
        ("2000년대 말 힙합 추천해줘", (2007, 2009)),
        ("2010년대 중반 아이돌 노래", (2014, 2016)),
        ("2020년대초반 신나는 노래", (2020, 2023)),
        ("00년대 초반 감성", (2000, 2003)),
        ("80년대 후반 록", (1987, 1989)),
        ("90년대 중후반 R&B", (1994, 1999)),
        ("2010년대 초중반 걸그룹", (2010, 2016)),
        ("2000년대 초 힙합", (2000, 2003)),
        ("90년대 말기 감성", (1997, 1999)),
        ("2010년대 막바지 노래", (2017, 2019)),
    ],
)
def test_decade_part(message, expected):
    assert pattern_year_range(message) == expected


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("2015년쯤 유행하던 노래", (2013, 2017)),
        ("2010년 전후 K-Pop", (2008, 2012)),
        ("2018년 즈음 노래", (2016, 2020)),
        ("2005년 무렵 록", (2003, 2007)),
    ],
)
def test_around_year(message, expected):
    assert pattern_year_range(message) == expected


@pytest.mark.parametrize(
    "message",
    [
        "90년대 노래",                         # 연대 전체는 Jev가 잘 고른다
        "2010년대 이후 나온 힙합",
        "2010년 전에 나온 노래",                # "전"(이전)은 "전후"가 아니다
        "후반전에 듣기 좋은 노래",
        "2시쯤 듣기 좋은 노래",
        "2000원쯤 하는 커피",
        "2010년대 후반부터 2022년까지 나온 노래",  # 연도가 여럿이면 Jev에 맡긴다
    ],
)
def test_no_pattern(message):
    assert pattern_year_range(message) is None


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ("new", ["조금 더 밝게"]),
        ("continue", ["카페에서 공부할 때 들을 노래", "좀 더 잔잔하게", "조금 더 밝게"]),
        ("repeat", ["카페에서 공부할 때 들을 노래", "좀 더 잔잔하게"]),
    ],
)
def test_next_search_parts(choice, expected):
    previous = ["카페에서 공부할 때 들을 노래", "좀 더 잔잔하게"]
    answers = {"continues": {"choice": choice, "confidence": 0.9}}
    assert next_search_parts(previous, "조금 더 밝게", answers) == expected


def test_next_search_parts_keeps_recent_messages_only():
    previous = ["새벽 감성 노래", "좀 더 신나게", "힙합으로"]
    answers = {"continues": {"choice": "continue", "confidence": 0.9}}
    assert next_search_parts(previous, "조금 덜 시끄럽게", answers) == ["좀 더 신나게", "힙합으로", "조금 덜 시끄럽게"]


def test_next_search_parts_without_previous_starts_new():
    answers = {"continues": {"choice": "repeat", "confidence": 0.9}}
    assert next_search_parts([], "하나 더 줘", answers) == ["하나 더 줘"]


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ({"by_lyrics": 0.96, "by_ranking": 0.1, "by_number": 0.1, "by_topic": 0.1}, True),
        ({"by_lyrics": 0.1, "by_ranking": 0.1, "by_number": 0.1, "by_topic": 0.62}, False),  # "생일 노래" 오탐 수준
        ({"by_lyrics": 0.1, "by_ranking": 0.1, "by_number": 0.1, "by_topic": 0.1}, False),
    ],
)
def test_asks_unsupported(scores, expected):
    assert asks_unsupported({name: {"noul": value} for name, value in scores.items()}) is expected


@pytest.mark.parametrize(
    ("choice", "confidence", "expected"),
    [
        ("none", 0.9, None),
        ("2", 0.9, 2),
        ("more", 0.9, 6),
        ("2", 0.8, 2),     # 기준(COUNT_CONFIDENT)과 같으면 채택
        ("2", 0.79, None),  # 기준 바로 아래면 거름
        ("1", 0.57, None),  # "영어 이메일 하나 써주고"가 곡 수로 새던 수준
    ],
)
def test_requested_count(choice, confidence, expected):
    # 6곡 이상은 최대치(5)+1로 넘겨 track_limit은 5곡으로 자르고 count_notice는 최대 안내를 낸다.
    assert requested_count({"choice": choice, "confidence": confidence}) == expected


def test_year_range_treats_this_year_as_open_end():
    this_year = date.today().year
    answers = {"year_from": {"choice": str(this_year - 3), "confidence": 0.9},
               "year_to": {"choice": str(this_year), "confidence": 0.9}}
    assert year_range(answers) == (this_year - 3, None)


def test_resolve_years_prefers_code_pattern_over_jev():
    # Jev는 연대 전체(1990~1999)를 골랐지만, 메시지에 "후반"이 있으면 코드 결과를 쓴다.
    answers = {"year_from": {"choice": "1990", "confidence": 0.9},
               "year_to": {"choice": "1999", "confidence": 0.9}}
    assert resolve_years("90년대 후반 노래", answers) == (1997, 1999)
    assert resolve_years("90년대 노래", answers) == (1990, 1999)
