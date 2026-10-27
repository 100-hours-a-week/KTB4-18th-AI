"""고속 모드 판단 로직 중 Jev 호출 없이 코드로 정하는 부분."""

from datetime import date

import pytest

from backend.recommendation.fast import pattern_year_range, resolve_years, year_range


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
