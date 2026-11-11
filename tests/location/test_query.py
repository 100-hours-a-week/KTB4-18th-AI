"""LLM 출력의 장소 단어가 정해진 보기 안에서만 나오는지, 보조 단어를 규칙대로 모으는지 검증한다."""

import pytest
from pydantic import ValidationError

from backend.location.query import LocationQuery, support_words
from backend.location.schemas import LocationContext, NatureHit


def context(nature=(), traits=()):
    return LocationContext(
        provider="kakao_osm", region=None, density="busy", nature=list(nature), place_traits=list(traits),
        time_of_day="afternoon", weather="clear", temperature_c=15.0,
    )


def test_support_words_follow_nature_then_traits():
    """남한산성: 숲(주 단어)은 빼고, 거리를 보지 않는 산 → 고궁 순. 맞는 보기가 없는 특징(테마거리)은 건너뛴다"""
    ctx = context(
        [NatureHit(kind="forest", distance_m=58), NatureHit(kind="mountain", distance_m=470, name="청량산")],
        ["테마거리", "고궁,궁", "먹자골목"],
    )
    assert support_words(ctx, "숲 산책로") == ["산", "돌담길"]  # 최대 2개라 먹자골목은 빠진다


def test_support_words_skip_far_nature():
    ctx = context([NatureHit(kind="river", distance_m=56, name="불광천"), NatureHit(kind="park", distance_m=301)])
    assert support_words(ctx, "번화가") == ["강"]


@pytest.mark.parametrize(("name", "expected"), [("한강", "한강"), ("석촌호수", "호수"), ("의암호", "호수"), ("불광천", "강")])
def test_support_words_name_river(name, expected):
    ctx = context([NatureHit(kind="river", distance_m=0, name=name)])
    assert support_words(ctx, "공원") == [expected]


def test_no_support_words_without_matching_features():
    """개롱역처럼 지형도 맞는 특징도 없으면 주 단어만 쓴다"""
    assert support_words(context(traits=["테마거리", "도보여행"]), "번화가") == []


def test_place_must_be_one_of_words():
    """보기에 없는 자유 문장(이름이 섞인 표현 포함)은 스키마 위반이다"""
    with pytest.raises(ValidationError):
        LocationQuery(place="해운대 바닷가", message="해운대예요.")


def test_schema_sends_words_as_enum():
    """json_schema strict로 보내는 스키마에 보기가 enum으로 들어가야 모델이 그 안에서 고른다"""
    schema = LocationQuery.model_json_schema()
    assert "해수욕장" in schema["properties"]["place"]["enum"]
