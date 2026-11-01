"""검색어에서 LLM에 넘긴 지형·지역 이름을 지우는 규칙을 검증한다."""

import pytest

from backend.location.query import strip_names
from backend.location.schemas import LocationContext, NatureHit


def ctx(region=None, nature=()):
    return LocationContext(
        provider="kakao_osm", region=region, density="moderate", nature=list(nature), place_traits=[],
        time_of_day="afternoon", weather="cloudy", temperature_c=15.0,
    )


@pytest.mark.parametrize(("query", "context", "expected"), [
    # 실제로 새어 나온 사례
    ("흐린 오후 설악산 숲길에서 듣기 좋은 노래", ctx(nature=[NatureHit(kind="mountain", distance_m=0, name="설악산")]),
     "흐린 오후 산 숲길에서 듣기 좋은 노래"),
    # 꼬리말을 뗀 이름만 써도 지운다
    ("광안리 바다를 보며 듣는 노래", ctx(nature=[NatureHit(kind="sea", distance_m=80, name="광안리해수욕장")]),
     "바닷가 바다를 보며 듣는 노래"),
    # 긴 이름을 먼저 바꿔 "해수욕장"이 남지 않는다
    ("광안리해수욕장에서 듣는 노래", ctx(nature=[NatureHit(kind="sea", distance_m=80, name="광안리해수욕장")]),
     "바닷가에서 듣는 노래"),
    # 지역명은 시·구·동 모두 지운다
    ("속초 대포동의 흐린 오후에 듣는 노래", ctx(region="강원특별자치도 속초시 대포동"),
     "흐린 오후에 듣는 노래"),
    ("서울 송파구 먹자골목에서 듣는 노래", ctx(region="서울특별시 송파구 방이2동"),
     "먹자골목에서 듣는 노래"),
])
def test_names_are_removed(query, context, expected):
    assert strip_names(query, context) == expected


def test_short_stem_is_kept():
    """"중구"에서 "중"만 떼어 지우면 "중간" 같은 일반 단어가 깨진다"""
    query = "도심 한가운데 중간 휴식에 듣는 노래"
    assert strip_names(query, ctx(region="서울특별시 중구 명동")) == query


def test_query_without_names_is_unchanged():
    query = "비 오는 오후 강가에서 듣기 좋은 잔잔한 노래"
    context = ctx(region="서울특별시 마포구 망원1동", nature=[NatureHit(kind="river", distance_m=50, name="한강")])
    assert strip_names(query, context) == query
