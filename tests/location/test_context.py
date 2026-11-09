"""카카오·OSM 측정값을 LLM 입력(장소 특징·지형·시간대)으로 정리하는 규칙을 검증한다."""

from datetime import datetime

import pytest

from backend.location.context import (
    KAKAO_DENSITY_CODES, _kakao_name, build_context, combine_kakao_osm, summarize_kakao, summarize_osm_nature,
    summarize_weather, weather_unknown,
)
from backend.location.schemas import NatureHit, PlaceSummary


@pytest.mark.parametrize(("name", "category", "expected"), [
    # 끝 덩어리(시설·구간)를 뺀다 (망원에서 실제로 "배달존"이 문장에 나왔다)
    ("망원한강공원 배달존", "여행 > 공원시설물", "망원한강공원"),
    ("경주역사유적지구 대릉원지구", "여행 > 관광,명소 > 문화유적", "경주역사유적지구"),
    ("남한산성둘레길 1코스", "여행 > 관광,명소 > 남한산성둘레길", "남한산성둘레길"),
    ("청초호수공원 입구", "교통,수송 > 입출구", "청초호수공원"),
    # 가게 지점은 지점 이름에서 "점"만 뺀다
    ("투썸플레이스 세종호수공원점", "음식점 > 카페 > 커피전문점 > 투썸플레이스", "세종호수공원"),
    ("CU 속초호수공원점", "가정,생활 > 편의점 > CU", "속초호수공원"),
    # 지점이 아닌 "점": 여행·등산 분류는 끝 덩어리를 뺀다
    ("관악산 호수공원분기점", "스포츠,레저 > 등산 > 등산정보", "관악산"),
    ("한양도성 순성길 조망점", "여행 > 관광,명소 > 전망대", "한양도성 순성길"),
    ("스머프 매직 포레스트 홍대점", "여행 > 관광,명소 > 테마파크", "스머프 매직 포레스트"),
    # "점"을 빼면 한 글자만 남는 단어는 지점이 아니다
    ("의암공원 매점", "가정,생활 > 슈퍼마켓", "의암공원"),
    # 한 덩어리는 그대로
    ("서울숲", "여행 > 공원 > 도시근린공원", "서울숲"),
    ("롯데백화점", "가정,생활 > 백화점", "롯데백화점"),
])
def test_kakao_name(name, category, expected):
    assert _kakao_name({"place_name": name, "category_name": category}) == expected


def nature(parks):
    density = {code: {"meta": {"total_count": 0}, "documents": []} for code in KAKAO_DENSITY_CODES}
    summary = summarize_kakao("서울특별시 마포구 망원1동", density, [], parks)
    return [(hit.kind, hit.name) for hit in summary.nature]


def doc(name, category, distance=50):
    return {"place_name": name, "category_name": category, "distance": str(distance)}


@pytest.mark.parametrize(("place", "expected"), [
    (doc("망원한강공원 배달존", "여행 > 공원시설물"), [("river", "한강"), ("park", "망원한강공원")]),
    (doc("서울숲", "여행 > 공원 > 도시근린공원"), [("park", "서울숲")]),
    # 물가 단어가 든 가게는 물가 근거로만 쓰고, 공원 분류가 아니라 공원으로는 안 본다
    (doc("CU 속초호수공원점", "가정,생활 > 편의점 > CU"), [("river", "속초호수공원")]),
    # 작은 공원은 시설물이어도 뺀다
    (doc("바람산어린이공원 화장실", "여행 > 공원시설물"), []),
    (doc("마을마당 화장실", "여행 > 공원시설물"), []),
    # 공원 분류가 아니면 이름에 "공원"이 있어도 뺀다
    (doc("스타벅스 서울숲공원점", "음식점 > 카페 > 커피전문점"), []),
])
def test_park_and_waterside(place, expected):
    assert sorted(nature([place])) == sorted(expected)


@pytest.mark.parametrize(("clock", "expected"), [
    ("00:00", "dawn"), ("05:59", "dawn"),
    ("06:00", "morning"), ("10:59", "morning"),
    ("11:00", "afternoon"), ("16:59", "afternoon"),
    ("17:00", "evening"), ("20:59", "evening"),  # 겨울에 어두워도 저녁이다
    ("21:00", "night"), ("23:59", "night"),
])
def test_time_of_day_follows_clock(clock, expected):
    now = datetime.fromisoformat(f"2026-12-20T{clock}")
    assert summarize_weather({"current": {"time": now.isoformat(), "weather_code": 0, "temperature_2m": 1.0}}).time_of_day == expected
    assert weather_unknown(now).time_of_day == expected


def kakao(attractions, shops=0):
    """가게 수 shops개(밀도)와 관광명소 목록으로 카카오 요약을 만든다"""
    density = {code: {"meta": {"total_count": 0}, "documents": []} for code in KAKAO_DENSITY_CODES}
    density[KAKAO_DENSITY_CODES[0]]["meta"]["total_count"] = shops
    return summarize_kakao("서울특별시 송파구 잠실동", density, attractions, [])


@pytest.mark.parametrize(("shops", "expected"), [
    (50, ["테마파크"]),  # moderate: 상권 밖 테마파크로 본다
    (600, []),  # very_busy: 도심 실내 놀이시설(키즈카페 등)일 수 있어 뺀다
])
def test_kakao_theme_park_only_outside_busy_area(shops, expected):
    summary = kakao([doc("스머프 매직 포레스트 홍대점", "여행 > 관광,명소 > 테마파크", 258)], shops)
    assert summary.place_traits == expected


def measure(nearest=None, relief_m=0):
    return {"nearest": nearest or {}, "landuse": [], "protected": [], "peaks": [],
            "relief": {"min_m": 0, "max_m": relief_m, "relief_m": relief_m}}


def item(distance, area=0, name=None, sub=None):
    return {"distance_m": distance, "area_m2": area, "name": name, "sub": sub, "ele": None}


def test_osm_large_theme_park_kept_in_busy_area():
    """카카오는 very_busy라 테마파크를 뺐지만 OSM 면적이 기준 이상이면 다시 받는다. 작은 면은 받지 않는다"""
    osm = measure({"theme_park": [item(280, 40_279, "롯데월드")], "zoo": [item(100, 21, "동물병원")]})
    summary = combine_kakao_osm(kakao([doc("롯데월드 어드벤처", "여행 > 관광,명소 > 테마파크", 424)], 600), osm)
    assert summary.place_traits == ["테마파크"]


@pytest.mark.parametrize(("relief_m", "expected"), [
    (0, []),  # 산이 없으면 도심 하천이라 계곡으로 보지 않는다
    (300, ["stream", "mountain"]),
])
def test_osm_stream_needs_mountain(relief_m, expected):
    osm = measure({"stream": [item(50, sub="stream")]}, relief_m)
    assert [hit.kind for hit in summarize_osm_nature(osm, "moderate")] == expected


def weather():
    return summarize_weather({"current": {"time": "2026-11-06T15:00", "weather_code": 0, "temperature_2m": 15.0}})


@pytest.mark.parametrize(("nature", "expected"), [
    ([NatureHit(kind="sea", distance_m=100)], ["섬"]),
    ([NatureHit(kind="river", distance_m=100, name="한강")], []),  # 여의도 같은 강 섬
])
def test_island_needs_sea(nature, expected):
    places = PlaceSummary(region=None, density="moderate", nature=nature, place_traits=["섬"])
    assert build_context("kakao", places, weather()).place_traits == expected


def kakao_with_station(attractions, shops, station="범계역 4호선"):
    summary_input = {code: {"meta": {"total_count": 0}, "documents": []} for code in KAKAO_DENSITY_CODES}
    summary_input[KAKAO_DENSITY_CODES[0]]["meta"]["total_count"] = shops
    summary_input["SW8"] = {"meta": {"total_count": 1}, "documents": [{"place_name": station}]}
    return summarize_kakao("경기도 안양시 동안구 범계동", summary_input, attractions, [])


@pytest.mark.parametrize(("attractions", "shops", "expected"), [
    # 500m 안 관광명소 이름이 먼저
    ([doc("평촌로데오거리", "여행 > 관광,명소 > 테마거리", 300)], 600, "평촌로데오거리"),
    # 관광명소가 멀거나, 걸러지는 도심 실내 놀이시설뿐이면 지하철역(노선은 뗀다)
    ([doc("평촌로데오거리", "여행 > 관광,명소 > 테마거리", 800)], 600, "범계역"),
    ([doc("스머프 매직 포레스트 홍대점", "여행 > 관광,명소 > 테마파크", 50)], 600, "범계역"),
])
def test_landmark_attraction_then_station(attractions, shops, expected):
    assert kakao_with_station(attractions, shops).landmark == expected
