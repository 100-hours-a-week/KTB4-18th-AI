"""카카오 장소 이름을 사용자 문장에 쓸 이름으로 줄이는 규칙을 검증한다."""

from datetime import datetime

import pytest

from backend.location.context import (
    KAKAO_DENSITY_CODES, _kakao_name, summarize_kakao, summarize_weather, weather_unknown,
)


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
