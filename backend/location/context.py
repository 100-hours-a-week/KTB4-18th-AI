"""장소·날씨 원본 응답을 LocationContext로 바꾸는 규칙.

외부 호출은 하지 않는다. 입력은 각 API의 응답을 그대로 받는다.
임계값은 scripts/location/probe_*.py로 받은 한국 좌표 7곳 결과를 보고 정했다.
"""

from datetime import datetime, timedelta

from backend.location.schemas import (
    Density, LocationContext, Nature, NatureHit, PlaceSummary, Provider, TimeOfDay, WeatherSummary,
)

MAX_TRAITS = 5

# 자연 지형을 그 장소의 특징으로 인정하는 거리(m). 붐빌수록 건물에 가려 덜 보인다고 보고 줄인다.
# 산은 멀리서도 보여서 한적한 곳에서는 넓게 잡는다.
# 예: 해리단길(busy)에서 약 800m 떨어진 해운대 해변은 인정하지 않는다
NATURE_RADIUS_M: dict[Nature, dict[str, int]] = {
    "sea":      {"quiet": 500, "moderate": 300, "busy": 150},
    # 강·호수는 수면이 넓게 트여 있고 등록된 점이 수면 한가운데라 바다보다 넓게 잡는다 (예: 석촌호수 동호 314m)
    "river":    {"quiet": 500, "moderate": 400, "busy": 350},
    # 계곡·개울은 폭이 좁아 가까이 있어야 보인다
    "stream":   {"quiet": 300, "moderate": 200, "busy": 150},
    "forest":   {"quiet": 300, "moderate": 200, "busy": 100},
    "park":     {"quiet": 300, "moderate": 200, "busy": 100},
    "mountain": {"quiet": 1000, "moderate": 1000, "busy": 300},
}

# ── 카카오 ──────────────────────────────────────────────
# 밀도: 반경 500m의 음식점(FD6)+카페(CE7)+편의점(CS2) 전체 개수
# 측정값: 강남역 1076 · 해운대 367 · 여의도 205 · 경포 72 · 평창읍 66 · 북한산 0 · 김제평야 0
# busy 하한은 처음 80이었는데, 정부세종청사 87·성산일출봉 117이 체감은 moderate라 150으로 올렸다
KAKAO_DENSITY_CODES = ("FD6", "CE7", "CS2")
KAKAO_DENSITY_LEVELS = ((10, "quiet"), (150, "moderate"), (500, "busy"))

# 관광명소(AT4) category_name의 마지막 단계에 이 단어가 들어 있으면 자연 지형으로 본다
# 위에서부터 먼저 맞는 것을 쓴다. "산림욕장"이 mountain에 걸리지 않도록 forest를 먼저 둔다
# "섬"은 바다 섬(동백섬)과 강 섬(여의도)이 섞여 있어 쓰지 않는다
KAKAO_NATURE_WORDS: dict[Nature, tuple[str, ...]] = {
    "sea": ("해수욕장", "해변"),
    "forest": ("숲", "휴양림", "수목원", "산림욕장"),
    # 한국 국립공원은 대부분 산이라 park보다 먼저 mountain으로 본다 (예: 월악산국립공원 덕주지구)
    "mountain": ("산", "봉우리", "오름", "능선", "국립공원"),
    "stream": ("계곡", "폭포"),
    "river": ("호수", "강", "저수지"),
    "park": ("공원",),
}

SEA_PARK_WORDS = ("해상", "해안")

# 특징은 검색어에 들어갈 단어라 가까운 것만 쓴다. 2km 개수 순이면 판교에 "테마파크", 잠실대교 위에 "먹자골목"이 붙었다
TRAIT_NEAR_M = 500
# 공방이 관광명소로 등록된 "도자기,도예촌", 자연 분류에서 뺀 "섬"은 분위기 단어로 쓰지 않는다
KAKAO_TRAIT_NOISE = ("도자기,도예촌", "섬", "섬(내륙)")

# "공원" 키워드 결과 중 동네마다 있는 작은 공원은 장소의 특징으로 보지 않는다 (예: 신촌 바람산어린이공원)
KAKAO_SMALL_PARK_WORDS = ("어린이공원", "마을마당", "소공원")
# 강·호수는 점이 없거나 수면 한가운데 한 점이라 거리가 부정확하다.
# 이름에 이 단어가 있는 장소는 분류와 관계없이(진출입로·화장실 포함) 물가로 본다 (예: 한강공원잠실대교북단 진출입로)
KAKAO_WATERSIDE_WORDS = ("한강", "호수", "강변", "천변")

# ── Google ─────────────────────────────────────────────
# 밀도: 반경 500m 거리순 20개 중 가장 먼 장소까지의 거리. 20개가 안 차면 한적한 곳
# 측정값: 강남역 13m · 평창읍 57m · 해운대 66m · 여의도 162m · 경포 164m · 북한산 18개
GOOGLE_DENSITY_LEVELS = ((30, "very_busy"), (100, "busy"), (300, "moderate"))

# 사용자 등록 잡음이 적은 유형은 1개만 있어도 인정한다
GOOGLE_RELIABLE_TYPES: dict[str, Nature] = {
    "beach": "sea",
    "mountain_peak": "mountain",
    "woods": "forest",
    "lake": "river",
}
# 강남역에서 회사·식당이 park로, 온천이 river로 등록된 사례가 있어 이름도 같이 확인한다
GOOGLE_NOISY_TYPES: dict[str, tuple[Nature, tuple[str, ...]]] = {
    "park": ("park", ("공원", "park")),
    "city_park": ("park", ("공원", "park")),
    "national_park": ("mountain", ("국립공원", "national park")),
    "state_park": ("park", ("공원", "park")),
    "river": ("river", ("강", "하천", "river")),
    "hiking_area": ("mountain", ("산", "둘레길", "등산")),
    "nature_preserve": ("forest", ("숲", "산", "봉")),
}


def _level(value, levels, default):
    """levels의 (상한, 등급) 중 value가 처음으로 들어가는 등급"""
    for limit, name in levels:
        if value < limit:
            return name
    return default


def _visible_nature(
    hits: list[tuple[Nature, int, str | None]], density: Density, radius=NATURE_RADIUS_M,
) -> list[NatureHit]:
    """(종류, 거리, 이름) 목록 중 밀도별 인정 거리 안에 있는 것만, 종류별 가장 가까운 것 하나씩 가까운 순으로"""
    level = "busy" if density == "very_busy" else density
    nearest: dict[Nature, NatureHit] = {}
    for kind, distance, name in hits:
        if distance <= radius[kind][level] and (kind not in nearest or distance < nearest[kind].distance_m):
            nearest[kind] = NatureHit(kind=kind, distance_m=distance, name=name)
    return sorted(nearest.values(), key=lambda hit: hit.distance_m)


def summarize_kakao(
    region: str | None, density: dict[str, dict], attractions: list[dict], parks: list[dict],
) -> PlaceSummary:
    """density: 카테고리 코드별 카테고리 검색 응답(500m), attractions: AT4 문서 목록(2km, 거리순),
    parks: "공원" 키워드 검색 문서 목록(500m)"""
    count = sum(density[code]["meta"]["total_count"] for code in KAKAO_DENSITY_CODES)
    level = _level(count, KAKAO_DENSITY_LEVELS, "very_busy")

    hits, traits = [], []
    for doc in attractions:
        kind = doc["category_name"].split(" > ")[-1]
        tag = next((t for t, words in KAKAO_NATURE_WORDS.items() if any(w in kind for w in words)), None)
        # 바다 국립공원(한려해상·태안해안 등)은 산이 아니다
        if tag == "mountain" and any(w in doc["place_name"] for w in SEA_PARK_WORDS):
            tag = "sea"
        if tag is None:
            if int(doc["distance"]) <= TRAIT_NEAR_M and kind not in KAKAO_TRAIT_NOISE:
                traits.append(kind)  # attractions가 거리순이라 가까운 순으로 쌓인다
        else:
            hits.append((tag, int(doc["distance"]), doc["place_name"]))

    for doc in parks:
        name, distance = doc["place_name"], int(doc["distance"])
        if any(w in name for w in KAKAO_WATERSIDE_WORDS):
            # "한강공원잠실대교북단상류측계단 진출입로" 같은 시설 이름은 문장에 못 쓰니 강 이름만 남긴다
            hits.append(("river", distance, "한강" if "한강" in name else None))
        # "여행 > 공원", "여행 > 공원시설물"만 인정한다. "OO공원점" 같은 음식점·카페는 분류에서 걸러진다
        if doc["category_name"].startswith("여행 > 공원") and not any(w in name for w in KAKAO_SMALL_PARK_WORDS):
            hits.append(("park", distance, name))

    return PlaceSummary(
        region=region,
        density=level,
        nature=_visible_nature(hits, level),
        place_traits=list(dict.fromkeys(traits))[:MAX_TRAITS],
    )


def _google_nature_tag(place: dict) -> Nature | None:
    ptype = place.get("primaryType")
    if ptype in GOOGLE_RELIABLE_TYPES:
        return GOOGLE_RELIABLE_TYPES[ptype]
    if ptype in GOOGLE_NOISY_TYPES:
        tag, words = GOOGLE_NOISY_TYPES[ptype]
        name = place["displayName"]["text"].lower()
        return tag if any(w in name for w in words) else None
    return None


def summarize_google(dense: list[dict], nature_places: list[dict]) -> PlaceSummary:
    """dense: 500m 거리순 검색, nature_places: 자연 유형 2km 검색. 둘 다 distance_m이 붙은 장소 목록"""
    if len(dense) < 20:
        density = "quiet"
    else:
        density = _level(dense[-1]["distance_m"], GOOGLE_DENSITY_LEVELS, "quiet")

    hits = [
        (tag, p["distance_m"], p["displayName"]["text"])
        for p in nature_places if (tag := _google_nature_tag(p)) is not None
    ]

    return PlaceSummary(
        region=None,  # Google은 지역명을 받지 않는다. 필요하면 카카오 행정동 조회를 붙인다
        density=density,
        nature=_visible_nature(hits, density),
        # 가까운 순으로 옷가게·보험사 같은 가게 유형이 먼저 잡혀 분위기 신호로 약하다. 비교용이라 비워 둔다
        place_traits=[],
    )


# ── OSM ───────────────────────────────────────────────
# 입력은 scripts/location/measure_osm.py가 잰 값(종류별 가까운 지형 목록 + 주변 고도 기복)이다.
# 밀도는 OSM으로 잴 수 없어 카카오 값을 받는다.

# 연못·하수처리장·배수로 등은 강·호수로 보지 않는다 (예: 화랑공원 생태호수)
OSM_SMALL_WATER = {"pond", "wastewater", "basin", "ditch", "fishpond", "drain", "reflecting_pool", "moat"}
# 동네 녹지·조경 연못은 장소의 특징으로 보지 않는다. 선(강 중심선·개울)은 면적이 0이라 거르지 않는다
# 측정값: 역삼공원 0.5ha · 오금공원 20.6ha · 방이동백제고분군 4.2ha · 잠실새내역 이름 없는 수면 70㎡ · 석촌호수 14.6ha
OSM_MIN_AREA_M2 = {"park": 20_000, "forest": 100_000, "river": 10_000}
# 주변 ±1km 최고·최저 고도 차이가 이 이상일 때만 산으로 본다. 근처 봉우리 유무는 보지 않는다
# 측정값: 남한산성 312 · 송계계곡 434 / 해리단길 165 · 한남동 120 · 판교 80 · 서강대 87
MOUNTAIN_MIN_RELIEF_M = 250
# OSM은 실제 물가까지 재므로, 카카오의 수면 중심점 때문에 넓혔던 강·호수 거리를 바다와 같게 되돌린다
# 예: 방이맛골 → 석촌호수 278m는 빼고, 롯데월드타워 150m·체스터톤스 → 청초호 136m는 남긴다
OSM_NATURE_RADIUS_M = {**NATURE_RADIUS_M, "river": NATURE_RADIUS_M["sea"]}

# 산 이름. 틀린 이름(서울대에서 "돌산")은 없느니만 못해서 확실할 때만 붙인다
PROTECTED_SUFFIXES = ("국립공원", "도립공원", "군립공원")
PEAK_NEAR_M = 200  # 이 안의 봉우리면 그 봉우리 위에 있다고 본다 (예: 북한산 백운대 21m)
# 봉우리 점수 = 거리 ÷ (봉우리 높이 − 주변 최저 고도)^1.5. 높이 솟은 산일수록 멀리서도 그 산 영향권으로 본다
# 측정값: 남한산성 → 청량산 0.074 · 서울대 → 관악산 0.141 (우정봉 0.354)
PEAK_SCORE_MAX = 0.2


DENSITY_ORDER: tuple[Density, ...] = ("quiet", "moderate", "busy", "very_busy")


def adjust_density_in_park(density: Density, measure: dict) -> Density:
    """넓은 공원 안에 있고 busy 이상이면 한 단계 한산하게 본다.

    500m 반경 개수에는 공원 밖 상권이 섞인다. 공원 안에서는 덜 붐비게 느껴진다
    (예: 경주 대릉원 395개 busy → moderate, 여수 이순신광장 323개 busy → moderate)
    섞일 상권이 적은 moderate 이하는 낮추지 않는다 (예: 춘천 공지천 36개는 moderate 유지)
    """
    parks = measure["nearest"].get("park", [])
    inside = any(p["distance_m"] == 0 and p["area_m2"] >= OSM_MIN_AREA_M2["park"] for p in parks)
    if not inside or density not in ("busy", "very_busy"):
        return density
    return DENSITY_ORDER[DENSITY_ORDER.index(density) - 1]


def _mountain_name(measure: dict) -> str | None:
    """1) 국립·도립공원 구역 안 2) 200m 안 봉우리 3) 봉우리 점수가 기준 이하인 산. 다 아니면 None"""
    for park in measure["protected"]:
        if any(w in park for w in SEA_PARK_WORDS):
            continue
        name = park.replace(" ", "")
        for suffix in PROTECTED_SUFFIXES:
            name = name.removesuffix(suffix)
        # "경주 국립공원" 같은 역사 공원은 산 이름이 아니다
        if "산" in name:
            return name

    peaks = measure["peaks"]
    near = [p for p in peaks if p["distance_m"] <= PEAK_NEAR_M]
    if near:
        return min(near, key=lambda p: p["distance_m"])["name"]

    base = measure["relief"]["min_m"]
    scored = [(p["distance_m"] / (p["ele"] - base) ** 1.5, p["name"]) for p in peaks if p["ele"] > base]
    if scored and min(scored)[0] <= PEAK_SCORE_MAX:
        return min(scored)[1]
    return None


def summarize_osm_nature(measure: dict, density: Density) -> list[NatureHit]:
    hits = []
    for kind, items in measure["nearest"].items():
        if kind == "mountain":
            continue
        for item in items:
            # 수면(natural=water)으로 그려진 개울은 stream으로 옮긴다 (예: 판교 운중천 쪽 water=stream)
            tag = "stream" if kind == "river" and item["sub"] == "stream" else kind
            if tag == "river" and item["sub"] in OSM_SMALL_WATER:
                continue
            if item["area_m2"] and item["area_m2"] < OSM_MIN_AREA_M2.get(tag, 0):
                continue
            hits.append((tag, item["distance_m"], item["name"]))
    nature = _visible_nature(hits, density, OSM_NATURE_RADIUS_M)

    # 산은 거리 대신 기복으로 판정한다. 거리는 참고용으로 가장 가까운 봉우리까지
    # 고도 API가 실패하면 relief가 None이고, 산 판정만 건너뛴다
    relief = measure.get("relief")
    if relief and relief["relief_m"] >= MOUNTAIN_MIN_RELIEF_M:
        peaks = measure["nearest"].get("mountain", [])
        distance = peaks[0]["distance_m"] if peaks else 0
        nature.append(NatureHit(kind="mountain", distance_m=distance, name=_mountain_name(measure)))
    return nature


def combine_kakao_osm(kakao: PlaceSummary, measure: dict) -> PlaceSummary:
    """판정은 OSM, 사람에게 보여줄 말(지역·이름·특징)은 카카오가 맡는다.

    - 밀도: 카카오 개수 + OSM 공원 안 보정
    - 자연: OSM. 바다만 카카오 해수욕장으로 보충한다(사람이 붙인 분류라 오탐이 거의 없다)
    - 이름: 같은 종류를 카카오도 잡았으면 카카오 이름(한국어 장소명), 아니면 OSM 이름
    - 특징: 카카오 관광 특징 + OSM 용도지역이 상업·공업이면 업무지구. 용도지역은 비어 있는 곳이 많아 있을 때만 쓴다
    """
    density = adjust_density_in_park(kakao.density, measure)
    nature = summarize_osm_nature(measure, density)

    kakao_by_kind = {hit.kind: hit for hit in kakao.nature}
    if "sea" in kakao_by_kind and all(hit.kind != "sea" for hit in nature):
        nature.append(kakao_by_kind["sea"])
    # 산 이름은 카카오 "○○산"에 뒷동산이 섞여(서울대 → 돌산) OSM 규칙(_mountain_name)만 쓴다
    nature = [
        hit.model_copy(update={"name": kakao_by_kind[hit.kind].name})
        if hit.kind != "mountain" and hit.kind in kakao_by_kind and kakao_by_kind[hit.kind].name else hit
        for hit in nature
    ]

    traits = list(kakao.place_traits)
    if {"commercial", "industrial"} & set(measure["landuse"]):
        traits.insert(0, "업무지구")

    return PlaceSummary(
        region=kakao.region,
        density=density,
        nature=sorted(nature, key=lambda hit: hit.distance_m),
        place_traits=traits[:MAX_TRAITS],
    )


# WMO 날씨 코드 → 단순 분류. https://open-meteo.com/en/docs 의 WMO Weather interpretation codes
def _weather_kind(code: int):
    if code <= 1:
        return "clear"
    if code <= 3:
        return "cloudy"
    if code in (45, 48):
        return "fog"
    if 71 <= code <= 77 or code in (85, 86):
        return "snow"
    if code >= 95:
        return "storm"
    return "rain"


def _time_of_day(now: datetime, sunrise: datetime, sunset: datetime) -> TimeOfDay:
    if now < sunrise - timedelta(minutes=30) or now > sunset + timedelta(minutes=30):
        return "night"
    if now < sunrise + timedelta(hours=1):
        return "dawn"
    if now.hour < 12:
        return "morning"
    if now < sunset - timedelta(hours=1):
        return "afternoon"
    return "sunset"


def summarize_weather(forecast: dict) -> WeatherSummary:
    """Open-Meteo 응답(current + daily sunrise·sunset, 현지 시간대)"""
    now = datetime.fromisoformat(forecast["current"]["time"])
    sunrise = datetime.fromisoformat(forecast["daily"]["sunrise"][0])
    sunset = datetime.fromisoformat(forecast["daily"]["sunset"][0])
    time_of_day = _time_of_day(now, sunrise, sunset)

    return WeatherSummary(
        time_of_day=time_of_day,
        weather=_weather_kind(forecast["current"]["weather_code"]),
        temperature_c=forecast["current"]["temperature_2m"],
    )


def weather_unknown(now: datetime) -> WeatherSummary:
    """날씨 API 실패 시. 시간대는 한국 평균 일출 6:30·일몰 18:30으로 어림하고 날씨는 비워 둔다"""
    sunrise = now.replace(hour=6, minute=30, second=0, microsecond=0)
    sunset = now.replace(hour=18, minute=30, second=0, microsecond=0)
    return WeatherSummary(time_of_day=_time_of_day(now, sunrise, sunset), weather=None, temperature_c=None)


def build_context(provider: Provider, places: PlaceSummary, weather: WeatherSummary) -> LocationContext:
    return LocationContext(provider=provider, **places.model_dump(), **weather.model_dump())
