"""위치 추천의 외부 API 호출: 카카오 로컬, Open-Meteo 날씨. 고도는 파일에서 읽는다(backend/location/elevation.py).

모두 비동기다. 서비스가 asyncio.gather로 동시에 부르므로 전체 대기는 가장 느린 호출 하나 정도다.
응답은 가공하지 않고 그대로 돌려준다. 판정은 backend/location/context.py가 한다.
"""

import asyncio
import os

import httpx

KAKAO_CATEGORY_URL = "https://dapi.kakao.com/v2/local/search/category.json"
KAKAO_KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"
KAKAO_REGION_URL = "https://dapi.kakao.com/v2/local/geo/coord2regioncode.json"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

DENSITY_RADIUS_M = 500
ATTRACTION_RADIUS_M = 2000
# 밀도 판정은 FD6·CE7·CS2만 쓰지만, 나머지는 실험 페이지·대시보드에서 동네 성격을 볼 때 쓴다
DENSITY_CODES = ("FD6", "CE7", "CS2", "AD5", "CT1", "AT4", "SW8")
# 관광명소(AT4)로 잡히지 않는 시설. 카카오 키워드 검색은 분류 이름도 찾아서 "프로야구장"처럼 분류 이름으로 묻는다
# ("야구장"으로 물으면 잠실에서 떡볶이집·화장실 15건에 잠실야구장이 밀려났다). 이름만 겹치는 가게는 context가 분류로 거른다
FACILITY_QUERIES = ("공항", "기차역", "프로야구장", "종합운동장", "스키장", "시장")
# 스키장은 슬로프가 수 km라 1km까지 받는다. 인정 거리는 시설마다 context.FACILITIES가 정한다
FACILITY_RADIUS_M = 1000
# 스키장은 리조트 건물마다 "하이원스키장 힐콘도A동"처럼 이름에 스키장이 붙어, 첫 페이지 15건이 콘도로 차고
# 진짜 스키장(301m)은 2페이지에 있었다. 이런 검색어만 관광명소처럼 3페이지까지 받는다
FACILITY_PAGES = {"스키장": 3}


class KakaoNotConfigured(Exception):
    """KAKAO_REST_API_KEY가 없다."""


def kakao_client() -> httpx.AsyncClient:
    key = os.getenv("KAKAO_REST_API_KEY", "").strip()
    if not key or key.startswith("<"):
        raise KakaoNotConfigured()
    return httpx.AsyncClient(timeout=30, headers={"Authorization": f"KakaoAK {key}"})


async def _get(client: httpx.AsyncClient, url: str, params: dict) -> dict:
    res = await client.get(url, params=params)
    res.raise_for_status()
    return res.json()


def _until_end(pages: list[dict]) -> list[dict]:
    """한꺼번에 요청한 페이지들의 문서를 이어 붙이고, 끝난 페이지(is_end) 뒤는 버린다"""
    documents = []
    for page in pages:
        documents += page["documents"]
        if page["meta"]["is_end"]:
            break
    return documents


async def fetch_kakao(lat: float, lng: float) -> dict:
    """summarize_kakao에 넣을 원본: 행정동, 카테고리별 500m 검색, 관광명소 2km(최대 45개), 공원 키워드 500m,
    시설 키워드 1km(FACILITY_QUERIES), 학교 500m(대학교 이름)"""
    async with kakao_client() as client:
        def category(code, radius, page=1):
            return _get(client, KAKAO_CATEGORY_URL, {
                "category_group_code": code, "x": lng, "y": lat, "radius": radius,
                "sort": "distance", "page": page, "size": 15,
            })

        def keyword(query, radius=DENSITY_RADIUS_M, page=1):
            return _get(client, KAKAO_KEYWORD_URL, {
                "query": query, "x": lng, "y": lat, "radius": radius, "sort": "distance", "page": page, "size": 15,
            })

        # 여러 페이지가 필요한 검색(관광명소, 스키장)은 페이지도 한꺼번에 요청한다
        facility_pages = [(query, page) for query in FACILITY_QUERIES for page in range(1, FACILITY_PAGES.get(query, 1) + 1)]
        region, parks, schools, *rest = await asyncio.gather(
            _get(client, KAKAO_REGION_URL, {"x": lng, "y": lat}),
            keyword("공원"),
            category("SC4", DENSITY_RADIUS_M),
            *(keyword(query, FACILITY_RADIUS_M, page) for query, page in facility_pages),
            *(category(code, DENSITY_RADIUS_M) for code in DENSITY_CODES),
            *(category("AT4", ATTRACTION_RADIUS_M, page) for page in (1, 2, 3)),
        )

    facility_results, rest = rest[:len(facility_pages)], rest[len(facility_pages):]
    facilities = {
        query: _until_end([result for (q, _), result in zip(facility_pages, facility_results) if q == query])
        for query in FACILITY_QUERIES
    }
    facilities["학교"] = schools["documents"]
    density = dict(zip(DENSITY_CODES, rest[:len(DENSITY_CODES)]))
    attractions = _until_end(rest[len(DENSITY_CODES):])

    regions = region["documents"]
    dong = next((r for r in regions if r["region_type"] == "H"), regions[0] if regions else None)
    return {
        "region": dong["address_name"] if dong else None,
        "density": density,
        "attractions": attractions,
        "parks": parks["documents"],
        "facilities": facilities,
    }


async def fetch_weather(lat: float, lng: float) -> dict:
    """summarize_weather에 넣을 Open-Meteo 응답"""
    async with httpx.AsyncClient(timeout=30) as client:
        return await _get(client, WEATHER_URL, {
            "latitude": lat, "longitude": lng,
            "current": "temperature_2m,weather_code,is_day,precipitation,cloud_cover",
            "timezone": "Asia/Seoul", "forecast_days": 1,
        })

