"""위치 추천의 외부 API 호출: 카카오 로컬, Open-Meteo 날씨·고도.

모두 비동기다. 서비스가 asyncio.gather로 동시에 부르므로 전체 대기는 가장 느린 호출 하나 정도다.
응답은 가공하지 않고 그대로 돌려준다. 판정은 backend/location/context.py가 한다.
"""

import asyncio
import math
import os

import httpx

KAKAO_CATEGORY_URL = "https://dapi.kakao.com/v2/local/search/category.json"
KAKAO_KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"
KAKAO_REGION_URL = "https://dapi.kakao.com/v2/local/geo/coord2regioncode.json"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"

DENSITY_RADIUS_M = 500
ATTRACTION_RADIUS_M = 2000
# 밀도 판정은 FD6·CE7·CS2만 쓰지만, 나머지는 실험 페이지·대시보드에서 동네 성격을 볼 때 쓴다
DENSITY_CODES = ("FD6", "CE7", "CS2", "AD5", "CT1", "AT4", "SW8")
# 9×9 격자, 250m 간격 → ±1km. 고도 API는 좌표 하나를 호출 1회로 세서 81회로 계산된다
RELIEF_GRID_STEP_M = 250


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


async def fetch_kakao(lat: float, lng: float) -> dict:
    """summarize_kakao에 넣을 원본: 행정동, 카테고리별 500m 검색, 관광명소 2km(최대 45개), 공원 키워드 500m"""
    async with kakao_client() as client:
        def category(code, radius, page=1):
            return _get(client, KAKAO_CATEGORY_URL, {
                "category_group_code": code, "x": lng, "y": lat, "radius": radius,
                "sort": "distance", "page": page, "size": 15,
            })

        # 관광명소는 3페이지까지 받아야 해서 페이지도 한꺼번에 요청하고, 끝난 페이지(is_end) 뒤는 버린다
        region, parks, *rest = await asyncio.gather(
            _get(client, KAKAO_REGION_URL, {"x": lng, "y": lat}),
            _get(client, KAKAO_KEYWORD_URL, {
                "query": "공원", "x": lng, "y": lat, "radius": DENSITY_RADIUS_M, "sort": "distance", "size": 15,
            }),
            *(category(code, DENSITY_RADIUS_M) for code in DENSITY_CODES),
            *(category("AT4", ATTRACTION_RADIUS_M, page) for page in (1, 2, 3)),
        )

    density = dict(zip(DENSITY_CODES, rest[:len(DENSITY_CODES)]))
    attractions = []
    for page in rest[len(DENSITY_CODES):]:
        attractions += page["documents"]
        if page["meta"]["is_end"]:
            break

    regions = region["documents"]
    dong = next((r for r in regions if r["region_type"] == "H"), regions[0] if regions else None)
    return {
        "region": dong["address_name"] if dong else None,
        "density": density,
        "attractions": attractions,
        "parks": parks["documents"],
    }


async def fetch_weather(lat: float, lng: float) -> dict:
    """summarize_weather에 넣을 Open-Meteo 응답"""
    async with httpx.AsyncClient(timeout=30) as client:
        return await _get(client, WEATHER_URL, {
            "latitude": lat, "longitude": lng,
            "current": "temperature_2m,weather_code,is_day,precipitation,cloud_cover",
            "daily": "sunrise,sunset", "timezone": "Asia/Seoul", "forecast_days": 1,
        })


async def fetch_relief(lat: float, lng: float) -> dict:
    """주변 ±1km 격자의 해발고도와 최고·최저 차이(기복). 산 판정과 산 이름 고르기에 쓴다"""
    d_lat = RELIEF_GRID_STEP_M / 110_540
    d_lng = RELIEF_GRID_STEP_M / (111_320 * math.cos(math.radians(lat)))
    offsets = range(-4, 5)
    lats = [round(lat + i * d_lat, 5) for i in offsets for _ in offsets]
    lngs = [round(lng + j * d_lng, 5) for _ in offsets for j in offsets]
    async with httpx.AsyncClient(timeout=30) as client:
        data = await _get(client, ELEVATION_URL, {
            "latitude": ",".join(map(str, lats)), "longitude": ",".join(map(str, lngs)),
        })
    elevation = data["elevation"]
    return {"min_m": min(elevation), "max_m": max(elevation), "relief_m": max(elevation) - min(elevation)}
