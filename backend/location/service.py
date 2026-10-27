"""위치 추천 흐름: 외부 소스를 동시에 부르고, 실패한 소스는 빼고 판정한 뒤, 검색어를 만들어 곡을 찾는다.

장소 소스별 실패 처리 (위키: 한쪽 컨텍스트만 확보해도 추천을 계속한다)
- 카카오 실패: OSM만으로 자연 판정. 밀도는 moderate로 두고 지역명·특징은 비운다
- OSM 없음(파일·shapely 미설치) 또는 실패: 카카오만으로 판정
- 카카오·OSM 둘 다 실패: 503 LOCATION_CONTEXT_UNAVAILABLE
- 고도 실패: 산 판정만 건너뛴다 / 날씨 실패: 날씨 없이 진행한다

검색어·검색 실패는 V1·V2와 같은 503으로 끝낸다
- LLM 미설정·공급자 오류·시간 초과: MODEL_UNAVAILABLE / 응답이 스키마와 다름: MODEL_RESPONSE_INVALID
- 임베딩·DB: MODEL_UNAVAILABLE·MUSIC_CATALOG_UNAVAILABLE
- 소스 하나가 30초를 넘으면: 그 소스만 실패로 보고 위 규칙대로 진행한다
- 요청 전체가 90초를 넘으면: LOCATION_TIMEOUT
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import openai
from openai import OpenAI
from pydantic import ValidationError

from backend.core.errors import _error, _model_unavailable
from backend.location import osm, sources
from backend.location.context import (
    build_context, combine_kakao_osm, summarize_kakao, summarize_osm_nature, summarize_weather, weather_unknown,
)
from backend.location.query import llm_query
from backend.location.schemas import (
    LocationContext, LocationRecommendationRequest, LocationRecommendationResponse, PlaceSummary, Provider,
)
from backend.providers.openrouter import OPENROUTER_BASE_URL, get_embedding_client
from backend.recommendation.embedding import embed_query
from backend.recommendation.retrieval import vector_recommendation

# NOTE: uvicorn 로거를 써야 별도 설정 없이 INFO 줄이 컨테이너 로그에 남는다(STT·사진과 같은 방식).
logger = logging.getLogger("uvicorn.error")
KST = ZoneInfo("Asia/Seoul")
SOURCE_TIMEOUT_S = 30
REQUEST_TIMEOUT_S = 90  # 소스·검색어·검색 세 단계가 각각 30초까지 걸릴 수 있다
LLM_TIMEOUT_S = 30.0  # V1 채팅의 LLM 클라이언트와 같다


async def _gather_sources(lat: float, lng: float) -> dict:
    """소스를 동시에 부른다. 실패하거나 시간을 넘긴 소스는 None으로 바꾸고 이름만 로그에 남긴다"""
    calls = {
        "kakao": sources.fetch_kakao(lat, lng),
        "weather": sources.fetch_weather(lat, lng),
        "relief": sources.fetch_relief(lat, lng),
    }
    if osm.available():
        # OSM 측정은 CPU 계산이라 이벤트 루프를 막지 않게 스레드에서 돌린다
        # NOTE: 시간을 넘겨도 스레드는 취소되지 않고 끝까지 실행된 뒤 버려진다
        calls["osm"] = asyncio.to_thread(osm.measure, lat, lng)

    timed = [asyncio.wait_for(call, SOURCE_TIMEOUT_S) for call in calls.values()]
    results = await asyncio.gather(*timed, return_exceptions=True)
    raw = {"osm": None}
    for name, result in zip(calls, results):
        if isinstance(result, Exception):
            logger.warning("Location source failed: source=%s error=%s", name, type(result).__name__)
            result = None
        raw[name] = result
    return raw


def _places(raw: dict) -> tuple[PlaceSummary, Provider]:
    measure = raw["osm"]
    if measure is not None:
        measure["relief"] = raw["relief"]  # None이면 산 판정만 건너뛴다

    kakao_raw = raw["kakao"]
    if kakao_raw is not None:
        kakao = summarize_kakao(kakao_raw["region"], kakao_raw["density"], kakao_raw["attractions"], kakao_raw["parks"])
        if measure is None:
            return kakao, "kakao"
        return combine_kakao_osm(kakao, measure), "kakao_osm"

    if measure is not None:
        # 밀도는 카카오 가게 수로만 잰다. 모르면 판정이 극단으로 가지 않도록 중간값을 둔다
        nature = summarize_osm_nature(measure, "moderate")
        return PlaceSummary(region=None, density="moderate", nature=nature, place_traits=[]), "osm"

    raise _error(503, "LOCATION_CONTEXT_UNAVAILABLE", "주변 장소 정보를 확인하지 못했습니다. 잠시 후 다시 시도하세요.")


def _music_query(ctx: LocationContext) -> str:
    key = os.getenv("OPENROUTER_LLM_API_KEY", "").strip()
    if not key or key.startswith("<") or not os.getenv("LLM_MODEL", "").strip():
        raise _model_unavailable()
    client = OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL, timeout=LLM_TIMEOUT_S, max_retries=0)
    try:
        return llm_query(client, ctx).music_query
    except openai.APIError as error:  # 공급자 오류·연결 실패·시간 초과
        logger.warning("Location query model unavailable: error=%s", type(error).__name__)
        raise _model_unavailable() from error
    except ValidationError as error:  # 빈 응답·잘린 JSON·스키마 위반
        raise _error(503, "MODEL_RESPONSE_INVALID", "모델 응답을 해석하지 못했습니다. 잠시 후 다시 시도하세요.") from error


def _search(query: str) -> list:
    return vector_recommendation(embed_query(get_embedding_client(), query))


async def _recommend(lat: float, lng: float) -> tuple[list, Provider, dict]:
    started = time.monotonic()
    raw = await _gather_sources(lat, lng)
    timings = {"sources_ms": round((time.monotonic() - started) * 1000)}

    places, provider = _places(raw)
    if raw["weather"] is not None:
        weather = summarize_weather(raw["weather"])
    else:
        weather = weather_unknown(datetime.now(KST).replace(tzinfo=None))
    ctx = build_context(provider, places, weather)

    # OpenAI SDK·DB 호출은 동기 함수라 스레드에서 돌린다
    query = await asyncio.to_thread(_music_query, ctx)
    timings["query_ms"] = round((time.monotonic() - started) * 1000)
    tracks = await asyncio.to_thread(_search, query)
    return tracks, provider, timings


async def recommend(request: LocationRecommendationRequest) -> LocationRecommendationResponse:
    started = time.monotonic()
    try:
        async with asyncio.timeout(REQUEST_TIMEOUT_S):
            tracks, provider, timings = await _recommend(request.latitude, request.longitude)
    except TimeoutError as error:
        # NOTE: 스레드에서 돌던 LLM·DB 호출은 취소되지 않고 끝까지 실행된 뒤 버려진다.
        logger.warning("Location request timed out: elapsed_ms=%d", round((time.monotonic() - started) * 1000))
        raise _error(503, "LOCATION_TIMEOUT", "위치 추천 처리 시간이 초과되었습니다. 잠시 후 다시 시도하세요.") from error

    # NOTE: 좌표·지역명·검색어는 위치를 드러내므로 남기지 않는다. 경로·결과·소요 시간만 기록한다.
    logger.info(
        "Location request: provider=%s tracks=%d sources_ms=%d query_ms=%d total_ms=%d",
        provider, len(tracks), timings["sources_ms"], timings["query_ms"],
        round((time.monotonic() - started) * 1000),
    )
    return LocationRecommendationResponse(tracks=tracks)
