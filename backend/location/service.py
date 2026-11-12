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
import functools
import logging
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import openai
from openai import OpenAI
from pydantic import ValidationError

from backend.core.errors import _error, _model_unavailable
from backend.location import elevation, osm, sources
from backend.location.context import (
    build_context, combine_kakao_osm, summarize_kakao, summarize_osm_nature, summarize_weather, weather_unknown,
)
from backend.location.query import LocationQuery, llm_query, support_words
from backend.location.schemas import (
    LocationContext, LocationRecommendationRequest, LocationRecommendationResponse, PlaceSummary, Provider,
)
from backend.providers.openrouter import OPENROUTER_BASE_URL, get_embedding_client
from backend.recommendation.embedding import embed_query
from backend.location.blend import (
    POOL_SIZE, choose_tracks, ingredient_phrase, place_phrase, place_vector, query_vector, slot_phrases,
)
from backend.recommendation.adapter import search_space
from backend.recommendation.retrieval import candidate_pools_db

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
    }
    if osm.available():
        # OSM 측정은 CPU 계산이라 이벤트 루프를 막지 않게 스레드에서 돌린다
        # NOTE: 시간을 넘겨도 스레드는 취소되지 않고 끝까지 실행된 뒤 버려진다
        calls["osm"] = asyncio.to_thread(osm.measure, lat, lng)
        # 고도 기복은 OSM 산 판정에만 쓴다. 파일에서 81칸을 읽는 디스크 작업이라 스레드에서 돌린다
        calls["relief"] = asyncio.to_thread(elevation.relief, lat, lng)

    timed = [asyncio.wait_for(call, SOURCE_TIMEOUT_S) for call in calls.values()]
    results = await asyncio.gather(*timed, return_exceptions=True)
    raw = {"osm": None, "relief": None}
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
        kakao = summarize_kakao(
            kakao_raw["region"], kakao_raw["density"], kakao_raw["attractions"], kakao_raw["parks"],
            kakao_raw["facilities"],
        )
        if measure is None:
            return kakao, "kakao"
        return combine_kakao_osm(kakao, measure), "kakao_osm"

    if measure is not None:
        # 밀도는 카카오 가게 수로만 잰다. 모르면 판정이 극단으로 가지 않도록 중간값을 둔다
        nature = summarize_osm_nature(measure, "moderate")
        return PlaceSummary(region=None, density="moderate", nature=nature, place_traits=[]), "osm"

    raise _error(503, "LOCATION_CONTEXT_UNAVAILABLE", "주변 장소 정보를 확인하지 못했습니다. 잠시 후 다시 시도하세요.")


def _location_query(ctx: LocationContext) -> LocationQuery:
    """LLM 1회로 장소 단어(place)와 응답 문장(message)을 함께 만든다"""
    key = os.getenv("OPENROUTER_LLM_API_KEY", "").strip()
    if not key or key.startswith("<") or not os.getenv("LLM_MODEL", "").strip():
        raise _model_unavailable()
    client = OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL, timeout=LLM_TIMEOUT_S, max_retries=0)
    try:
        return llm_query(client, ctx)
    except openai.APIError as error:  # 공급자 오류·연결 실패·시간 초과
        logger.warning("Location query model unavailable: error=%s", type(error).__name__)
        raise _model_unavailable() from error
    except ValidationError as error:  # 빈 응답·잘린 JSON·스키마 위반
        raise _error(503, "MODEL_RESPONSE_INVALID", "모델 응답을 해석하지 못했습니다. 잠시 후 다시 시도하세요.") from error


@functools.cache
def _fixed_embedding(text: str) -> tuple[float, ...]:
    """장소 단어·날씨말·시간 문장은 정해진 몇 개뿐이라 프로세스마다 처음 한 번만 임베딩한다. 실패하면 캐시하지 않는다"""
    return tuple(embed_query(get_embedding_client(), text))


def _blend_search(
    place: str, supports: list[str], weather: str | None, temperature_c: float | None, time_of_day: str, month: int,
) -> list:
    """장소 단어(주·보조)·날씨말·특별 칸 문장을 따로 임베딩해 칸별로 곡을 고른다"""
    space = search_space()
    word_qs = [query_vector(space, _fixed_embedding(place_phrase(word))) for word in [place, *supports]]
    ingredient_q = query_vector(space, _fixed_embedding(ingredient_phrase(weather, temperature_c, month)))
    slot_qs = [query_vector(space, _fixed_embedding(p)) for p in slot_phrases(weather, time_of_day, month)]

    pools, vectors = candidate_pools_db([*word_qs, ingredient_q, *slot_qs], space, POOL_SIZE)
    word_pools, ingredient_pool, slot_pools = pools[:len(word_qs)], pools[len(word_qs)], pools[len(word_qs) + 1:]
    # 장소 후보는 단어마다 찾은 곡을 합친 것. 섞은 장소 벡터는 이 안에서 순서를 정하는 데만 쓴다(blend.place_vector)
    place_pool = list({track.track_id: track for pool in word_pools for track in pool}.values())
    return choose_tracks(
        space, place_vector(space, word_qs[0], word_qs[1:]), ingredient_q, slot_qs,
        {"place": place_pool, "ingredient": ingredient_pool, "slots": slot_pools}, vectors,
    )


async def _recommend(lat: float, lng: float) -> tuple[LocationQuery, list, Provider, dict]:
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
    query = await asyncio.to_thread(_location_query, ctx)
    timings["query_ms"] = round((time.monotonic() - started) * 1000)
    tracks = await asyncio.to_thread(
        _blend_search, query.place, support_words(ctx, query.place), ctx.weather, ctx.temperature_c,
        ctx.time_of_day, datetime.now(KST).month,
    )
    return query, tracks, provider, timings


async def recommend(request: LocationRecommendationRequest) -> LocationRecommendationResponse:
    started = time.monotonic()
    try:
        async with asyncio.timeout(REQUEST_TIMEOUT_S):
            query, tracks, provider, timings = await _recommend(request.latitude, request.longitude)
    except TimeoutError as error:
        # NOTE: 스레드에서 돌던 LLM·DB 호출은 취소되지 않고 끝까지 실행된 뒤 버려진다.
        logger.warning(
            "Location request timed out: request_id=%s elapsed_ms=%d",
            request.request_id, round((time.monotonic() - started) * 1000),
        )
        raise _error(503, "LOCATION_TIMEOUT", "위치 추천 처리 시간이 초과되었습니다. 잠시 후 다시 시도하세요.") from error

    # NOTE: 좌표·지역명·검색어·message는 위치를 드러내므로 남기지 않는다. 경로·결과·소요 시간만 기록한다.
    logger.info(
        "Location request: request_id=%s provider=%s tracks=%d sources_ms=%d query_ms=%d total_ms=%d",
        request.request_id, provider, len(tracks), timings["sources_ms"], timings["query_ms"],
        round((time.monotonic() - started) * 1000),
    )
    return LocationRecommendationResponse(message=query.message, tracks=tracks)
