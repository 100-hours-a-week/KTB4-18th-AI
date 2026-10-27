"""위치 추천 API의 요청 검증·소스별 실패 처리·오류 사유·로그를 외부 호출 없이 검증한다."""

import asyncio
import logging

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from backend.location import service
from backend.location.query import LocationQuery
from backend.location.sources import DENSITY_CODES
from backend.main import app
from db.types import Track

URL = "/v1/location/recommendations"
HAEUNDAE = {"latitude": 35.15871, "longitude": 129.16042}
# calls 픽스처가 가짜로 바꾸기 전의 실제 함수. LLM 오류 분기를 검증할 때 되돌려 쓴다
REAL_MUSIC_QUERY = service._music_query


def kakao_raw(food_count=300):
    density = {code: {"meta": {"total_count": 0}, "documents": []} for code in DENSITY_CODES}
    density["FD6"]["meta"]["total_count"] = food_count
    return {
        "region": "부산광역시 해운대구 우1동",
        "density": density,
        "attractions": [{"category_name": "여행 > 관광,명소 > 해수욕장,해변", "distance": "53", "place_name": "해운대해수욕장"}],
        "parks": [],
    }


def weather_raw():
    return {
        "current": {"time": "2026-10-27T14:00", "temperature_2m": 20.0, "weather_code": 0},
        "daily": {"sunrise": ["2026-10-27T06:40"], "sunset": ["2026-10-27T17:35"]},
    }


def track():
    return Track(track_id="1", title="바다", artist="가수", artwork_url=None, preview_url=None, store_url=None)


@pytest.fixture
def calls(monkeypatch):
    """외부 소스·검색어·검색을 가짜로 바꾼다. 테스트는 이 dict를 고쳐 실패를 흉내 낸다.
    검색어 단계에 넘어간 LocationContext는 state["contexts"]에 쌓인다(응답에는 컨텍스트가 없다)."""
    state = {"kakao": kakao_raw(), "weather": weather_raw(), "relief": {"min_m": 0, "max_m": 30, "relief_m": 30},
             "osm": None, "contexts": [], "searched": []}

    def source(name):
        async def fetch(_lat, _lng):
            if isinstance(state[name], Exception):
                raise state[name]
            return state[name]
        return fetch

    monkeypatch.setattr(service.sources, "fetch_kakao", source("kakao"))
    monkeypatch.setattr(service.sources, "fetch_weather", source("weather"))
    monkeypatch.setattr(service.sources, "fetch_relief", source("relief"))
    monkeypatch.setattr(service.osm, "available", lambda: state["osm"] is not None)
    monkeypatch.setattr(service.osm, "measure", lambda _lat, _lng: state["osm"])

    def music_query(ctx):
        state["contexts"].append(ctx)
        return "맑은 오후 바닷가에서 듣기 좋은 노래"

    def search(query):
        state["searched"].append(query)
        return [track()]

    monkeypatch.setattr(service, "_music_query", music_query)
    monkeypatch.setattr(service, "_search", search)
    return state


@pytest.fixture
def llm(monkeypatch):
    """_music_query의 실제 분기를 타도록 키·모델을 넣고 llm_query만 바꿔 끼운다"""
    monkeypatch.setenv("OPENROUTER_LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setattr(service, "_music_query", REAL_MUSIC_QUERY)

    def use(behavior):
        monkeypatch.setattr(service, "llm_query", lambda _client, _ctx: behavior())
    return use


def reason(response):
    return response.json()["details"]["reason"]


def test_success_returns_only_tracks(calls):
    response = TestClient(app).post(URL, json=HAEUNDAE)

    assert response.status_code == 200
    assert response.json() == {"tracks": [track().model_dump()]}
    ctx = calls["contexts"][0]
    assert ctx.provider == "kakao"
    assert [hit.kind for hit in ctx.nature] == ["sea"]
    assert ctx.weather == "clear" and ctx.time_of_day == "afternoon"


def test_osm_is_combined_when_available(calls):
    calls["osm"] = {
        "nearest": {"park": [{"distance_m": 0, "area_m2": 50_000, "name": "큰공원", "sub": "park", "ele": None}]},
        "landuse": ["commercial"], "protected": [], "peaks": [],
    }

    TestClient(app).post(URL, json=HAEUNDAE)

    ctx = calls["contexts"][0]
    assert ctx.provider == "kakao_osm"
    assert {hit.kind for hit in ctx.nature} == {"park", "sea"}
    assert ctx.place_traits[0] == "업무지구"
    # 넓은 공원 안이고 busy(300개)라 한 단계 한산하게 본다
    assert ctx.density == "moderate"


def test_both_place_sources_failing_returns_503(calls):
    calls["kakao"] = RuntimeError("kakao down")

    response = TestClient(app).post(URL, json=HAEUNDAE)

    assert response.status_code == 503
    assert reason(response) == "LOCATION_CONTEXT_UNAVAILABLE"
    assert response.json()["message"].endswith("다시 시도하세요.")
    assert calls["searched"] == []


def test_weather_failure_continues_without_weather(calls):
    calls["weather"] = RuntimeError("weather down")

    response = TestClient(app).post(URL, json=HAEUNDAE)

    assert response.status_code == 200
    assert calls["contexts"][0].weather is None
    assert calls["contexts"][0].temperature_c is None


def test_relief_failure_skips_only_mountain(calls):
    calls["relief"] = RuntimeError("elevation down")
    calls["osm"] = {"nearest": {}, "landuse": [], "protected": [], "peaks": []}

    response = TestClient(app).post(URL, json=HAEUNDAE)

    assert response.status_code == 200
    assert "mountain" not in {hit.kind for hit in calls["contexts"][0].nature}


def test_llm_not_configured_returns_model_unavailable(calls, monkeypatch):
    monkeypatch.delenv("OPENROUTER_LLM_API_KEY", raising=False)
    monkeypatch.setattr(service, "_music_query", REAL_MUSIC_QUERY)

    response = TestClient(app).post(URL, json=HAEUNDAE)

    assert response.status_code == 503
    assert reason(response) == "MODEL_UNAVAILABLE"
    assert calls["searched"] == []


def test_llm_provider_error_returns_model_unavailable(calls, llm):
    def fail():
        raise openai.APIConnectionError(request=httpx.Request("POST", "https://test.invalid"))
    llm(fail)

    response = TestClient(app).post(URL, json=HAEUNDAE)

    assert response.status_code == 503
    assert reason(response) == "MODEL_UNAVAILABLE"


def test_llm_invalid_response_returns_model_response_invalid(calls, llm):
    llm(lambda: LocationQuery.model_validate_json(""))

    response = TestClient(app).post(URL, json=HAEUNDAE)

    assert response.status_code == 503
    assert reason(response) == "MODEL_RESPONSE_INVALID"


def test_unexpected_llm_bug_is_not_swallowed(calls, llm):
    def bug():
        raise KeyError("programming error")
    llm(bug)

    with pytest.raises(KeyError):
        TestClient(app).post(URL, json=HAEUNDAE)


def test_request_over_time_limit_returns_timeout(calls, monkeypatch):
    async def slow(_lat, _lng):
        await asyncio.sleep(1)
        return kakao_raw()

    monkeypatch.setattr(service.sources, "fetch_kakao", slow)
    monkeypatch.setattr(service, "REQUEST_TIMEOUT_S", 0.05)

    response = TestClient(app).post(URL, json=HAEUNDAE)

    assert response.status_code == 503
    assert reason(response) == "LOCATION_TIMEOUT"


@pytest.mark.parametrize("body", [
    {"latitude": 95, "longitude": 129.1},
    {"latitude": 35.1},
    {**HAEUNDAE, "exclude_track_ids": ["1"]},
])
def test_invalid_request_is_rejected(calls, body):
    response = TestClient(app).post(URL, json=body)

    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_REQUEST"
    assert calls["searched"] == []


def test_log_has_no_location(calls, caplog):
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        TestClient(app).post(URL, json=HAEUNDAE)

    assert "Location request: provider=kakao tracks=1" in caplog.text
    for leaked in ("35.1", "129.1", "해운대", "바닷가"):
        assert leaked not in caplog.text
