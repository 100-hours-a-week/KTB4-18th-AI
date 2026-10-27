"""위치 컨텍스트와 위치 추천 API 스키마. 장소 공급자(카카오·Google·OSM)와 관계없이 같은 형식을 쓴다."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from db.types import Track

# kakao_osm: 밀도·지역·특징·이름은 카카오, 자연 지형 판정은 OSM + 고도 기복
# kakao·osm: 한쪽 공급자가 실패하거나 없을 때 남은 쪽만으로 판정한 결과
Provider = Literal["kakao", "google", "osm", "kakao_osm"]
Density = Literal["quiet", "moderate", "busy", "very_busy"]
# sea: 바다·해변, river: 강·호수·저수지, stream: 계곡·개울
Nature = Literal["sea", "river", "stream", "mountain", "forest", "park"]
TimeOfDay = Literal["dawn", "morning", "afternoon", "sunset", "night"]
Weather = Literal["clear", "cloudy", "fog", "rain", "snow", "storm"]


class NatureHit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Nature
    distance_m: int
    name: str | None = None  # 안내 문장용. 이름 없는 지형이 많다


class PlaceSummary(BaseModel):
    """공급자별 장소 원본을 정규화한 결과."""

    model_config = ConfigDict(extra="forbid")
    region: str | None
    density: Density
    nature: list[NatureHit]  # 종류별 가장 가까운 것만, 가까운 순
    place_traits: list[str] = Field(max_length=5)  # 자연 외 대표 장소 유형. 현재 카카오만 채운다


class WeatherSummary(BaseModel):
    """날씨 API가 실패하면 weather·temperature_c는 None이다. 없는 날씨를 지어내지 않는다"""

    model_config = ConfigDict(extra="forbid")
    time_of_day: TimeOfDay
    weather: Weather | None
    temperature_c: float | None


class LocationContext(BaseModel):
    """LLM에 넘기는 위치 특징. 정확한 좌표는 포함하지 않는다."""

    model_config = ConfigDict(extra="forbid")
    provider: Provider
    region: str | None
    density: Density
    nature: list[NatureHit]
    place_traits: list[str]
    time_of_day: TimeOfDay
    weather: Weather | None
    temperature_c: float | None


# ── API ───────────────────────────────────────────────
class LocationRecommendationRequest(BaseModel):
    """새 격자에 들어왔다고 FE·BE가 판단한 좌표. 표시 조건(10분 주기·하루 3회 등)은 FE·BE가 맡는다"""

    model_config = ConfigDict(extra="forbid")
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class LocationRecommendationResponse(BaseModel):
    tracks: list[Track]  # V1·V2와 같은 공통 track 객체, 0~5개
