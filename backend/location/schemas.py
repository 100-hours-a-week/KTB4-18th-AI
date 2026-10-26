"""위치 컨텍스트 스키마. 장소 공급자(카카오·Google)와 관계없이 같은 형식을 쓴다."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Provider = Literal["kakao", "google"]
Density = Literal["quiet", "moderate", "busy", "very_busy"]
# sea: 바다·해변, river: 강·호수·저수지, stream: 계곡·개울
Nature = Literal["sea", "river", "stream", "mountain", "forest", "park"]
TimeOfDay = Literal["dawn", "morning", "afternoon", "sunset", "night"]
Weather = Literal["clear", "cloudy", "fog", "rain", "snow", "storm"]


class NatureHit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Nature
    distance_m: int


class PlaceSummary(BaseModel):
    """공급자별 장소 원본을 정규화한 결과."""

    model_config = ConfigDict(extra="forbid")
    region: str | None
    density: Density
    nature: list[NatureHit]  # 종류별 가장 가까운 것만, 가까운 순
    place_traits: list[str] = Field(max_length=5)  # 자연 외 대표 장소 유형. 현재 카카오만 채운다


class WeatherSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    time_of_day: TimeOfDay
    weather: Weather
    temperature_c: float


class LocationContext(BaseModel):
    """LLM에 넘기는 위치 특징. 정확한 좌표는 포함하지 않는다."""

    model_config = ConfigDict(extra="forbid")
    provider: Provider
    region: str | None
    density: Density
    nature: list[NatureHit]
    place_traits: list[str]
    time_of_day: TimeOfDay
    weather: Weather
    temperature_c: float
