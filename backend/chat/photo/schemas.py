"""사진 장면 분석 결과 스키마."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Scene(BaseModel):
    """VLM 출력과 검색에 쓰는 제한된 장면 정보."""

    model_config = ConfigDict(extra="forbid", strict=True)
    usable: bool
    reject_reason: Literal["scene_unreadable", "not_a_place"] | None
    place_type: str = Field(max_length=80)
    visual_elements: list[str] = Field(max_length=6)
    lighting: Literal["daylight", "sunset", "night", "unknown"]
    mood: list[str] = Field(max_length=4)
    music_query: str = Field(max_length=200)
    recommendation_context: str = Field(max_length=300)

    @model_validator(mode="after")
    def consistent_result(self):
        """적합 판정과 거절 사유·검색 문장의 모순을 거절한다."""
        if self.usable and (self.reject_reason is not None or not self.music_query.strip() or not self.recommendation_context.strip()):
            raise ValueError("Usable scene needs a query and no rejection")
        if not self.usable and (self.reject_reason is None or self.music_query):
            raise ValueError("Rejected scene needs a reason and no query")
        if any(not value.strip() or len(value) > 60 for value in self.visual_elements + self.mood):
            raise ValueError("Invalid scene tag")
        return self
