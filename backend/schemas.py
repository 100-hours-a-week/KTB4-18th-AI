"""Spring Backend와 AI 서버 사이의 API 스키마.

현재는 V1 API 계약을 기준으로 Pydantic 모델을 정의한다.
Spring Backend에서 API 계약이 변경되면 이 파일도 함께 갱신해야 한다.
"""

from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class UserContext(BaseModel):
    """추천 품질을 보조하는 선택적 사용자 정보 모델."""

    # TODO: Spring Backend에서 사용자 정보를 수집하는 정책이 변경되면 이 모델을 함께 갱신
    age: int | None = None
    gender: str | None = None
    preferred_genres: list[str] | None = None


class ChatRequest(BaseModel):
    """사용자가 전송한 텍스트 기반 음악 추천 요청 모델."""

    thread_id: UUID     # TODO: Spring Backend에서 대화 맥락을 관리하는 정책이 변경되면 이 모델을 함께 갱신
    request_id: UUID    # TODO: Spring Backend에서 중복 요청을 처리하는 정책이 변경되면 이 모델을 함께 갱신
    message: str = Field(max_length=300)
    user_context: UserContext | None = None

    # message 필드에 대한 아래 함수를 message 전용 검증기로 등록하는 데코레이터
    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("입력은 비어있을 수 없습니다.")
        return value


class Track(BaseModel):
    """추천곡에 대한 Pydantic 모델."""

    # TODO: iTunes API에서 제공하는 다른 필드가 필요하면 API 계약과 함께 검토
    track_id: str  # iTunes track ID
    title: str  # iTunes track title
    artist: str  # iTunes track artist
    artwork_url: str | None  # iTunes album artwork URL (600x600)
    preview_url: str | None  # iTunes track preview URL (30초 미리듣기)
    store_url: str  # iTunes Store 링크
    reason: str  # 현재 입력과 해당 곡이 어울리는 이유


class Voice(BaseModel):
    """Spring Backend가 전달한 음성 파일에 대한 Pydantic 모델."""

    # TODO: Spring Backend에서 음성 파일을 전달하는 정책이 변경되면 이 모델을 함께 갱신.
    # + Pydantic 검증기 구현이 필요.
    filename: str
    content_type: str
    data: bytes


class HealthResponse(BaseModel):
    """AI 서버 상태 확인 응답 모델."""

    status: str
