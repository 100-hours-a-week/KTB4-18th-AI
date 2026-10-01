from fastapi import APIRouter, Response
from pydantic import BaseModel
from backend.health.readiness import database_is_ready


class HealthResponse(BaseModel):
    """AI 서버 상태 확인 응답 모델."""

    status: str

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    """외부 의존성 확인 없이 서버 생존 여부를 반환한다."""

    return HealthResponse(status="ok")


@router.get(
    "/readiness",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "DB 또는 추천 데이터 준비 안 됨"}},
    tags=["health"],
)
def readiness(response: Response) -> HealthResponse:
    """DB 연결과 추천 가능 곡 존재 여부를 확인해 배포 준비 상태를 반환한다."""
    ready = database_is_ready()
    response.status_code = 200 if ready else 503
    return HealthResponse(status="ready" if ready else "not_ready")
