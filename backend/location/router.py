from fastapi import APIRouter

from backend.http.schemas import ErrorResponse
from backend.location.schemas import LocationRecommendationRequest, LocationRecommendationResponse
from backend.location.service import recommend

router = APIRouter()


@router.post(
    "/v1/location/recommendations",
    response_model=LocationRecommendationResponse,
    responses={
        422: {"model": ErrorResponse, "description": "위도·경도 누락 또는 범위 오류 등 요청값 검증 실패"},
        503: {
            "model": ErrorResponse,
            "description": "LOCATION_CONTEXT_UNAVAILABLE(장소 정보 확보 실패), MODEL_UNAVAILABLE(LLM·임베딩), "
                           "MODEL_RESPONSE_INVALID(LLM 응답 형식), MUSIC_CATALOG_UNAVAILABLE(DB), "
                           "LOCATION_TIMEOUT(90초 초과)",
        },
    },
    tags=["location"],
)
async def location_recommendations(request: LocationRecommendationRequest) -> LocationRecommendationResponse:
    """새 지역에 들어온 사용자의 좌표로 주변 장소·날씨를 판단해 어울리는 곡을 최대 5개 추천한다."""

    return await recommend(request)
