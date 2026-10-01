from fastapi import APIRouter, UploadFile
from backend.stt.schemas import TranscriptionResponse
from backend.stt.service import transcribe_audio
from backend.http.schemas import ErrorResponse

router = APIRouter()


@router.post(
    "/v1/transcriptions",
    response_model=TranscriptionResponse,
    responses={
        422: {
            "model": ErrorResponse,
            "description": "audio 파일 누락 등 요청값 검증 실패",
        },
        400: {"model": ErrorResponse, "description": "빈 음성, 길이 초과 또는 인식된 발화 없음"},
        413: {"model": ErrorResponse, "description": "음성 파일 용량 초과"},
        415: {"model": ErrorResponse, "description": "지원하지 않거나 손상된 음성 파일"},
        503: {"model": ErrorResponse, "description": "전사 서비스 사용 불가"},
    },
    tags=["transcriptions"],
)
def transcriptions(audio: UploadFile) -> TranscriptionResponse:
    """녹음 파일을 받아 입력창에 표시할 전사 초안을 JSON으로 반환한다."""

    # NOTE: FE는 최대 60초 녹음 후 multipart의 audio 필드로 파일을 전송한다.
    return TranscriptionResponse(transcript=transcribe_audio(audio))
