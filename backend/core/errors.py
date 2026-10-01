from fastapi import HTTPException


def _model_unavailable() -> HTTPException:
    """V1 모델 장애 응답을 반환."""

    return HTTPException(
        status_code=503,
        detail={
            "code": "SERVICE_UNAVAILABLE",
            "message": "일시적으로 AI 기능을 사용할 수 없습니다.",
            "details": {"reason": "MODEL_UNAVAILABLE"},
        },
    )


def _catalog_unavailable() -> HTTPException:
    """V1 음악 카탈로그 장애 응답을 반환한다."""

    return HTTPException(
        status_code=503,
        detail={
            "code": "SERVICE_UNAVAILABLE",
            "message": "일시적으로 음악 정보를 조회할 수 없습니다.",
            "details": {"reason": "MUSIC_CATALOG_UNAVAILABLE"},
        },
    )


def _error(status: int, reason: str, message: str) -> HTTPException:
    """V1 공통 오류 형식으로 전사 실패를 표현한다."""
    codes = {400: "INVALID_REQUEST", 413: "PAYLOAD_TOO_LARGE",
             415: "UNSUPPORTED_MEDIA_TYPE", 503: "SERVICE_UNAVAILABLE"}
    return HTTPException(status_code=status, detail={
        "code": codes[status], "message": message, "details": {"reason": reason},
    })


def _unavailable() -> HTTPException:
    return _error(503, "TRANSCRIPTION_SERVICE_UNAVAILABLE", "일시적으로 음성 전사를 사용할 수 없습니다.")
