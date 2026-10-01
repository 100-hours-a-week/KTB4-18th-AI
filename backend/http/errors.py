from typing import Any
from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


async def validation_error_handler(
    _request: Request, _error: RequestValidationError
) -> JSONResponse:
    """Pydantic 요청 검증 실패를 V1 공통 오류 응답으로 변환한다."""

    return JSONResponse(
        status_code=400 if _request.url.path == "/v1/chat/messages" else 422,
        content={
            "code": "INVALID_REQUEST",
            "message": "요청값이 올바르지 않습니다.",
            "details": None,
        },
    )


async def http_error_handler(_request: Request, error: HTTPException) -> JSONResponse:
    """내부 HTTP 오류를 Spring Backend가 처리할 공통 형식으로 반환한다."""

    if isinstance(error.detail, dict):
        content: dict[str, Any] = error.detail
    else:
        content = {
            "code": "SERVICE_UNAVAILABLE" if error.status_code == 503 else "REQUEST_FAILED",
            "message": str(error.detail),
            "details": None,
        }
    return JSONResponse(status_code=error.status_code, content=content)
