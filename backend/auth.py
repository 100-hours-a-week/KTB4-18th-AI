"""Spring → AI 요청의 서버 간 전용 키 검증."""

import os
import secrets
from typing import Annotated

from fastapi import Header, HTTPException


def service_api_key() -> str:
    """인증 설정 누락 시 서버 시작과 요청 처리를 차단한다."""
    key = os.getenv("AI_SERVICE_API_KEY", "")
    if not key.strip():
        raise RuntimeError("AI_SERVICE_API_KEY 설정이 필요합니다.")
    return key


def require_service_api_key(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> None:
    """Bearer 형식과 비밀값을 검사하고 인증 실패를 공통 오류로 반환한다."""
    expected_key = service_api_key()
    parts = authorization.split() if authorization else []
    if (
        len(parts) != 2
        or parts[0].casefold() != "bearer"
        or not secrets.compare_digest(parts[1].encode("utf-8"), expected_key.encode("utf-8"))
    ):
        raise HTTPException(
            status_code=401,
            detail={
                "code": "UNAUTHORIZED",
                "message": "서버 인증에 실패했습니다.",
                "details": None,
            },
        )
