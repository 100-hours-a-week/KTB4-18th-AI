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
    supplied_key: Annotated[str | None, Header(alias="X-AI-API-Key")] = None,
) -> None:
    """키 누락·불일치를 동일한 공통 오류로 반환한다."""
    expected_key = service_api_key()
    if supplied_key is None or not secrets.compare_digest(
        supplied_key.encode("utf-8"), expected_key.encode("utf-8")
    ):
        raise HTTPException(
            status_code=401,
            detail={
                "code": "UNAUTHORIZED",
                "message": "서버 인증에 실패했습니다.",
                "details": None,
            },
        )
