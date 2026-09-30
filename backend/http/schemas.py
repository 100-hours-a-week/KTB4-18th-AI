from typing import Any

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """API 공통 오류 응답."""

    code: str
    message: str
    details: dict[str, Any] | None = None
