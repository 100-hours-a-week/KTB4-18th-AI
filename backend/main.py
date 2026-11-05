"""Spring Backend용 V1 API와 로컬 테스트 UI를 제공하는 FastAPI 서버."""

from pathlib import Path
from typing import cast

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response
from starlette.types import ExceptionHandler, Scope

from backend.chat.router import router as chat_router
from backend.chat.photo.router import router as photo_router
from backend.core.config import PROJECT_ROOT
from backend.health.router import router as health_router
from backend.http.errors import http_error_handler, validation_error_handler
from backend.stt.router import router as stt_router
from backend.observability import RequestTracingMiddleware, configure_tracing

# ===== 애플리케이션 설정 =====
load_dotenv(PROJECT_ROOT / ".env", override=False)
configure_tracing()

app = FastAPI(title="머문음 AI 채팅", version="0.2.0")
app.add_middleware(RequestTracingMiddleware)
# ===== API 엔드포인트 =====
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(photo_router)
app.include_router(stt_router)
# ===== API 오류 처리 =====
app.add_exception_handler(RequestValidationError, cast(ExceptionHandler, validation_error_handler))
app.add_exception_handler(HTTPException, cast(ExceptionHandler, http_error_handler))

# ===== 로컬 테스트 UI =====
STATIC_DIR = Path(__file__).parent / "static"


class RevalidatedStaticFiles(StaticFiles):
    """콘솔을 고친 뒤에도 브라우저가 예전 페이지를 쓰지 않도록 매번 서버에 변경 여부를 묻게 한다."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        # NOTE: no-cache는 저장을 막지 않는다. 바뀌지 않았으면 304로 저장본을 그대로 쓴다.
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/app", RevalidatedStaticFiles(directory=STATIC_DIR, html=True), name="app")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """로컬 테스트 UI로 이동한다."""

    return RedirectResponse(url="/app/")
