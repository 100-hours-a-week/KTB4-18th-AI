"""Spring Backend용 V1 API와 로컬 테스트 UI를 제공하는 FastAPI 서버."""

from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.recommendation import prepare_recommendation, stream_answer
from backend.schemas import ChatRequest, HealthResponse, ErrorResponse


# ===== 애플리케이션 설정 =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)

app = FastAPI(title="머문음 AI 채팅", version="0.2.0")


# ===== API 엔드포인트 =====
@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    """AI 서버 실행 상태를 반환함(Health Check)."""

    return HealthResponse(status="ok")


@app.post(
    "/v1/chat/messages",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"text/event-stream": {}}},
        422: {
            "model": ErrorResponse,
            "description": "요청값 검증 실패",
        },
        503: {
            "model": ErrorResponse,
            "description": "모델 또는 음악 검색 서비스 사용 불가",
        },
    },
    tags=["chat"],
)
def chat(body: ChatRequest) -> StreamingResponse:
    """추천 문장과 완성된 곡 목록을 Spring Backend에 SSE로 반환한다."""

    # TODO: request_id 중복 처리와 thread_id 대화 맥락은 책임 범위 확정 후 연결한다.
    client, search_context, tracks = prepare_recommendation(body.message)
    return StreamingResponse(
        stream_answer(client, body.message, search_context, tracks, body.user_context),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/v1/transcriptions", include_in_schema=False, tags=["transcriptions"])
def transcriptions() -> None:
    """Spring Backend가 전달한 음성 파일을 텍스트로 변환한다.

    현재는 구현하지 않고,
    클라이언트 측에서 녹음된 음성 파일 전송 정책(어떤 파일 확장자, 통신 방법)을 결정한 뒤 구현 예정.
    에러 처리 관련 상태 코드와 메세지는 확정 후 추가 수정이 필요.
    """

    # TODO: STT API 사용에 대한 근거를 조금 더 보충한 뒤, 사용할 API or 라이브러리를 확정지어 이어서 구현.
    raise HTTPException(status_code=501, detail="Not Implemented")


# ===== API 오류 처리 =====
@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    _request: Request, _error: RequestValidationError
) -> JSONResponse:
    """Pydantic 요청 검증 실패를 V1 공통 오류 응답으로 변환한다."""

    return JSONResponse(
        status_code=422,
        content={
            "code": "INVALID_REQUEST",
            "message": "요청값이 올바르지 않습니다.",
            "details": None,
        },
    )


@app.exception_handler(HTTPException)
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


# ===== 로컬 테스트 UI =====
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/app", StaticFiles(directory=STATIC_DIR, html=True), name="app")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """로컬 테스트 UI로 이동한다."""

    return RedirectResponse(url="/app/")
