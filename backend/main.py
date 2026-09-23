"""Spring Backend용 V1 API와 로컬 테스트 UI를 제공하는 FastAPI 서버."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.recommendation import _catalog_unavailable, prepare_recommendation, sse, stream_answer
from backend.schemas import ChatRequest, ErrorResponse, HealthResponse, TranscriptionResponse
from backend.transcriptions import transcribe_audio

# ===== 애플리케이션 설정 =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)

app = FastAPI(title="머문음 AI 채팅", version="0.2.0")


# ===== API 엔드포인트 =====
@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    """AI 서버 실행 상태를 반환함(Health Check)."""

    return HealthResponse(status="ok")


@app.get(
    "/health/ready",
    response_model=HealthResponse,
    responses={503: {"model": ErrorResponse, "description": "DB 연결 불가"}},
    tags=["health"],
)
def readiness() -> HealthResponse:
    """DB(pgvector) 연결까지 확인하는 readiness 엔드포인트."""

    try:
        # NOTE: main.py의 load_dotenv()보다 먼저 실행되면 db.models가 읽는
        # DATABASE_URL이 아직 없을 수 있어, 요청 처리 시점까지 import를 늦춘다.
        from sqlalchemy import text

        from db.models import SessionLocal

        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
    except Exception as error:
        raise _catalog_unavailable() from error

    return HealthResponse(status="ok")


@app.post(
    "/v1/chat/messages",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"text/event-stream": {}}},
        400: {
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

    # TODO: request_id 중복 처리는 책임 범위 확정 후 연결한다.
    # thread_id는 LangGraph checkpointer의 대화 세션 키로 흘려보낸다.
    result = prepare_recommendation(body.message, str(body.thread_id))

    # NOTE: 추천이 아니면 단순 str로 받으니 이렇게 처리한다.
    if isinstance(result, str):
        # TODO: guide/clarify/lookup 분기가 생기기 전까지의 임시 처리.
        def temporary_answer() -> Iterator[str]:
            yield sse("text", {"delta": result})
            yield sse("tracks", {"tracks": []})
            yield sse("done", {})

        return StreamingResponse(
            temporary_answer(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    client, tracks = result
    return StreamingResponse(
        stream_answer(client, body.message, tracks, body.user_context),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post(
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


# ===== API 오류 처리 =====
@app.exception_handler(RequestValidationError)
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
