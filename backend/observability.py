"""머문음 환경별 요청 추적과 공급자가 반환한 사용량 기록."""

import json
import logging
import os
import time

from langsmith import get_current_run_tree, trace, tracing_context
from langsmith.utils import get_env_var, get_tracer_agent_environment, get_tracer_agent_id, get_tracer_project

ENVIRONMENTS = {"local", "development", "staging", "production"}
TRACED_PATHS = {"/v1/chat/messages", "/v1/chat/images", "/v1/transcriptions"}
logger = logging.getLogger("uvicorn.error")


def configure_tracing() -> None:
    """Agent 환경에 연결된 프로젝트로 SDK와 LangGraph의 목적지를 통일한다."""
    environment = os.getenv("APP_ENV", "local").strip().lower()
    if environment not in ENVIRONMENTS:
        raise ValueError("APP_ENV must be local, development, staging or production")
    os.environ["APP_ENV"] = environment
    enabled = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    # LangGraph 자동 tracer가 Agent 주소와 project를 동시에 보내는 SDK 조합을 피한다.
    for key in ("LANGSMITH_AGENT_ID", "LANGSMITH_AGENT_ENVIRONMENT", "LANGCHAIN_PROJECT"):
        os.environ.pop(key, None)
    os.environ["LANGSMITH_PROJECT"] = f"meomuneum-{environment}"
    for key in ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2"):
        os.environ[key] = str(enabled).lower()
    get_env_var.cache_clear()
    get_tracer_project.cache_clear()
    get_tracer_agent_id.cache_clear()
    get_tracer_agent_environment.cache_clear()


def record_usage(usage, model: str) -> None:
    """추정 단가 대신 실제 응답의 토큰·비용을 기록하며 누락 비용은 0으로 채우지 않는다."""
    run = get_current_run_tree()
    if run is None:
        return
    run.add_metadata({"ls_provider": "openrouter", "ls_model_name": model})
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    if not isinstance(usage, dict):
        run.add_metadata({"usage_available": False, "cost_available": False})
        return
    tokens = {}
    for destination, names in {
        "input_tokens": ("input_tokens", "prompt_tokens"),
        "output_tokens": ("output_tokens", "completion_tokens"),
        "total_tokens": ("total_tokens",),
    }.items():
        value = next((usage[n] for n in names if usage.get(n) is not None), None)
        if value is not None:
            tokens[destination] = value
    for destination, names in {
        "input_token_details": ("input_tokens_details", "prompt_tokens_details"),
        "output_token_details": ("output_tokens_details", "completion_tokens_details"),
    }.items():
        details = next((usage[n] for n in names if isinstance(usage.get(n), dict)), {})
        tokens[destination] = {target: details[source] for source, target in
                               (("cached_tokens", "cache_read"), ("reasoning_tokens", "reasoning"), ("audio_tokens", "audio"))
                               if details.get(source) is not None}
    if usage.get("cost") is not None:
        tokens["total_cost"] = usage["cost"]
    run.set(usage_metadata=tokens)
    run.add_metadata({"usage_available": bool(tokens.get("total_tokens") or "input_tokens" in tokens),
                      "cost_available": usage.get("cost") is not None})


def stream_result(chunks: list[str]) -> dict:
    """안내 생성 span에는 SSE 문법 대신 완성 문장을 보관한다."""
    text = []
    for chunk in chunks:
        for block in chunk.split("\n\n"):
            lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
            if lines.get("event") == "text":
                text.append(json.loads(lines["data"]).get("delta", ""))
    return {"text": "".join(text)}


class RequestTracingMiddleware:
    """SSE가 끝날 때까지 요청 span을 유지하고 첫 안내·카드·거절 도착을 기록한다."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") not in TRACED_PATHS:
            return await self.app(scope, receive, send)
        enabled = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
        # 모의 테스트와 명시적으로 추적을 끈 실행은 LangGraph 자동 추적까지 차단한다.
        with tracing_context(enabled=enabled, project_name=os.getenv("LANGSMITH_PROJECT")):
            if not enabled:
                return await self.app(scope, receive, send)
            started = time.perf_counter()
            elapsed = lambda: round((time.perf_counter() - started) * 1000, 2)
            output, pending, request_body = {"done": False}, "", bytearray()
            environment = os.getenv("APP_ENV", "local")
            with trace("머문음 요청", inputs={"path": scope["path"]},
                       project_name=os.environ["LANGSMITH_PROJECT"],
                       _end_on_exit=False,
                       metadata={"environment": environment,
                                 "revision_id": os.getenv("APP_REVISION", "unknown"),
                                 "measurement_scope": "asgi_request_to_stream_end"}) as run:
                async def traced_receive():
                    message = await receive()
                    if message["type"] == "http.request" and scope["path"] == "/v1/chat/messages":
                        if len(request_body) < 16384:
                            request_body.extend(message.get("body", b"")[:16384 - len(request_body)])
                        if not message.get("more_body"):
                            try:
                                body = json.loads(request_body)
                                run.add_inputs({k: body[k] for k in ("message", "thread_id", "request_id", "user_context") if k in body})
                                run.add_metadata({"thread_id": body.get("thread_id"), "request_id": body.get("request_id")})
                            except (ValueError, TypeError):
                                pass
                    return message

                async def traced_send(message):
                    nonlocal pending
                    if message["type"] == "http.response.start":
                        output["http_status"] = message["status"]
                    elif message["type"] == "http.response.body":
                        pending += message.get("body", b"").decode("utf-8", errors="replace")
                        while "\n\n" in pending:
                            block, pending = pending.split("\n\n", 1)
                            fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
                            event = fields.get("event")
                            if event in ("text", "tracks", "rejected", "verdict"):
                                output.setdefault({"text": "first_text_ms", "tracks": "cards_ms",
                                                   "rejected": "rejection_ms", "verdict": "verdict_ms"}[event], elapsed())
                            if event in ("done", "error", "rejected", "verdict", "scene", "tracks"):
                                try:
                                    output[event] = json.loads(fields.get("data", "{}")) if event != "done" else True
                                except ValueError:
                                    output["invalid_event"] = True
                            if event == "text":
                                try:
                                    output["text"] = output.get("text", "") + json.loads(fields.get("data", "{}")).get("delta", "")
                                except ValueError:
                                    output["invalid_event"] = True
                        if not message.get("more_body") and scope["path"] == "/v1/transcriptions":
                            try:
                                output["transcription"] = json.loads(pending)
                            except ValueError:
                                pass
                            output["done"] = output.get("http_status", 500) < 400
                    await send(message)

                try:
                    await self.app(scope, traced_receive, traced_send)
                except BaseException as error:
                    output["exception"] = type(error).__name__
                    raise
                finally:
                    output["total_ms"] = elapsed()
                    output["outcome"] = ("error" if output.get("error") or output.get("http_status", 500) >= 400 else
                                         "provider_blocked" if output.get("rejected", {}).get("reason") == "provider_blocked" else
                                         "rejected" if output.get("rejected") else "completed" if output["done"] else "aborted")
                    run.end(outputs=output, error=output.get("exception"))
                    try:
                        # 입력은 receive 이후에 붙으므로 초기 post뿐 아니라 종료 patch에도 포함한다.
                        run.patch(exclude_inputs=False)
                    except Exception as error:
                        logger.warning("Request trace could not be sent: %s", type(error).__name__)
