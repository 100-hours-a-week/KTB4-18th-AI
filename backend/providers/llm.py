"""LLM 호출 경로 — 단계별로 주 경로(OpenRouter) 실패 시 오픈웨이트(Modal vLLM)로 폴백한다.

임베딩·DB까지 모두 죽는 전체 장애가 아니라 "일부 단계만 실패하는" 경우를 가정한다.
호출하는 쪽(classify·assign_reasons·stream_answer)은 지금처럼 client.responses.create만
부른다. FallbackLLM이 같은 모양으로 받아 경로를 고르고, 폴백 경로에서는 vLLM이 확실히
지원하는 chat completions로 바꿔 호출한 뒤 결과를 Responses API 모양으로 되돌려준다.

환경변수:
  LLM_FALLBACK_BASE_URL    Modal 엔드포인트 + /v1 (비어 있으면 폴백 비활성)
  LLM_FALLBACK_API_KEY     serving/modal_vllm.py의 secret에 넣은 VLLM_API_KEY
  LLM_FALLBACK_MODEL       vLLM served model name (예: Qwen/Qwen3-8B-FP8)
  LLM_FALLBACK_SLOTS       폴백할 단계, 쉼표 구분 (classify,reason,answer). 기본 answer
  LLM_FALLBACK_EXTRA_BODY  chat completions에 그대로 넣을 JSON (예: Qwen3 thinking 끄기)
  LLM_FALLBACK_TIMEOUT     폴백 요청 타임아웃(초). 기본 60
  LLM_ROUTE                primary(기본) | fallback — fallback이면 폴백 단계를 처음부터 오픈웨이트로
  LLM_FAULT_SLOTS          실험용: 이 단계의 주 경로를 일부러 실패시킨다 (부분 장애 재현)
  LLM_METRICS_PATH         호출마다 경로·지연·토큰을 JSONL로 남길 파일 경로
"""

import json
import logging
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from openai import OpenAI

logger = logging.getLogger("llm.route")

# NOTE: 호출부 시그니처를 바꾸지 않으려고 단계(slot)를 요청 모양에서 추론한다. 스트리밍은
# 최종 답변뿐이고, 나머지는 json_schema 이름으로 구분한다. 새 LLM 호출을 추가하면
# 여기에 이름을 등록해야 폴백 대상이 된다.
_SCHEMA_SLOTS = {"intent_classification": "classify", "song_reasons": "reason"}
_FAIL_EVENTS = ("response.failed", "response.incomplete", "error")


class InjectedFault(RuntimeError):
    """LLM_FAULT_SLOTS로 주 경로를 일부러 실패시킬 때 쓰는 예외."""


class PrimaryStreamFailed(RuntimeError):
    """주 경로 스트림이 정상 완료되지 못했다. text_sent면 이미 글자가 사용자에게 나간 상태다."""

    def __init__(self, reason: str, text_sent: bool) -> None:
        super().__init__(reason)
        self.reason = reason
        self.text_sent = text_sent


@dataclass
class Fallback:
    client: OpenAI
    model: str
    extra_body: dict | None = None


def slot_of(kwargs: dict) -> str:
    """responses.create 인자로 어느 단계의 호출인지 판별한다."""

    if kwargs.get("stream"):
        return "answer"
    name = ((kwargs.get("text") or {}).get("format") or {}).get("name")
    return _SCHEMA_SLOTS.get(name, "unknown")


def _env(name: str) -> str:
    """환경변수를 읽되, .env.example의 <YOUR_...> placeholder는 설정되지 않은 것으로 본다."""

    value = os.getenv(name, "").strip()
    return "" if value.startswith("<") else value


def _env_set(name: str, default: str = "") -> set[str]:
    return {s.strip() for s in (_env(name) or default).split(",") if s.strip()}


def with_fallback(primary: Any) -> Any:
    """폴백·장애 주입 설정이 있으면 primary를 FallbackLLM으로 감싸고, 없으면 그대로 돌려준다."""

    base_url = _env("LLM_FALLBACK_BASE_URL")
    fault_slots = _env_set("LLM_FAULT_SLOTS")
    if not base_url and not fault_slots:
        return primary

    fallback = None
    if base_url:
        extra = _env("LLM_FALLBACK_EXTRA_BODY")
        fallback = Fallback(
            client=OpenAI(
                api_key=_env("LLM_FALLBACK_API_KEY") or "EMPTY",
                base_url=base_url,
                timeout=float(os.getenv("LLM_FALLBACK_TIMEOUT", "60")),
                # NOTE: Modal은 컨테이너가 0대일 때 요청을 기다리게 하지 않고 503으로 바로
                # 거절한다(부팅은 그 요청을 계기로 시작됨). 몇 초 재시도로는 수 분짜리
                # 부팅을 기다릴 수 없어 재시도하지 않고 바로 실패시킨다.
                max_retries=0,
            ),
            model=os.environ["LLM_FALLBACK_MODEL"],
            extra_body=json.loads(extra) if extra else None,
        )

    return FallbackLLM(
        primary=primary,
        fallback=fallback,
        fallback_slots=_env_set("LLM_FALLBACK_SLOTS", "answer"),
        force_fallback=os.getenv("LLM_ROUTE", "primary").strip() == "fallback",
        fault_slots=fault_slots,
        metrics_path=os.getenv("LLM_METRICS_PATH", "").strip() or None,
    )


class FallbackLLM:
    """OpenAI client처럼 responses.create를 받으면서 단계별로 경로를 고른다."""

    def __init__(
        self,
        primary: Any,
        fallback: Fallback | None,
        fallback_slots: set[str],
        force_fallback: bool = False,
        fault_slots: set[str] | None = None,
        metrics_path: str | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.fallback_slots = fallback_slots
        self.force_fallback = force_fallback
        self.fault_slots = fault_slots or set()
        self.metrics_path = metrics_path
        # NOTE: 호출부가 client.responses.create(...)로 부르므로 자기 자신을 responses로 둔다.
        self.responses = self

    def create(self, **kwargs: Any) -> Any:
        slot = slot_of(kwargs)
        if kwargs.get("stream"):
            return _EventStream(self._stream(slot, kwargs))
        return self._complete(slot, kwargs)

    # ── 경로 선택 ─────────────────────────────────────────────

    def _can_fallback(self, slot: str) -> bool:
        return self.fallback is not None and slot in self.fallback_slots

    def _skip_primary(self, slot: str) -> bool:
        return self.force_fallback and self._can_fallback(slot)

    def _primary_create(self, slot: str, kwargs: dict) -> Any:
        if slot in self.fault_slots:
            raise InjectedFault(f"injected fault: {slot}")
        return self.primary.responses.create(**kwargs)

    def _complete(self, slot: str, kwargs: dict) -> Any:
        """분기·이유 생성처럼 한 번에 받는 호출. 주 경로가 실패하면 폴백한다."""

        if self._skip_primary(slot):
            return self._complete_fallback(slot, kwargs, reason="forced")
        try:
            return self._complete_primary(slot, kwargs)
        except Exception as error:
            if not self._can_fallback(slot):
                raise
            logger.warning("[%s] 주 경로 실패(%s) → 오픈웨이트 폴백", slot, type(error).__name__)
            return self._complete_fallback(slot, kwargs, reason=type(error).__name__)

    def _complete_primary(self, slot: str, kwargs: dict) -> Any:
        """OpenRouter를 호출하고 결과를 기록한다. 실패하면 기록한 뒤 예외를 그대로 올린다."""

        start = time.perf_counter()
        try:
            response = self._primary_create(slot, kwargs)
        except Exception as error:
            self._record(slot, "primary", start, error=error)
            raise
        self._record(slot, "primary", start, usage=_responses_usage(response))
        return response

    def _complete_fallback(self, slot: str, kwargs: dict, reason: str) -> Any:
        """Modal vLLM을 호출하고, 결과를 classify 등이 읽는 output_text 모양으로 감싼다."""

        start = time.perf_counter()
        try:
            completion = self.fallback.client.chat.completions.create(
                **chat_create_kwargs(kwargs, self.fallback.model, self.fallback.extra_body))
        except Exception as error:
            self._record(slot, "fallback", start, reason=reason, error=error)
            raise
        self._record(slot, "fallback", start, reason=reason, usage=_chat_usage(completion))
        return SimpleNamespace(output_text=completion.choices[0].message.content or "")

    def _stream(self, slot: str, kwargs: dict) -> Iterator[Any]:
        """최종 답변 스트리밍. 첫 글자가 나가기 전에 주 경로가 실패하면 폴백한다."""

        if self._skip_primary(slot):
            yield from self._stream_fallback(slot, kwargs, reason="forced")
            return
        try:
            yield from self._stream_primary(slot, kwargs)
        except PrimaryStreamFailed as failure:
            # NOTE: 이미 일부 문장이 사용자에게 나간 뒤에는 다른 모델로 이어 쓸 수 없어서
            # 폴백하지 않는다. 예외는 호출부(stream_answer)가 error 이벤트로 바꾼다.
            if failure.text_sent or not self._can_fallback(slot):
                raise
            logger.warning("[%s] 주 경로 실패(%s) → 오픈웨이트 폴백", slot, failure.reason)
            yield from self._stream_fallback(slot, kwargs, reason=failure.reason)

    def _stream_primary(self, slot: str, kwargs: dict) -> Iterator[Any]:
        """OpenRouter 스트림 이벤트를 그대로 흘려보낸다. 정상 완료가 아니면 PrimaryStreamFailed를 던진다."""

        start = time.perf_counter()
        ttft = None  # 첫 글자가 나간 시점. None이면 아직 사용자에게 아무것도 안 나갔다
        reason = "incomplete"  # 완료 이벤트 없이 스트림이 끝나면 이 사유로 실패
        try:
            with self._primary_create(slot, kwargs) as stream:
                for event in stream:
                    if event.type == "response.output_text.delta":
                        if ttft is None:
                            ttft = time.perf_counter() - start
                        yield event
                    elif event.type == "response.completed":
                        usage = _responses_usage(getattr(event, "response", None))
                        self._record(slot, "primary", start, ttft=ttft, usage=usage)
                        yield event
                        return
                    elif event.type in _FAIL_EVENTS:
                        reason = event.type
                        break
        except Exception as error:
            reason = type(error).__name__

        self._record(slot, "primary", start, ttft=ttft, error=reason)
        raise PrimaryStreamFailed(reason, text_sent=ttft is not None)

    def _stream_fallback(self, slot: str, kwargs: dict, reason: str) -> Iterator[Any]:
        """Modal vLLM 스트림을 받아 Responses API 이벤트 모양으로 바꿔 흘려보낸다."""

        start = time.perf_counter()
        ttft = None
        usage = None
        try:
            stream = self.fallback.client.chat.completions.create(
                **chat_create_kwargs(kwargs, self.fallback.model, self.fallback.extra_body))
            for chunk in stream:
                if chunk.usage:
                    usage = _chat_usage(chunk)
                if chunk.choices and chunk.choices[0].delta.content:
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    yield SimpleNamespace(type="response.output_text.delta",
                                          delta=chunk.choices[0].delta.content)
        except Exception as error:
            self._record(slot, "fallback", start, reason=reason, ttft=ttft, error=error)
            raise
        self._record(slot, "fallback", start, reason=reason, ttft=ttft, usage=usage)
        yield SimpleNamespace(type="response.completed")

    # ── 기록 ─────────────────────────────────────────────────

    def _record(
        self,
        slot: str,
        route: str,
        start: float,
        reason: str | None = None,
        ttft: float | None = None,
        usage: dict | None = None,
        error: Exception | str | None = None,
    ) -> None:
        """호출 한 번의 경로·지연·토큰을 로그로 남기고, 설정되면 JSONL 파일에도 쌓는다."""

        record = {
            "ts": time.time(),
            "slot": slot,
            "route": route,
            "model": self.fallback.model if route == "fallback" else os.getenv("LLM_MODEL"),
            "fallback_reason": reason,
            "latency_s": round(time.perf_counter() - start, 3),
            "ttft_s": round(ttft, 3) if ttft is not None else None,
            "ok": error is None,
            "error": error if error is None or isinstance(error, str) else type(error).__name__,
            **(usage or {}),
        }
        logger.info(json.dumps(record, ensure_ascii=False))
        if self.metrics_path:
            with open(self.metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


class _EventStream:
    """제너레이터를 `with stream: for event in stream` 모양으로 쓸 수 있게 감싼다."""

    def __init__(self, events: Iterator[Any]) -> None:
        self._events = events

    def __enter__(self) -> "_EventStream":
        return self

    def __exit__(self, *_args: Any) -> bool:
        self._events.close()
        return False

    def __iter__(self) -> Iterator[Any]:
        return self._events


# ── Responses API → chat completions 변환 ────────────────────

def chat_messages(kwargs: dict) -> list[dict]:
    """responses.create의 instructions·input을 chat completions의 messages로 바꾼다."""

    messages = []
    if kwargs.get("instructions"):
        messages.append({"role": "system", "content": kwargs["instructions"]})
    content = kwargs.get("input")
    if isinstance(content, str):
        messages.append({"role": "user", "content": content})
    else:
        # input이 [{"role": ..., "content": ...}] 목록인 경우(대화 기록 등)
        messages.extend({"role": m["role"], "content": m["content"]} for m in content or [])
    return messages


def chat_response_format(kwargs: dict) -> dict | None:
    """responses.create의 text.format(json_schema)을 chat completions의 response_format으로 바꾼다."""

    fmt = (kwargs.get("text") or {}).get("format")
    if not fmt or fmt.get("type") != "json_schema":
        return None
    return {"type": "json_schema", "json_schema": {
        "name": fmt["name"], "schema": fmt["schema"], "strict": fmt.get("strict", True),
    }}


def chat_create_kwargs(kwargs: dict, model: str, extra_body: dict | None = None) -> dict[str, Any]:
    """responses.create에 넘어온 인자 전체를 chat.completions.create 인자로 바꾼다."""

    converted: dict[str, Any] = {"model": model, "messages": chat_messages(kwargs)}
    response_format = chat_response_format(kwargs)
    if response_format:
        converted["response_format"] = response_format
    if kwargs.get("stream"):
        converted["stream"] = True
        converted["stream_options"] = {"include_usage": True}
    if extra_body:
        converted["extra_body"] = extra_body
    return converted


def _responses_usage(response: Any) -> dict | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return {"input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None)}


def _chat_usage(completion: Any) -> dict | None:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return None
    return {"input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens}
