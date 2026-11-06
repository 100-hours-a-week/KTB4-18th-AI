"""평가 프로세스의 OpenRouter 응답만 관측한다. 요청 본문·인증 헤더는 저장하지 않는다."""

import contextvars
import json
import time
from datetime import datetime, timezone
from unittest.mock import patch
import httpx

CASE = contextvars.ContextVar("evaluation_case", default=None)


def parse_response(raw):
    objects = []
    try:
        objects.append(json.loads(raw))
    except (ValueError, UnicodeDecodeError):
        for line in raw.decode("utf-8", errors="replace").splitlines():
            if line.startswith("data: "):
                try:
                    objects.append(json.loads(line[6:]))
                except ValueError:
                    pass
    result = {}
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        obj = obj.get("response", obj)
        if not isinstance(obj, dict):
            continue
        for key in ("id", "model", "usage"):
            if obj.get(key) is not None:
                result[key] = obj[key]
    return result


def summarize_usage(calls):
    """공급자가 반환한 토큰만 합산한다. 누락 필드를 0으로 만들지 않는다."""
    totals = {}
    for key, aliases in {
        "input_tokens": ("input_tokens", "prompt_tokens"),
        "output_tokens": ("output_tokens", "completion_tokens"),
    }.items():
        values = [
            next(
                (
                    c.get("usage", {}).get(a)
                    for a in aliases
                    if c.get("usage", {}).get(a) is not None
                ),
                None,
            )
            for c in calls
        ]
        if values and all(value is not None for value in values):
            totals[key] = sum(values)
    return totals


class Capture:
    def __init__(self):
        self.records, self.pending = [], []

    def finish(self, request, response, raw, start, case):
        parsed = parse_response(raw)
        record = {
            "case_id": case,
            "endpoint": request.url.path,
            "http_status": response.status_code,
            "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            **parsed,
        }
        self.records.append(record)
        if parsed.get("id"):
            self.pending.append((record, request.headers.get("authorization", "")))

    def __enter__(self):
        owner = self
        self.patches = []
        # NOTE: openai SDK 3.x는 httpx가 아니라 httpx2로 요청한다. 둘 다 가로채야 텍스트 LLM·임베딩 비용이 잡힌다.
        modules = [httpx]
        try:
            import httpx2

            modules.append(httpx2)
        except ImportError:
            pass

        def interested(request):
            return request.method == "POST" and request.url.host == "openrouter.ai"

        for module in modules:
            self.patches += self._patch(module, owner, interested)
        for item in self.patches:
            item.start()
        self.started = datetime.now(timezone.utc).isoformat()
        self.clock = time.perf_counter()
        return self

    @staticmethod
    def _patch(module, owner, interested):
        original_sync, original_async = module.Client.send, module.AsyncClient.send

        def sync_send(client, request, **kwargs):
            if not interested(request):
                return original_sync(client, request, **kwargs)
            start, case = time.perf_counter(), CASE.get()
            response = original_sync(client, request, **kwargs)
            if response.is_stream_consumed:
                owner.finish(request, response, response.content, start, case)
            else:
                inner = response.stream

                # NOTE: 서비스는 완료 이벤트를 받으면 반복을 멈추고 닫는다. close에서도 한 번만 기록한다.
                class Stream(module.SyncByteStream):
                    parts, done = [], False

                    def record(self):
                        if not self.done:
                            self.done = True
                            owner.finish(request, response, b"".join(self.parts), start, case)

                    def __iter__(self):
                        self.parts = []
                        try:
                            for part in inner:
                                self.parts.append(part)
                                yield part
                        finally:
                            self.record()

                    def close(self):
                        self.record()
                        inner.close()

                response.stream = Stream()
            return response

        async def async_send(client, request, **kwargs):
            if not interested(request):
                return await original_async(client, request, **kwargs)
            start, case = time.perf_counter(), CASE.get()
            response = await original_async(client, request, **kwargs)
            if response.is_stream_consumed:
                owner.finish(request, response, response.content, start, case)
            else:
                inner = response.stream

                class Stream(module.AsyncByteStream):
                    parts, done = [], False

                    def record(self):
                        if not self.done:
                            self.done = True
                            owner.finish(request, response, b"".join(self.parts), start, case)

                    async def __aiter__(self):
                        self.parts = []
                        try:
                            async for part in inner:
                                self.parts.append(part)
                                yield part
                        finally:
                            self.record()

                    async def aclose(self):
                        self.record()
                        await inner.aclose()

                response.stream = Stream()
            return response

        return [
            patch.object(module.Client, "send", sync_send),
            patch.object(module.AsyncClient, "send", async_send),
        ]

    def __exit__(self, *args):
        self.wall_s = round(time.perf_counter() - self.clock, 3)
        for item in self.patches:
            item.stop()

    def save(self, path):
        # 세대별 실제 비용을 조회한다. 아직 집계되지 않은 호출은 추정값으로 대체하지 않는다.
        with httpx.Client(timeout=15) as client:
            for record, authorization in self.pending:
                usage = record.get("usage") or {}
                if usage.get("cost") is not None:
                    record.update(cost_usd=usage["cost"], cost_source="response_usage")
                    continue
                try:
                    response = client.get(
                        "https://openrouter.ai/api/v1/generation",
                        params={"id": record["id"]},
                        headers={"Authorization": authorization},
                    )
                    if response.status_code == 200:
                        data = response.json()["data"]
                        record.update(
                            cost_usd=data.get("total_cost"),
                            cost_source="generation_api",
                            provider=data.get("provider_name"),
                            native_tokens_prompt=data.get("native_tokens_prompt"),
                            native_tokens_completion=data.get("native_tokens_completion"),
                        )
                    else:
                        record["cost_lookup_status"] = response.status_code
                except httpx.HTTPError as error:
                    record["cost_lookup_error"] = type(error).__name__
        self.pending.clear()
        path.write_text(
            json.dumps(
                {"started": self.started, "wall_s": self.wall_s, "calls": self.records},
                ensure_ascii=False,
                indent=2,
            )
        )
