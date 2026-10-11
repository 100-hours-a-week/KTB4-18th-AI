"""오픈웨이트 엔드포인트에 실제 파이프라인 프롬프트(분기·답변)를 보내 보는 확인용 스크립트.

분기는 json_schema로 형식이 지켜지는지, 답변은 스트리밍 TTFT가 얼마인지 본다.
.env의 LLM_FALLBACK_BASE_URL·LLM_FALLBACK_API_KEY·LLM_FALLBACK_MODEL을 읽는다.

사용: .venv/bin/python serving/smoke_test.py ["사용자 메시지"]
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from openai import OpenAI  # noqa: E402

from backend.chat.streaming import stream_answer  # noqa: E402
from backend.providers.llm import chat_create_kwargs  # noqa: E402
from backend.recommendation import intent  # noqa: E402
from db.types import Track  # noqa: E402

SAMPLE_TRACKS = [
    Track(track_id="1", title="밤편지", artist="IU", artwork_url=None, preview_url=None,
          reason="잔잔한 어쿠스틱 사운드가 차분한 분위기와 어울려요."),
    Track(track_id="2", title="Square (2017)", artist="백예린", artwork_url=None, preview_url=None,
          reason="몽환적인 보컬이 퇴근길의 여유로운 분위기와 닮았어요."),
]


class CaptureClient:
    """OpenAI client인 척 responses.create 인자만 받아두고 멈춘다.

    classify·stream_answer가 실제로 만드는 프롬프트를 그대로 가져오려고 쓴다.
    """

    def __init__(self) -> None:
        self.kwargs: dict | None = None
        self.responses = self

    def create(self, **kwargs):
        self.kwargs = kwargs
        raise RuntimeError("captured")


def capture_classify(message: str) -> dict:
    client = CaptureClient()
    try:
        intent.classify(client, [message])
    except Exception:
        pass
    return client.kwargs


def capture_answer(message: str) -> dict:
    client = CaptureClient()
    list(stream_answer(client, message, SAMPLE_TRACKS, None))
    return client.kwargs


def main() -> None:
    message = sys.argv[1] if len(sys.argv) > 1 else "퇴근길에 듣기 좋은 잔잔한 노래 추천해줘"
    client = OpenAI(base_url=os.environ["LLM_FALLBACK_BASE_URL"],
                    api_key=os.environ["LLM_FALLBACK_API_KEY"], timeout=300, max_retries=0)
    model = os.environ["LLM_FALLBACK_MODEL"]
    extra = json.loads(os.getenv("LLM_FALLBACK_EXTRA_BODY")
                       or '{"chat_template_kwargs": {"enable_thinking": false}}')

    print(f"■ 분기 (json_schema) — {message}")
    start = time.perf_counter()
    completion = client.chat.completions.create(**chat_create_kwargs(capture_classify(message), model, extra))
    elapsed = time.perf_counter() - start
    content = completion.choices[0].message.content
    try:
        print(json.dumps(json.loads(content), ensure_ascii=False, indent=2))
    except json.JSONDecodeError:
        print("JSON 파싱 실패:", content)
    print(f"  {elapsed:.2f}초, 입력 {completion.usage.prompt_tokens} / 출력 {completion.usage.completion_tokens} 토큰\n")

    print("■ 답변 (stream)")
    start = time.perf_counter()
    ttft = None
    usage = None
    for chunk in client.chat.completions.create(**chat_create_kwargs(
            {**capture_answer(message), "stream": True}, model, extra)):
        if chunk.usage:
            usage = chunk.usage
        if chunk.choices and chunk.choices[0].delta.content:
            ttft = ttft or time.perf_counter() - start
            print(chunk.choices[0].delta.content, end="", flush=True)
    elapsed = time.perf_counter() - start
    print(f"\n  TTFT {ttft:.2f}초, 총 {elapsed:.2f}초, "
          f"입력 {usage.prompt_tokens} / 출력 {usage.completion_tokens} 토큰")


if __name__ == "__main__":
    main()
