"""단계별 오픈웨이트 폴백 라우터(FallbackLLM) 동작 확인."""

import json
from types import SimpleNamespace

import pytest

from backend.chat.streaming import stream_answer
from backend.providers import llm
from backend.providers.llm import Fallback, FallbackLLM, with_fallback
from backend.recommendation import intent
from db.types import Track

TRACK = Track(track_id="1", title="song", artist="artist", artwork_url=None,
              preview_url=None, reason="reason")


class FakePrimary:
    """OpenRouter client 대역. error가 있으면 responses.create에서 던진다."""

    def __init__(self, result=None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.calls: list[dict] = []
        self.responses = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.result


class FakeStream:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        return iter(self.events)


class FakeVLLM:
    """Modal vLLM 엔드포인트 대역. chat.completions.create만 흉내 낸다."""

    def __init__(self, content: str = "", deltas: tuple[str, ...] = ()):
        self.content = content
        self.deltas = deltas
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            chunks = [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=d))], usage=None)
                      for d in self.deltas]
            chunks.append(SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2)))
            return iter(chunks)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


def router(primary, vllm, slots=("answer",), **options) -> FallbackLLM:
    return FallbackLLM(primary=primary, fallback=Fallback(client=vllm, model="open-model"),
                       fallback_slots=set(slots), **options)


def event_names(stream) -> list[str]:
    return [event.split("\n", 1)[0].removeprefix("event: ") for event in stream]


def test_without_fallback_config_primary_client_is_used_as_is():
    """폴백 설정이 없으면 감싸지 않아 기존 동작이 그대로 유지된다."""
    primary = FakePrimary()

    assert with_fallback(primary) is primary


def test_env_example_placeholders_keep_fallback_off(monkeypatch):
    """.env.example을 그대로 복사해 <YOUR_...> 값이 남아 있으면 폴백을 켜지 않는다."""
    monkeypatch.setenv("LLM_FALLBACK_BASE_URL", "<YOUR_VLLM_BASE_URL>")
    monkeypatch.setenv("LLM_FALLBACK_EXTRA_BODY", "<YOUR_VLLM_EXTRA_SETTINGS>")
    primary = FakePrimary()

    assert with_fallback(primary) is primary


def test_classify_falls_back_with_converted_messages_and_schema(monkeypatch):
    """분기 호출이 실패하면 같은 지시·입력·json_schema를 chat completions 모양으로 바꿔 폴백한다."""
    monkeypatch.setenv("LLM_MODEL", "openai/test-model")
    monkeypatch.setattr(intent, "known_genres", lambda: [])
    classification = {
        "intent": "guide", "recommend_has_enough_info": False, "recommend_unsupported_condition": False,
        "recommend_genres": None, "recommend_min_year": None, "recommend_max_year": None,
        "recommend_query": "뭐 할 수 있어?", "lookup_song": None, "lookup_song_alt": None,
        "lookup_artist": None, "requested_count": None, "has_non_music_request": False,
        "response_style": None,
    }
    vllm = FakeVLLM(content=json.dumps(classification, ensure_ascii=False))

    result = intent.classify(router(FakePrimary(error=TimeoutError()), vllm, slots=("classify",)),
                             ["뭐 할 수 있어?"])

    assert result["intent"] == "guide"
    sent = vllm.calls[0]
    assert sent["model"] == "open-model"
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    assert sent["response_format"]["json_schema"]["name"] == "intent_classification"


def test_slot_outside_fallback_slots_raises_primary_error():
    """폴백 대상이 아닌 단계는 주 경로 실패를 그대로 올려 기존 오류 처리를 따른다."""
    vllm = FakeVLLM(content="{}")
    client = router(FakePrimary(error=TimeoutError()), vllm, slots=("answer",))

    with pytest.raises(TimeoutError):
        client.responses.create(text={"format": {"type": "json_schema", "name": "song_reasons", "schema": {}}})
    assert vllm.calls == []


def test_answer_stream_falls_back_before_first_token(monkeypatch):
    """첫 글자가 나가기 전에 주 경로가 실패하면 오픈웨이트 답변으로 이어서 정상 완료한다."""
    monkeypatch.setenv("LLM_MODEL", "test-model")
    vllm = FakeVLLM(deltas=("안녕", "하세요"))

    events = list(stream_answer(router(FakePrimary(error=ConnectionError()), vllm), "request", [TRACK], None))

    assert event_names(events) == ["text", "text", "tracks", "done"]
    assert '"안녕"' in events[0]
    assert vllm.calls[0]["stream"] is True


def test_answer_stream_does_not_fall_back_after_first_token(monkeypatch):
    """이미 글자가 나간 뒤 실패하면 다른 모델로 이어 쓰지 않고 실패로 끝낸다."""
    monkeypatch.setenv("LLM_MODEL", "test-model")
    primary = FakePrimary(result=FakeStream([
        SimpleNamespace(type="response.output_text.delta", delta="hello"),
        SimpleNamespace(type="response.failed"),
    ]))
    vllm = FakeVLLM(deltas=("안녕",))

    events = list(stream_answer(router(primary, vllm), "request", [TRACK], None))

    assert event_names(events) == ["text", "error"]
    assert vllm.calls == []


def test_fault_injection_fails_only_selected_slot_primary(monkeypatch):
    """LLM_FAULT_SLOTS로 고른 단계만 주 경로를 건너뛴 듯 실패시켜 부분 장애를 재현한다."""
    monkeypatch.setenv("LLM_MODEL", "test-model")
    primary = FakePrimary(result=FakeStream([]))
    vllm = FakeVLLM(deltas=("폴백",))

    events = list(stream_answer(router(primary, vllm, fault_slots={"answer"}), "request", [TRACK], None))

    assert event_names(events) == ["text", "tracks", "done"]
    assert primary.calls == []


def test_forced_route_skips_primary_for_fallback_slots_only():
    """LLM_ROUTE=fallback(오프라인 평가용)은 폴백 대상 단계만 처음부터 오픈웨이트로 보낸다."""
    primary = FakePrimary(result=SimpleNamespace(output_text="primary"))
    vllm = FakeVLLM(content="open")
    client = router(primary, vllm, slots=("classify",), force_fallback=True)

    classify_kwargs = {"text": {"format": {"type": "json_schema", "name": "intent_classification", "schema": {}}}}
    reason_kwargs = {"text": {"format": {"type": "json_schema", "name": "song_reasons", "schema": {}}}}

    assert client.responses.create(**classify_kwargs).output_text == "open"
    assert client.responses.create(**reason_kwargs).output_text == "primary"


def test_metrics_record_route_and_fallback_reason(monkeypatch, tmp_path):
    """호출마다 단계·경로·폴백 사유·토큰을 JSONL로 남긴다."""
    monkeypatch.setenv("LLM_MODEL", "test-model")
    path = tmp_path / "metrics.jsonl"
    client = router(FakePrimary(error=TimeoutError()), FakeVLLM(deltas=("a",)), metrics_path=str(path))

    list(stream_answer(client, "request", [TRACK], None))

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(r["slot"], r["route"], r["ok"]) for r in records] == [
        ("answer", "primary", False), ("answer", "fallback", True)]
    assert records[1]["fallback_reason"] == "TimeoutError"
    assert records[1]["output_tokens"] == 2


def test_with_fallback_reads_env(monkeypatch):
    """환경변수로 폴백 엔드포인트·대상 단계·thinking 옵션을 설정한다."""
    monkeypatch.setenv("LLM_FALLBACK_BASE_URL", "https://example.modal.run/v1")
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "Qwen/Qwen3-8B-FP8")
    monkeypatch.setenv("LLM_FALLBACK_SLOTS", "answer, classify")
    monkeypatch.setenv("LLM_FALLBACK_EXTRA_BODY", '{"chat_template_kwargs": {"enable_thinking": false}}')

    client = with_fallback(FakePrimary())

    assert isinstance(client, llm.FallbackLLM)
    assert client.fallback_slots == {"answer", "classify"}
    assert client.fallback.model == "Qwen/Qwen3-8B-FP8"
    assert client.fallback.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
