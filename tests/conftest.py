import pytest

# NOTE: main.py가 .env를 읽기 때문에, 로컬 .env에 폴백 설정이 있으면 테스트 중에도
# 라우터가 켜져 주 경로 실패 시 실제 Modal 엔드포인트를 호출하게 된다. 테스트에서는
# 항상 꺼두고, 폴백을 시험하는 테스트만 직접 설정한다.
_ROUTING_ENV = (
    "LLM_FALLBACK_BASE_URL", "LLM_FALLBACK_API_KEY", "LLM_FALLBACK_MODEL", "LLM_FALLBACK_SLOTS",
    "LLM_FALLBACK_EXTRA_BODY", "LLM_FALLBACK_TIMEOUT", "LLM_ROUTE", "LLM_FAULT_SLOTS", "LLM_METRICS_PATH",
)


@pytest.fixture(autouse=True)
def _no_llm_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ROUTING_ENV:
        monkeypatch.delenv(name, raising=False)
