"""OpenRouter Decisions API로 Jev(결정 전용 모델)를 호출한다.

NOTE: Jev는 chat/completions가 아니라 전용 엔드포인트(/api/alpha/decisions)를 써서
OpenAI 클라이언트 대신 httpx로 직접 호출한다. 키는 LLM과 같은 OPENROUTER_LLM_API_KEY.
"""

import os

import httpx

from backend.core.errors import _model_unavailable

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"

# NOTE: 요청마다 TLS 연결을 새로 맺지 않도록 프로세스 동안 클라이언트 하나를 재사용한다.
# 고속 모드용이라 timeout을 짧게 두고, 재시도는 하지 않는다.
_client = httpx.Client(timeout=5.0)


def decide(state: str | list, questions: dict) -> dict:
    """state(판단할 맥락)와 questions를 보내 질문 이름별 answers를 반환한다."""

    api_key = os.getenv("OPENROUTER_LLM_API_KEY", "").strip()
    model = os.getenv("JEV_MODEL", "").strip()
    if not api_key or api_key.startswith("<") or not model:
        raise _model_unavailable()

    try:
        response = _client.post(
            DECISIONS_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "state": state, "questions": questions},
        )
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise _model_unavailable() from error

    return response.json()["answers"]
