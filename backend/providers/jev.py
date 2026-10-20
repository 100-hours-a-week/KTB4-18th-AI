"""OpenRouter Decisions API로 Jev(결정 전용 모델)를 호출한다.

NOTE: Jev는 chat/completions가 아니라 전용 엔드포인트(/api/alpha/decisions)를 써서
OpenAI 클라이언트 대신 httpx로 직접 호출한다. 키는 LLM과 같은 OPENROUTER_LLM_API_KEY.
"""

import os

import httpx

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"

# NOTE: 요청마다 TLS 연결을 새로 맺지 않도록 프로세스 동안 클라이언트 하나를 재사용한다.
# 고속 모드용이라 timeout을 짧게 두고, 재시도는 하지 않는다.
_client = httpx.Client(timeout=5.0)


class JevUnavailable(Exception):
    """Jev를 쓸 수 없음(설정 없음, 호출 실패, 응답 형식 이상). 호출 측은 classify로 폴백한다."""


def decide(state: str | list, questions: dict) -> dict:
    """state(판단할 맥락)와 questions를 보내 질문 이름별 answers를 반환한다."""

    api_key = os.getenv("OPENROUTER_LLM_API_KEY", "").strip()
    model = os.getenv("JEV_MODEL", "").strip()
    if not api_key or api_key.startswith("<") or not model:
        raise JevUnavailable("OPENROUTER_LLM_API_KEY 또는 JEV_MODEL이 없음")

    # NOTE: alpha 엔드포인트라 응답 형식이 바뀔 수 있어서 JSON 파싱 실패도 장애로 본다.
    try:
        response = _client.post(
            DECISIONS_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "state": state, "questions": questions},
        )
        response.raise_for_status()
        return response.json()["answers"]
    except (httpx.HTTPError, ValueError, KeyError) as error:
        raise JevUnavailable(str(error)) from error
