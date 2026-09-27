"""테스트는 실제 비밀값 대신 서버 인증 전용 모의 값을 사용한다."""

import pytest


@pytest.fixture(autouse=True)
def service_auth_environment(monkeypatch):
    monkeypatch.setenv("AI_SERVICE_API_KEY", "test-service-key")
