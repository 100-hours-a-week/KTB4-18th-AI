"""모의 테스트가 개발·운영 트레이스에 전송되지 않게 한다."""

import os

import pytest
from langsmith import tracing_context

for key in ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2"):
    os.environ[key] = "false"


@pytest.fixture(autouse=True)
def disable_external_tracing():
    with tracing_context(enabled=False):
        yield
