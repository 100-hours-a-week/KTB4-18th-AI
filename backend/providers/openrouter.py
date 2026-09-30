"""OpenRouter 임베딩 클라이언트 생성."""

import os
from openai import OpenAI
from backend.core.errors import _model_unavailable

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_embedding_client() -> OpenAI:
    """OPENROUTER_EMBEDDING_API_KEY로 embed_query용 OpenAI(OpenRouter) client를 생성한다."""

    api_key = os.getenv("OPENROUTER_EMBEDDING_API_KEY", "").strip()
    if not api_key or api_key.startswith("<"):
        raise _model_unavailable()
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL,
                  timeout=30.0, max_retries=0)
