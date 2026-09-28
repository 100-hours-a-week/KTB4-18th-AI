"""검색 질의 임베딩."""

import math
import os

from openai import OpenAI
from backend.core.errors import _model_unavailable


def embed_query(client: OpenAI, message: str) -> list[float]:
    """사용자 쿼리를 OpenRouter 경유 gemini-embedding-2 벡터로 바꾼다."""

    model = os.getenv("EMBEDDING_MODEL")
    if model != "google/gemini-embedding-2":
        raise _model_unavailable()

    # NOTE: tracks.emb_gemini의 3072차원과 맞춘다. 임베딩만으로 DB 초기화를 요구하지 않는다.
    # TODO: OpenRouter가 Gemini 임베딩에 dimensions를 실제로 반영하는지,
    # 결과 벡터가 direct Gemini API 호출과 수치까지 동일한지 실측 확인 필요.
    try:
        response = client.embeddings.create(
            model=model, input=message, dimensions=3072, encoding_format="float",
        )
        # NOTE: 반환된 벡터가 쓰레기 값인 경우 추가
        embedded_query = response.data[0].embedding if response.data else None
        if (not embedded_query or len(embedded_query) != 3072
                or not all(math.isfinite(x) for x in embedded_query) or not any(embedded_query)):
            raise ValueError("Invalid embedding")
    except Exception as error:
        raise _model_unavailable() from error
    if not embedded_query:
        raise _model_unavailable()
    return embedded_query
