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
        return validate_embedding(response)
    except Exception as error:
        raise _model_unavailable() from error


def validate_embedding(response) -> list[float]:
    """동기·비동기 호출이 같은 차원과 유한 벡터 계약을 사용한다."""
    vector = response.data[0].embedding if response.data else None
    if (not vector or len(vector) != 3072
            or not all(math.isfinite(x) for x in vector) or not any(vector)):
        raise ValueError("Invalid embedding")
    return vector
