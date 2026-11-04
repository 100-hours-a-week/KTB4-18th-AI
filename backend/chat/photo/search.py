"""사진 전용 비동기 임베딩·검색. 취소 시 호출과 DB 연결을 함께 정리한다."""
import os

import httpx
from openai import AsyncOpenAI
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from backend.core.errors import _model_unavailable, _catalog_unavailable
from backend.providers.openrouter import OPENROUTER_BASE_URL
from backend.recommendation.adapter import search_space, to_v4_query
from backend.recommendation.embedding import validate_embedding

EMBEDDING_TIMEOUT_S = 20
SEARCH_TIMEOUT_S = 15


async def embed_photo_query(query: str) -> list[float]:
    key = os.getenv('OPENROUTER_EMBEDDING_API_KEY', '').strip()
    if not key or key.startswith('<') or os.getenv('EMBEDDING_MODEL') != 'google/gemini-embedding-2':
        raise _model_unavailable()
    try:
        async with AsyncOpenAI(api_key=key, base_url=OPENROUTER_BASE_URL,
                               timeout=httpx.Timeout(15, connect=3), max_retries=0) as client:
            response = await client.embeddings.create(
                model='google/gemini-embedding-2', input=query,
                dimensions=3072, encoding_format='float')
            return validate_embedding(response)
    except Exception as error:
        raise _model_unavailable() from error


def photo_search_engine():
    # 별도 연결을 쓰므로 기존 풀의 오래된 연결·대기 작업에 묶이지 않는다.
    # NullPool은 취소·완료 후 연결을 재사용하지 않고 닫는다.
    from db.models import DATABASE_URL
    return create_async_engine(
        make_url(DATABASE_URL).set(drivername='postgresql+psycopg'), poolclass=NullPool,
        connect_args={'connect_timeout': 5, 'options': '-c statement_timeout=12000 -c lock_timeout=3000 -c default_transaction_read_only=on'},
    )


async def search_photo_tracks(vector: list[float], exclude_ids: set[int]):
    from db.search import search
    engine = photo_search_engine()
    try:
        space = search_space()
        query = to_v4_query(vector) if space == 'v4' else vector
        async with AsyncSession(engine) as session:
            # SQLAlchemy가 기존 검색 함수를 비동기 psycopg 연결에 연결한다.
            return await session.run_sync(search, query, exclude_ids=exclude_ids, space=space, k=5)
    except Exception as error:
        raise _catalog_unavailable() from error
    finally:
        await engine.dispose()
