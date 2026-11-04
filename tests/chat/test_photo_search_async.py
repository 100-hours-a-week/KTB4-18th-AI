import asyncio
from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI

from backend.chat.photo import search


@pytest.mark.parametrize('dimension', [3072, 512])
def test_async_embedding_validates_provider_result(monkeypatch, dimension):
    monkeypatch.setenv('OPENROUTER_EMBEDDING_API_KEY', 'test-key')
    monkeypatch.setenv('EMBEDDING_MODEL', 'google/gemini-embedding-2')
    def handle(request):
        assert request.url.path == '/api/v1/embeddings'
        return httpx.Response(200, json={'data': [{'index': 0, 'object': 'embedding', 'embedding': [0.1]*dimension}], 'model': 'google/gemini-embedding-2', 'object': 'list'})
    monkeypatch.setattr(search, 'AsyncOpenAI', lambda **kwargs: AsyncOpenAI(
        **kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))))
    if dimension == 3072:
        assert len(asyncio.run(search.embed_photo_query('test query'))) == dimension
    else:
        from fastapi import HTTPException
        with pytest.raises(HTTPException): asyncio.run(search.embed_photo_query('test query'))


def test_cancelled_database_search_closes_session_and_engine(monkeypatch):
    exited = []
    class Session:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): exited.append('session')
        async def run_sync(self, *args, **kwargs): await asyncio.Event().wait()
    async def dispose(): exited.append('engine')
    monkeypatch.setattr(search, 'photo_search_engine', lambda: SimpleNamespace(dispose=dispose))
    monkeypatch.setattr(search, 'AsyncSession', lambda engine: Session())
    monkeypatch.setattr(search, 'search_space', lambda: 'gemini')
    async def run():
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.02): await search.search_photo_tracks([1.0], set())
    asyncio.run(run())
    assert exited == ['session', 'engine']
