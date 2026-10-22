"""일간 임베딩 배치가 임베딩(1단계) 뒤에 emb_v4(2단계)를 채우는 순서. 실제 DB·API 없이 함수를 바꿔 끼워 검사한다."""

from contextlib import nullcontext

import pytest


@pytest.fixture
def embed(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://test:test@localhost/test")
    monkeypatch.setenv("OPENROUTER_EMBEDDING_API_KEY", "embedding-test-key")
    from batch import embed
    calls = []
    monkeypatch.setattr(embed, "SessionLocal", lambda: nullcontext())
    monkeypatch.setattr(embed, "embed_new", lambda key, limit, dry_run: calls.append("embed"))
    monkeypatch.setattr(embed, "count_missing", lambda s: calls.append("count") or 3)
    monkeypatch.setattr(embed, "fill_missing", lambda: calls.append("fill") or 3)
    embed.calls = calls
    return embed


def test_fills_v4_after_embedding(embed):
    embed.run()
    assert embed.calls == ["embed", "count", "fill"]


def test_dry_run_only_counts_v4(embed):
    embed.run(dry_run=True)
    assert embed.calls == ["embed", "count"]


def test_v4_failure_ends_with_error(embed, monkeypatch):
    """2단계 실패는 예외로 끝나 종료 코드가 0이 아니다 (1단계 emb_gemini는 이미 커밋된 뒤)."""

    def broken():
        raise RuntimeError("v4_ref_queries가 비어 있다")

    monkeypatch.setattr(embed, "fill_missing", broken)
    with pytest.raises(RuntimeError):
        embed.run()
    assert embed.calls == ["embed", "count"]
