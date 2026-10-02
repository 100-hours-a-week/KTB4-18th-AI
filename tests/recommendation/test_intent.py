import json
from types import SimpleNamespace

from backend.recommendation import intent


def classification_json(**overrides: object) -> str:
    base = {
        "intent": "recommend",
        "recommend_has_enough_info": True,
        "recommend_unsupported_condition": False,
        "recommend_genres": None,
        "recommend_min_year": None,
        "recommend_query": "슬픈 발라드",
        "lookup_song": None,
        "lookup_song_alt": None,
        "lookup_artist": None,
        "requested_count": None,
        "has_non_music_request": False,
        "response_style": None,
    }
    base.update(overrides)
    return json.dumps(base, ensure_ascii=False)


def fake_client(output_text: str) -> SimpleNamespace:
    return SimpleNamespace(responses=SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(output_text=output_text)
    ))


def test_classify_drops_genres_missing_from_db(monkeypatch):
    """DB에 없는 장르("Ballad", "발라드")가 하드 필터로 들어가 검색 결과가 0곡이 되지 않게 버린다."""
    monkeypatch.setenv("LLM_MODEL", "openai/test-model")
    monkeypatch.setattr(intent, "known_genres", lambda: ["K-Pop", "R&B/Soul"])

    result = intent.classify(fake_client(classification_json(recommend_genres=["Ballad", "발라드"])), ["슬픈 발라드 추천해줘"])

    assert result["recommend_genres"] is None


def test_classify_keeps_known_genres_in_db_spelling(monkeypatch):
    """DB에 있는 장르는 대소문자만 달라도 DB 표기로 맞춰 남긴다."""
    monkeypatch.setenv("LLM_MODEL", "openai/test-model")
    monkeypatch.setattr(intent, "known_genres", lambda: ["K-Pop", "R&B/Soul"])

    result = intent.classify(fake_client(classification_json(recommend_genres=["k-pop", "Ballad"])), ["케이팝 노래"])

    assert result["recommend_genres"] == ["K-Pop"]
