import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend import main, recommendation
from backend.graph import nodes
from backend.schemas import Track

THREAD_ID = "11111111-1111-4111-8111-111111111111"
REQUEST_ID = "22222222-2222-4222-8222-222222222222"


def request_body(message: str = "퇴근길 음악") -> dict[str, str]:
    """메시지를 요청 JSON으로 바꿔준다."""
    return {
        "thread_id": THREAD_ID,
        "request_id": REQUEST_ID,
        "message": message,
    }


def classification(intent: str, **overrides: object) -> dict:
    """classify()가 반환하는 형태의 dict를 만든다. 기본값은 전부 '해당 없음'."""
    base = {
        "intent": intent,
        "recommend_has_enough_info": True,
        "recommend_unsupported_condition": False,
        "recommend_genres": None,
        "recommend_min_year": None,
        "lookup_song": None,
        "lookup_artist": None,
    }
    base.update(overrides)
    return base


def recommend_classification(**overrides: object) -> dict:
    return classification("recommend", **overrides)


def make_track_row(track_id: int = 999, title: str = "TOMBOY", artist: str = "HYUKOH") -> SimpleNamespace:
    """db.search.lookup()이 반환하는 TrackRow를 흉내낸다."""
    return SimpleNamespace(
        track_id=track_id,
        title=title,
        artist=artist,
        artwork_url="https://example.com/art.jpg",
        preview_url="https://example.com/preview.m4a",
        store_url=f"https://music.apple.com/track/{track_id}",
    )


def make_track(track_id: str = "123", reason: str = "폴백 이유") -> Track:
    """pgvector가 반환한 가짜 Track이다."""
    return Track(
        track_id=track_id,
        title="TOMBOY",
        artist="HYUKOH",
        artwork_url="https://example.com/art.jpg",
        preview_url="https://example.com/preview.m4a",
        store_url="https://music.apple.com/track/123",
        reason=reason,
    )


class FakeStream:
    """openai Stream이 지원하는 컨텍스트 매니저 프로토콜을 흉내낸다."""

    def __init__(self, events: list[object]) -> None:
        self._events = events

    """리스트로 Stream을 흉내내기 위해선 enter와 exit을 정의한다"""
    def __enter__(self) -> "FakeStream":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def __iter__(self):
        return iter(self._events)


class FakeResponses:
    """가짜 openai 생성. 'stream' 여부에 따라 스트리밍 아니면 추천 이유 생성이다."""
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return FakeStream(
                [
                    SimpleNamespace(type="response.output_text.delta", delta="퇴근길에 "),
                    SimpleNamespace(type="response.output_text.delta", delta="어울리는 곡이에요."),
                    SimpleNamespace(type="response.completed"),
                ]
            )
        # reason_node가 호출하는 곡별 추천 이유 생성(assign_reasons)의 유일한 비스트리밍 호출
        return SimpleNamespace(output_text=json.dumps({"123": "퇴근길에 잘 어울리는 곡이에요."}))


class FakeOpenAI:
    def __init__(self) -> None:
        self.responses = FakeResponses()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, FakeOpenAI]:
    """파이프라인의 일부만 바꾸게 하는 애가 얘다."""
    fake_openai = FakeOpenAI()
    monkeypatch.setenv("OPENROUTER_LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "gpt-5.6-luna")
    monkeypatch.setattr(recommendation, "OpenAI", lambda **kwargs: fake_openai)

    # NOTE: classify_node는 OpenAI 구조화 출력으로 intent·조건을 뽑는데, 여기 fixture의
    # FakeOpenAI는 이유 생성용 JSON만 흉내내므로 classify는 목으로 고정해
    # recommend(검색 가능) 경로를 계속 타게 한다.
    monkeypatch.setattr(nodes, "classify", lambda client, messages: recommend_classification())

    # NOTE: embed_node/search_node는 각각 Gemini 임베딩 API와 PostgreSQL(pgvector) 조회라는
    # 외부 I/O 경계이므로, 그 경계에서 가짜 값을 주입해 그래프 배선(embed → search →
    # reason)은 실제 코드로 검증한다.
    monkeypatch.setattr(nodes, "get_embedding_client", lambda: object())
    monkeypatch.setattr(nodes, "embed_query", lambda embedding_client, message: [0.1, 0.2, 0.3])
    monkeypatch.setattr(
        nodes,
        "vector_recommendation",
        lambda query_vector, message, exclude_ids=None, genres=None, min_year=None: (
            [make_track()],
            {"123": {"lofi": 0.9, "chill": 0.7}},
        ),
    )

    return TestClient(main.app), fake_openai


def test_chat_streams_text_before_completed_track_cards(
    client: tuple[TestClient, FakeOpenAI],
) -> None:
    test_client, fake_openai = client
    response = test_client.post(
        "/v1/chat/messages",
        json=request_body(" 퇴근길 음악 "),
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert 'event: text\ndata: {"delta": "퇴근길에 "}' in response.text
    assert "event: tracks" in response.text
    assert '"preview_url": "https://example.com/preview.m4a"' in response.text
    """최종 답변 스트리밍이 곡별 추천 이유 생성보다 먼저 오는가?"""
    assert response.text.index("event: text") < response.text.index("event: tracks")
    assert "event: done" in response.text
    # 1) reason_node의 곡별 추천 이유 생성  2) 최종 답변 스트리밍
    assert len(fake_openai.responses.calls) == 2
    assert test_client.post("/chat/stream", json=request_body()).status_code == 404


def test_chat_no_match_returns_empty_tracks(
    client: tuple[TestClient, FakeOpenAI], monkeypatch: pytest.MonkeyPatch
) -> None:
    """매치되는 음악이 없을 때를 보여준다."""
    test_client, fake_openai = client
    monkeypatch.setattr(
        nodes, "vector_recommendation",
        lambda query_vector, message, exclude_ids=None, genres=None, min_year=None: ([], {}),
    )

    response = test_client.post("/v1/chat/messages", json=request_body())

    assert response.status_code == 200
    assert "event: text" in response.text
    assert 'event: tracks\ndata: {"tracks": []}' in response.text
    assert "event: done" in response.text
    # 추천곡이 없으면 이유 생성·최종 답변 생성 모두 LLM을 호출하지 않는다.
    assert fake_openai.responses.calls == []


def test_chat_music_catalog_unavailable_returns_503(
    client: tuple[TestClient, FakeOpenAI], monkeypatch: pytest.MonkeyPatch
) -> None:
    """DB 장애가 있을 때 오류가 잘 나오는지 확인한다."""
    test_client, _ = client

    def failing_vector_recommendation(
        query_vector: list[float], message: str, exclude_ids=None, genres=None, min_year=None,
    ):
        raise recommendation._catalog_unavailable()

    monkeypatch.setattr(nodes, "vector_recommendation", failing_vector_recommendation)
    response = test_client.post("/v1/chat/messages", json=request_body())

    assert response.status_code == 503
    assert response.json()["details"]["reason"] == "MUSIC_CATALOG_UNAVAILABLE"


def test_chat_rejects_blank_message(client: tuple[TestClient, FakeOpenAI]) -> None:
    """빈 메시지를 받으면 오류가 잘 나오는지 확인한다."""
    test_client, _ = client
    response = test_client.post("/v1/chat/messages", json=request_body("   "))

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_REQUEST"


def test_chat_requires_openrouter_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """openrouter api key가 없으면 오류가 잘 나오는지 확인한다."""
    monkeypatch.delenv("OPENROUTER_LLM_API_KEY", raising=False)
    response = TestClient(main.app).post("/v1/chat/messages", json=request_body("안녕"))

    assert response.status_code == 503


def test_chat_requires_openrouter_embedding_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """embedding용 openrouter api key가 없으면 오류가 잘 나오는지 확인한다."""
    monkeypatch.setenv("OPENROUTER_LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "openai/test-model")
    monkeypatch.setattr(nodes, "classify", lambda client, messages: recommend_classification())
    monkeypatch.delenv("OPENROUTER_EMBEDDING_API_KEY", raising=False)
    monkeypatch.setattr(recommendation, "OpenAI", lambda **kwargs: FakeOpenAI())
    response = TestClient(main.app).post("/v1/chat/messages", json=request_body("안녕"))

    assert response.status_code == 503


def test_health_does_not_call_external_services() -> None:
    """health 엔드포인트가 외부 호출 없이도 status가 잘 나오는지 확인한다."""
    response = TestClient(main.app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize(
    ("intent", "expected_text"),
    [
        ("guide", recommendation.GUIDE_TEXT),
        ("clarify", recommendation.CLARIFY_TEXT["intent_unclear"]),
        ("out_of_scope", recommendation.OUT_OF_SCOPE_TEXT),
    ],
)
def test_static_intents_skip_search_and_embedding(client, monkeypatch, intent, expected_text):
    """guide/clarify/out_of_scope는 고정 문구만 내려주고 임베딩·DB를 전혀 안 건드린다."""
    test_client, fake_openai = client
    monkeypatch.setattr(nodes, "classify", lambda client, messages: classification(intent))

    def unexpected(*args, **kwargs):
        pytest.fail("정적 안내 분기에서 임베딩 또는 DB 검색 실행")

    monkeypatch.setattr(nodes, "get_embedding_client", unexpected)
    monkeypatch.setattr(nodes, "vector_recommendation", unexpected)
    monkeypatch.setattr(nodes, "lookup_tracks_db", unexpected)
    response = test_client.post("/v1/chat/messages", json=request_body("안녕"))

    assert response.status_code == 200
    assert expected_text in response.text
    assert 'event: tracks\ndata: {"tracks": []}' in response.text
    assert "event: done" in response.text
    assert fake_openai.responses.calls == []


def test_recommend_unsupported_condition_returns_guide_text(client, monkeypatch):
    """필수 조건을 지원 못 하면 검색 없이 바로 안내 문구로 끝난다."""
    test_client, fake_openai = client
    monkeypatch.setattr(
        nodes, "classify",
        lambda client, messages: recommend_classification(recommend_unsupported_condition=True),
    )

    def unexpected(*args, **kwargs):
        pytest.fail("지원 불가능한 조건인데 임베딩 또는 DB 검색 실행")

    monkeypatch.setattr(nodes, "get_embedding_client", unexpected)
    monkeypatch.setattr(nodes, "vector_recommendation", unexpected)
    response = test_client.post("/v1/chat/messages", json=request_body("이 아티스트 신곡만 추천해줘"))

    assert response.status_code == 200
    assert recommendation.GUIDE_TEXT in response.text
    assert fake_openai.responses.calls == []


def test_recommend_insufficient_info_returns_clarify_text(client, monkeypatch):
    """검색 단서가 없으면 검색 없이 바로 확인 질문으로 끝난다."""
    test_client, fake_openai = client
    monkeypatch.setattr(
        nodes, "classify",
        lambda client, messages: recommend_classification(recommend_has_enough_info=False),
    )

    def unexpected(*args, **kwargs):
        pytest.fail("정보 부족인데 임베딩 또는 DB 검색 실행")

    monkeypatch.setattr(nodes, "get_embedding_client", unexpected)
    monkeypatch.setattr(nodes, "vector_recommendation", unexpected)
    response = test_client.post("/v1/chat/messages", json=request_body("추천해줘"))

    assert response.status_code == 200
    assert recommendation.CLARIFY_TEXT["recommend_insufficient_info"] in response.text
    assert fake_openai.responses.calls == []


def test_recommend_forwards_genres_and_min_year_to_search(client, monkeypatch):
    """classify가 뽑은 장르·연도 조건이 실제 검색 호출로 전달되는지 확인한다."""
    test_client, _ = client
    monkeypatch.setattr(
        nodes, "classify",
        lambda client, messages: recommend_classification(
            recommend_genres=["K-Pop"], recommend_min_year=2020,
        ),
    )
    calls: list[dict[str, object]] = []

    def fake_vector_recommendation(query_vector, message, exclude_ids=None, genres=None, min_year=None):
        calls.append({"genres": genres, "min_year": min_year})
        return [make_track()], {"123": {"lofi": 0.9}}

    monkeypatch.setattr(nodes, "vector_recommendation", fake_vector_recommendation)
    response = test_client.post("/v1/chat/messages", json=request_body("케이팝 신나는 노래"))

    assert response.status_code == 200
    assert calls == [{"genres": ["K-Pop"], "min_year": 2020}]


def test_lookup_found_streams_lookup_answer(client, monkeypatch):
    """제목+아티스트로 곡이 하나로 좁혀지면 조회 결과 카드와 함께 답변을 스트리밍한다."""
    test_client, fake_openai = client
    monkeypatch.setattr(
        nodes, "classify",
        lambda client, messages: classification("lookup", lookup_song="Dynamite", lookup_artist="BTS"),
    )

    def unexpected(*args, **kwargs):
        pytest.fail("lookup 요청에서 임베딩 실행")

    monkeypatch.setattr(nodes, "get_embedding_client", unexpected)
    monkeypatch.setattr(
        nodes, "lookup_tracks_db",
        lambda song_title=None, artist=None, exclude_ids=None: [
            make_track_row(track_id=1, title="Dynamite", artist="BTS")
        ],
    )
    response = test_client.post("/v1/chat/messages", json=request_body("Dynamite BTS 언제 나온 곡이야?"))

    assert response.status_code == 200
    assert "event: tracks" in response.text
    assert '"artist": "BTS"' in response.text
    assert "event: done" in response.text


def test_lookup_title_only_multiple_artists_are_all_shown_as_found(client, monkeypatch):
    """제목만 있는데 여러 아티스트가 걸려도(동명이곡) 되묻지 않고 전부 조회 결과 카드로 보여준다."""
    test_client, fake_openai = client
    monkeypatch.setattr(
        nodes, "classify",
        lambda client, messages: classification("lookup", lookup_song="Dynamite"),
    )

    def unexpected(*args, **kwargs):
        pytest.fail("lookup 요청에서 임베딩 실행")

    monkeypatch.setattr(nodes, "get_embedding_client", unexpected)
    monkeypatch.setattr(
        nodes, "lookup_tracks_db",
        lambda song_title=None, artist=None, exclude_ids=None: [
            make_track_row(track_id=1, title="Dynamite", artist="BTS"),
            make_track_row(track_id=2, title="Dynamite", artist="ITZY"),
        ],
    )
    response = test_client.post("/v1/chat/messages", json=request_body("Dynamite 누가 불렀어?"))

    assert response.status_code == 200
    assert '"artist": "BTS"' in response.text
    assert '"artist": "ITZY"' in response.text
    assert "event: done" in response.text


def test_lookup_not_found_returns_not_found_text(client, monkeypatch):
    """대상은 있는데 DB에 없으면 정보 없음 안내로 끝난다."""
    test_client, fake_openai = client
    monkeypatch.setattr(
        nodes, "classify",
        lambda client, messages: classification("lookup", lookup_song="존재하지않는곡"),
    )

    def unexpected(*args, **kwargs):
        pytest.fail("정보 없는 lookup 요청에서 임베딩 실행")

    monkeypatch.setattr(nodes, "get_embedding_client", unexpected)
    monkeypatch.setattr(nodes, "lookup_tracks_db", lambda song_title=None, artist=None, exclude_ids=None: [])
    response = test_client.post("/v1/chat/messages", json=request_body("존재하지않는곡 정보 알려줘"))

    assert response.status_code == 200
    assert recommendation.NOT_FOUND_TEXT in response.text
    assert fake_openai.responses.calls == []


def test_lookup_no_target_asks_without_db_call(client, monkeypatch):
    """제목·아티스트를 아예 못 뽑았으면 DB 조회 없이 바로 되묻는다."""
    test_client, fake_openai = client
    monkeypatch.setattr(nodes, "classify", lambda client, messages: classification("lookup"))

    def unexpected(*args, **kwargs):
        pytest.fail("대상 없는 lookup 요청에서 임베딩 또는 DB 조회 실행")

    monkeypatch.setattr(nodes, "get_embedding_client", unexpected)
    monkeypatch.setattr(nodes, "lookup_tracks_db", unexpected)
    response = test_client.post("/v1/chat/messages", json=request_body("그 노래 정보 좀"))

    assert response.status_code == 200
    assert 'event: tracks\ndata: {"tracks": []}' in response.text
    assert fake_openai.responses.calls == []


def test_local_chat_page_uses_v1_fields() -> None:
    """테스트용 HTML 페이지가 v1에 맞게 나오는지 확인한다."""
    response = TestClient(main.app).get("/app/")

    assert response.status_code == 200
    assert 'const CHAT_URL = "/v1/chat/messages"' in response.text
    assert "thread_id" in response.text
    assert "request_id" in response.text
    assert "response.body.getReader()" in response.text
    assert "addTrackCards" in response.text
    assert "store_url" in response.text
