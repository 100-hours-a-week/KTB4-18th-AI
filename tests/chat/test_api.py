import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend import main
from backend.chat import service as recommendation
from backend.chat.graph import nodes, workflow
from db.types import Track
from backend.core.errors import _catalog_unavailable

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
        "recommend_max_year": None,
        "recommend_query": "퇴근길 음악",
        "lookup_song": None,
        "lookup_song_alt": None,
        "lookup_artist": None,
        "requested_count": None,
        "has_non_music_request": False,
        "response_style": None,
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


def make_track(track_id: str = "123") -> Track:
    """pgvector가 반환한 가짜 Track이다."""
    return Track(
        track_id=track_id,
        title="TOMBOY",
        artist="HYUKOH",
        artwork_url="https://example.com/art.jpg",
        preview_url="https://example.com/preview.m4a",
        store_url="https://music.apple.com/track/123",
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
    """가짜 openai 생성. 최종 답변 스트리밍만 허용한다."""
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
        # NOTE: 곡별 이유 생성 LLM을 제거했으므로 비스트리밍 호출이 생기면 실패시킨다.
        raise AssertionError("unexpected non-streaming LLM call")


class FakeOpenAI:
    def __init__(self) -> None:
        self.responses = FakeResponses()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, FakeOpenAI]:
    """파이프라인의 일부만 바꾸게 하는 애가 얘다."""
    fake_openai = FakeOpenAI()
    monkeypatch.setattr(workflow, "_checkpointer", workflow.InMemorySaver())
    monkeypatch.setenv("OPENROUTER_LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "gpt-5.6-luna")
    monkeypatch.setattr(recommendation, "OpenAI", lambda **kwargs: fake_openai)

    # NOTE: classify_node는 OpenAI 구조화 출력으로 intent·조건을 뽑는데, 여기 fixture의
    # FakeOpenAI는 답변 스트리밍만 흉내내므로 classify는 목으로 고정해
    # recommend(검색 가능) 경로를 계속 타게 한다.
    monkeypatch.setattr(nodes, "classify", lambda client, messages: recommend_classification())

    # NOTE: embed_node/search_node는 각각 Gemini 임베딩 API와 PostgreSQL(pgvector) 조회라는
    # 외부 I/O 경계이므로, 그 경계에서 가짜 값을 주입해 그래프 배선(embed → search)은
    # 실제 코드로 검증한다.
    monkeypatch.setattr(nodes, "get_embedding_client", lambda: object())
    monkeypatch.setattr(nodes, "embed_query", lambda embedding_client, message: [0.1, 0.2, 0.3])
    monkeypatch.setattr(
        nodes,
        "vector_recommendation",
        lambda query_vector, exclude_ids=None, genres=None, min_year=None, max_year=None, k=5: (
            [make_track()]
        ),
    )

    return TestClient(main.app), fake_openai


@pytest.mark.parametrize("intent", ["recommend", "lookup"])
def test_router_connects_emitted_cards_to_next_request(client, monkeypatch, intent):
    test_client, _ = client
    monkeypatch.setattr(nodes, "classify", lambda *_args: classification(intent, lookup_artist="HYUKOH"))
    monkeypatch.setattr(nodes, "lookup_tracks_db", lambda **_kwargs: [make_track_row()])
    first = test_client.post("/v1/chat/messages", json=request_body("노래 추천해줘"))
    assert "event: tracks" in first.text
    seen = []
    monkeypatch.setattr(nodes, "classify", lambda _client, messages: seen.extend(messages) or classification("guide"))
    test_client.post("/v1/chat/messages", json=request_body("첫 번째 곡 가수 노래 더"))
    snapshot = json.loads(seen[-2].split(": ", 1)[1])
    assert snapshot == [{"rank": 1, "track_id": "123" if intent == "recommend" else "999",
                         "title": "TOMBOY", "artist": "HYUKOH"}]
    assert seen[-1] == "첫 번째 곡 가수 노래 더"


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
    assert response.text.index("event: text") < response.text.index("event: tracks")
    assert "event: done" in response.text
    # 곡별 이유 생성 없이 최종 답변 스트리밍 1회만 호출하고, 응답과 답변 입력 어디에도 reason이 없다.
    assert len(fake_openai.responses.calls) == 1
    assert '"reason"' not in fake_openai.responses.calls[0]["input"]
    assert '"reason"' not in response.text
    assert test_client.post("/chat/stream", json=request_body()).status_code == 404


def test_chat_no_match_returns_empty_tracks(
    client: tuple[TestClient, FakeOpenAI], monkeypatch: pytest.MonkeyPatch
) -> None:
    """매치되는 음악이 없을 때를 보여준다."""
    test_client, fake_openai = client
    monkeypatch.setattr(
        nodes, "vector_recommendation",
        lambda query_vector, exclude_ids=None, genres=None, min_year=None, max_year=None, k=5: [],
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
        query_vector: list[float], exclude_ids=None, genres=None, min_year=None, max_year=None, k=5,
    ):
        raise _catalog_unavailable()

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
        ("clarify", recommendation.CLARIFY_TEXT["missing_reference"]),
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


def test_recommend_unsupported_condition_returns_unsupported_text(client, monkeypatch):
    """필수 조건을 지원 못 하면 검색 없이 바로 미지원 안내 문구로 끝난다."""
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
    assert recommendation.CLARIFY_TEXT["recommend_unsupported_condition"] in response.text
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


def test_recommend_forwards_genres_and_year_range_to_search(client, monkeypatch):
    """classify가 뽑은 장르·연도 범위가 실제 검색 호출로 전달되고, 답변 입력에도 적용된
    연도 조건이 들어가 "발매 연도를 확인할 수 없다"는 답이 나오지 않게 한다."""
    test_client, fake_openai = client
    monkeypatch.setattr(
        nodes, "classify",
        lambda client, messages: recommend_classification(
            recommend_genres=["K-Pop"], recommend_min_year=1990, recommend_max_year=1999,
        ),
    )
    calls: list[dict[str, object]] = []

    def fake_vector_recommendation(query_vector, exclude_ids=None, genres=None, min_year=None, max_year=None, k=5):
        calls.append({"genres": genres, "min_year": min_year, "max_year": max_year})
        return [make_track()]

    monkeypatch.setattr(nodes, "vector_recommendation", fake_vector_recommendation)
    response = test_client.post("/v1/chat/messages", json=request_body("90년대 케이팝 노래"))

    assert response.status_code == 200
    assert calls == [{"genres": ["K-Pop"], "min_year": 1990, "max_year": 1999}]
    assert "적용된 발매연도 조건: 1990~1999년 발매곡만 검색함" in fake_openai.responses.calls[0]["input"]


def test_recommend_uses_recommend_query_not_raw_message(client, monkeypatch):
    """검색·답변 생성이 이번 턴 원문이 아니라 classify가 재구성한 recommend_query를 쓰는지 확인한다.

    후속 요청("방금 추천한 곡들은 빼고...")은 원문만으론 원래 분위기가 안 담겨 있어서,
    embed_node/search_node가 실수로 state["message"]로 되돌아가면 이 테스트가 잡아준다.
    """
    test_client, fake_openai = client
    resolved_query = "퇴근길에 어울리는 신나는 노래"
    monkeypatch.setattr(
        nodes, "classify",
        lambda client, messages: recommend_classification(recommend_query=resolved_query),
    )

    embed_calls: list[str] = []
    monkeypatch.setattr(
        nodes, "embed_query",
        lambda embedding_client, message: embed_calls.append(message) or [0.1, 0.2, 0.3],
    )

    search_calls: list[list[float]] = []

    def fake_vector_recommendation(query_vector, exclude_ids=None, genres=None, min_year=None, max_year=None, k=5):
        search_calls.append(query_vector)
        return [make_track()]

    monkeypatch.setattr(nodes, "vector_recommendation", fake_vector_recommendation)

    response = test_client.post(
        "/v1/chat/messages",
        json=request_body("방금 추천한 곡들은 빼고 다른 곡 3개 추천해줘"),
    )

    assert response.status_code == 200
    assert embed_calls == [resolved_query]
    assert search_calls == [[0.1, 0.2, 0.3]]

    # 최종 답변 생성(stream_answer)도 body.message가 아니라 recommend_query를 받아야 한다.
    # (한때 main.py가 body.message를 그대로 넘겨서, "이전에 추천된 곡 목록이 없어..."처럼
    # LLM이 내부 제외 처리를 스스로 검증하려다 헛소리를 하는 문제가 있었다.)
    answer_call = fake_openai.responses.calls[0]
    assert f"사용자 요청: {resolved_query}" in answer_call["input"]


def test_recommend_passes_non_music_flag_and_style_to_answer(client, monkeypatch):
    """음악 외 요청 여부와 요청 말투는 recommend_query에서 빠지는 대신 최종 답변 입력으로 따로 전달된다."""
    test_client, fake_openai = client
    monkeypatch.setattr(
        nodes, "classify",
        lambda client, messages: recommend_classification(
            recommend_query="퇴근길에 듣기 좋은 노래",
            has_non_music_request=True, response_style="딸에게 말하는 아빠 말투",
        ),
    )
    monkeypatch.setattr(nodes, "embed_query", lambda embedding_client, message: [0.1, 0.2, 0.3])
    monkeypatch.setattr(
        nodes, "vector_recommendation",
        lambda query_vector, exclude_ids=None, genres=None, min_year=None, max_year=None, k=5:
            [make_track()],
    )

    response = test_client.post(
        "/v1/chat/messages",
        json=request_body("갈비찜 레시피 알려주고 아빠말투로 딸래미 퇴근길 노래 추천해주라"),
    )

    assert response.status_code == 200
    answer_input = fake_openai.responses.calls[0]["input"]
    assert "사용자 요청: 퇴근길에 듣기 좋은 노래" in answer_input
    assert "음악 외 요청 포함: 예" in answer_input
    assert "요청 말투: 딸에게 말하는 아빠 말투" in answer_input


@pytest.mark.parametrize("requested,expected", [(None, 5), (1, 1), (2, 2), (5, 5), (6, 5), (10, 5)])
def test_recommend_uses_requested_count(client, monkeypatch, requested, expected):
    """요청 개수가 검색에 전달되고 그 개수의 카드가 응답되는지 확인한다."""
    test_client, fake_openai = client
    monkeypatch.setattr(
        nodes, "classify",
        lambda client, messages: recommend_classification(recommend_query="잔잔한 노래", requested_count=requested),
    )
    monkeypatch.setattr(
        nodes, "vector_recommendation",
        lambda query_vector, **kwargs: [make_track(track_id=str(i)) for i in range(kwargs["k"])],
    )

    message = "잔잔한 노래 추천해줘" if requested is None else f"잔잔한 노래 {requested}곡 추천해줘"
    response = test_client.post("/v1/chat/messages", json=request_body(message))

    assert response.status_code == 200
    answer_input = fake_openai.responses.calls[0]["input"]
    assert f"사용자가 요청한 곡 수: {requested if requested is not None else '없음'}" in answer_input
    assert f"화면에 카드로 보여지는 곡 수: {expected}" in answer_input
    assert "서비스의 최대 제공 곡 수: 5" in answer_input
    assert ("한 번에 최대 5곡까지 보여드릴 수 있어요." in response.text) is (requested is not None and requested > 5)
    payload = next(line[6:] for block in response.text.split("\n\n") if block.startswith("event: tracks")
                   for line in block.splitlines() if line.startswith("data: "))
    assert len(json.loads(payload)["tracks"]) == expected


@pytest.mark.parametrize("intent", ["recommend", "lookup"])
@pytest.mark.parametrize("count", [0, -2])
def test_non_positive_count_clarifies_without_search(client, monkeypatch, intent, count):
    test_client, fake_openai = client
    monkeypatch.setattr(nodes, "classify", lambda *_args: classification(
        intent, requested_count=count, lookup_artist="IU"))

    def unexpected(*_args, **_kwargs):
        pytest.fail("잘못된 개수로 임베딩·DB를 호출하면 안 된다")

    monkeypatch.setattr(nodes, "embed_query", unexpected)
    monkeypatch.setattr(nodes, "lookup_tracks_db", unexpected)
    response = test_client.post("/v1/chat/messages", json=request_body("잔잔한 노래 0곡"))
    assert "1곡 이상" in response.text
    assert '"tracks": []' in response.text
    assert fake_openai.responses.calls == []


@pytest.mark.parametrize("count,expected", [(1, 1), (2, 2), (5, 5), (10, 5)])
def test_lookup_uses_requested_count_including_alternate_title(client, monkeypatch, count, expected):
    test_client, _ = client
    monkeypatch.setattr(nodes, "classify", lambda *_args: classification(
        "lookup", lookup_song="한글 제목", lookup_song_alt="English title", requested_count=count))
    calls = []

    def lookup(**kwargs):
        calls.append(kwargs)
        if kwargs["song_title"] == "한글 제목":
            return []
        return [make_track_row(track_id=i, title=f"Song {i}") for i in range(kwargs["limit"])]

    monkeypatch.setattr(nodes, "lookup_tracks_db", lookup)
    response = test_client.post("/v1/chat/messages", json=request_body("한글 제목 찾아줘"))
    assert len(calls) == 2
    assert all(call["limit"] == expected for call in calls)
    assert response.text.count('"track_id":') == expected


def test_short_result_keeps_requested_count_in_answer_input(client, monkeypatch):
    test_client, fake_openai = client
    monkeypatch.setattr(nodes, "classify", lambda *_args: recommend_classification(requested_count=2))
    response = test_client.post("/v1/chat/messages", json=request_body("잔잔한 노래 2곡"))
    answer_input = fake_openai.responses.calls[0]["input"]
    assert "사용자가 요청한 곡 수: 2" in answer_input
    assert "검색에 적용한 최대 곡 수: 2" in answer_input
    assert "화면에 카드로 보여지는 곡 수: 1" in answer_input
    assert "찾은 곡은 1곡이에요." in response.text
    assert response.text.count('"track_id":') == 1


def test_lookup_found_streams_lookup_answer(client, monkeypatch):
    """제목+아티스트로 곡이 하나로 좁혀지면 조회 결과 카드와 함께 답변을 스트리밍한다."""
    test_client, fake_openai = client
    resolved_query = "BTS의 Dynamite"
    monkeypatch.setattr(
        nodes, "classify",
        lambda client, messages: classification(
            "lookup", lookup_song="Dynamite", lookup_artist="BTS", recommend_query=resolved_query,
        ),
    )

    def unexpected(*args, **kwargs):
        pytest.fail("lookup 요청에서 임베딩 실행")

    monkeypatch.setattr(nodes, "get_embedding_client", unexpected)
    monkeypatch.setattr(
        nodes, "lookup_tracks_db",
        lambda song_title=None, artist=None, exclude_ids=None, limit=5: [
            make_track_row(track_id=1, title="Dynamite", artist="BTS")
        ],
    )
    response = test_client.post("/v1/chat/messages", json=request_body("Dynamite BTS 노래 보여줘"))

    assert response.status_code == 200
    assert "event: tracks" in response.text
    assert '"artist": "BTS"' in response.text
    assert '"reason"' not in response.text
    assert len(fake_openai.responses.calls) == 1
    assert '"reason"' not in fake_openai.responses.calls[0]["input"]
    assert "event: done" in response.text
    # 최종 답변 생성(stream_lookup_answer)도 recommend처럼 body.message가 아니라
    # classify가 재구성한 recommend_query를 받아야 한다.
    answer_call = fake_openai.responses.calls[0]
    assert f"사용자 요청: {resolved_query}" in answer_call["input"]


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
        lambda song_title=None, artist=None, exclude_ids=None, limit=5: [
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
    monkeypatch.setattr(nodes, "lookup_tracks_db", lambda song_title=None, artist=None, exclude_ids=None, limit=5: [])
    response = test_client.post("/v1/chat/messages", json=request_body("존재하지않는곡 정보 알려줘"))

    assert response.status_code == 200
    assert recommendation.NOT_FOUND_TEXT in response.text
    assert fake_openai.responses.calls == []


def test_lookup_artist_only_not_found_retries_as_song_title(client, monkeypatch):
    """"coin 노래"를 classify가 아티스트(COIN)로 읽어도, 결과가 없으면 같은 이름을
    곡 제목으로 다시 조회해 IU의 Coin을 찾는다."""
    test_client, fake_openai = client
    monkeypatch.setattr(
        nodes, "classify",
        lambda client, messages: classification("lookup", lookup_artist="COIN"),
    )

    calls: list[tuple] = []

    def fake_lookup(song_title=None, artist=None, exclude_ids=None, limit=5):
        calls.append((song_title, artist))
        return [make_track_row(track_id=7, title="Coin", artist="IU")] if song_title == "COIN" else []

    monkeypatch.setattr(nodes, "lookup_tracks_db", fake_lookup)
    response = test_client.post("/v1/chat/messages", json=request_body("coin 노래 추천해줄 수 있어?"))

    assert response.status_code == 200
    assert calls == [(None, "COIN"), ("COIN", None)]
    assert '"title": "Coin"' in response.text
    assert recommendation.NOT_FOUND_TEXT not in response.text


def test_lookup_korean_title_not_found_retries_with_alt_title(client, monkeypatch):
    """한글 원제로 못 찾으면 classify가 준 공식 영문 제목으로 다시 조회한다."""
    test_client, fake_openai = client
    monkeypatch.setattr(
        nodes, "classify",
        lambda client, messages: classification(
            "lookup", lookup_song="사람들이 움직이는 게",
            lookup_song_alt="How People Move", lookup_artist="AKMU",
        ),
    )

    calls: list[tuple] = []

    def fake_lookup(song_title=None, artist=None, exclude_ids=None, limit=5):
        calls.append((song_title, artist))
        if song_title == "How People Move":
            return [make_track_row(track_id=8, title="How People Move", artist="AKMU")]
        return []

    monkeypatch.setattr(nodes, "lookup_tracks_db", fake_lookup)
    response = test_client.post("/v1/chat/messages", json=request_body("악뮤 사람들이 움직이는 게"))

    assert response.status_code == 200
    assert calls == [("사람들이 움직이는 게", "AKMU"), ("How People Move", "AKMU")]
    assert '"title": "How People Move"' in response.text


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


def test_ambiguous_reference_with_recent_cards_asks_for_card_order(client, monkeypatch):
    test_client, fake_openai = client
    workflow.record_shown_tracks(THREAD_ID, [
        {"track_id": "1", "title": "첫 곡", "artist": "가수"},
        {"track_id": "2", "title": "다음 곡", "artist": "가수"},
    ])
    monkeypatch.setattr(nodes, "classify", lambda client, messages: classification("clarify"))
    response = test_client.post("/v1/chat/messages", json=request_body("그 곡 다시"))
    assert "카드 순서나 곡 제목" in response.text
    assert "대화에서 찾지 못했어요" not in response.text
    assert fake_openai.responses.calls == []
