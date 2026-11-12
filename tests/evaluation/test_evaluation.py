"""외부 호출 없이 채점 경계와 기존 라우터 실행 어댑터를 검증한다."""

from unittest.mock import patch
import asyncio

from evaluation.grading.scoring import score_turn, route_of
import pytest


def actual(**kw):
    return dict(done=True, state={}, tracks=[], **kw)


def test_null_missing_and_genre():
    assert score_turn({"min_year": None}, actual())["min_year"]["score"] == 0
    assert (
        score_turn(
            {"min_year": None}, dict(done=True, state={"recommend_min_year": None}, tracks=[])
        )["min_year"]["score"]
        == 1
    )
    assert "min_year" not in score_turn({}, actual())


def test_no_vacuous_artist_pass():
    assert score_turn({"lookup_artist": "IU"}, actual())["lookup_artist"]["score"] == 0


def test_routes_priority():
    assert route_of(
        {
            "intent": "recommend",
            "recommend_unsupported_condition": True,
            "recommend_has_enough_info": False,
        }
    ) == ["recommend", "unsupported"]
    assert route_of({"intent": "clarify"}) == ["clarify", "missing_reference"]


def test_unknown_expectation_fails():
    with pytest.raises(ValueError):
        score_turn({"typo": 1}, actual())


def test_router_static_response(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite://")
    monkeypatch.setenv("LANGSMITH_TRACING_V2", "false")
    from evaluation.execution.runner import run_turn
    from backend.chat.service import ChatOutcome
    from uuid import uuid4
    from types import SimpleNamespace

    with (
        patch(
            "backend.chat.router.prepare_recommendation",
            return_value=ChatOutcome(kind="static", text="사용 안내"),
        ),
        patch("backend.chat.graph.workflow.build_graph") as graph,
    ):
        graph.return_value.get_state.return_value = SimpleNamespace(values={"intent": "guide"})
        result = asyncio.run(run_turn({"message": "사용법"}, uuid4(), tmp_path, {}))
    assert result["done"] and result["text"] == "사용 안내"
    assert result["route"] == ["guide", "guide"]
    assert result["cards_ms"] is not None


def test_telemetry_sync_and_async():
    import httpx
    from evaluation.execution.telemetry import Capture, CASE, parse_response

    raw = b'data: {"id":"gen-test","usage":{"cost":0.01}}\n\ndata: [DONE]\n'
    assert parse_response(raw)["usage"]["cost"] == 0.01
    CASE.set("pilot")
    with Capture() as capture:
        with httpx.Client(
            transport=httpx.MockTransport(lambda req: httpx.Response(200, content=raw))
        ) as client:
            with client.stream("POST", "https://openrouter.ai/api/v1/chat/completions") as response:
                assert response.read() == raw

        async def check():
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(lambda req: httpx.Response(200, content=raw))
            ) as client:
                async with client.stream(
                    "POST", "https://openrouter.ai/api/v1/chat/completions"
                ) as response:
                    assert await response.aread() == raw

        asyncio.run(check())
    assert len(capture.records) == 2
    assert all(r["case_id"] == "pilot" and r["id"] == "gen-test" for r in capture.records)


def test_case_metrics_skip_unscored():
    from evaluation.grading.metrics import case_metrics

    turn = {
        "actual": {"done": True, "total_ms": 10.0},
        "scores": {
            "service_success": {"status": "scored", "score": 1},
            "routes": {"status": "scored", "score": 1},
            "lookup_track_ids_if_found": {"status": "not_applicable", "score": None},
        },
    }
    m = case_metrics({"turns": [turn], "expected_turns": 1}, 0.002)
    assert m["route_acc"] == 1 and m["contract_pass"] == 1 and "lookup_acc" not in m


def test_telemetry_records_stream_closed_early():
    import httpx
    from evaluation.execution.telemetry import Capture, CASE

    raw = b'data: {"id":"gen-early","usage":{"cost":0.02}}\n\ndata: {"more":1}\n'
    CASE.set("early")
    with Capture() as capture:
        # 생성기 본문이어야 실제 스트림처럼 소비되지 않은 상태로 돌아온다.
        with httpx.Client(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(200, content=iter(raw.split(b"\n\n")))
            )
        ) as client:
            with client.stream("POST", "https://openrouter.ai/api/v1/responses") as response:
                next(response.iter_raw())
            assert len(capture.records) == 1
    assert len(capture.records) == 1 and capture.records[0]["case_id"] == "early"


def test_format_and_exclusion_checks():
    base = dict(
        done=True,
        state={},
        tracks=[{"track_id": "1", "artist": "a"}],
        route=["recommend", "tracks"],
    )
    scores = score_turn(
        {}, dict(base, text="**추천**"), metrics=["markdown", "exclusion"], previous={"1"}
    )
    assert scores["plain_text"]["score"] == 0 and scores["previous_tracks_excluded"]["score"] == 0
    assert (
        score_turn({}, dict(base, text="P!nk — What About Us"), metrics=["markdown"])["plain_text"][
            "score"
        ]
        == 1
    )
    lookup = dict(
        base, route=["lookup", "lookup_found"], state={"lookup_artist": "a", "lookup_song": "song"}
    )
    assert (
        score_turn({}, lookup, metrics=["exclusion"], previous={"1"})["previous_tracks_excluded"][
            "status"
        ]
        == "not_applicable"
    )


def test_case_scoring_remembers_all_previous_turns():
    from evaluation.grading.scoring import score_case

    case = {
        "metadata": {"metrics": ["exclusion"]},
        "turns": [
            {
                "expected": {},
                "actual": dict(done=True, state={}, tracks=[{"track_id": tid, "artist": "a"}]),
            }
            for tid in ("1", "2", "1")
        ],
    }
    score_case(case)
    assert case["turns"][1]["scores"]["previous_tracks_excluded"]["score"] == 1
    assert case["turns"][2]["scores"]["previous_tracks_excluded"]["score"] == 0


def test_contract_unknown_and_latency_are_separate():
    from evaluation.grading.metrics import case_metrics

    case = {
        "expected_turns": 1,
        "turns": [
            {
                "actual": {"done": True},
                "scores": {
                    "service_success": {"status": "scored", "score": 1},
                    "routes": {"status": "scored", "score": 1},
                    "verdict_deadline_s": {"status": "scored", "score": 0},
                },
            }
        ],
    }
    assert case_metrics(case)["contract_pass"] == 1
    case["turns"][0]["scores"]["returned_min_year"] = {
        "status": "insufficient_evidence",
        "score": None,
    }
    assert "contract_pass" not in case_metrics(case)
    case["turns"][0]["scores"]["routes"]["score"] = 0
    assert case_metrics(case)["contract_pass"] == 0


def track(genre):
    return {"track_id": genre, "title": "t", "artist": "a", "genre": genre}


def recommended(text, *genres):
    return dict(
        done=True,
        state={},
        route=["recommend", "tracks"],
        text=text,
        tracks=[track(g) for g in genres],
    )


def test_markdown_is_checked_without_opt_in():
    scores = score_turn({}, recommended("**굵게** 추천해요", "Pop"))
    assert scores["plain_text"]["score"] == 0


def test_genre_claim_fails_when_cards_are_another_genre():
    scores = score_turn(
        {},
        recommended("차분한 클래식 음악을 찾아봤어요.", "Metal", "K-Pop", "K-Pop", "Jazz", "Pop"),
    )
    assert scores["genre_claim"]["score"] == 0


def test_genre_claim_fails_when_only_a_minority_of_cards_match():
    scores = score_turn(
        {},
        recommended("뉴에이지 음악을 추천해요.", "New Age", "K-Pop", "Soundtrack", "Jazz", "Pop"),
    )
    assert scores["genre_claim"]["score"] == 0


def test_genre_claim_groups_catalog_spellings():
    scores = score_turn(
        {}, recommended("힙합 곡을 골랐어요.", "Hip-Hop/Rap", "Korean Hip-Hop", "Rap", "Pop")
    )
    assert scores["genre_claim"]["score"] == 1


def test_genre_claim_skips_answer_without_genre_word():
    assert "genre_claim" not in score_turn({}, recommended("차분한 음악을 찾아봤어요.", "Metal"))


def test_genre_claim_allows_honest_notice():
    scores = score_turn({}, recommended("클래식 곡은 찾지 못해 차분한 곡으로 골랐어요.", "Metal"))
    assert "genre_claim" not in scores


def test_genre_claim_ignores_lookup_answers():
    case = dict(recommended("재즈 곡을 찾았어요.", "Pop"), route=["lookup", "lookup_found"])
    assert "genre_claim" not in score_turn({}, case)


def test_genre_claim_counts_toward_functional_requirement():
    from evaluation.grading.metrics import case_metrics
    from evaluation.grading.scoring import score_case

    case = score_case(
        {
            "metadata": {"metrics": []},
            "expected_turns": 1,
            "turns": [{"expected": {}, "actual": recommended("클래식을 추천해요.", "Metal")}],
        }
    )
    assert case_metrics(case)["contract_pass"] == 0


def test_genre_claim_does_not_read_eobsi_as_notice():
    scores = score_turn({}, recommended("부담 없이 듣는 클래식을 골랐어요.", "Metal"))
    assert scores["genre_claim"]["score"] == 0


def test_genre_claim_allows_not_mainly_notice():
    text = "표시된 곡들은 클래식 위주가 아니라 분위기가 다를 수 있어요."
    assert "genre_claim" not in score_turn({}, recommended(text, "Metal"))


def test_genre_claim_passes_when_two_of_five_cards_match():
    scores = score_turn(
        {},
        recommended("재즈나 어쿠스틱 음악을 추천해요.", "Jazz", "K-Pop", "Blues", "K-Pop", "Jazz"),
    )
    assert scores["genre_claim"]["score"] == 1


def test_card_missing_from_the_catalog_is_a_failed_turn_even_with_its_own_genre(monkeypatch):
    import evaluation.execution.runner as runner

    monkeypatch.setattr(runner, "catalog", lambda tracks: runner.merge_catalog(tracks, {}))
    out = {"tracks": [{"track_id": "1", "title": "t", "artist": "a", "genre": "Pop"}]}
    runner.attach_catalog(out)
    assert out["error"] == "catalog_LookupError"
    assert score_turn({}, dict(out, done=True))["service_success"]["score"] == 0


def test_catalog_failure_is_a_failed_turn_and_keeps_the_first_error(monkeypatch):
    import evaluation.execution.runner as runner

    def broken(tracks):
        raise RuntimeError("db")

    monkeypatch.setattr(runner, "catalog", broken)
    out = {"tracks": [{"track_id": "1"}]}
    runner.attach_catalog(out)
    assert out["error"] == "catalog_RuntimeError"
    first = {"tracks": [{"track_id": "1"}], "error": "stream_error"}
    runner.attach_catalog(first)
    assert first["error"] == "stream_error"


def test_verified_cards_keep_the_turn_clean(monkeypatch):
    from types import SimpleNamespace
    import evaluation.execution.runner as runner

    row = SimpleNamespace(genre=None, release_date=None, mood_tags=None)
    monkeypatch.setattr(runner, "catalog", lambda tracks: runner.merge_catalog(tracks, {"1": row}))
    out = {"tracks": [{"track_id": "1"}]}
    runner.attach_catalog(out)
    assert "error" not in out and out["tracks"][0]["genre"] is None
