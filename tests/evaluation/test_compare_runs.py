import json

import pytest

from evaluation.compare import NotComparable, compare
from evaluation.report import render


DONE = {
    "finished": "time",
    "db_unchanged": True,
    "mode": "normal",
    "split": None,
    "dataset_version": "v",
    "dataset_sha256": {"text": "d", "stt": "d"},
    "assets_sha256": {"text": "a", "stt": "a"},
    "metric_version": "m",
    "rubric_version": "r",
    "judge_model": "j",
    "judge_settings": {},
    "judge_prompt_sha256": "p",
    "eval_package_sha256": "e",
    "lossless_png_transport_ids": [],
    "concurrency": 2,
    "scope": "s",
    "models": {},
    "search_embedding": "gemini",
    "agentic_chat": False,
    "db_before": {"catalog_sha": "c", "embedding_sha": "e"},
}


def turn(text="안내", tracks=("1",), faith=1, cards_ms=2000, genre="Pop"):
    return {
        "expected": {"routes": [["recommend", "tracks"]]},
        "actual": {
            "done": True,
            "state": {},
            "route": ["recommend", "tracks"],
            "text": text,
            "input": {},
            "tracks": [{"track_id": t, "title": "t", "artist": t, "genre": genre} for t in tracks],
            "cards_ms": cards_ms,
            "first_text_ms": cards_ms,
            "total_ms": cards_ms,
        },
        "initial_judge": {"status": "ok", "faithfulness": faith},
    }


def case(id_, turns, cost=0.01, metrics=None):
    return {
        "id": id_,
        "metadata": {"metrics": []},
        "expected_turns": len(turns),
        "turns": turns,
        "service_cost_usd": cost,
        "metrics": metrics or {"요청 완료율": 1},
    }


def write(folder, cases, manifest=None):
    folder.mkdir()
    (folder / "text.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases)
    )
    (folder / "manifest.json").write_text(
        json.dumps({**DONE, **(manifest or {"search_embedding": "gemini"})})
    )
    return folder


def wrong_route():
    bad = turn()
    bad["actual"]["route"] = ["guide", "guide"]
    return bad


def test_counts_functional_pass_and_names_changed_cases(tmp_path):
    base = write(tmp_path / "a", [case("T-1", [wrong_route()]), case("T-2", [turn()])])
    new = write(tmp_path / "b", [case("T-1", [turn()]), case("T-2", [turn()])])
    report = compare(base, new)
    row = report["functional"]["text"]
    assert (row["base"], row["new"], row["total"]) == (1, 2, 2)
    assert row["improved"] == ["T-1"] and row["regressed"] == []


def test_rescoring_applies_new_checks_to_both_runs(tmp_path):
    claim = turn(text="클래식을 추천해요.", genre="Metal")
    base = write(tmp_path / "a", [case("T-1", [claim])])
    new = write(tmp_path / "b", [case("T-1", [turn()])])
    assert compare(base, new)["functional"]["text"]["base"] == 0


def test_lists_failed_checks_only_for_new_run(tmp_path):
    base = write(tmp_path / "a", [case("T-1", [turn()])])
    new = write(tmp_path / "b", [case("T-1", [wrong_route()])])
    assert compare(base, new)["failures"] == {"T-1": ["routes"]}


def test_faithfulness_uses_only_turns_with_cards(tmp_path):
    fixed = turn(tracks=(), faith=0)
    base = write(tmp_path / "a", [case("T-1", [turn(faith=0), fixed])])
    new = write(tmp_path / "b", [case("T-1", [turn(faith=1), fixed])])
    row = compare(base, new)["faithfulness"]["text"]
    assert (row["base"], row["new"], row["total"]) == (0, 1, 1)


def test_song_repeat_uses_case_share(tmp_path):
    same = [case(f"T-{i}", [turn(tracks=("1", str(i + 10)))]) for i in range(4)]
    mixed = [case(f"T-{i}", [turn(tracks=(str(i), str(i + 10)))]) for i in range(4)]
    report = compare(write(tmp_path / "a", same), write(tmp_path / "b", mixed))
    assert report["song_repeat"] == {"base": 1.0, "new": 0.25}


def test_undeclared_condition_change_stops_the_comparison(tmp_path):
    base = write(tmp_path / "a", [case("T-1", [turn()])], {"search_embedding": "gemini"})
    new = write(tmp_path / "b", [case("T-1", [turn()])], {"search_embedding": "v4"})
    with pytest.raises(NotComparable, match="search_embedding"):
        compare(base, new)


def test_declared_experiment_variable_is_allowed(tmp_path):
    base = write(tmp_path / "a", [case("T-1", [turn()])], {"search_embedding": "gemini"})
    new = write(tmp_path / "b", [case("T-1", [turn()])], {"search_embedding": "v4"})
    report = compare(base, new, vary=("search_embedding",))
    assert report["conditions"] == {"search_embedding": ["gemini", "v4"]}


def test_unknown_variable_name_is_rejected(tmp_path):
    base = write(tmp_path / "a", [case("T-1", [turn()])])
    with pytest.raises(ValueError, match="embeding"):
        compare(base, base, vary=("embeding",))


def test_embedding_change_is_separate_from_catalog_change(tmp_path):
    db = lambda catalog, embedding: {
        "db_before": {
            "sha": catalog + embedding,
            "catalog_sha": catalog,
            "embedding_sha": embedding,
        }
    }
    base = write(tmp_path / "a", [case("T-1", [turn()])], db("c", "e1"))
    new = write(tmp_path / "b", [case("T-1", [turn()])], db("c", "e2"))
    assert compare(base, new, vary=("embedding",))["conditions"] == {"embedding": ["e1", "e2"]}
    other = write(tmp_path / "c", [case("T-1", [turn()])], db("c2", "e2"))
    with pytest.raises(NotComparable, match="catalog"):
        compare(new, other, vary=("embedding",))


def test_dataset_hash_is_compared_only_for_shared_kinds(tmp_path):
    base = write(
        tmp_path / "a", [case("T-1", [turn()])], {"dataset_sha256": {"text": "x", "stt": "y"}}
    )
    new = write(tmp_path / "b", [case("T-1", [turn()])], {"dataset_sha256": {"text": "x"}})
    assert compare(base, new)["conditions"] == {}
    changed = write(tmp_path / "c", [case("T-1", [turn()])], {"dataset_sha256": {"text": "z"}})
    with pytest.raises(NotComparable, match="dataset_sha256"):
        compare(base, changed)


def test_warns_when_completion_breaks(tmp_path):
    base = write(tmp_path / "a", [case("T-1", [turn()])])
    new = write(tmp_path / "b", [case("T-1", [turn()], metrics={"요청 완료율": 0})])
    assert any("요청 완료율" in w for w in compare(base, new)["warnings"])


def test_render_shows_all_six_metrics(tmp_path):
    base = write(tmp_path / "a", [case("T-1", [turn()])])
    text = render(compare(base, base))
    for name in (
        "기능 요구 충족",
        "안내 근거 충실도",
        "장면 해석",
        "곡 반복",
        "추천 카드 대기",
        "서비스 비용",
    ):
        assert name in text


def test_condition_recorded_on_one_side_only_cannot_be_verified(tmp_path):
    base = write(tmp_path / "a", [case("T-1", [turn()])], {"search_embedding": None})
    new = write(tmp_path / "b", [case("T-1", [turn()])], {"search_embedding": "gemini"})
    with pytest.raises(NotComparable, match="search_embedding"):
        compare(base, new)


def test_condition_recorded_on_neither_side_stops_the_comparison(tmp_path):
    base = write(tmp_path / "a", [case("T-1", [turn()])], {"search_embedding": None})
    with pytest.raises(NotComparable, match="search_embedding"):
        compare(base, base)
    assert any(
        "search_embedding" in item
        for item in compare(base, base, allow_incomplete=True)["incomplete"]
    )


def test_kind_without_a_recorded_hash_stops_the_comparison(tmp_path):
    base = write(tmp_path / "a", [case("T-1", [turn()])], {"dataset_sha256": {"stt": "d"}})
    with pytest.raises(NotComparable, match="dataset_sha256"):
        compare(base, base)


def test_workflow_change_must_be_declared(tmp_path):
    base = write(tmp_path / "a", [case("T-1", [turn()])], {"agentic_chat": False})
    new = write(tmp_path / "b", [case("T-1", [turn()])], {"agentic_chat": True})
    with pytest.raises(NotComparable, match="agentic_chat"):
        compare(base, new)
    assert compare(base, new, vary=("agentic_chat",))["conditions"] == {
        "agentic_chat": [False, True]
    }


def test_missing_manifest_stops_the_comparison(tmp_path):
    base = write(tmp_path / "a", [case("T-1", [turn()])])
    new = write(tmp_path / "b", [case("T-1", [turn()])])
    (new / "manifest.json").unlink()
    with pytest.raises(NotComparable, match="manifest"):
        compare(base, new)


def test_case_missing_from_one_run_stops_the_comparison(tmp_path):
    base = write(tmp_path / "a", [case("T-1", [turn()]), case("T-2", [turn()])])
    new = write(tmp_path / "b", [case("T-1", [turn()])])
    with pytest.raises(NotComparable, match="T-2"):
        compare(base, new)
    assert compare(base, new, allow_incomplete=True)["functional"]["text"]["total"] == 1


def test_run_with_fewer_cases_than_planned_is_incomplete(tmp_path):
    cases = [case("T-1", [turn()])]
    base = write(tmp_path / "a", cases, {"planned": {"text": 2}})
    with pytest.raises(NotComparable, match="예정 2"):
        compare(base, base)


def test_failed_service_case_is_incomplete(tmp_path):
    broken = turn()
    broken["actual"].update(done=False, error="Boom")
    base = write(tmp_path / "a", [case("T-1", [turn()])])
    new = write(tmp_path / "b", [case("T-1", [broken])])
    with pytest.raises(NotComparable, match="서비스 성공 0/1"):
        compare(base, new)


def test_coverage_counts_planned_run_success_and_judging(tmp_path):
    unjudged = turn()
    unjudged.pop("initial_judge")
    base = write(tmp_path / "a", [case("T-1", [turn(), unjudged])], {"planned": {"text": 1}})
    row = compare(base, base, allow_incomplete=True)["coverage"]["text"]["base"]
    assert row == {"planned": 1, "run": 1, "service_ok": 1, "judged": 1, "judge_total": 2}


def test_run_without_any_judging_is_not_incomplete(tmp_path):
    plain = turn()
    plain.pop("initial_judge")
    base = write(tmp_path / "a", [case("T-1", [plain])])
    assert compare(base, base)["coverage"]["text"]["new"]["judged"] == 0


def test_denominators_count_only_cases_with_that_check(tmp_path):
    counted = turn(tracks=("1", "2"))
    counted["expected"]["cards"] = 2
    base = write(tmp_path / "a", [case("T-1", [counted]), case("T-2", [turn()])])
    row = compare(base, base)["denominators"]["text"]
    assert row["cards_ok"]["new"] == (1, 1)
    assert row["route_acc"]["new"] == (2, 2)


def test_latency_is_paired_by_feature_and_same_turn(tmp_path):
    fast = turn(cards_ms=1000)
    fast["actual"]["first_text_ms"] = 500
    slow = turn(cards_ms=9000)
    slow["actual"]["first_text_ms"] = 8000
    no_time = turn(cards_ms=None)
    no_time["actual"]["first_text_ms"] = None
    base = write(tmp_path / "a", [case("T-1", [fast]), case("T-2", [slow])])
    new = write(tmp_path / "b", [case("T-1", [fast]), case("T-2", [no_time])])
    row = compare(base, new)["latency"]["text"]["first_text_ms"]
    assert (row["n"], row["excluded"]) == (1, 1)
    assert row["base"]["median"] == row["new"]["median"] == 0.5


def test_cost_per_request_counts_turns_and_flags_missing_cost(tmp_path):
    base = write(tmp_path / "a", [case("T-1", [turn(), turn()], cost=0.04)])
    new = write(tmp_path / "b", [case("T-1", [turn(), turn()], cost=None)])
    row = compare(base, new)["request_cost"]["text"]
    assert row["base"] == {"per_request": 0.02, "requests": 2, "missing": 0}
    assert row["new"] == {"per_request": None, "requests": 2, "missing": 1}


def test_saved_verdict_is_kept_next_to_the_rescored_one(tmp_path):
    claim = turn(text="클래식을 추천해요.", genre="Metal")
    base = write(
        tmp_path / "a", [case("T-1", [claim], metrics={"요청 완료율": 1, "기능 요구 충족률": 1})]
    )
    assert compare(base, base)["rescored"]["text"] == {"base": ["T-1"], "new": ["T-1"]}


def test_duplicate_records_use_the_last_attempt(tmp_path):
    base = write(tmp_path / "a", [case("T-1", [wrong_route()]), case("T-1", [turn()])])
    assert compare(base, base)["functional"]["text"] == {
        "base": 1,
        "new": 1,
        "total": 1,
        "improved": [],
        "regressed": [],
    }


def test_render_shows_coverage_denominators_and_per_feature_numbers(tmp_path):
    base = write(tmp_path / "a", [case("T-1", [turn()])])
    text = render(compare(base, base))
    for name in ("완료 현황", "지표별 분모", "첫 안내 대기", "요청당 비용"):
        assert name in text


def test_fast_mode_run_is_blocked_even_when_the_manifest_does_not_say_so(tmp_path):
    fast = turn()
    fast["actual"]["state"] = {"fast_mode": True}
    base = write(tmp_path / "a", [case("T-1", [turn()])], {"mode": "normal"})
    new = write(tmp_path / "b", [case("T-1", [fast])], {"mode": "normal"})
    old = write(tmp_path / "c", [case("T-1", [fast])], {})
    with pytest.raises(NotComparable, match="mode"):
        compare(base, new)
    with pytest.raises(NotComparable, match="mode"):
        compare(base, old)


def test_changed_attachments_stop_the_comparison(tmp_path):
    base = write(tmp_path / "a", [case("T-1", [turn()])], {"assets_sha256": {"text": "x"}})
    new = write(tmp_path / "b", [case("T-1", [turn()])], {"assets_sha256": {"text": "y"}})
    with pytest.raises(NotComparable, match="assets_sha256"):
        compare(base, new)


def test_unfinished_run_is_incomplete(tmp_path):
    base = write(tmp_path / "a", [case("T-1", [turn()])], {"finished": None})
    with pytest.raises(NotComparable, match="끝나지 않은"):
        compare(base, base)


def test_database_change_during_run_is_incomplete(tmp_path):
    base = write(tmp_path / "a", [case("T-1", [turn()])], {"db_unchanged": False})
    with pytest.raises(NotComparable, match="DB"):
        compare(base, base)


def test_incomplete_status_blocks_the_comparison(tmp_path):
    base = write(tmp_path / "a", [case("T-1", [turn()])], {"status": "incomplete"})
    with pytest.raises(NotComparable, match="미완료"):
        compare(base, base)


def test_kind_present_on_one_side_blocks_unless_kinds_are_chosen(tmp_path):
    base = write(tmp_path / "a", [case("T-1", [turn()])])
    new = write(tmp_path / "b", [case("T-1", [turn()])])
    (new / "stt.jsonl").write_text(json.dumps({"id": "S-1", "success": True}) + "\n")
    with pytest.raises(NotComparable, match="음성"):
        compare(base, new)
    assert list(compare(base, new, kinds=("text",))["functional"]) == ["text"]


def test_judged_run_needs_every_turn_judged(tmp_path):
    plain = turn()
    plain.pop("initial_judge")
    base = write(tmp_path / "a", [case("T-1", [plain])], {"no_judge": False})
    with pytest.raises(NotComparable, match="채점 완료 0/1"):
        compare(base, base)


def test_measurement_conditions_cannot_be_waived(tmp_path):
    base = write(tmp_path / "a", [case("T-1", [turn()])], {"rubric_version": "a"})
    new = write(tmp_path / "b", [case("T-1", [turn()])], {"rubric_version": "b"})
    with pytest.raises(ValueError, match="rubric_version"):
        compare(base, new, vary=("rubric_version",))
    with pytest.raises(NotComparable, match="rubric_version"):
        compare(base, new, allow_incomplete=True)


def test_evaluator_code_change_stops_the_comparison(tmp_path):
    base = write(tmp_path / "a", [case("T-1", [turn()])], {"eval_package_sha256": "a"})
    new = write(tmp_path / "b", [case("T-1", [turn()])], {"eval_package_sha256": "b"})
    with pytest.raises(NotComparable, match="eval_package_sha256"):
        compare(base, new)


def test_old_run_without_a_record_is_reference_only(tmp_path):
    base = write(tmp_path / "a", [case("T-1", [turn()])], {"search_embedding": None})
    new = write(tmp_path / "b", [case("T-1", [turn()])], {"search_embedding": "gemini"})
    report = compare(base, new, allow_incomplete=True)
    assert any("search_embedding" in item for item in report["incomplete"])


def test_fast_mode_is_never_reference_comparable(tmp_path):
    fast = turn()
    fast["actual"]["state"] = {"fast_mode": True}
    base = write(tmp_path / "a", [case("T-1", [turn()])], {"note": "old"})
    new = write(tmp_path / "b", [case("T-1", [fast])], {"note": "old"})
    with pytest.raises(NotComparable, match="mode"):
        compare(base, new, allow_incomplete=True)


def test_retry_cost_and_attempts_are_reported(tmp_path):
    failed = case("T-1", [turn()], cost=0.03)
    failed["turns"][0]["actual"].update(done=False, error="Boom")
    final = case("T-1", [turn()], cost=0.01)
    base = write(tmp_path / "a", [final])
    (base / "text.service.jsonl").write_text("".join(json.dumps(c) + "\n" for c in (failed, final)))
    report = compare(base, base)
    assert report["attempts"]["text"]["new"] == {
        "attempts": 2,
        "retried": ["T-1"],
        "failed": 1,
        "cost": 0.04,
        "missing": 0,
    }
    assert report["service_cost"]["new"]["known"] == pytest.approx(0.04)
    assert report["request_cost"]["text"]["new"]["per_request"] == 0.01
    assert "재시도" in render(report)


def test_quality_metrics_are_left_out_when_a_side_was_not_judged(tmp_path):
    plain = turn()
    plain.pop("initial_judge")
    base = write(tmp_path / "a", [case("T-1", [turn()])], {"no_judge": False})
    new = write(tmp_path / "b", [case("T-1", [plain])], {"no_judge": True})
    report = compare(base, new)
    assert (
        report["quality_skipped"] is True
        and report["faithfulness"] == {}
        and report["scene"] is None
    )
    assert report["functional"]["text"]["total"] == 1
    assert "비교 안 함" in render(report) and "장면 해석" not in render(report)


def test_quality_metrics_stay_when_both_sides_were_judged(tmp_path):
    base = write(tmp_path / "a", [case("T-1", [turn()])], {"no_judge": False})
    report = compare(base, base)
    assert report["quality_skipped"] is False and report["faithfulness"]["text"]["total"] == 1


def test_chosen_kinds_limit_cost_warnings_and_failures(tmp_path):
    base = write(tmp_path / "a", [case("T-1", [turn()], cost=0.01)])
    new = write(tmp_path / "b", [case("T-1", [turn()], cost=0.01)])
    other = case("P-1", [wrong_route()], cost=9, metrics={"요청 완료율": 0})
    (base / "photo.jsonl").write_text(json.dumps(other, ensure_ascii=False) + "\n")
    (base / "photo.service.jsonl").write_text(json.dumps(other, ensure_ascii=False) + "\n")
    (new / "photo.jsonl").write_text(json.dumps(other, ensure_ascii=False) + "\n")
    report = compare(base, new, kinds=("text",))
    assert report["service_cost"]["base"] == {"known": 0.01, "missing": 0}
    assert report["failures"] == {} and report["warnings"] == []


def test_voice_only_comparison_renders_without_song_figures(tmp_path):
    base = write(tmp_path / "a", [case("T-1", [turn()])])
    new = write(tmp_path / "b", [case("T-1", [turn()])])
    for folder in (base, new):
        (folder / "stt.jsonl").write_text(
            json.dumps({"id": "S-1", "success": True, "metrics": {}}) + "\n"
        )
    assert "곡 반복: 최다 곡이 나온 사례 비율 | 없음 | 없음" in render(
        compare(base, new, kinds=("stt",))
    )


def test_unknown_kind_name_is_rejected(tmp_path):
    base = write(tmp_path / "a", [case("T-1", [turn()])])
    with pytest.raises(ValueError, match="photos"):
        compare(base, base, kinds=("text", "photos"))


def test_test_split_report_shows_scores_without_case_names(tmp_path):
    good, bad = case("T-1", [turn()]), case("T-1", [wrong_route()], metrics={"요청 완료율": 0})
    base = write(tmp_path / "a", [good], {"split": "test"})
    new = write(tmp_path / "b", [bad], {"split": "test"})
    (new / "text.service.jsonl").write_text(json.dumps(bad) + "\n" + json.dumps(bad) + "\n")
    text = render(compare(base, new))
    assert "T-1" not in text and "1/1 | 0/1 | -1건" in text and "내용 숨김" in text


def test_test_split_stop_message_has_no_case_names(tmp_path):
    base = write(tmp_path / "a", [case("T-1", [turn()]), case("T-2", [turn()])], {"split": "test"})
    new = write(tmp_path / "b", [case("T-1", [turn()])], {"split": "test"})
    with pytest.raises(NotComparable) as stopped:
        compare(base, new)
    assert "T-2" not in str(stopped.value) and "1건" in str(stopped.value)
