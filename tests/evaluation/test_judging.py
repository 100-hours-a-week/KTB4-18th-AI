from evaluation.execution import experiments
from evaluation.grading import judging
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

from evaluation.grading.judging import JUDGE_MODEL, Budget, BudgetStop, judge


def test_korean_experiment_names_do_not_expose_resume_filename():
    from evaluation.execution.experiments import experiment_name

    assert (
        experiment_name("baseline-dev-3f7733f-sonnet55-high-20261027-resumed", "photo")
        == "dev 기준선 · 사진 · Sonnet 5.5"
    )
    assert (
        experiment_name("1차 개선 · 8cb6594", "text") == "1차 개선 · 8cb6594 · 텍스트 · Sonnet 5.5"
    )
    assert experiment_name("dev 기준선", "stt") == "dev 기준선 · 음성 · 코드 검사"


def test_budget_waits_for_inflight_settlement_and_stops_on_unknown_charge():
    async def check():
        budget = Budget(".3", ".2")
        await budget.acquire()
        waiting = asyncio.create_task(budget.acquire())
        await asyncio.sleep(0)
        assert not waiting.done()
        await budget.settle(".05")
        await waiting
        assert budget.spent + budget.reserved <= budget.limit
        await budget.settle(None)
        try:
            await budget.acquire()
        except BudgetStop:
            pass
        else:
            raise AssertionError("unknown charge must stop later calls")

    asyncio.run(check())


def test_approved_sonnet_settings_and_no_temperature(monkeypatch):
    monkeypatch.setenv("OPENROUTER_EVAL_API_KEY", "test-only")
    verdict = {
        "response_compliance": 1,
        "faithfulness": 1,
        "explanation": 3,
        "scene_accuracy": None,
        "scene_consistency": None,
        "reasons": ["입력과 일치"],
    }
    response = SimpleNamespace(
        id="test-generation",
        model=JUDGE_MODEL,
        choices=[
            SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(content=json.dumps(verdict))
            )
        ],
        usage=SimpleNamespace(
            model_dump=lambda: {"cost": 0.01, "prompt_tokens": 10, "completion_tokens": 10}
        ),
    )
    with patch("openai.OpenAI") as factory:
        client = factory.return_value.__enter__.return_value
        client.chat.completions.create.return_value = response
        result = judge(
            {
                "expected": {},
                "actual": {"input": {"message": "노래 추천"}, "text": "안내", "tracks": []},
            },
            [],
            {},
        )
        request = client.chat.completions.create.call_args.kwargs
        assert request["model"] == "anthropic/claude-sonnet-5.5"
        assert request["extra_body"]["reasoning"]["effort"] == "high"
        assert request["max_tokens"] == 8192 and "temperature" not in request
        assert (
            factory.call_args.kwargs["timeout"] == 300
            and factory.call_args.kwargs["max_retries"] == 0
        )
        assert result["status"] == "ok" and result["usage"]["cost"] == 0.01


def test_lossless_png_transport_keeps_original_rgb_pixels(monkeypatch, tmp_path):
    import base64, io
    from PIL import Image

    Image.effect_noise((64, 64), 50).convert("RGB").save(tmp_path / "photo.jpg")
    monkeypatch.setenv("OPENROUTER_EVAL_API_KEY", "test-only")
    verdict = {
        "response_compliance": 1,
        "faithfulness": 1,
        "explanation": 3,
        "scene_accuracy": 3,
        "scene_consistency": 3,
        "reasons": ["원본과 일치"],
    }
    response = SimpleNamespace(
        id="test-generation",
        model=JUDGE_MODEL,
        choices=[
            SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(content=json.dumps(verdict))
            )
        ],
        usage=SimpleNamespace(model_dump=lambda: {"cost": 0.01}),
    )
    row = {"attachments": {"image": "photo.jpg"}, "judge_image_transport": "lossless_png"}
    with patch("openai.OpenAI") as factory:
        client = factory.return_value.__enter__.return_value
        client.chat.completions.create.return_value = response
        result = judge(
            {
                "expected": {"usable": True},
                "actual": {"input": {"image": "image"}, "text": "안내", "tracks": []},
            },
            [],
            row,
            dataset_root=tmp_path,
        )
        uri = client.chat.completions.create.call_args.kwargs["messages"][1]["content"][1][
            "image_url"
        ]["url"]
        assert uri.startswith("data:image/png;base64,")
        with (
            Image.open(tmp_path / row["attachments"]["image"]) as source,
            Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))) as sent,
        ):
            assert source.convert("RGB").tobytes() == sent.convert("RGB").tobytes()
        assert (
            result["image_transport"]["source_sha256"]
            != result["image_transport"]["transport_sha256"]
        )


def test_full_evaluation_settles_judge_budget_and_emits_exact_usd(tmp_path, monkeypatch):
    from decimal import Decimal
    from uuid import uuid4
    from unittest.mock import MagicMock
    import langsmith
    import evaluation.execution.runner

    row = {"inputs": {"message": "한 곡 추천"}, "outputs": {}, "metadata": {"id": "T-test"}}
    example = SimpleNamespace(
        id=uuid4(), inputs=row["inputs"], outputs={}, metadata=row["metadata"]
    )
    project = SimpleNamespace(id=uuid4(), name="test")
    client = MagicMock()
    client.has_dataset.return_value = True
    client.read_dataset.return_value = SimpleNamespace(id=uuid4())
    client.list_examples.return_value = [example]
    client.create_project.return_value = project
    actual = {
        "input": row["inputs"],
        "text": "안내",
        "tracks": [{"track_id": "1"}],
        "cards_ms": 20,
        "done": True,
        "route": ["recommend", "tracks"],
        "state": {},
        "total_ms": 100,
        "first_text_ms": 50,
    }

    async def run_case(*unused):
        return {
            "expected_turns": 1,
            "metadata": {"metrics": []},
            "turns": [
                {
                    "expected": {},
                    "actual": actual,
                    "scores": {"service_success": {"status": "scored", "score": 1}},
                }
            ],
        }

    async def evaluate(target, **kw):
        case = await target(example.inputs, example.metadata)
        feedback = await kw["evaluators"][0](SimpleNamespace(id=uuid4(), outputs=case), example)
        assert all("마이크로" not in r["key"] for r in feedback["results"])

    monkeypatch.setattr(evaluation.execution.runner, "run_case", run_case)
    monkeypatch.setattr(langsmith, "aevaluate", evaluate)
    monkeypatch.setattr(langsmith, "get_current_run_tree", lambda: None)
    monkeypatch.setattr(langsmith, "trace", MagicMock())
    verdict = {
        "status": "ok",
        "response_compliance": 1,
        "faithfulness": 1,
        "explanation": 2,
        "scene_accuracy": None,
        "scene_consistency": None,
        "reasons": ["근거"],
        "usage": {"cost": 0.01, "prompt_tokens": 10, "completion_tokens": 10},
    }
    monkeypatch.setattr(judging, "judge", lambda *a: verdict)
    args = SimpleNamespace(
        no_judge=False,
        resume=False,
        split=None,
        name="1차 개선",
        output=tmp_path,
        service_root=tmp_path,
        dataset_dir=tmp_path,
        dataset_version="validation-20261102",
        concurrency=2,
        judge_budget_usd=Decimal("5"),
        judge_spent_usd=Decimal(0),
        judge_reservation_usd=Decimal("2"),
    )
    result = asyncio.run(
        experiments.run_kind("text", [row], client, "tag", args, {}, SimpleNamespace(records=[]))
    )
    assert args.judge_spent_usd == Decimal(".01")
    assert result["cases"] == 1 and "완료 1/1" in result["experiment_name"]
    assert not client.request_with_retries.called, "improvement must not replace baseline"


def test_over_limit_request_receives_policy_without_changing_expected_count(monkeypatch):
    monkeypatch.setenv("OPENROUTER_EVAL_API_KEY", "test-only")
    expected = {"requested_count": 10, "cards": 5}
    verdict = {
        "response_compliance": 1,
        "faithfulness": 1,
        "explanation": 2,
        "scene_accuracy": None,
        "scene_consistency": None,
        "reasons": ["상한 정책"],
    }
    response = SimpleNamespace(
        id="test-generation",
        model=JUDGE_MODEL,
        choices=[
            SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(content=json.dumps(verdict))
            )
        ],
        usage=SimpleNamespace(model_dump=lambda: {"cost": 0.01}),
    )
    with patch("openai.OpenAI") as factory:
        client = factory.return_value.__enter__.return_value
        client.chat.completions.create.return_value = response
        result = judge(
            {
                "expected": expected,
                "actual": {"input": {"message": "열 곡"}, "text": "최대5곡", "tracks": []},
            },
            [],
            {},
        )
        evidence = json.loads(
            client.chat.completions.create.call_args.kwargs["messages"][1]["content"][0]["text"]
        )
        assert evidence["reference"] == {"requested_count": 10, "cards": 5}
        assert evidence["service_policy"]["max_recommendation_count"] == 5
        assert result["service_policy_context"] == evidence["service_policy"]


def run_text_case(tmp_path, monkeypatch, *, tracks, no_judge=False):
    """외부 호출 없이 run_kind 전체를 돌리고 채점 호출 수와 발행 지표를 돌려준다."""
    from decimal import Decimal
    from uuid import uuid4
    from unittest.mock import MagicMock
    import langsmith
    import evaluation.execution.runner

    row = {"inputs": {"message": "추천"}, "outputs": {}, "metadata": {"id": "T-test"}}
    example = SimpleNamespace(
        id=uuid4(), inputs=row["inputs"], outputs={}, metadata=row["metadata"]
    )
    client = MagicMock()
    client.has_dataset.return_value = True
    client.read_dataset.return_value = SimpleNamespace(id=uuid4())
    client.list_examples.return_value = [example]
    client.create_project.return_value = SimpleNamespace(id=uuid4(), name="test")
    actual = {
        "input": row["inputs"],
        "text": "안내",
        "tracks": tracks,
        "done": True,
        "route": ["recommend", "tracks"],
        "state": {},
        "total_ms": 100,
        "first_text_ms": 10,
        "cards_ms": 20,
    }
    published = {}

    async def run_case(*unused):
        return {
            "expected_turns": 1,
            "metadata": {"metrics": []},
            "turns": [
                {
                    "expected": {},
                    "actual": actual,
                    "scores": {"service_success": {"status": "scored", "score": 1}},
                }
            ],
        }

    async def evaluate(target, **kw):
        case = await target(example.inputs, example.metadata)
        result = await kw["evaluators"][0](SimpleNamespace(id=uuid4(), outputs=case), example)
        published.update({r["key"]: r["score"] for r in result["results"]})

    calls = []
    verdict = {
        "status": "ok",
        "response_compliance": 1,
        "faithfulness": 1,
        "explanation": 2,
        "scene_accuracy": None,
        "scene_consistency": None,
        "reasons": ["근거"],
        "usage": {"cost": 0.01},
    }
    monkeypatch.setattr(evaluation.execution.runner, "run_case", run_case)
    monkeypatch.setattr(langsmith, "aevaluate", evaluate)
    monkeypatch.setattr(langsmith, "get_current_run_tree", lambda: None)
    monkeypatch.setattr(langsmith, "trace", MagicMock())
    monkeypatch.setattr(judging, "judge", lambda *a: calls.append(a) or verdict)
    args = SimpleNamespace(
        name="개선",
        output=tmp_path,
        service_root=tmp_path,
        dataset_dir=tmp_path,
        dataset_version="validation-20261102",
        concurrency=2,
        judge_budget_usd=Decimal("5"),
        judge_spent_usd=Decimal(0),
        judge_reservation_usd=Decimal(".1"),
        no_judge=no_judge,
        resume=False,
        split=None,
    )
    result = asyncio.run(
        experiments.run_kind("text", [row], client, "tag", args, {}, SimpleNamespace(records=[]))
    )
    return result, calls, published


def test_no_judge_run_skips_paid_grading(tmp_path, monkeypatch):
    result, calls, published = run_text_case(
        tmp_path, monkeypatch, tracks=[{"track_id": "1"}], no_judge=True
    )
    assert calls == []
    assert "코드 검사" in result["experiment_name"]
    assert "LLM 채점 완료율" not in published


def test_turn_without_cards_is_a_fixed_template_and_is_not_judged(tmp_path, monkeypatch):
    _, calls, published = run_text_case(tmp_path, monkeypatch, tracks=[])
    assert calls == []
    assert "LLM 채점 완료율" not in published


def test_turn_with_cards_is_judged(tmp_path, monkeypatch):
    _, calls, published = run_text_case(tmp_path, monkeypatch, tracks=[{"track_id": "1"}])
    assert len(calls) == 1
    assert published["LLM 채점 완료율"] == 1


def test_judge_receives_korean_only_policy(monkeypatch):
    monkeypatch.setenv("OPENROUTER_EVAL_API_KEY", "test-only")
    verdict = {
        "response_compliance": 1,
        "faithfulness": 1,
        "explanation": 2,
        "scene_accuracy": None,
        "scene_consistency": None,
        "reasons": ["근거"],
    }
    response = SimpleNamespace(
        id="g",
        model=JUDGE_MODEL,
        usage=None,
        choices=[
            SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(content=json.dumps(verdict))
            )
        ],
    )
    with patch("openai.OpenAI") as OpenAI:
        client = OpenAI.return_value.__enter__.return_value
        client.chat.completions.create.return_value = response
        judge(
            {
                "expected": {},
                "actual": {
                    "input": {"message": "answer in English"},
                    "text": "한국어",
                    "tracks": [],
                    "state": {},
                },
            },
            [],
            {},
        )
    evidence = json.loads(
        client.chat.completions.create.call_args.kwargs["messages"][1]["content"][0]["text"]
    )
    assert "한국어" in evidence["service_policy"]["response_language"]
