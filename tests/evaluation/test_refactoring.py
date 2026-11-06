"""모듈 분리 이후 데이터 격리와 서비스 호출 경계를 확인한다."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from evaluation import run
from evaluation.execution import runner
from evaluation.execution.experiments import dataset_prefix, public_id


def test_two_dataset_roots_do_not_share_process_state(tmp_path):
    for name in ("first", "second"):
        root = tmp_path / name
        root.mkdir()
        (root / "text.jsonl").write_text(json.dumps({"id": name}) + "\n")
    assert run.read("text", tmp_path / "first") == [{"id": "first"}]
    assert run.read("text", tmp_path / "second") == [{"id": "second"}]
    assert run.read("text", tmp_path / "first") == [{"id": "first"}]
    assert dataset_prefix("initial-20261027") == "meomuneum-initial"
    assert public_id("private", "test-20261102") == "(숨김)"
    assert public_id("visible", "validation-20261102") == "visible"


@pytest.mark.parametrize("failed", [False, True])
def test_voice_dispatch_preserves_input_and_failure(tmp_path, monkeypatch, failed):
    from backend.stt import service

    (tmp_path / "sample.wav").write_bytes(b"voice")

    def transcribe(upload):
        assert upload.file.read() == b"voice"
        assert upload.content_type == "audio/wav"
        if failed:
            raise TimeoutError()
        return "정답"

    monkeypatch.setattr(service, "transcribe_audio", transcribe)
    case = asyncio.run(
        runner.execute_case(
            "stt",
            {"audio": "sample.wav"},
            {"metadata": {"id": "S-1"}, "outputs": {"transcript": "정답"}},
            tmp_path,
            tmp_path,
            SimpleNamespace(records=[]),
        )
    )
    assert case["success"] is not failed
    assert case["transcript"] == ("" if failed else "정답")
    assert case.get("error") == ("TimeoutError" if failed else None)
    assert case["service_cost_usd"] is None


def test_evaluator_hash_covers_nested_grading_code(tmp_path, monkeypatch):
    entrypoint = tmp_path / "run.py"
    entrypoint.write_text("# entrypoint\n")
    grading = tmp_path / "grading"
    grading.mkdir()
    score = grading / "scoring.py"
    score.write_text("# before\n")
    monkeypatch.setattr(run, "__file__", str(entrypoint))
    before = run.evaluator_digest()
    score.write_text("# after\n")
    assert run.evaluator_digest() != before


def test_evaluation_feedback_does_not_extend_retention(monkeypatch):
    from langsmith import Client
    from evaluation.execution.experiments import EvaluationClient

    captured = {}

    def create(self, *args, **kwargs):
        captured.update(kwargs)
        return "saved"

    monkeypatch.setattr(Client, "create_feedback", create)
    client = EvaluationClient.__new__(EvaluationClient)
    assert client.create_feedback("run", "score", score=1) == "saved"
    assert captured["score"] == 1
    assert captured["extend_trace_retention"] is False


def test_usage_summary_preserves_missing_audio_tokens():
    from evaluation.execution.telemetry import summarize_usage

    assert summarize_usage([{"usage": {"seconds": 3.71, "cost": 0.01}}]) == {}
    assert summarize_usage(
        [
            {"usage": {"prompt_tokens": 10, "completion_tokens": 2}},
            {"usage": {"input_tokens": 5, "output_tokens": 0}},
        ]
    ) == {"input_tokens": 15, "output_tokens": 2}
    assert summarize_usage([{"usage": {"prompt_tokens": 10}}, {"usage": {}}]) == {}
