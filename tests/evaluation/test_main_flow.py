from evaluation.grading import judging
from evaluation.report import render

"""main()을 외부 호출 없이 끝까지 돌린다. 인자, manifest, 재개, 비교가 이어지는지 본다."""

import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import httpx
import langsmith
import pytest

from evaluation import run as batch
from evaluation import compare as comparison
import evaluation.execution.runner

COMMIT = [
    "git",
    "-c",
    "user.name=t",
    "-c",
    "user.email=t@t",
    "commit",
    "-q",
    "--allow-empty",
    "-m",
    "x",
]
VERDICT = {
    "status": "ok",
    "response_compliance": 1,
    "faithfulness": 1,
    "explanation": 2,
    "scene_accuracy": None,
    "scene_consistency": None,
    "reasons": ["근거"],
    "usage": {"cost": 0.01},
}


class Response:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


class Http:
    """키·모델·에이전트 조회만 대신한다."""

    send = None

    def __init__(self, **_):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def get(self, url, **_):
        if url.endswith("/key"):
            return Response({"data": {"limit": 20, "limit_remaining": 5}})
        if url.endswith("/models"):
            return Response(
                {
                    "data": [
                        {
                            "id": batch.JUDGE_MODEL,
                            "context_length": 1000,
                            "pricing": {"prompt": ".000002", "completion": ".00001"},
                        }
                    ]
                }
            )
        return Response({"tag_value_id": "tag"})


class Remote(MagicMock):
    """데이터셋 등록과 조회를 메모리에 남긴다."""

    def has_dataset(self, dataset_name):
        return dataset_name in self.stored

    def create_dataset(self, name, **_):
        self.stored[name] = SimpleNamespace(id=name, examples=[])
        return self.stored[name]

    def read_dataset(self, dataset_name):
        return self.stored[dataset_name]

    def create_examples(self, dataset_id, examples):
        from datetime import datetime, timedelta, timezone

        saved = datetime.now(timezone.utc) + timedelta(
            seconds=5
        )  # 서버 반영 시각이 호출 시각보다 늦다.
        self.stored[dataset_id].examples += [
            SimpleNamespace(
                id=uuid4(),
                created_at=saved,
                modified_at=saved,
                **{k: e[k] for k in ("inputs", "outputs", "metadata", "split")},
            )
            for e in examples
        ]

    def list_examples(self, dataset_id, **_):
        return self.stored[dataset_id].examples

    def create_project(self, name, **_):
        return SimpleNamespace(id=uuid4(), name=name)

    def read_project(self, project_id):
        return SimpleNamespace(id=project_id, name="이전")


@pytest.fixture
def flow(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in ("LANGSMITH_API_KEY", "OPENROUTER_EVAL_API_KEY", "LANGSMITH_TRACING"):
        monkeypatch.setenv(name, "test-only")
    for name in list(sys.modules):
        if name == "backend" or name.startswith(("backend.", "db.")):
            monkeypatch.delitem(sys.modules, name)
    service = tmp_path / "service"
    service.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=service, check=True)
    subprocess.run(COMMIT, cwd=service, check=True)
    data = tmp_path / "data"
    data.mkdir()
    (tmp_path / ".env").write_text("")
    rows = [
        {
            "inputs": {"turns": [{"message": i}]},
            "outputs": {"turns": [{"routes": [["recommend", "tracks"]]}]},
            "metadata": {"id": i, "metrics": []},
            "split": split,
        }
        for i, split in (("T-1", ["validation", "smoke"]), ("T-2", "validation"))
    ]
    (data / "text.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    remote = Remote()
    remote.stored = {}
    state = SimpleNamespace(verdicts=[], remote=remote, tmp=tmp_path, service=service, data=data)

    async def run_case(row, _):
        turn = {
            "expected": row["outputs"]["turns"][0],
            "actual": {
                "input": {},
                "text": "안내",
                "tracks": [{"track_id": "1", "title": "t", "artist": "a"}],
                "done": True,
                "state": {},
                "route": ["recommend", "tracks"],
                "total_ms": 100,
                "first_text_ms": 10,
                "cards_ms": 20,
            },
        }
        from evaluation.grading.scoring import score_case

        return score_case(
            {
                "id": row["metadata"]["id"],
                "split": row["split"],
                "metadata": row["metadata"],
                "expected_turns": 1,
                "turns": [turn],
                "outputs": row["outputs"],
                "attachments": {},
            }
        )

    async def evaluate(target, **kw):
        for example in kw["data"]:
            case = await target(example.inputs, example.metadata)
            await kw["evaluators"][0](SimpleNamespace(id=uuid4(), outputs=case), example)

    monkeypatch.setattr(httpx, "Client", Http)
    monkeypatch.setattr(batch, "EvaluationClient", lambda: remote)
    monkeypatch.setattr(langsmith, "aevaluate", evaluate)
    monkeypatch.setattr(langsmith, "get_current_run_tree", lambda: None)
    monkeypatch.setattr(langsmith, "trace", MagicMock())
    monkeypatch.setattr(evaluation.execution.runner, "run_case", run_case)
    monkeypatch.setattr(
        batch,
        "db_fingerprint",
        lambda: {"sha": "x", "catalog_sha": "c", "embedding_sha": "e"},
    )
    monkeypatch.setattr(
        judging, "judge", lambda *_: state.verdicts.pop(0) if state.verdicts else VERDICT
    )

    def run(output, *extra, version="validation-20261102"):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "batch.py",
                "--env-file",
                str(tmp_path / ".env"),
                "--service-root",
                str(service),
                "--dataset-dir",
                str(data),
                "--dataset-version",
                version,
                "--output",
                str(tmp_path / output),
                "--name",
                "실험",
                "--kinds",
                "text",
                "--execute",
                *extra,
            ],
        )
        batch.main()
        return json.loads((tmp_path / output / "manifest.json").read_text())

    state.run = run
    return state


def test_a_paid_failure_then_resume_finishes_with_every_charge_counted(flow):
    flow.verdicts = [{"status": "evaluation_error", "error": "Timeout", "usage": {"cost": 0.03}}]
    first = flow.run("a", "--split", "validation")
    assert (
        first["status"] == "incomplete"
        and first["planned"] == {"text": 2}
        and first["split"] == "validation"
    )
    assert first["judge_spent_usd"] == pytest.approx(0.04)
    final = flow.run("a", "--split", "validation", "--resume")
    assert final["status"] == "complete" and len(final["resumed"]) == 1
    assert final["judge_spent_usd"] == pytest.approx(0.05) and final["judge_cost_missing"] == 0


def test_a_second_run_into_the_same_folder_is_refused(flow):
    flow.run("a", "--split", "smoke", "--no-judge")
    with pytest.raises(AssertionError, match="fresh"):
        flow.run("a", "--split", "smoke", "--no-judge")


def test_two_finished_runs_can_be_compared_and_code_check_runs_skip_quality(flow):
    flow.run("a", "--split", "validation")
    flow.run("b", "--split", "validation", "--no-judge")
    report = comparison.compare(flow.tmp / "a", flow.tmp / "b")
    assert report["functional"]["text"]["total"] == 2 and report["quality_skipped"] is True
    assert "비교 안 함" in render(report)


def test_test_data_hides_case_names_even_without_the_split_option(flow):
    flow.run("a", version="test-20261102")
    flow.run("b", version="test-20261102")
    rows = [json.loads(line) for line in (flow.tmp / "b/text.jsonl").read_text().splitlines()]
    rows[0]["turns"][0]["actual"]["route"] = ["guide", "guide"]
    (flow.tmp / "b/text.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    text = render(comparison.compare(flow.tmp / "a", flow.tmp / "b"))
    assert "T-1" not in text and "-1건" in text


def test_labels_are_tied_to_the_catalog_they_were_written_for(flow):
    (flow.data / "catalog_sha.txt").write_text("other\n")
    with pytest.raises(AssertionError, match="catalog_changed_since_labels"):
        flow.run("a", "--no-judge")
    (flow.data / "catalog_sha.txt").write_text("c\n")
    assert flow.run("b", "--no-judge")["status"] == "complete"


def test_test_case_names_stay_out_of_the_console(flow, capsys):
    flow.run("a", "--no-judge", version="test-20261102")
    assert "T-1" not in capsys.readouterr().out
    flow.run("b", "--no-judge")
    assert "T-1" in capsys.readouterr().out


def test_new_dataset_is_tagged_at_the_time_its_examples_were_saved(flow):
    flow.run("a", "--no-judge")
    stored = flow.remote.stored["meomuneum-validation-20261102-text"]
    tagged = flow.remote.update_dataset_tag.call_args.kwargs
    assert tagged["tag"] == "validation-20261102"
    assert tagged["as_of"] == max(e.modified_at for e in stored.examples)
