"""LangSmith 데이터셋 연결과 기능별 Experiment 실행. CLI 설정은 run.py가 담당한다."""

import copy
import json
import time

from langsmith import Client

from ..results import read_saved, append_case, service_ok, finished, needs_judge
from ..grading.metrics import feedback
from ..grading.judging import JUDGE_MODEL, RUBRIC_VERSION, Budget, grade_case
from .runner import execute_case
from .telemetry import summarize_usage


class EvaluationClient(Client):
    """평가 점수를 기록할 때 프로젝트의 기본 보관 등급을 유지한다."""

    def create_feedback(self, *args, **kwargs):
        kwargs["extend_trace_retention"] = False
        return super().create_feedback(*args, **kwargs)


VERSION = "initial-metrics-20261031-v2"


def dataset_prefix(version):
    return "meomuneum-initial" if version == "initial-20261027" else f"meomuneum-{version}"


FUNCTION_NAMES = {
    "text": "텍스트",
    "photo": "사진",
    "dialogue": "멀티턴",
    "stt": "음성",
    "faults": "입력·장애",
}


def experiment_name(title, kind, judged=True):
    title = "dev 기준선" if title.startswith("baseline-dev-3f7733f") else title
    mode = "코드 검사" if kind in ("stt", "faults") or not judged else "Sonnet 5.5"
    return f"{title} · {FUNCTION_NAMES[kind]} · {mode}"


def select_split(rows, split):
    """split은 문자열이나 목록이다. smoke는 validation 사례에 함께 표시한다."""
    if not split:
        return rows
    return [
        r
        for r in rows
        if split in ([r["split"]] if isinstance(r.get("split"), str) else r.get("split") or [])
    ]


def public_id(case_id, dataset_version):
    """test 사례 ID는 화면에 남기지 않는다. ID로 원본 후보를 찾을 수 있다."""
    return "(숨김)" if str(dataset_version).startswith("test-") else case_id


def tag_dataset(client, dataset_id, expected, version):
    """예제가 저장된 시각에 버전 태그를 붙인다. 호출 시각으로 붙이면 서버 반영 전이라 빈 버전이 된다."""
    for _ in range(10):
        saved = list(client.list_examples(dataset_id=dataset_id))
        if len(saved) == expected:
            break
        time.sleep(1)
    assert len(saved) == expected, "remote_examples_not_ready"
    client.update_dataset_tag(
        dataset_id=dataset_id,
        as_of=max(e.modified_at or e.created_at for e in saved),
        tag=version,
    )


def connect_dataset(client, agent_tag, kind, rows, dataset_root, version):
    """기존 버전과 기대값을 대조하고, 없는 데이터셋만 등록한다."""
    name = f"{dataset_prefix(version)}-{kind}"
    if client.has_dataset(dataset_name=name):
        dataset = client.read_dataset(dataset_name=name)
    else:
        dataset = client.create_dataset(
            name,
            description=f"머문음 평가 {version} {kind}. AI 검수, 사람 정답 검수 아님. {VERSION}",
            metadata={"version": version, "metric_version": VERSION},
            tag_value_ids=[agent_tag],
        )
        examples = []
        for row in rows:
            example = {k: row[k] for k in ("inputs", "outputs", "metadata", "split")}
            if row.get("attachments"):
                example["attachments"] = {
                    k: (
                        "audio/wav" if k == "audio" else "image/jpeg",
                        (dataset_root / rel).read_bytes(),
                    )
                    for k, rel in row["attachments"].items()
                }
            examples.append(example)
        client.create_examples(dataset_id=dataset.id, examples=examples)
        tag_dataset(client, dataset.id, len(rows), version)
    remote = list(client.list_examples(dataset_id=dataset.id, as_of=version))
    local = {r["metadata"]["id"]: r for r in rows}
    assert len(remote) == len(rows) and all(
        e.inputs == local[e.metadata["id"]]["inputs"]
        and e.outputs == local[e.metadata["id"]]["outputs"]
        for e in remote
    ), "remote_dataset_mismatch"
    return dataset, remote


async def run_kind(kind, rows, client, agent_tag, args, metadata, capture):
    from langsmith import aevaluate, get_current_run_tree
    from .telemetry import CASE

    dataset, remote = connect_dataset(
        client, agent_tag, kind, rows, args.dataset_dir, args.dataset_version
    )
    local = {r["metadata"]["id"]: r for r in rows}
    judged = kind not in ("stt", "faults") and not args.no_judge
    project_name = experiment_name(args.name, kind, judged)
    metadata = dict(metadata, kind=kind)
    if not judged:
        for key in ("judge_model", "judge_settings", "judge_prompt_sha256", "rubric_version"):
            metadata.pop(key, None)
    selected = {r["metadata"]["id"] for r in select_split(rows, args.split)}
    service_path, result_path, project_path, judge_path = (
        args.output / f"{kind}{suffix}"
        for suffix in (".service.jsonl", ".jsonl", ".project.json", ".judge.jsonl")
    )
    saved_service, saved_results = read_saved(service_path), read_saved(result_path)
    kept = {i for i, c in saved_results.items() if i in selected and finished(kind, c, judged)}
    previous = json.loads(project_path.read_text()) if args.resume and project_path.exists() else []
    if previous and kept == selected:
        project = client.read_project(project_id=previous[-1])
        return {
            "dataset_id": str(dataset.id),
            "experiment_id": str(project.id),
            "experiment_name": project.name,
            "cases": len(kept),
            "planned": len(selected),
            "status": "complete",
        }
    # 재개마다 새 실험을 만들어 사례당 최종 결과 하나만 남긴다. 저장 응답과 판정을 재사용하므로 모델을 다시 호출하지 않는다.
    attempt = len(previous) + 1
    if previous:
        project_name, metadata = (
            f"{project_name} · 시도 {attempt}",
            dict(metadata, attempt=attempt, supersedes=previous[-1]),
        )
    project = client.create_project(
        project_name,
        reference_dataset_id=dataset.id,
        metadata=metadata,
        num_examples=len(selected),
        num_repetitions=1,
        tag_value_ids=[agent_tag],
    )
    project_path.write_text(json.dumps([*previous, str(project.id)]))
    results = []
    budget = None
    if judged:
        budget = Budget(args.judge_budget_usd - args.judge_spent_usd, args.judge_reservation_usd)
        if "P-035" in local:
            local["P-035"] = dict(local["P-035"], judge_image_transport="lossless_png")

    async def target(inputs, metadata):
        row = local[metadata["id"]]
        CASE.set(row["metadata"]["id"])
        reusable = saved_service.get(row["metadata"]["id"]) or saved_results.get(
            row["metadata"]["id"]
        )
        if reusable and service_ok(kind, reusable):
            case = copy.deepcopy(reusable)
        else:
            case = await execute_case(
                kind, inputs, row, args.dataset_dir, args.service_root, capture
            )
            append_case(service_path, case)
        service_calls = case.get("service_calls") or []
        run = get_current_run_tree()
        if run and service_calls:
            usage = summarize_usage(service_calls)
            if case["service_cost_usd"] is not None:
                usage["total_cost"] = case["service_cost_usd"]
            run.set(usage_metadata=usage)
        return case

    async def evaluator(run, example):
        case = copy.deepcopy(run.outputs)
        CASE.set(example.metadata["id"] + ":judge")
        if judged:
            await grade_case(
                case,
                local[example.metadata["id"]],
                saved_results,
                budget,
                judge_path,
                project,
                client,
                args.dataset_dir,
            )
        case["metrics"] = feedback(kind, case)
        case["run_id"] = str(run.id)
        results.append(case)
        if case["id"] not in kept:
            append_case(result_path, case)
        print(kind, public_id(case["id"], args.dataset_version), "saved", flush=True)
        details = json.dumps(
            {
                "case_id": case["id"],
                "judge_model": JUDGE_MODEL if budget else None,
                "rubric_version": RUBRIC_VERSION if budget else None,
                "turn_judgments": [
                    {k: v for k, v in t.get("initial_judge", {}).items() if k != "usage"}
                    for t in case.get("turns", [])
                ],
            },
            ensure_ascii=False,
        )
        return {
            "results": [
                dict(
                    key=k,
                    score=v,
                    comment=details,
                    **({"value": {"exact_usd": v}} if k.endswith("(USD)") else {}),
                )
                for k, v in case["metrics"].items()
            ]
        }

    if selected:
        await aevaluate(
            target,
            data=[e for e in remote if e.metadata["id"] in selected],
            evaluators=[evaluator],
            experiment=project,
            client=client,
            max_concurrency=1 if kind == "faults" else args.concurrency,
        )
    # 강제 종료가 아니어도 사례가 빠질 수 있어 결과 파일 기준으로 완료를 판정한다.
    status = (
        "complete"
        if len(results) == len(selected) and all(finished(kind, c, judged) for c in results)
        else "incomplete"
    )
    if budget:
        args.judge_spent_usd += budget.spent
        total = sum(needs_judge(t) for r in results for t in r["turns"])
        scored = sum(
            t.get("initial_judge", {}).get("status") == "ok" for r in results for t in r["turns"]
        )
        project_name = (
            f"{project_name} · {'완료' if status == 'complete' else '중단'} {scored}/{total}"
        )
        client.update_project(project.id, name=project_name)
        assert not budget.stop, budget.stop
    # 다음 회귀 평가에서 바로 비교할 수 있게 데이터셋의 기준 실험으로 지정한다.
    if status == "complete":
        for old_id in previous:
            old_name = client.read_project(project_id=old_id).name
            if not old_name.startswith("[이전 시도]"):
                client.update_project(old_id, name=f"[이전 시도] {old_name}")
    if (
        args.name.startswith("baseline-")
        and not args.no_judge
        and status == "complete"
        and len(selected) == len(rows)
    ):
        client.request_with_retries(
            "PATCH", f"/datasets/{dataset.id}", json={"baseline_experiment_id": str(project.id)}
        )
    return {
        "dataset_id": str(dataset.id),
        "experiment_id": str(project.id),
        "experiment_name": project_name,
        "cases": len(results),
        "planned": len(selected),
        "status": status,
    }
