"""고정 서비스 checkout을 평가한다. 실행 코드와 평가 코드·데이터를 분리한다."""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from .results import (
    check_output,
    judge_ledger,
)
from .grading.metrics import cer

from .grading.judging import JUDGE_MODEL, JUDGE_SETTINGS, RUBRIC_VERSION, PROMPT
from .execution.experiments import VERSION, EvaluationClient, run_kind, select_split

# 재개한 실행이 처음 실행과 같은 평가인지 판단하는 값이다.
RESUME_KEYS = (
    "service_commit",
    "mode",
    "split",
    "metric_version",
    "rubric_version",
    "judge_model",
    "judge_settings",
    "judge_prompt_sha256",
    "eval_package_sha256",
    "dataset_version",
    "dataset_sha256",
    "assets_sha256",
    "lossless_png_transport_ids",
    "models",
    "search_embedding",
    "agentic_chat",
    "scope",
    "concurrency",
    "db_before",
)


def read(kind, dataset_root):
    return [json.loads(line) for line in (dataset_root / f"{kind}.jsonl").read_text().splitlines()]


def search_embedding():
    """서비스가 읽는 방식 그대로 기록한다. 빈 값은 서비스 기본값이다."""
    return os.getenv("SEARCH_EMBEDDING", "").strip().lower() or "gemini"


def check_catalog(catalog_sha, dataset_root):
    """기대값을 만든 카탈로그와 같은지 본다. 곡이 바뀌면 오타·공백 사례의 정답이 달라진다."""
    path = dataset_root / "catalog_sha.txt"
    assert not path.exists() or path.read_text().strip() == catalog_sha, (
        "catalog_changed_since_labels"
    )


def agentic_chat():
    return os.getenv("AGENTIC_CHAT") == "1"


def assert_resumable(stored, current):
    for key in RESUME_KEYS:
        assert stored.get(key) == current.get(key), f"resume_condition_changed: {key}"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assets_digest(kind, rows, dataset_root):
    """JSONL 밖의 사진·음성 파일까지 묶는다. 기록된 해시와 다르면 실행을 멈춘다."""
    index = dataset_root / "photo/manifest.json"
    known = json.loads(index.read_text()) if kind == "photo" and index.exists() else {}
    h = hashlib.sha256()
    for row in sorted(rows, key=lambda r: r["metadata"]["id"]):
        for _, relative in sorted((row.get("attachments") or {}).items()):
            # 모든 첨부가 실행 전에 이 함수를 지나므로 경로 검사는 여기서 한 번만 한다.
            path = (dataset_root / relative).resolve()
            assert path.is_relative_to(dataset_root.resolve()), (
                f"attachment_outside_dataset: {relative}"
            )
            found = digest(path)
            expected = known.get(relative) or row["metadata"].get("sha256")
            assert expected in (None, found), f"attachment_changed: {relative}"
            h.update(f"{relative}:{found}\n".encode())
    return h.hexdigest()


def evaluator_digest():
    """실행기와 채점 코드를 함께 묶는다. 채점 코드만 바뀌어도 값이 달라진다."""
    h = hashlib.sha256()
    here = Path(__file__).resolve()
    for path in sorted(here.parent.rglob("*.py")):
        h.update(path.read_bytes())
    return h.hexdigest()


def db_fingerprint():
    from db.models import SessionLocal
    from sqlalchemy import text

    with SessionLocal() as session:
        row = session.execute(
            text("""select count(*) as tracks, count(emb_gemini) as embedded,
          md5(string_agg(md5(concat_ws('|',track_id,title,artist,genre,release_date,preview_url,store_url,mood_tags::text,emb_gemini::text,emb_v4::text)),'' order by track_id)) as sha,
          md5(string_agg(md5(concat_ws('|',track_id,title,artist,genre,release_date,preview_url,store_url,mood_tags::text)),'' order by track_id)) as catalog_sha,
          md5(string_agg(md5(concat_ws('|',track_id,emb_gemini::text,emb_v4::text)),'' order by track_id)) as embedding_sha
          from tracks""")
        ).one()
        return dict(row._mapping)


def main():
    from dotenv import load_dotenv

    p = argparse.ArgumentParser()
    p.add_argument("--env-file", type=Path, required=True)
    p.add_argument("--service-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--kinds", default="text,photo,dialogue,stt,faults")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument(
        "--no-judge", action="store_true", help="LLM 채점 없이 서비스 실행과 코드 검사만 한다"
    )
    p.add_argument("--dataset-dir", type=Path, required=True, help="평가 데이터 폴더")
    p.add_argument("--dataset-version", required=True)
    p.add_argument("--split", help="validation, test, smoke처럼 이 split의 사례만 실행한다")
    p.add_argument(
        "--resume",
        action="store_true",
        help="같은 조건의 출력 폴더에서 끝나지 않은 사례만 이어서 실행한다",
    )
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    load_dotenv(args.env_file)
    assert args.execute, "--execute required"
    args.dataset_dir = args.dataset_dir.resolve()
    assert cer("가 나!", "가나") == 0 and cer("가나", "가다") == 0.5
    args.service_root = args.service_root.resolve()
    assert not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=args.service_root, text=True
    ).strip(), "service checkout must be clean"
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.service_root, text=True
    ).strip()
    for key in ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2"):
        os.environ[key] = "false"
    for key in ("LANGSMITH_AGENT_ID", "LANGSMITH_AGENT_ENVIRONMENT"):
        os.environ.pop(key, None)
    sys.path.insert(0, str(args.service_root))
    import httpx

    client = EvaluationClient()
    with httpx.Client(timeout=20) as http:
        response = http.get(
            "https://api.smith.langchain.com/api/v1/agents/meomuneum",
            headers={"x-api-key": os.environ["LANGSMITH_API_KEY"]},
        )
        response.raise_for_status()
        agent_tag = response.json()["tag_value_id"]
    args.judge_spent_usd = Decimal(0)
    args.judge_budget_usd = Decimal(0)
    args.judge_reservation_usd = Decimal(0)
    if not args.no_judge:
        with httpx.Client(timeout=30) as http:
            response = http.get(
                "https://openrouter.ai/api/v1/key",
                headers={"Authorization": "Bearer " + os.environ["OPENROUTER_EVAL_API_KEY"]},
            )
            response.raise_for_status()
            key = response.json()["data"]
            assert key["limit"] == 20, "approved key limit changed"
            response = http.get("https://openrouter.ai/api/v1/models")
            response.raise_for_status()
            model = next(m for m in response.json()["data"] if m["id"] == JUDGE_MODEL)
            assert Decimal(model["pricing"]["prompt"]) <= Decimal(".000002") and Decimal(
                model["pricing"]["completion"]
            ) <= Decimal(".00001")
            assert (
                Decimal(model["pricing"].get("request", "0")) == 0
                and Decimal(model["pricing"].get("image", "0")) == 0
            )
        args.judge_budget_usd = Decimal(str(key["limit_remaining"]))
        args.judge_reservation_usd = Decimal(model["context_length"]) * Decimal(
            ".000002"
        ) + Decimal(JUDGE_SETTINGS["max_tokens"]) * Decimal(".00001")
    kinds = args.kinds.split(",")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    check_output(args.output, args.resume)
    if not args.resume:
        (args.output / "evaluator.py").write_bytes(Path(__file__).read_bytes())
        (args.output / "judge-prompt.txt").write_text(PROMPT)
    assert not [
        m
        for n, m in sys.modules.items()
        if n.startswith(("backend", "db."))
        and getattr(m, "__file__", None)
        and not Path(m.__file__).is_relative_to(args.service_root)
    ], "wrong service imported"
    metadata = {
        "service_commit": commit,
        "revision_id": commit,
        "service_clean": True,
        "environment": "local",
        "metric_version": VERSION,
        "rubric_version": RUBRIC_VERSION,
        "judge_model": JUDGE_MODEL,
        "judge_settings": JUDGE_SETTINGS,
        "judge_prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "eval_sha256": digest(Path(__file__)),
        "eval_package_sha256": evaluator_digest(),
        "mode": "normal",
        "split": args.split,
        "no_judge": args.no_judge,
        "dataset_version": args.dataset_version,
        "dataset_sha256": {kind: digest(args.dataset_dir / f"{kind}.jsonl") for kind in kinds},
        "assets_sha256": {
            kind: assets_digest(kind, read(kind, args.dataset_dir), args.dataset_dir)
            for kind in kinds
        },
        "models": {
            k: os.getenv(k)
            for k in (
                "LLM_MODEL",
                "EMBEDDING_MODEL",
                "PHOTO_VLM_MODEL",
                "PHOTO_FALLBACK_MODEL",
                "STT_MODEL",
            )
        },
        "search_embedding": search_embedding(),
        "agentic_chat": agentic_chat(),
        "scope": "in_process_router_not_network_or_FE",
        "concurrency": args.concurrency,
        "db_before": db_fingerprint(),
        "planned": {
            kind: len(select_split(read(kind, args.dataset_dir), args.split)) for kind in kinds
        },
        "judge_budget_usd": float(args.judge_budget_usd),
        "lossless_png_transport_ids": ["P-035"],
        "service_source": str(args.service_root),
        "started": datetime.now(timezone.utc).isoformat(),
        "experiments": {},
    }
    assert all(metadata["planned"].values()), "empty split"
    check_catalog(metadata["db_before"]["catalog_sha"], args.dataset_dir)
    if args.resume:
        stored = json.loads(manifest_path.read_text())
        assert_resumable(stored, metadata)
        metadata = {
            k: v
            for k, v in stored.items()
            if k not in ("finished", "status", "db_after", "db_unchanged", "judge_spent_usd")
        }
        metadata.update(
            judge_budget_usd=float(args.judge_budget_usd),
            no_judge=args.no_judge,
            resumed=[*stored.get("resumed", []), datetime.now(timezone.utc).isoformat()],
        )
    (args.output / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    from .execution.telemetry import Capture

    with Capture() as capture:
        for kind in kinds:
            metadata["experiments"][kind] = asyncio.run(
                run_kind(
                    kind,
                    read(kind, args.dataset_dir),
                    client,
                    agent_tag,
                    args,
                    dict(metadata),
                    capture,
                )
            )
            (args.output / "manifest.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2)
            )
    # 재개마다 호출 기록을 따로 남겨 이전 시도의 사용량을 덮어쓰지 않는다.
    capture.save(
        args.output
        / (
            "telemetry.json"
            if not args.resume
            else f"telemetry.resume-{len(metadata['resumed'])}.json"
        )
    )
    ledger = judge_ledger(args.output)
    metadata.update(
        judge_spent_usd=float(ledger["spent"]),
        judge_cost_missing=ledger["missing"],
        db_after=db_fingerprint(),
        finished=datetime.now(timezone.utc).isoformat(),
    )
    metadata["db_unchanged"] = metadata["db_before"] == metadata["db_after"]
    metadata["status"] = (
        "complete"
        if all(metadata["experiments"][kind]["status"] == "complete" for kind in kinds)
        else "incomplete"
    )
    assert metadata["db_unchanged"], "database_changed_during_evaluation"
    (args.output / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps(metadata["experiments"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
