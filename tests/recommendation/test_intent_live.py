"""실제 모델 비교용. RUN_LIVE_MODEL_TESTS=1일 때만 유료 호출한다."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
from dotenv import dotenv_values
from openai import OpenAI

from backend.providers.openrouter import OPENROUTER_BASE_URL
from backend.recommendation import intent


@pytest.mark.skipif(os.getenv("RUN_LIVE_MODEL_TESTS") != "1", reason="실제 모델 평가는 명시적으로 실행")
def test_intent_against_fixed_cases(monkeypatch, tmp_path):
    settings = dotenv_values(".env") | dict(os.environ)
    model = settings["LLM_MODEL"]
    monkeypatch.setenv("LLM_MODEL", model)
    genres = ["Jazz", "K-Pop", "Pop", "R&B/Soul", "Rock"]
    monkeypatch.setattr(intent, "known_genres", lambda: genres)
    cases_path = Path(__file__).with_name("intent_cases.json")
    cases = json.loads(cases_path.read_text())
    output = Path(os.getenv("INTENT_EVAL_OUTPUT", str(tmp_path / "intent-eval.json")))
    report = {
        "model": model,
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "genres": genres,
        "prompt_sha256": hashlib.sha256(intent.classify_instructions(genres).encode()).hexdigest(),
        "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "results": [],
    }
    # 재실행으로 기준선 파일을 덮어쓰지 않는다.
    with output.open("x") as saved, OpenAI(
        api_key=settings["OPENROUTER_LLM_API_KEY"], base_url=OPENROUTER_BASE_URL,
        max_retries=0, timeout=30,
    ) as client:
        try:
            for case in cases:
                actual = intent.classify(client, case["conversation"])
                mismatches = {
                    key: {"expected": value, "actual": actual.get(key)}
                    for key, value in case["expected"].items() if actual.get(key) != value
                }
                for key in case.get("non_null", []):
                    if not actual.get(key):
                        mismatches[key] = {"expected": "non-empty", "actual": actual.get(key)}
                report["results"].append({"id": case["id"], "actual": actual, "mismatches": mismatches})
        finally:
            json.dump(report, saved, ensure_ascii=False, indent=2)
    failures = {row["id"]: row["mismatches"] for row in report["results"] if row["mismatches"]}
    print(f"intent evaluation: {len(cases) - len(failures)}/{len(cases)}, report={output}")
    assert not failures, failures
