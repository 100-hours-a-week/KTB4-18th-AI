"""실행 결과 파일과 완료 판정. 저장 형식과 채점 대상 규칙을 유지한다."""

import json
from decimal import Decimal
from pathlib import Path


def read_attempts(path):
    """시도 기록 전체를 돌려준다. 강제 종료로 깨진 줄은 실행하지 않은 것으로 본다."""
    attempts = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                attempts.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return attempts


def read_saved(path):
    """사례별 마지막 기록을 돌려준다."""
    return {case["id"]: case for case in read_attempts(path)}


def check_output(output, resume):
    """한 폴더에는 한 실행만 둔다. 다른 기능을 이어 돌리면 manifest가 덮어써져 커밋이 섞인다."""
    started = (output / "manifest.json").exists()
    assert not resume or started, "nothing to resume"
    assert resume or not started and not any(output.glob("*.jsonl")), (
        "choose fresh output or --resume"
    )


def append_case(path, case):
    if path.exists() and not path.read_bytes().endswith(b"\n"):
        with path.open("a") as out:
            out.write("\n")
    with path.open("a") as out:
        out.write(json.dumps(case, ensure_ascii=False, default=str) + "\n")


def service_ok(kind, case):
    if kind in ("stt", "faults"):
        return bool(case.get("success"))
    turns = case.get("turns", [])
    return (
        bool(turns)
        and len(turns) == case.get("expected_turns")
        and all(t["actual"].get("done") and not t["actual"].get("error") for t in turns)
    )


def finished(kind, case, judged):
    return service_ok(kind, case) and (
        not judged
        or all(
            t.get("initial_judge", {}).get("status") == "ok"
            for t in case["turns"]
            if needs_judge(t)
        )
    )


def needs_judge(turn):
    """곡 카드가 없는 턴은 고정 문구이므로 LLM 채점 대상이 아니다."""
    actual = turn["actual"]
    return bool(actual.get("done") and not actual.get("error") and actual.get("tracks"))


def judge_ledger(folder):
    """채점 호출마다 남긴 기록을 더한다. 실패한 호출과 재개 전 시도의 비용도 들어간다."""
    costs = [
        c.get("cost_usd")
        for path in Path(folder).glob("*.judge.jsonl")
        for c in read_attempts(path)
    ]
    return {
        "spent": sum((Decimal(str(c)) for c in costs if c is not None), Decimal(0)),
        "missing": sum(c is None for c in costs),
    }
