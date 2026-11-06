"""두 평가 결과 폴더를 같은 코드 규칙으로 다시 채점해 사람이 볼 지표만 비교한다. 모델을 호출하지 않는다."""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from .results import needs_judge, read_attempts, read_saved, service_ok
from .grading.metrics import CONTRACT_KEYS, GROUPS, case_metrics, percentile, top_track_case_share
from .grading.scoring import score_case

KINDS = ("text", "photo", "dialogue")
ALL_KINDS = (*KINDS, "stt", "faults")
LABELS = {
    "text": "텍스트",
    "photo": "사진",
    "dialogue": "멀티턴",
    "stt": "음성",
    "faults": "입력·장애",
}
ALERTS = (
    "요청 완료율",
    "대화 완료율",
    "전사 성공률",
    "구조화 응답 성공률",
    "LLM 채점 완료율",
    "장애 처리 통과율",
)
CER = "문자 오류율(CER)"
CER_ALERT = 0.02
# 측정 기준이 바뀌면 점수의 뜻이 달라진다. 이 값이 다르면 새 평가 시리즈로 다시 측정한다.
MEASUREMENT = (
    "mode",
    "split",
    "dataset_version",
    "dataset_sha256",
    "assets_sha256",
    "metric_version",
    "rubric_version",
    "judge_model",
    "judge_settings",
    "judge_prompt_sha256",
    "eval_package_sha256",
    "lossless_png_transport_ids",
    "concurrency",
    "scope",
    "catalog",
)
# 개선 실험에서 일부러 바꿀 수 있는 서비스 조건이다.
SERVICE_VARIABLES = ("models", "search_embedding", "embedding", "agentic_chat")
CONDITIONS = (*MEASUREMENT, *SERVICE_VARIABLES)
# 전체 사례를 돌린 실행은 split이 없다.
OPTIONAL = ("split",)
CONTRACT_NAME = "기능 요구 충족률"
WAITS = (("first_text_ms", "첫 안내 대기"), ("cards_ms", "추천 카드 대기"))


class NotComparable(Exception):
    """의도하지 않은 조건 차이나 미완료 실행이 있어 수치를 비교할 수 없다."""


def names(ids, hide):
    return f"{len(ids)}건" if hide else " ".join(ids)


def contract(case):
    return case_metrics(case).get("contract_pass")


def load(folder):
    runs = {}
    for kind in ALL_KINDS:
        path = Path(folder) / f"{kind}.jsonl"
        if path.exists():
            runs[kind] = list(read_saved(path).values())
    for kind in KINDS:
        for case in runs.get(kind, []):
            case["saved_contract"] = case.get("metrics", {}).get(CONTRACT_NAME)
            score_case(case)
    return runs


def manifest(folder):
    path = Path(folder) / "manifest.json"
    if not path.exists():
        raise NotComparable(f"manifest 없음: {folder}")
    return json.loads(path.read_text())


def run_mode(record, runs):
    """manifest가 아니라 실제 턴 기록을 우선한다. 옛 실행에는 mode 기록이 없다."""
    fast = any(
        (t["actual"].get("state") or {}).get("fast_mode")
        for run in runs.values()
        for c in run
        for t in c.get("turns", [])
    )
    return "fast" if fast else record.get("mode")


def facts(record, kinds):
    out = {key: record.get(key) for key in CONDITIONS}
    for key in ("dataset_sha256", "assets_sha256"):
        if out[key]:
            # 고른 기능의 해시가 하나라도 없으면 대조할 수 없다.
            out[key] = {k: out[key][k] for k in kinds} if kinds <= out[key].keys() else None
    db = record.get("db_before") or {}
    # 예전 실행은 곡 정보와 벡터를 한 지문으로 기록했다.
    out["catalog"], out["embedding"] = (
        (db.get("catalog_sha"), db.get("embedding_sha"))
        if "catalog_sha" in db
        else (db.get("sha"), db.get("sha"))
    )
    return out


def conditions(a, b, kinds):
    a, b = facts(a, kinds), facts(b, kinds)
    differ = {key: [a[key], b[key]] for key in CONDITIONS if a[key] != b[key]}
    return differ, [
        key for key in CONDITIONS if a[key] is None and b[key] is None and key not in OPTIONAL
    ]


def changed(key, pair):
    """양쪽 값이 모두 있으면 실제 변경이다. 한쪽 기록이 없으면 대조할 수 없을 뿐이다."""
    return None not in pair or (key == "mode" and "fast" in pair)


def coverage(kind, run, planned):
    turns = [t for case in run for t in case.get("turns", [])]
    targets = [t for t in turns if needs_judge(t)]
    return {
        "planned": len(run) if planned is None else planned,
        "run": len(run),
        "service_ok": sum(service_ok(kind, case) for case in run),
        "judged": sum(t.get("initial_judge", {}).get("status") == "ok" for t in targets),
        "judge_total": len(targets),
    }


def preflight(label, record):
    """실행이 끝까지 완료됐고 실행 중 DB가 그대로였는지 본다."""
    reasons = []
    if not record.get("finished"):
        reasons.append(f"{label}: 끝나지 않은 실행")
    if record.get("db_unchanged") is not True:
        reasons.append(f"{label}: 실행 전후 DB 동일 기록 없음")
    if record.get("status") == "incomplete":
        reasons.append(f"{label}: 미완료로 기록된 실행")
    return reasons


def incomplete_reasons(label, row, missing, no_judge, hide):
    reasons = []
    if row["run"] < row["planned"]:
        reasons.append(f"{label}: 예정 {row['planned']}건 중 {row['run']}건 실행")
    if row["service_ok"] < row["run"]:
        reasons.append(f"{label}: 서비스 성공 {row['service_ok']}/{row['run']}")
    # 옛 실행은 채점 여부 기록이 없어 일부만 채점된 경우만 잡는다.
    judge_missing = (
        row["judged"] < row["judge_total"]
        if no_judge is False
        else 0 < row["judged"] < row["judge_total"] and no_judge is None
    )
    if judge_missing:
        reasons.append(f"{label}: 채점 완료 {row['judged']}/{row['judge_total']}")
    if missing:
        reasons.append(f"{label}: 다른 쪽에 없는 사례 {names(sorted(missing), hide)}")
    return reasons


def shared(run, ids):
    return [case for case in run if case["id"] in ids]


def functional(base, new):
    a, b = ({c["id"]: contract(c) for c in run} for run in (base, new))
    ids = sorted(a.keys() & b.keys())
    return {
        "base": sum(a[i] == 1 for i in ids),
        "new": sum(b[i] == 1 for i in ids),
        "total": len(ids),
        "improved": [i for i in ids if a[i] != 1 and b[i] == 1],
        "regressed": [i for i in ids if a[i] == 1 and b[i] != 1],
    }


def rescored(run):
    """저장 당시 판정과 현재 규칙의 판정이 다른 사례다. 서비스 변화가 아니라 규칙 변화다."""
    return [
        c["id"]
        for c in run
        if c.get("saved_contract") is not None and c["saved_contract"] != contract(c)
    ]


def denominators(base, new):
    """지표마다 실제로 검사한 수를 센다. 텍스트 30건이어도 곡 수 검사는 3건일 수 있다."""

    def count(run, names):
        scores = [
            s["score"]
            for c in run
            for t in c["turns"]
            for n, s in t["scores"].items()
            if n in names and s["status"] == "scored"
        ]
        return sum(scores), len(scores)

    rows = {
        key: {"base": count(base, names), "new": count(new, names)} for key, names in GROUPS.items()
    }
    return {key: row for key, row in rows.items() if row["base"][1] or row["new"][1]}


def judged(run, key):
    """곡 카드가 있는 턴의 LLM 판정만 모은다. 고정 문구는 채점 대상이 아니다."""
    out = {}
    for case in run:
        for index, turn in enumerate(case["turns"]):
            verdict = turn.get("initial_judge") or {}
            if (
                turn["actual"].get("tracks")
                and verdict.get("status") == "ok"
                and verdict.get(key) is not None
            ):
                out[(case["id"], index)] = verdict[key]
    return out


def faithfulness(base, new):
    a, b = judged(base, "faithfulness"), judged(new, "faithfulness")
    keys = sorted(a.keys() & b.keys())
    return {
        "base": sum(a[k] == 1 for k in keys),
        "new": sum(b[k] == 1 for k in keys),
        "total": len(keys),
        "changed": [k[0] for k in keys if a[k] != b[k]],
    }


def scene(base, new):
    a, b = judged(base, "scene_accuracy"), judged(new, "scene_accuracy")
    keys = sorted(a.keys() & b.keys())
    if not keys:
        return None
    return {
        "base": statistics.mean(a[k] for k in keys),
        "new": statistics.mean(b[k] for k in keys),
        "total": len(keys),
        "changed": [k[0] for k in keys if a[k] != b[k]],
    }


def recommended(runs):
    return [case for kind in KINDS for case in runs.get(kind, [])]


def song_repeat(runs):
    return top_track_case_share(recommended(runs))


def waits(base, new, key):
    """두 실행 모두 값이 있는 같은 턴만 비교한다. 느린 요청이 실패로 빠져 빨라 보이는 착시를 막는다."""

    def values(run):
        return {
            (c["id"], i): t["actual"][key] / 1000
            for c in run
            for i, t in enumerate(c["turns"])
            if needs_judge(t) and t["actual"].get(key) is not None
        }

    def summary(found, keys):
        numbers = [found[k] for k in keys]
        return {"median": statistics.median(numbers), "p90": percentile(numbers, 0.9)}

    a, b = values(base), values(new)
    keys = a.keys() & b.keys()
    if not keys:
        return None
    return {
        "n": len(keys),
        "excluded": len(a.keys() ^ b.keys()),
        "base": summary(a, keys),
        "new": summary(b, keys),
    }


def attempts(folder, kind):
    """재시도까지 포함한 시도 기록이다. 최종 결과만 보면 실패한 호출의 비용이 사라진다."""
    records = read_attempts(Path(folder) / f"{kind}.service.jsonl")
    if not records:
        return None
    counts = Counter(c["id"] for c in records)
    costs = [c.get("service_cost_usd") for c in records]
    return {
        "attempts": len(records),
        "retried": sorted(i for i, n in counts.items() if n > 1),
        "failed": sum(not service_ok(kind, c) for c in records),
        "cost": sum(c for c in costs if c is not None),
        "missing": sum(c is None for c in costs),
    }


def request_cost(run):
    costs = [c.get("service_cost_usd") for c in run]
    requests = sum(len(c.get("turns") or [c]) for c in run)
    missing = sum(cost is None for cost in costs)
    return {
        "per_request": None if missing or not requests else sum(costs) / requests,
        "requests": requests,
        "missing": missing,
    }


def service_cost(folder, runs):
    """시도 기록이 있으면 재시도까지 더한다. 옛 실행은 최종 결과만 있다."""
    known, missing = 0, 0
    for kind in (k for k in (*KINDS, "stt") if k in runs):
        ledger = attempts(folder, kind)
        if ledger:
            known, missing = known + ledger["cost"], missing + ledger["missing"]
        else:
            costs = [c.get("service_cost_usd") for c in runs.get(kind, [])]
            known, missing = (
                known + sum(c for c in costs if c is not None),
                missing + sum(c is None for c in costs),
            )
    return {"known": known, "missing": missing}


def mean_cer(runs):
    values = [c["metrics"][CER] for c in runs.get("stt", []) if CER in c.get("metrics", {})]
    return statistics.mean(values) if values else None


def warnings(new_folder, new, base, hide):
    found = []
    for name in ALERTS:
        below = [
            c["id"] for run in new.values() for c in run if c.get("metrics", {}).get(name, 1) < 1
        ]
        if below:
            found.append(f"{name} 미달: {names(below, hide)}")
    a, b = mean_cer(base), mean_cer(new)
    if a is not None and b is not None and b - a > CER_ALERT:
        found.append(f"문자 오류율 상승: {a:.2%} → {b:.2%}")
    missing = service_cost(new_folder, new)["missing"]
    if missing:
        found.append(f"서비스 비용 누락 {missing}건. 합계는 최솟값이다")
    return found


def failures(new):
    """실패한 사례만 어떤 세부 검사에서 걸렸는지 보여 준다."""
    found = {}
    for case in recommended(new):
        names = [
            name
            for turn in case["turns"]
            for name, score in turn["scores"].items()
            if name in CONTRACT_KEYS and score["status"] == "scored" and score["score"] == 0
        ]
        if names:
            found[case["id"]] = list(dict.fromkeys(names))
    return found


def hide_cases(report):
    """test는 점수만 본다. 어떤 사례가 틀렸는지 알면 test가 개선 작업에 새어 나간다."""
    count = lambda ids: [f"{len(ids)}건"] if ids else []
    for row in (
        *report["functional"].values(),
        *report["faithfulness"].values(),
        report["scene"] or {},
    ):
        for key in {"improved", "regressed", "changed"} & row.keys():
            row[key] = count(row[key])
    for sides in report["rescored"].values():
        for side, ids in sides.items():
            sides[side] = count(ids)
    for sides in report["attempts"].values():
        for row in filter(None, sides.values()):
            row["retried"] = count(row["retried"])
    if report["failures"]:
        report["failures"] = {f"{len(report['failures'])}건": ["내용 숨김"]}
    return report


def compare(base_folder, new_folder, vary=(), allow_incomplete=False, kinds=None):
    not_allowed = set(vary) - set(SERVICE_VARIABLES)
    if not_allowed:
        raise ValueError(
            f"--vary로 바꿀 수 없는 조건: {', '.join(sorted(not_allowed))}. 측정 기준이 바뀌면 새 시리즈로 측정한다."
        )
    wrong = set(kinds or ()) - set(ALL_KINDS)
    if wrong:
        raise ValueError(f"--kinds에 없는 기능: {', '.join(sorted(wrong))}")
    a, b = manifest(base_folder), manifest(new_folder)
    hide = any(
        r.get("split") == "test" or str(r.get("dataset_version")).startswith("test-")
        for r in (a, b)
    )
    base, new = load(base_folder), load(new_folder)
    chosen = [k for k in ALL_KINDS if k in (kinds or {*base, *new})]
    reasons = [*preflight("기준", a), *preflight("비교", b)]
    reasons += [f"{LABELS[k]}: 한쪽 실행에 없음" for k in chosen if k not in base or k not in new]
    both = [k for k in chosen if k in base and k in new]
    differ, unknown = conditions(
        dict(a, mode=run_mode(a, base)), dict(b, mode=run_mode(b, new)), set(both)
    )
    blocked = {key: pair for key, pair in differ.items() if key not in vary and changed(key, pair)}
    if blocked:
        raise NotComparable(
            f"조건 차이: {json.dumps(blocked, ensure_ascii=False)}. 서비스 변수만 --vary로 밝힐 수 있다."
        )
    reasons += [
        f"대조 불가 조건: {key}"
        for key, pair in differ.items()
        if key not in vary and not changed(key, pair)
    ]
    reasons += [f"대조 불가 조건: {key}" for key in unknown]
    # 고른 기능만 남긴다. 제외한 기능의 비용과 실패가 합계에 섞이면 안 된다.
    base, new = {k: base[k] for k in both}, {k: new[k] for k in both}
    report_coverage, common, ledgers = {}, {}, {}
    for kind in both:
        ids_a, ids_b = {c["id"] for c in base[kind]}, {c["id"] for c in new[kind]}
        common[kind] = ids_a & ids_b
        rows = {
            "base": coverage(kind, base[kind], (a.get("planned") or {}).get(kind)),
            "new": coverage(kind, new[kind], (b.get("planned") or {}).get(kind)),
        }
        report_coverage[kind] = rows
        ledgers[kind] = {"base": attempts(base_folder, kind), "new": attempts(new_folder, kind)}
        reasons += incomplete_reasons(
            f"기준 {LABELS[kind]}", rows["base"], ids_a - ids_b, a.get("no_judge"), hide
        )
        reasons += incomplete_reasons(
            f"비교 {LABELS[kind]}", rows["new"], ids_b - ids_a, b.get("no_judge"), hide
        )
    if reasons and not allow_incomplete:
        raise NotComparable("미완료 또는 대조 불가: " + "; ".join(reasons))
    kinds = [k for k in both if k in KINDS]
    base_cases = {k: shared(base[k], common[k]) for k in both}
    new_cases = {k: shared(new[k], common[k]) for k in both}
    cost_kinds = [k for k in both if k != "faults"]
    # 코드 검사만 한 쪽이 있으면 의미 채점 값이 없다. 기능·시간·비용만 비교한다.
    quality = not (a.get("no_judge") or b.get("no_judge"))
    report = {
        "conditions": differ,
        "quality_skipped": not quality,
        "incomplete": reasons,
        "coverage": report_coverage,
        "attempts": ledgers,
        "functional": {k: functional(base_cases[k], new_cases[k]) for k in kinds},
        "denominators": {k: denominators(base_cases[k], new_cases[k]) for k in kinds},
        "rescored": {
            k: {"base": rescored(base_cases[k]), "new": rescored(new_cases[k])} for k in kinds
        },
        "faithfulness": {k: faithfulness(base_cases[k], new_cases[k]) for k in kinds}
        if quality
        else {},
        "scene": scene(base_cases.get("photo", []), new_cases.get("photo", []))
        if quality
        else None,
        "latency": {
            k: {key: waits(base_cases[k], new_cases[k], key) for key, _ in WAITS} for k in kinds
        },
        "request_cost": {
            k: {"base": request_cost(base_cases[k]), "new": request_cost(new_cases[k])}
            for k in cost_kinds
        },
        "song_repeat": {"base": song_repeat(base_cases), "new": song_repeat(new_cases)},
        "service_cost": {
            "base": service_cost(base_folder, base),
            "new": service_cost(new_folder, new),
        },
        "warnings": warnings(new_folder, new, base, hide),
        "failures": failures(new),
    }
    return hide_cases(report) if hide else report


if __name__ == "__main__":
    from .report import render

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument(
        "--vary",
        default="",
        help=f"이번 실험에서 일부러 바꾼 서비스 조건: {', '.join(SERVICE_VARIABLES)}",
    )
    parser.add_argument(
        "--kinds",
        default="",
        help=f"비교할 기능만 고른다: {', '.join(ALL_KINDS)}. 없으면 양쪽 전체를 요구한다",
    )
    parser.add_argument(
        "--allow-incomplete", action="store_true", help="완료된 공통 사례만 참고용으로 비교한다"
    )
    args = parser.parse_args()
    try:
        print(
            render(
                compare(
                    args.base,
                    args.new,
                    vary=tuple(filter(None, args.vary.split(","))),
                    allow_incomplete=args.allow_incomplete,
                    kinds=tuple(filter(None, args.kinds.split(","))),
                )
            )
        )
    except (NotComparable, ValueError) as exc:
        raise SystemExit(f"비교 불가: {exc}")
