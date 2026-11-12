"""사례 단위 지표. 해당 없음·판정 불가는 0점으로 발행하지 않는다."""

from collections import Counter
import math
import unicodedata

from ..results import needs_judge

# 턴별 코드 채점 키를 대시보드 열 단위로 묶는다. 키 이름은 실험 간 비교를 위해 바꾸지 않는다.
GROUPS = {
    "route_acc": ("routes",),
    "slot_acc": (
        "min_year",
        "max_year",
        "requested_count",
        "has_non_music_request",
        "genres_allowed",
        "response_style_present",
    ),
    "lookup_acc": (
        "lookup_track_ids_any",
        "lookup_track_ids_if_found",
        "lookup_artist",
        "cards_title_contains",
    ),
    "cards_ok": ("cards",),
    "exclusion_ok": ("previous_tracks_excluded",),
    "format_ok": ("plain_text",),
    "claim_ok": ("genre_claim",),
    "constraint_ok": ("returned_min_year", "returned_max_year", "returned_genres_allowed"),
    "photo_verdict_acc": ("usable", "lighting_allowed", "reject_reason_allowed"),
    "verdict_in_deadline": ("verdict_deadline_s",),
}
LIST_KEYS = ("unique_track_ratio", "artist_diversity")
CONTRACT_KEYS = {n for key, names in GROUPS.items() if key != "verdict_in_deadline" for n in names}
LATENCY_KEYS = ("first_text_ms", "cards_ms", "total_ms", "verdict_ms")


def mean(values):
    return sum(values) / len(values) if values else None


def case_metrics(case, service_cost=None):
    """run_case 결과(+judge)를 고정 키 {key: score}로 줄인다. 값이 None인 키는 발행하지 않는다."""
    turns = case.get("turns", [])
    ok = (
        bool(turns)
        and len(turns) == case["expected_turns"]
        and all(t["actual"].get("done") and not t["actual"].get("error") for t in turns)
    )
    out = {
        "service_success": mean([t["scores"]["service_success"]["score"] for t in turns])
        if turns
        else 0,
        "case_completion": int(ok),
    }
    scored = lambda names: [
        s["score"]
        for t in turns
        for n, s in t["scores"].items()
        if n in names and s["status"] == "scored"
    ]
    for key, names in GROUPS.items():
        out[key] = mean(scored(names))
    contract = scored(CONTRACT_KEYS)
    incomplete = any(
        s["status"] in ("insufficient_evidence", "execution_error")
        for t in turns
        for n, s in t["scores"].items()
        if n in CONTRACT_KEYS
    )
    out["contract_pass"] = (
        0 if not ok or any(v == 0 for v in contract) else None if incomplete or not contract else 1
    )
    for key in LIST_KEYS:
        out[key] = mean(scored({key}))
    for key in LATENCY_KEYS:
        out[key] = mean([t["actual"][key] for t in turns if t["actual"].get(key) is not None])
    out["service_cost_usd"] = service_cost
    return {k: v for k, v in out.items() if v is not None}


def percentile(values, q):
    values = sorted(values)
    return values[math.ceil(len(values) * q) - 1] if values else None


def top_track_case_share(cases):
    """가장 자주 나온 곡이 몇 개 사례에 나왔는지 본다. 사례 안 중복은 제외한다."""
    with_cards = [c for c in cases if any(t["actual"].get("tracks") for t in c.get("turns", []))]
    counts = Counter(
        tid
        for c in with_cards
        for tid in {x["track_id"] for t in c["turns"] for x in t["actual"].get("tracks", [])}
    )
    top = counts.most_common(1)
    return top[0][1] / len(with_cards) if top else None


def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def cer(reference, actual):
    a, b = normalize(reference), normalize(actual)
    assert a
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1] / len(a)


def feedback(kind, case):
    if kind in ("faults", "stt"):
        values = {
            "장애 처리 통과율" if kind == "faults" else "전사 성공률": int(case["success"]),
            "전체 처리 시간(s)": case["total_ms"] / 1000,
        }
        if kind == "stt" and case["success"]:
            values["문자 오류율(CER)"] = cer(case["reference"], case["transcript"])
    else:
        raw = case_metrics(case, case.get("service_cost_usd"))
        names = {
            "case_completion": "대화 완료율" if kind == "dialogue" else "요청 완료율",
            "route_acc": "의도 분류 정확도",
            "slot_acc": "조건 추출 정확도",
            "constraint_ok": "명시 조건 충족률",
            "cards_ok": "요청 곡 수 준수율",
            "exclusion_ok": "이전 곡 제외율",
            "contract_pass": "기능 요구 충족률",
        }
        values = {name: raw[key] for key, name in names.items() if key in raw}
        turns = case["turns"]
        if kind == "photo":
            for key in ("의도 분류 정확도", "조건 추출 정확도"):
                values.pop(key, None)
            values["이미지 입력 판정 정확도"] = (
                sum(t["scores"].get("usable", {}).get("score", 0) == 1 for t in turns)
                / case["expected_turns"]
            )
            values["구조화 응답 성공률"] = (
                sum(bool(t["actual"].get("scene")) for t in turns) / case["expected_turns"]
            )
        for source, label in (
            ("first_text_ms", "첫 안내 대기 시간(s)"),
            ("total_ms", "전체 처리 시간(s)"),
        ):
            numbers = [
                t["actual"][source] / 1000 for t in turns if t["actual"].get(source) is not None
            ]
            if numbers:
                values[label] = sum(numbers) / len(numbers)
        cards = [
            t["actual"]["cards_ms"] / 1000
            for t in turns
            if t["actual"].get("tracks")
            and t["actual"].get("done")
            and not t["actual"].get("error")
        ]
        if cards:
            values["추천 카드 대기 시간(s)"] = sum(cards) / len(cards)
        rejected = [
            t["actual"]["total_ms"] / 1000
            for t in turns
            if t["expected"].get("usable") is False
            and t["scores"].get("usable", {}).get("score") == 1
            and t["actual"].get("done")
            and not t["actual"].get("error")
        ]
        if rejected:
            values["정상 거절 처리 시간(s)"] = sum(rejected) / len(rejected)
        judges = [
            t["initial_judge"] for t in turns if t.get("initial_judge", {}).get("status") == "ok"
        ]
        for key, label in (
            ("response_compliance", "응답 요구 준수율"),
            ("faithfulness", "안내 근거 충실도"),
            ("explanation", "안내 적절성 점수"),
            ("scene_accuracy", "장면 해석 점수"),
            ("scene_consistency", "검색 문장·설명 일관성 점수"),
        ):
            numbers = [j[key] for j in judges if j.get(key) is not None]
            if numbers:
                values[label] = sum(numbers) / len(numbers)
        applicable = sum(needs_judge(t) for t in turns)
        if applicable and any("initial_judge" in t for t in turns):
            values["LLM 채점 완료율"] = len(judges) / applicable
    if case.get("service_cost_usd") is not None:
        values["서비스 비용(USD)"] = case["service_cost_usd"]
    if case.get("judge_cost_usd") is not None:
        values["평가 비용(USD)"] = case["judge_cost_usd"]
    return values
