"""고정 기대값 채점. 누락 기대값은 미채점이며 null 기대값은 실제 null을 요구한다."""

import re

# 안내문의 장르 표현과 카탈로그 장르 표기를 잇는다. 오탐을 줄이려고 뜻이 분명한 단어만 둔다.
GENRE_WORDS = {
    "클래식": ("classical",),
    "재즈": ("jazz",),
    "힙합": ("hip-hop", "rap"),
    "뉴에이지": ("new age",),
    "메탈": ("metal",),
    "컨트리": ("country",),
}
NOTICE = re.compile(r"없(?!이)|못|부족|아니")
GENRE_CLAIM_MIN_SHARE = (
    0.4  # '재즈나 어쿠스틱'처럼 여러 장르를 말한 안내를 고려해 5곡 중 2곡부터 통과시킨다.
)


def result(status, score=None, reason=""):
    return {"status": status, "score": score, "reason": reason}


def score_turn(expected, actual, *, metrics=(), previous=()):
    scores = {}
    if actual.get("error") or not actual.get("done"):
        return {
            "service_success": result("scored", 0),
            "contract": result("execution_error", reason=actual.get("error", "missing_done")),
        }
    scores["service_success"] = result("scored", 1)
    state, scene = actual.get("state", {}), actual.get("scene", {})
    tracks = actual.get("tracks", [])
    route = actual.get("route")

    def check(key, ok):
        scores[key] = result("scored", int(bool(ok)))

    for key, value in expected.items():
        if key == "routes":
            check(key, route in value)
        elif key in ("min_year", "max_year", "requested_count", "has_non_music_request"):
            field = {"min_year": "recommend_min_year", "max_year": "recommend_max_year"}.get(
                key, key
            )
            check(key, field in state and state[field] == value)
        elif key == "genres_allowed":
            got = state.get("recommend_genres")
            check(
                key,
                "recommend_genres" in state
                and (got is None if value is None else bool(got) and set(got) <= set(value)),
            )
        elif key == "cards":
            check(key, len(tracks) == value)
        elif key == "response_style_present":
            check(key, bool(state.get("response_style")) == value)
        elif key in ("lookup_track_ids_any", "lookup_track_ids_if_found"):
            if key.endswith("if_found") and route != ["lookup", "lookup_found"]:
                scores[key] = result("not_applicable")
            else:
                check(key, bool({str(x) for x in value} & {t["track_id"] for t in tracks}))
        elif key == "lookup_artist":
            check(
                key,
                bool(tracks)
                and all(
                    re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", t["artist"], re.I)
                    for t in tracks
                ),
            )
        elif key == "cards_title_contains":
            check(
                key, bool(tracks) and all(value.casefold() in t["title"].casefold() for t in tracks)
            )
        elif key == "usable":
            check(key, "usable" in scene and scene["usable"] == value)
        elif key in ("lighting_allowed", "reject_reason_allowed"):
            check(
                key,
                scene.get(
                    {"lighting_allowed": "lighting", "reject_reason_allowed": "reject_reason"}[key]
                )
                in value,
            )
        elif key == "verdict_deadline_s":
            check(
                key, actual.get("verdict_ms") is not None and actual["verdict_ms"] <= value * 1000
            )
        elif key not in ("visible_facts", "must_not", "mood_examples", "checks", "followup_turns"):
            raise ValueError(f"Unknown expectation: {key}")
    # 추출 조건뿐 아니라 실제 카드도 DB 정보로 검증한다.
    for key, field, predicate in (
        ("min_year", "release_year", lambda a, b: a >= b),
        ("max_year", "release_year", lambda a, b: a <= b),
        ("genres_allowed", "genre", lambda a, b: a in b),
    ):
        if expected.get(key) is not None:
            name = "returned_" + key
            if not tracks:
                scores[name] = result("not_applicable", reason="no_tracks; see route/cards")
            elif any(t.get(field) is None for t in tracks):
                scores[name] = result("insufficient_evidence", reason="missing_catalog_metadata")
            else:
                check(name, all(predicate(t[field], expected[key]) for t in tracks))
    if tracks:
        ids = [t["track_id"] for t in tracks]
        scores["unique_track_ratio"] = result("scored", len(set(ids)) / len(ids))
        scores["artist_diversity"] = result(
            "scored", len({t["artist"] for t in tracks}) / len(tracks)
        )
    # 곡명의 별표·밑줄은 허용하고 실제 마크다운 구조만 검사한다.
    text = actual.get("text", "")
    markdown = r"(?m)^\s{0,3}(?:#{1,6}\s|[-*+]\s|>\s|[0-9]+[.)]\s)|```|\*\*[^\n]+\*\*|\[[^]\n]+\]\([^)\n]+\)"
    check("plain_text", not re.search(markdown, text))
    claim = genre_claim_ok(text, tracks) if tracks and route and route[0] != "lookup" else None
    if claim is not None:
        check("genre_claim", claim)
    if "exclusion" in metrics and previous:
        exact_lookup = (
            route == ["lookup", "lookup_found"]
            and state.get("lookup_song")
            and state.get("lookup_artist")
        )
        if exact_lookup or not tracks:
            scores["previous_tracks_excluded"] = result(
                "not_applicable", reason="exact_lookup_or_no_tracks"
            )
        else:
            check(
                "previous_tracks_excluded",
                not ({str(t["track_id"]) for t in tracks} & set(previous)),
            )
    return scores


def genre_claim_ok(text, tracks):
    """안내문이 말한 장르가 카드와 맞는지 본다. 말하지 않았거나 못 찾았다고 알렸으면 None이다."""
    claims = [
        names
        for word, names in GENRE_WORDS.items()
        if any(
            re.search(word + "(?!한)", s) and not NOTICE.search(s)
            for s in re.split(r"[.!?\n]", text)
        )
    ]
    if not claims:
        return None
    genres = [(t.get("genre") or "").lower() for t in tracks]
    return all(
        sum(any(n in g for n in names) for g in genres) / len(genres) >= GENRE_CLAIM_MIN_SHARE
        for names in claims
    )


def score_case(case):
    """저장된 실행값만 다시 채점한다. 모델·DB는 호출하지 않는다."""
    previous = set()
    for turn in case["turns"]:
        turn["scores"] = score_turn(
            turn["expected"],
            turn["actual"],
            metrics=case["metadata"].get("metrics", []),
            previous=previous,
        )
        previous.update(str(t["track_id"]) for t in turn["actual"].get("tracks", []))
    return case


def route_of(state):
    intent = state.get("intent")
    if intent == "recommend":
        outcome = (
            "unsupported"
            if state.get("recommend_unsupported_condition")
            else "insufficient"
            if state.get("recommend_has_enough_info") is False
            else "tracks"
            if state.get("tracks")
            else "no_tracks"
        )
    elif intent == "lookup":
        outcome = "lookup_" + state.get("lookup_status", "unknown")
    elif intent == "clarify":
        outcome = "missing_reference"
    else:
        outcome = intent
    return [intent, outcome]
