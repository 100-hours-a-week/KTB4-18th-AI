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


# NOTE: "장소나 분위기를 알려주시면"처럼 정보를 나열하는 열린 질문은 걸리지 않게, 둘 중 고르게 하는 형태만 잡는다.
BINARY_CHOICE = re.compile(
    r"아니면|중(에서|에)? (어[느떤]|무엇|뭐|골라|하나)|쪽이 좋|어느 쪽|선호하는 쪽|\w/\w|\w+을지 [^.?!]*\w+을지"
    r"|(세요|까요|좋아요|나요|가요)[,，].*(세요|까요|좋아요|나요|가요)\?"
)
# "카페에 계신지 야외에 계신지 알려주세요"처럼 쉼표 없이 이어진 두 선택지. "어디에 있으신지 어떤 분위기를
# 원하시는지"처럼 열린 의문사로 여러 정보를 묻는 문장은 선택지가 아니므로 뺀다.
PARALLEL_CHOICE = re.compile(r"\w+지 (?:[^,.?!]* )?\w+지 (알려|골라|말씀|선택)")
OPEN_WORD = re.compile(r"어디|무엇|뭐|어떤|어떻게|언제|누구")


def binary_choice(text):
    return bool(BINARY_CHOICE.search(text)) or any(
        not OPEN_WORD.search(m.group(0)) for m in PARALLEL_CHOICE.finditer(text)
    )


def score_turn(expected, actual, *, metrics=(), previous=(), before=None):
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
        elif key == "asks":
            # 같은 에피소드면 이전 턴 대비, 새 에피소드면 0 대비 전달된 질문 수를 본다. 상태가 없으면 실패다.
            same = before is not None and before.get("episode") == state.get("episode")
            base = before.get("question_count", 0) if same else 0
            count = state.get("question_count")
            check(key, state.get("episode") is not None and count is not None
                  and (count == base + 1 if value else count == base))
        elif key == "episode":
            # 새 에피소드는 이전 번호(없으면 0)에서 정확히 1 늘어야 한다.
            now, prev = state.get("episode"), (before or {}).get("episode")
            new = now is not None and now == (prev or 0) + 1
            same = now is not None and prev is not None and prev == now
            check(key, new if value == "new" else same)
        elif key == "query_includes_any":
            check(key, any(word in (state.get("recommend_query") or "") for word in value))
        elif key == "no_binary_choice":
            check(key, (not binary_choice(actual.get("text", ""))) == value)
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
    previous, before = set(), None
    for turn in case["turns"]:
        turn["scores"] = score_turn(
            turn["expected"],
            turn["actual"],
            metrics=case["metadata"].get("metrics", []),
            previous=previous,
            before=before,
        )
        previous.update(str(t["track_id"]) for t in turn["actual"].get("tracks", []))
        before = turn["actual"].get("state")
    return case


def route_of(state):
    intent = state.get("intent")
    if intent == "recommend":
        outcome = (
            "unsupported"
            if state.get("recommend_unsupported_condition")
            else "tracks"
            if state.get("tracks")
            else "insufficient"
            if state.get("recommend_has_enough_info") is False
            else "no_tracks"
        )
    elif intent == "lookup":
        outcome = "lookup_" + state.get("lookup_status", "unknown")
    elif intent == "clarify":
        outcome = "missing_reference"
    else:
        outcome = intent
    return [intent, outcome]
