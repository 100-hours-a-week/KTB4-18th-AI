"""고속 모드 판단 로직: Jev 답을 해석해 의도를 정하고, LLM 없이 조회 곡을 찾는다.

첫 번째 Jev 호출의 질문(nodes.JEV_QUESTIONS + artist_question() + genre_question() +
year_questions())은 jev_node가 보낸다.
Jev는 텍스트를 만들지 못해서, 조회는 "아티스트를 목록에서 고르기 → 그 아티스트 곡 제목과
문자열 매칭 → 안 되면 곡 목록에서 고르기(두 번째 Jev 호출)" 순서로 하고, 그래도 확신이
없으면 None을 돌려 classify(LLM)에 맡긴다.
"""

import re
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import date

from backend.providers.jev import decide
from backend.providers.openrouter import get_embedding_client
from backend.recommendation.counts import MAX_TRACK_COUNT, track_limit
from backend.recommendation.embedding import embed_query
from backend.recommendation.intent import known_genres
from backend.recommendation.retrieval import lookup_tracks_db, seed_artist_titles_db, seed_artists_db

# NOTE: "조금 덜 시끄럽게"처럼 이번 메시지만으론 의도를 알 수 없는 후속 요청이 있어서
# 최근 대화도 같이 보여준다. Jev는 판단에만 쓰니 몇 개 더 넣어도 흐려지지 않는다.
JEV_CONTEXT_MESSAGES = 4

# NOTE: "악뮤 사람들이 움직이는 게"처럼 한글 제목이 문장처럼 보이면 Jev가 아티스트는
# 확신하면서(0.86) intent는 out_of_scope로 본다. 아티스트를 이만큼 확신하면 조회로 본다.
ARTIST_CONFIDENT = 0.8
# NOTE: 조회에 쓸 아티스트의 최소 확신. "butter 노래"에서 Jev가 NewJeans를 0.26으로 고르는
# 것처럼 낮은 확신의 아티스트로 찾으면 엉뚱한 곡이 나간다. 볼빨간사춘기(BOL4)는 0.57~0.79로
# 맞게 고르고 있어서 그보다 낮게 끊는다.
ARTIST_USABLE = 0.5
# NOTE: 제목을 말했는지(has_title)는 아티스트만 말한 문장이 0.04~0.11, 한글 제목 문장이
# 0.14~0.61로 나와서 그 사이에서 낮게 끊는다. 기준 아래로 떨어진 제목 문장은 아무 곡이나
# 나가는 조용한 오답이 되지만, 아티스트만 말한 문장이 넘어오면 2단계에서 "제목 없음"으로
# 걸러져 0.4초만 손해 보므로 낮은 쪽으로 잡는다.
TITLE_MAYBE = 0.1
# NOTE: "dynamite 노래 추천해줘"처럼 제목만 말하면 Jev가 아티스트를 못 고르고 intent를
# recommend로 봐서 임베딩 검색으로 샌다. 제목만 말한 문장은 has_title이 0.77~0.95, 일반
# 추천 문장 94개는 최대 0.13이라, 이 이상이면 조회로 보고 classify에 제목 추출을 맡긴다.
TITLE_LIKELY = 0.5
# NOTE: 2단계에서 "제목이 없어서 none"은 확신이 높고(0.92~0.96), 목록에서 고르긴 했지만
# 자신 없는 경우는 낮게 나온다("사람들이 움직이는 게"→How People Move 0.66, "밤편지"→
# Through the Night 0.2). 이보다 낮으면 classify에 맡긴다.
SONG_CONFIDENT = 0.6
# NOTE: 장르·연도는 검색의 하드 필터라 Jev가 자신 없게 고른 값은 쓰지 않고 임베딩 검색에
# 맡긴다(장르 필터로 5곡이 안 차면 search()가 장르를 빼고 다시 채우지만, 연도는 풀지 않는다).
FILTER_CONFIDENT = 0.5
# NOTE: DB의 가장 오래된 곡은 1992년이지만, "80년대 노래"를 classify처럼 1980~1989로 필터해
# 없다고 답하게 1980년부터 둔다.
YEAR_MIN = 1980

_seed_cache: list[str] | None = None


def known_seed_artists() -> list[str]:
    """seed 아티스트 목록을 프로세스 생애 동안 캐시해 반환한다.

    NOTE: 프로세스가 뜬 뒤 새 seed가 추가돼도 재시작 전까진 반영되지 않는다.
    """

    global _seed_cache
    if _seed_cache is None:
        _seed_cache = seed_artists_db()
    return _seed_cache


def jev_state(messages: list[str]) -> list[str]:
    """Jev에 넘길 state: 최근 메시지를 이전 대화/이번 메시지로 표시한다."""

    recent = messages[-JEV_CONTEXT_MESSAGES:]
    return [f"이전 대화: {m}" for m in recent[:-1]] + [f"이번 메시지(판단 대상): {recent[-1]}"]


def artist_question() -> dict:
    """첫 번째 Jev 호출에 합칠 아티스트 질문. 선택지는 seed 아티스트 목록 + none."""

    return {
        "type": "choice",
        "instructions": "음악 외 요청 부분(다른 주제로 언급된 사람 등)은 무시하고, 이번 메시지가 노래를 찾는 "
                        "대상으로 가리키는 아티스트. 한글 이름이면 같은 아티스트의 영문 표기를 고른다.",
        "criteria": {**{a: a for a in known_seed_artists()},
                     "none": "특정 아티스트를 말하지 않았거나 목록에 없음"},
    }


def genre_question() -> dict:
    """첫 번째 Jev 호출에 합칠 장르 질문. 선택지는 classify와 같은 DB 장르 목록 + none.

    NOTE: classify는 장르를 여러 개 줄 수 있지만 Jev Choice는 하나만 고른다. 단일 선택으로도
    추천 문장 94개 중 90개가 classify와 일치해서, 묶음("힙합 계열")은 필요해지면 넣는다.
    """

    return {
        "type": "choice",
        "instructions": "음악 외 요청 부분(응답 언어, 다른 주제 등)은 무시하고, 사용자가 원하는 노래의 장르. "
                        "목록에 맞는 장르가 없으면 none",
        "criteria": {**{g: g for g in known_genres()}, "none": "장르 언급이 없거나 목록에 없는 장르"},
    }


def genre_filter(answer: dict) -> list[str] | None:
    """Jev 장르 답을 검색 필터로 바꾼다. none이거나 자신 없으면 필터 없음."""

    if answer["choice"] == "none" or answer["confidence"] < FILTER_CONFIDENT:
        return None
    return [answer["choice"]]


def year_questions() -> dict:
    """첫 번째 Jev 호출에 합칠 발매연도 시작·끝 질문. 선택지는 none + YEAR_MIN~올해.

    NOTE: 처음엔 연대(1990s, recent 등)를 골랐는데 "2024년 이후"(→2020s, 2020~2023이 섞임),
    "2000년 이전"(→1990s, 그 전이 빠짐)처럼 구간으로 표현 못 하는 요청이 eval 연도 문장 10개 중
    4개였다. 연도별 선택지는 50개 정도라 Choice(최대 255개)에 들어간다.
    NOTE: Jev는 올해가 몇 년인지 몰라서 classify 프롬프트처럼 직접 알려준다.
    NOTE: "N년대 초반·중반·후반", "N년 전후·쯤"은 여기서 예시로 풀지 않고 pattern_year_range()가
    코드로 정한다.
    """

    this_year = date.today().year
    years = {str(y): f"{y}년" for y in range(YEAR_MIN, this_year + 1)}
    return {
        "year_from": {
            "type": "choice",
            "instructions": f"음악 외 요청 부분은 무시하고, 사용자가 원하는 노래 발매연도의 시작(이 해부터). "
                            f"올해는 {this_year}년. 예: '1990년대'→1990, '2020년 이후'→2020, "
                            f"'최신곡·요즘 나온 노래'→{this_year - 1}, '2015년 이전'→none",
            "criteria": {"none": "시작 연도 조건 없음", **years},
        },
        "year_to": {
            "type": "choice",
            "instructions": f"음악 외 요청 부분은 무시하고, 사용자가 원하는 노래 발매연도의 끝(이 해까지). "
                            f"올해는 {this_year}년. 예: '1990년대'→1999, '2015년 이전'→2014, "
                            f"'2020년 이후'→none, '최신곡'→none",
            "criteria": {"none": "끝 연도 조건 없음", **years},
        },
    }


# NOTE: 아래 두 패턴은 규칙이 명확해서 Jev 대신 코드로 연도를 정한다.
# - "N년대 초반·중반·후반": Jev가 예시로 준 연대("2010년대 후반"→2017~2019)만 맞히고 다른
#   연대("90년대 후반", "2000년대 말")는 연대 전체로 뭉갰다.
# - "N년 전후·쯤": Jev가 끝 연도 확신을 기준(0.5) 근처로 내서, 3번 중 1번은 끝이 버려져
#   "N년 이후 전부"가 됐다. 앞뒤 2년으로 정한다.
# NOTE: 긴 표현이 먼저 걸리도록 "초중반"을 "초반"보다, "말기"를 "말"보다 앞에 둔다.
DECADE_PART = re.compile(r"(\d{2,4})년대\s*(초중반|중후반|초반|중반|후반|말기|막바지|끝자락|말|초)")
DECADE_PART_YEARS = {
    "초중반": (0, 6), "중후반": (4, 9), "초반": (0, 3), "초": (0, 3), "중반": (4, 6),
    "후반": (7, 9), "말기": (7, 9), "막바지": (7, 9), "끝자락": (7, 9), "말": (7, 9),
}
AROUND_YEAR = re.compile(r"(\d{4})년\s*(전후|쯤|즈음|무렵|경)")
AROUND_YEARS = 2
# NOTE: "2010년대 후반부터 2022년까지"처럼 연도가 여럿이면 패턴 하나로 정하면 나머지 조건이
# 무시되므로, 연도 숫자가 2개 이상이면 코드로 정하지 않고 Jev에 맡긴다.
YEAR_MENTION = re.compile(r"\d{2,4}년")


def pattern_year_range(message: str) -> tuple[int, int] | None:
    """메시지의 "N년대 초반·중반·후반" 또는 "N년 전후·쯤"을 연도 범위로 바꾼다. 없으면 None."""

    if len(YEAR_MENTION.findall(message)) != 1:
        return None

    found = DECADE_PART.search(message)
    if found:
        digits, part = found.groups()
        decade = int(digits)
        # "90년대"→1990, "00년대"→2000, "10년대"→2010
        if decade < 100:
            decade += 1900 if decade >= 30 else 2000
        first, last = DECADE_PART_YEARS[part]
        return decade + first, decade + last

    found = AROUND_YEAR.search(message)
    if found:
        year = int(found.group(1))
        return year - AROUND_YEARS, year + AROUND_YEARS

    return None


def resolve_years(message: str, answers: dict) -> tuple[int | None, int | None]:
    """발매연도 필터: 메시지에 코드로 정할 패턴이 있으면 그 범위, 없으면 Jev 답."""

    return pattern_year_range(message) or year_range(answers)


def year_range(answers: dict) -> tuple[int | None, int | None]:
    """Jev 발매연도 시작·끝 답을 (min_year, max_year)로 바꾼다.

    한쪽이 none이거나 자신 없으면 그쪽은 조건 없음(None)으로 두고, 시작이 끝보다 늦으면
    잘못 고른 것이라 둘 다 버린다.
    NOTE: "최근 3년"을 Jev가 2023~2026으로도, 2023~none으로도 고른다. 올해까지는 끝이 없는
    것과 같아서 None으로 맞춘다(classify의 "최신곡"→작년/null과도 같은 모양).
    """

    def pick(answer: dict) -> int | None:
        if answer["choice"] == "none" or answer["confidence"] < FILTER_CONFIDENT:
            return None
        return int(answer["choice"])

    start, end = pick(answers["year_from"]), pick(answers["year_to"])
    if start and end and start > end:
        return None, None
    if end == date.today().year:
        end = None
    return start, end


# NOTE: 이어받기가 계속되면 검색 문장이 계속 길어져 이번 요청의 비중이 줄고 임베딩이 흐려지므로,
# 최근 메시지 3개까지만 이어 붙인다.
SEARCH_PARTS_MAX = 3


def candidate_parts(previous_parts: list[str], message: str) -> dict[str, list[str]]:
    """Jev 이어받기 답(new/continue/repeat)별 검색 문장 메시지 목록. " ".join한 것이 recommend_query가 된다.

    - new: 이번 메시지만
    - continue: 지난 목록 + 이번 메시지. 지난 목록에 이미 앞 턴들의 맥락이 들어 있어서
      "카페 공부 → 좀 더 잔잔하게 → 조금 더 밝게"도 세 턴 맥락이 다 들어간다.
    - repeat: 지난 목록 그대로. 이미 보여준 곡은 shown_track_ids로 빠지므로 같은 조건의 다른 곡이 나온다.
    지난 목록이 없으면(고속 모드 첫 추천) 셋 다 이번 메시지만이다.

    NOTE: "다른 곡으로 더 줘"를 이어 붙이면 그 문구가 임베딩을 흔들어 classify 재작성 결과와
    0/5곡 겹쳤다(빼면 5/5). 그래서 재요청은 문구를 붙이지 않는다.
    NOTE: Jev 답을 기다리지 않고 임베딩을 미리 시작하려고(embed_candidates) 세 경우를 다 만든다.
    """

    if not previous_parts:
        return {"new": [message], "continue": [message], "repeat": [message]}
    return {
        "new": [message],
        "continue": (previous_parts + [message])[-SEARCH_PARTS_MAX:],
        "repeat": previous_parts,
    }


def next_search_parts(previous_parts: list[str], message: str, answers: dict) -> list[str]:
    """Jev 이어받기 답에 맞는 검색 문장 메시지 목록."""

    return candidate_parts(previous_parts, message)[answers["continues"]["choice"]]


# NOTE: 고속 추천에서 Jev(약 0.45초)를 기다렸다가 임베딩(1.3~1.7초)하면 둘을 더한 만큼 걸린다.
# Jev 답을 모르는 채로 후보 문장을 미리 임베딩하면 둘 중 긴 쪽만큼만 걸린다. 둘 다 네트워크
# 대기라 스레드로 동시에 보내면 된다. 추천이 아닌 턴이면 벡터는 버린다(임베딩 1회는 1원도 안 듦).
# 요청마다 스레드를 새로 만들지 않도록 프로세스 동안 하나를 재사용한다.
_embed_pool = ThreadPoolExecutor(max_workers=8)


def embed_candidates(parts_by_choice: dict[str, list[str]]) -> dict[str, Future]:
    """후보 검색 문장들을 백그라운드 스레드에서 임베딩하기 시작한다. 문장 → Future(벡터).

    같은 문장은 한 번만 보낸다. .result()에서 임베딩 실패 예외(503)가 그대로 올라온다.
    """

    client = get_embedding_client()
    queries = {" ".join(parts) for parts in parts_by_choice.values()}
    return {query: _embed_pool.submit(embed_query, client, query) for query in queries}


# NOTE: 미지원 문장은 가장 낮아도 0.93이었고, 유일한 오탐 "생일 노래 추천해줘"는 주제 기준 0.62였다.
# 오탐이어도 classify가 지원 여부를 다시 판단해 느려질 뿐 틀리진 않지만, 0.7로 두면 그마저 줄어든다.
UNSUPPORTED_CONFIDENT = 0.7
UNSUPPORTED_QUESTIONS = ("by_lyrics", "by_ranking", "by_number", "by_topic")


def asks_unsupported(answers: dict) -> bool:
    """가사·순위·정확한 수치·주제처럼 우리가 가진 정보로 고를 수 없는 기준을 요구하는가."""

    return any(answers[name]["noul"] >= UNSUPPORTED_CONFIDENT for name in UNSUPPORTED_QUESTIONS)


# NOTE: "2곡만", "한 곡만", "20곡"처럼 실제로 곡 수를 말하면 확신이 1.0이고, "영어 이메일 하나 써주고"·
# "드라마 하나 추천해주고"의 "하나"가 곡 수로 새는 경우는 0.51~0.57이었다. 지시문에 이메일·드라마를
# 직접 적어 막으려 하자 오히려 0.63~0.86으로 더 샜다. 그래서 문구 대신 기준을 높게 둔다.
# "두세 곡만"(0.44~0.54)은 기본 곡 수로 간다.
COUNT_CONFIDENT = 0.8


def requested_count(answer: dict) -> int | None:
    """Jev 곡 수 답을 classify의 requested_count와 같은 모양으로 바꾼다. 말하지 않았거나 자신 없으면 None.

    6곡 이상(more)은 MAX_TRACK_COUNT + 1로 넘겨, track_limit()이 최대치로 자르고 count_notice()가
    "한 번에 최대 N곡까지" 안내를 내게 한다.
    """

    choice = answer["choice"]
    if choice == "none" or answer["confidence"] < COUNT_CONFIDENT:
        return None
    if choice == "more":
        return MAX_TRACK_COUNT + 1
    return int(choice)


def resolve_intent(answers: dict) -> str:
    """Jev intent를 그대로 쓰되, 아티스트를 확신하거나 제목을 말한 게 분명하면 조회로 본다."""

    artist = answers["artist"]
    if artist["choice"] != "none" and artist["confidence"] >= ARTIST_CONFIDENT:
        return "lookup"
    if answers["has_title"]["noul"] >= TITLE_LIKELY:
        return "lookup"
    return answers["intent"]["choice"]


def normalize(value: str) -> str:
    """괄호 부분((feat. ...), [Remix] 등)을 떼고 소문자로 바꾼 뒤 공백·구두점을 없앤다."""

    value = re.sub(r"[(\[].*?[)\]]", "", value)
    return re.sub(r"[\W_]+", "", value.lower())


def match_title(message: str, titles: list[str]) -> str | None:
    """메시지에 글자 그대로 들어 있는 제목 중 가장 긴 것을 고른다.

    NOTE: "Me", "Love" 같은 짧은 제목이 상관없는 메시지에 걸리지 않도록 정규화 후
    2글자 미만은 매칭하지 않는다.
    """

    target = normalize(message)
    hits = [t for t in titles if len(normalize(t)) >= 2 and normalize(t) in target]
    return max(hits, key=lambda t: len(normalize(t))) if hits else None


def find_lookup_rows(messages: list[str], answers: dict, shown_ids: set[int]) -> list | None:
    """1단계 답으로 조회할 곡을 찾는다. classify에 맡겨야 하면 None을 돌려준다.

    제목을 콕 집은 조회는 이전에 보여준 곡이어도 다시 보여주고, 아티스트만 말한
    "폭넓은" 조회만 턴 간 중복을 뺀다(lookup_node와 같은 정책). 곡 수는 lookup_node처럼
    track_limit()으로 정한다.
    """

    artist = answers["artist"]["choice"]
    if artist == "none" or answers["artist"]["confidence"] < ARTIST_USABLE:
        return None
    limit = track_limit(requested_count(answers["requested_count"]))

    if answers["has_title"]["noul"] < TITLE_MAYBE:
        return lookup_tracks_db(seed_artist=artist, exclude_ids=shown_ids, limit=limit) or None

    titles = seed_artist_titles_db(artist)
    found = match_title(messages[-1], titles)
    if found:
        return lookup_tracks_db(song_title=found, seed_artist=artist, limit=limit) or None

    song = decide(jev_state(messages), {"song": {
        "type": "choice",
        "instructions": "이번 메시지가 가리키는 곡. 한글 제목이면 같은 곡의 영문 공식 제목을 고른다.",
        "criteria": {**{f"t{i}": title for i, title in enumerate(titles)},
                     "none": "곡 제목을 말하지 않음"},
    }})["song"]
    if song["confidence"] < SONG_CONFIDENT:
        return None
    if song["choice"] == "none":
        return lookup_tracks_db(seed_artist=artist, exclude_ids=shown_ids, limit=limit) or None
    return lookup_tracks_db(song_title=titles[int(song["choice"][1:])], seed_artist=artist, limit=limit) or None
