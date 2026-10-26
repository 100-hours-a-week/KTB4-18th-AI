"""고속 모드 판단 로직: Jev 답을 해석해 의도를 정하고, LLM 없이 조회 곡을 찾는다.

첫 번째 Jev 호출의 질문(nodes.JEV_QUESTIONS + artist_question() + genre_question() +
year_questions())은 jev_node가 보낸다.
Jev는 텍스트를 만들지 못해서, 조회는 "아티스트를 목록에서 고르기 → 그 아티스트 곡 제목과
문자열 매칭 → 안 되면 곡 목록에서 고르기(두 번째 Jev 호출)" 순서로 하고, 그래도 확신이
없으면 None을 돌려 classify(LLM)에 맡긴다.
"""

import re
from datetime import date

from backend.providers.jev import decide
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
        "instructions": "이번 메시지가 가리키는 아티스트. 한글 이름이면 같은 아티스트의 영문 표기를 고른다.",
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
    NOTE: 초반·중반·후반 예시가 없을 땐 "90년대 후반"을 1990~1999로 뭉개거나, "2010년대 후반"을
    2017로 고르고도 확신이 0.34라 기준에서 버려졌다(15번 중 0번 성공). 그래서 예시를 넣었다.
    """

    this_year = date.today().year
    years = {str(y): f"{y}년" for y in range(YEAR_MIN, this_year + 1)}
    return {
        "year_from": {
            "type": "choice",
            "instructions": f"음악 외 요청 부분은 무시하고, 사용자가 원하는 노래 발매연도의 시작(이 해부터). "
                            f"올해는 {this_year}년. 예: '1990년대'→1990, '2010년대 초반'→2010, "
                            f"'2010년대 중반'→2014, '2010년대 후반·말'→2017, '2020년 이후'→2020, "
                            f"'최신곡·요즘 나온 노래'→{this_year - 1}, '2015년 이전'→none",
            "criteria": {"none": "시작 연도 조건 없음", **years},
        },
        "year_to": {
            "type": "choice",
            "instructions": f"음악 외 요청 부분은 무시하고, 사용자가 원하는 노래 발매연도의 끝(이 해까지). "
                            f"올해는 {this_year}년. 예: '1990년대'→1999, '2010년대 초반'→2013, "
                            f"'2010년대 중반'→2016, '2010년대 후반·말'→2019, '2015년 이전'→2014, "
                            f"'2020년 이후'→none, '최신곡'→none",
            "criteria": {"none": "끝 연도 조건 없음", **years},
        },
    }


def year_range(answers: dict) -> tuple[int | None, int | None]:
    """Jev 발매연도 시작·끝 답을 (min_year, max_year)로 바꾼다.

    한쪽이 none이거나 자신 없으면 그쪽은 조건 없음(None)으로 두고, 시작이 끝보다 늦으면
    잘못 고른 것이라 둘 다 버린다.
    """

    def pick(answer: dict) -> int | None:
        if answer["choice"] == "none" or answer["confidence"] < FILTER_CONFIDENT:
            return None
        return int(answer["choice"])

    start, end = pick(answers["year_from"]), pick(answers["year_to"])
    if start and end and start > end:
        return None, None
    return start, end


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
    "폭넓은" 조회만 턴 간 중복을 뺀다(lookup_node와 같은 정책).
    """

    artist = answers["artist"]["choice"]
    if artist == "none" or answers["artist"]["confidence"] < ARTIST_USABLE:
        return None

    if answers["has_title"]["noul"] < TITLE_MAYBE:
        return lookup_tracks_db(seed_artist=artist, exclude_ids=shown_ids) or None

    titles = seed_artist_titles_db(artist)
    found = match_title(messages[-1], titles)
    if found:
        return lookup_tracks_db(song_title=found, seed_artist=artist) or None

    song = decide(jev_state(messages), {"song": {
        "type": "choice",
        "instructions": "이번 메시지가 가리키는 곡. 한글 제목이면 같은 곡의 영문 공식 제목을 고른다.",
        "criteria": {**{f"t{i}": title for i, title in enumerate(titles)},
                     "none": "곡 제목을 말하지 않음"},
    }})["song"]
    if song["confidence"] < SONG_CONFIDENT:
        return None
    if song["choice"] == "none":
        return lookup_tracks_db(seed_artist=artist, exclude_ids=shown_ids) or None
    return lookup_tracks_db(song_title=titles[int(song["choice"][1:])], seed_artist=artist) or None
