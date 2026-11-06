"""장소·날씨·시간 검색어를 따로 임베딩해 점수를 섞어 곡을 고른다.

- 장소 칸: 사다리. 장소 점수 몫을 1.0에서 0.6까지 칸 수만큼 고르게 내리며 장소·날씨말 후보 안에서 고른다.
- 특별 칸: 시간(새벽·아침·밤), 날씨(비·눈)일 때 그 요소 후보 안에서 장소 점수 0.7로 섞어 1곡씩 고른다.
근거 실험: scripts/location/place_experiments/ (place_ladder.py, place_triangle.py, slot_pool.py)
"""

import numpy as np

from backend.recommendation.adapter import to_v4_query
from backend.recommendation.counts import DEFAULT_TRACK_COUNT
from db.vectors import l2norm

# NOTE: 날씨말 몫이 0.5까지 가면 후보 안의 허브 곡(여러 장소에 두루 가까운 곡)이 장소마다 이겨서 0.4에서 멈춘다
LADDER_TOP, LADDER_BOTTOM = 1.0, 0.6
SLOT_PLACE_WEIGHT = 0.7  # 0.5면 요소 대표곡 몇 개로 모이고, 1.0이면 요소 후보 끝자락 곡을 고른다
POOL_SIZE = 30

SLOT_TIMES = {"dawn": "새벽", "morning": "아침", "night": "밤"}  # 오후·저녁은 음악 뜻이 약해 칸을 주지 않는다


def phrase(words: str) -> str:
    return f"{words}에 듣기 좋은 노래"


def place_phrase(place: str) -> str:
    return f"{place}에서 듣기 좋은 노래"


def _snow_words(month: int) -> str:
    # "눈 오는 날"은 크리스마스 곡으로 쏠려(30곡 중 6곡) 12월에만 쓴다. "함박눈"은 1곡
    return "눈 오는 날" if month == 12 else "함박눈 내리는 날"


def ingredient_phrase(weather: str | None, temperature_c: float | None, month: int) -> str:
    """사다리에 섞는 날씨말. 비·눈이 아니면 기온, 기온이 무난한 10~20℃와 날씨를 모를 때는 맑음·흐림"""
    if weather in ("rain", "storm"):  # 천둥 문장은 세고 곡이 어수선하다
        return phrase("비 오는 날")
    if weather == "snow":
        return phrase(_snow_words(month))
    if temperature_c is not None:
        # 시원한(cool 가사로 읽힘)·쌀쌀한(추운과 같은 곡)은 따로 쓰지 않는다
        if temperature_c < 10:
            return phrase("추운 날")
        if temperature_c >= 30:
            return phrase("더운 날")
        if temperature_c >= 20:
            return phrase("따뜻한 날")
    if weather in ("cloudy", "fog"):  # 안개 문장은 세고 곡이 어수선하다
        return phrase("흐린 날")
    # NOTE: 날씨를 모를 때도 맑음을 섞는다. 장소만으로 고르면 가사가 장소 단어와 맞물린 곡이 그대로 나온다
    return phrase("맑은 날")


def slot_phrases(weather: str | None, time_of_day: str, month: int) -> list[str]:
    """특별 칸 검색어. 시간 칸이 먼저, 날씨 칸이 다음"""
    out = []
    if time_of_day in SLOT_TIMES:
        out.append(phrase(SLOT_TIMES[time_of_day]))
    if weather in ("rain", "storm"):
        out.append(phrase("비 오는 날"))
    elif weather == "snow":
        out.append(phrase(_snow_words(month)))
    return out


def ladder_weights(n: int) -> list[float]:
    """장소 칸마다의 장소 점수 몫. 3칸이면 [1.0, 0.8, 0.6]"""
    if n == 1:
        return [LADDER_TOP]
    return [round(w, 4) for w in np.linspace(LADDER_TOP, LADDER_BOTTOM, n)]


def query_vector(space: str, embedding: list[float]) -> np.ndarray:
    """검색어 임베딩 → 검색 공간의 검색어 벡터. db.search.track_vectors()의 곡 벡터와 내적하면 검색 점수다"""
    if space == "v4":
        return to_v4_query(embedding)
    return l2norm(np.asarray(embedding, dtype=np.float32))


def mix(space: str, a: np.ndarray, b: np.ndarray, weight: float) -> np.ndarray:
    """두 검색어 벡터의 점수를 weight : 1−weight로 섞는 검색 벡터.
    v4는 [q′, 1] 꼴이라 마지막 1(허브 보정 자리)은 섞지 않고 남긴다. 그래야 곡 점수 q′·s − β·r(s)를 그대로 섞은 값이 된다.
    """
    if space == "v4":
        return np.append(weight * a[:-1] + (1 - weight) * b[:-1], np.float32(1.0))
    return weight * a + (1 - weight) * b


def _same_song(track) -> tuple[str, str]:
    return track.artist.lower(), track.title.lower()  # db.search._dedup과 같은 기준


def _best(query: np.ndarray, candidates: list, vectors: dict, taken: set):
    fresh = [t for t in candidates if _same_song(t) not in taken]
    if not fresh:
        return None
    best = max(fresh, key=lambda t: float(query @ vectors[t.track_id]))
    taken.add(_same_song(best))
    return best


def choose_tracks(
    space: str,
    place: np.ndarray,
    ingredient: np.ndarray,
    slots: list[np.ndarray],
    pools: dict,
    vectors: dict,
    k: int = DEFAULT_TRACK_COUNT,
) -> list:
    """곡 k개. place·ingredient·slots는 검색 공간으로 바꾼 검색어 벡터.
    pools: "place"·"ingredient"·"slots"(slots와 같은 순서의 목록) 후보 곡, vectors: track_id → 곡 벡터(검색과 같은 점수가 내적이 되게)
    """
    taken: set = set()
    chosen = []
    ladder_pool = pools["place"] + pools["ingredient"]
    for weight in ladder_weights(k - len(slots)):
        track = _best(mix(space, place, ingredient, weight), ladder_pool, vectors, taken)
        if track is not None:
            chosen.append(track)
    for slot, pool in zip(slots, pools["slots"]):
        track = _best(mix(space, place, slot, SLOT_PLACE_WEIGHT), pool, vectors, taken)
        if track is not None:
            chosen.append(track)
    return chosen
