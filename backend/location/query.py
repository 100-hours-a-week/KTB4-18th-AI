"""LocationContext를 임베딩 검색어로 바꾼다. 비교 실험용으로 세 방식을 둔다.

- template_query: 규칙으로 문장을 조립한다. LLM 호출 없음. V1의 사용자 원문에 해당
- llm_query(style="situation"): V1 recommend_query처럼 상황·분위기 명사구로 재구성한다
- llm_query(style="sound"): 악기·질감·템포 같은 소리 묘사로 번역한다

검색은 v4 쿼리 어댑터(사용자 검색어 분포로 학습)를 거치므로 situation이 분포에 가장 가깝다는 가설이다.
어느 쪽을 쓸지는 scripts/location/compare_queries.py 결과를 보고 정한다.
"""

import json
import os

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from backend.location.schemas import LocationContext

WEATHER_WORDS = {"clear": "맑은", "cloudy": "흐린", "fog": "안개 낀", "rain": "비 오는", "snow": "눈 내리는", "storm": "폭풍우 치는"}
TIME_WORDS = {"dawn": "새벽", "morning": "아침", "afternoon": "오후", "sunset": "해 질 녘", "night": "밤"}
DENSITY_WORDS = {"quiet": "한적한", "moderate": "여유로운", "busy": "활기찬", "very_busy": "북적이는"}
# 밀도는 가게 수로 잰 낮 기준 상권 규모라, 붐빔을 뜻하는 단어는 낮·맑음/흐림일 때만 쓴다
# (예: 비 오는 밤 속초 청초호 전망대에 "활기찬"이 붙어 신나는 K-pop이 나왔다)
LIVELY_DENSITIES = ("busy", "very_busy")
LIVELY_TIMES = ("morning", "afternoon")
LIVELY_WEATHERS = ("clear", "cloudy")
# {n}은 지형 이름. 이름이 없거나 너무 길면 NATURE_PLAIN을 쓴다
NATURE_NAMED = {"sea": "{n} 바닷가", "river": "{n} 물가", "mountain": "{n} 산자락", "park": "{n} 공원"}
NATURE_PLAIN = {
    "sea": "바닷가", "river": "강가", "stream": "계곡 물소리가 들리는 곳",
    "mountain": "산자락", "forest": "숲길", "park": "공원",
}
MAX_NAME_LEN = 15  # "경주역사유적지구 대릉원지구"처럼 긴 이름은 검색어를 흐린다


def template_query(ctx: LocationContext) -> str:
    places = []
    for hit in ctx.nature[:3]:
        if hit.name and len(hit.name) <= MAX_NAME_LEN and hit.kind in NATURE_NAMED:
            places.append(NATURE_NAMED[hit.kind].format(n=hit.name))
        else:
            places.append(NATURE_PLAIN[hit.kind])
    if not places:
        places = ["빌딩 숲 사이 거리" if "업무지구" in ctx.place_traits else "도심 거리"]

    traits = [t for t in ctx.place_traits if t != "업무지구"][:2]
    near = f" ({', '.join(traits)} 근처)" if traits else ""
    lively_now = ctx.time_of_day in LIVELY_TIMES and ctx.weather in LIVELY_WEATHERS
    density = "" if ctx.density in LIVELY_DENSITIES and not lively_now else DENSITY_WORDS[ctx.density] + " "
    weather = WEATHER_WORDS[ctx.weather] + " " if ctx.weather else ""  # 날씨 API 실패 시 비어 있다
    return (
        f"{weather}{TIME_WORDS[ctx.time_of_day]}, "
        f"{density}{', '.join(places)}{near}에서 듣기 좋은 음악"
    )


class LocationQuery(BaseModel):
    """LLM 출력. 곡이 정해지기 전에 만드는 검색 조건이다."""

    model_config = ConfigDict(extra="forbid")
    music_query: str = Field(max_length=200)
    mood: list[str] = Field(max_length=4)


# 두 방식이 같이 쓰는 입력 해석 규칙
_COMMON_RULES = (
    "곡 제목이나 아티스트 이름은 쓰지 마세요. "
    "nature는 가까운 순이며 distance_m은 미터입니다. 300m보다 먼 지형은 약하게만 반영하세요. "
    "name이 없는 지형의 이름을 지어내지 마세요. "
    "daytime_commercial_scale은 반경 500m의 가게 수로 잰 낮 기준 상권 규모일 뿐, 지금 붐비는 정도가 아닙니다. "
    "지금의 분위기는 time_of_day·weather와 함께 판단하세요. 밤·새벽이거나 비·눈·안개일 때는 '활기찬', '신나는'처럼 "
    "붐빔을 단정하는 표현을 쓰지 마세요. quiet인데 북적이는처럼 측정값과 정반대인 분위기도 만들지 마세요. "
    "weather가 null이면 날씨를 측정하지 못한 것이니 날씨를 언급하지 마세요. "
    "place_traits는 근처 장소 유형이며 분위기 힌트로만 쓰세요. "
    "mood에는 music_query와 일치하는 분위기 단어를 1~4개 넣으세요."
)
INSTRUCTIONS = {
    # V1 classify의 recommend_query 규칙(backend/recommendation/intent.py)을 따른다
    "situation": (
        "사용자가 지금 있는 장소의 측정된 특징을 받아, 그 자리에 선 사람이 음악 검색창에 쓸 법한 "
        "한국어 검색 문장 하나를 만드세요. "
        "music_query는 원하는 음악을 묘사하는 명사구로 쓰세요(예: \"퇴근길에 듣기 좋은 잔잔한 노래\", "
        "\"비 오는 밤 바닷가에서 듣기 좋은 감성적인 노래\"). \"추천해줘\" 같은 요청 표현은 넣지 마세요. "
        "장소·날씨·시간대의 상황과 분위기를 담고, 악기·음색·템포 같은 소리 묘사는 넣지 마세요. "
        "이 장소를 다른 곳과 구별 짓는 특징(바다, 산, 강, 업무지구, 테마거리 등)이 있으면 하나 이상 살리고, "
        "name이 있는 지형은 이름을 그대로 써도 됩니다. 100자 이내로 쓰세요. " + _COMMON_RULES
    ),
    "sound": (
        "사용자가 지금 있는 장소의 측정된 특징을 받아, 그 자리에서 듣기 좋은 음악을 찾기 위한 한국어 검색 문장을 만드세요. "
        "music_query는 장소 묘사를 그대로 옮기지 말고 음악의 분위기·템포·질감·악기·장르 느낌으로 번역한 "
        "한 문장(200자 이내)입니다. " + _COMMON_RULES
    ),
}


def _llm_input(ctx: LocationContext) -> str:
    """좌표는 애초에 없다. 산은 기복으로 판정해 거리에 의미가 없으므로 이름만 넘긴다"""
    nature = [
        {"kind": h.kind, "name": h.name} if h.kind == "mountain" else h.model_dump()
        for h in ctx.nature
    ]
    # "density"라는 이름만 보고 LLM이 지금 붐빈다고 읽어서, 무엇을 잰 값인지 이름에 드러낸다
    payload = ctx.model_dump(exclude={"provider", "nature", "density"}) | {
        "daytime_commercial_scale": ctx.density, "nature": nature,
    }
    return json.dumps(payload, ensure_ascii=False)


def llm_query(client: OpenAI, ctx: LocationContext, style: str = "situation") -> LocationQuery:
    response = client.responses.create(
        model=os.environ["LLM_MODEL"],
        instructions=INSTRUCTIONS[style],
        input=_llm_input(ctx),
        text={"format": {
            "type": "json_schema", "name": "location_query", "strict": True,
            "schema": LocationQuery.model_json_schema(),
        }},
    )
    return LocationQuery.model_validate_json(response.output_text)
