"""의도 분류와 장르 참고 목록."""

import json
import os

from openai import OpenAI

from backend.core.errors import _model_unavailable

_genre_cache: list[str] | None = None


def known_genres() -> list[str]:
    """DB의 실제 장르 목록을 프로세스 생애 동안 캐시해 반환한다. classify 프롬프트가
    참고 목록으로 쓴다 — 카탈로그가 늘어나도 코드를 안 고쳐도 되게 하기 위함.

    NOTE: 프로세스가 뜬 뒤 새 장르가 추가돼도 재시작 전까진 반영되지 않는다.
    DB 조회에 실패하면 빈 리스트로 폴백해 classify 자체는 계속 동작하게 한다.
    """

    global _genre_cache
    if _genre_cache is None:
        try:
            # NOTE: main.py의 load_dotenv()보다 먼저 실행되면 db.models가 읽는
            # DATABASE_URL이 아직 없을 수 있어, 호출 시점까지 import를 늦춘다.
            from db.models import SessionLocal
            from db.search import known_genres as db_known_genres

            with SessionLocal() as session:
                _genre_cache = db_known_genres(session)
        except Exception:  # noqa: BLE001
            _genre_cache = []
    return _genre_cache


def classify_instructions(genres: list[str]) -> str:
    """사용자 메시지의 의도·대상·조건을 분류하기 위한 지시를 반환한다.

    TODO: recommend_unsupported_condition이 intent=recommend 전용 필드라서, lookup으로
    분류되는 요청(아티스트 한 명 특정) 안에서 같은 종류의 미지원 조건이 나올 때마다
    "이 경우엔 lookup 대신 recommend로 분류하라"는 예외를 계속 추가해야 한다(예:
    인지도/대중성 조건). 이런 예외가 더 쌓이면 unsupported_condition을 intent와
    무관하게 항상 채우는 독립 필드로 분리하는 구조 변경을 검토할 것.
    """

    genre_hint = ", ".join(genres) if genres else "(장르 목록을 가져오지 못했습니다 — 참고 없이 판단하세요)"

    return (
        "사용자의 음악 챗봇 메시지를 분석해 아래 필드를 모두 채우세요.\n\n"
        "intent는 다음 중 하나:\n"
        "- recommend: 특정 아티스트 한 명으로 한정하지 않고, 분위기·상황·장르·연도·"
        "템포 등을 바탕으로 음악을 추천받고 싶어하는 요청. 가사·차트 순위·정확한 "
        "BPM처럼 우리가 갖고 있지 않은 조건으로 물어봐도(예: \"차트 1위 곡 "
        "알려줘\"), 음악 추천/조회를 원하는 게 명백하면 recommend로 분류하고 "
        "recommend_unsupported_condition으로 처리하세요.\n"
        "- guide: 챗봇 사용법이나 지원 범위를 묻는 질문\n"
        "- clarify: 음악 추천/조회를 원하는 것은 분명한데 대상이나 지시어가 "
        "가리키는 게 conversation에 없어 되물어야 하는 경우(예: \"그 노래 말고 "
        "다른 거\", \"아까 추천해준 곡 다시\", \"저번에 말한 그 아티스트\"인데 "
        "conversation에 해당 내용이 없음). 이전 채팅방의 대화는 conversation에 "
        "들어오지 않으므로 그런 참조도 여기에 해당합니다. 음악과 무관하거나 의미를 "
        "알 수 없는 입력은 clarify가 아니라 out_of_scope입니다.\n"
        "- lookup: 특정 곡·아티스트 정보를 조회하거나, 특정 아티스트 한 명의 곡을 "
        "원하는 요청. '추천해줘'라는 표현을 쓰더라도 원하는 게 특정 아티스트 한 명의 "
        "곡으로 좁혀지면 lookup입니다(예: \"아이유 노래 추천해줘\", \"이 아티스트 "
        "신곡 알려줘\"). 단, 그 아티스트 곡 중 우리가 갖고 있지 않은 정보(인지도·"
        "대중성·가사·인기 순위 등)로 좁혀 달라는 요청이면 lookup이 아니라 "
        "recommend로 분류하고 recommend_unsupported_condition=true로 표시하세요 "
        "(예: \"이 아티스트 사람들이 잘 모르는 곡만\", \"덜 알려진 곡으로\").\n"
        "- out_of_scope: 음악 추천·조회와 무관한 요청. 인사말·잡담·다른 주제 "
        "질문은 물론, \"12345\"처럼 의미를 파악할 수 없는 입력도 여기 "
        "포함됩니다. 음악 외 요청이 섞여 있어도 음악 추천·조회 부분이 있으면 "
        "out_of_scope가 아니라 그 음악 부분을 기준으로 분류하세요(예: \"갈비찜 "
        "레시피 알려주고 뉴진스 노래 추천해줘\" → lookup). 음악 외 요청의 결과를 "
        "가리키는 말(예: \"오늘 뭐 먹을지 골라주고 그 음식에 어울리는 노래\"의 \"그 "
        "음식\")은 이전 대화를 참조하는 게 아니므로 clarify로 보내지 말고, 정해진 "
        "부분만으로 판단하세요.\n\n"
        "conversation 필드는 이 대화방의 메시지를 오래된 순서로 담고 있고, "
        "마지막 항목이 이번에 분류할 요청입니다. 앞의 메시지들은 '이 곡과 "
        "비슷한 노래' 같은 지시어를 이해하기 위한 맥락으로만 참고하세요.\n\n"
        "intent가 recommend일 때 추가로 채울 필드:\n"
        "- recommend_has_enough_info: 메시지(와 이어받는 conversation 맥락)에 "
        "아무 조건도 없이 \"추천해줘\", \"노래 알려줘\" 수준이면 false, 조건이 "
        "하나라도 있으면 true. 조건이 우리가 지원하지 않는 종류여도(예: \"맞춤법 "
        "관련된 노래\"처럼 노래의 주제·소재 조건) 조건은 있는 것이므로 true로 두고, "
        "지원 여부는 recommend_unsupported_condition으로만 판단하세요.\n"
        "- recommend_unsupported_condition: 사용자가 요구하는 '필수' 조건 중 우리가 "
        "들어줄 수 없는 게 있으면 true. 예:\n"
        "  · 특정 아티스트의 우리가 갖고 있지 않은 주관적인 정보의 카탈로그로 한정\n"
        "  (예: \"이 아티스트 사람들이 잘 모르는 곡만\")\n"
        "  · 가사 내용이나 상황·분위기로 바꿔 읽을 수 없는 노래의 주제·소재, 차트·인기 순위, "
        "정확한 BPM 수치처럼 우리가 갖고 있지 않은 정보로 조건을 제한(예: "
        "\"가사에 비가 나오는 노래\", \"맞춤법 관련된 노래\", \"가장 인기 많은 곡\", "
        "\"BPM 120인 곡\")\n"
        "  다음은 지원되므로 false입니다:\n"
        "  · 분위기·상황·장르·연도 요청. 표현이 아니라 의미로 구분하세요: 노래를 "
        "듣는 상황·분위기로 바꿔 읽을 수 있으면 그렇게 해석해 false로 두고, "
        "recommend_query도 상황·분위기로 풀어 쓰세요(예: \"생일 노래\" → \"생일을 "
        "축하하는 밝고 따뜻한 분위기의 노래\", \"노동요\", \"운동 관련 노래\", \"비 오는 "
        "날 듣기 좋은 노래\" → false). 가사 내용을 명시적으로 조건으로 걸거나(\"가사에 "
        "생일이 나오는 노래\"), 상황·분위기로 바꿀 수 없는 주제(\"맞춤법 관련된 "
        "노래\")일 때만 true입니다.\n"
        "  · \"가장 빠른\", \"제일 신나는\"처럼 순위 자체보다 분위기·템포감을 "
        "원하는 최상급 표현. 이런 경우 recommend_query에는 \"빠르고 신나는 "
        "노래\"처럼 분위기로 풀어서 쓰세요.\n"
        "  · 대화 기록에 없는 내용을 참조하는 경우는 이 필드가 아니라 intent를 "
        "clarify로 분류하세요.\n"
        f"- recommend_genres: 원하는 장르가 있으면 문자열 배열로. 가능하면 다음 "
        f"목록에서 가장 가까운 값을 고르고, 목록에 없는 명백한 장르명이면 그대로 "
        f"적어도 됩니다: {genre_hint}. 장르 언급이 없으면 null.\n"
        "- recommend_min_year: \"최신곡\", \"2020년 이후\" 같은 연도 하한이 있으면 "
        "정수로, 없으면 null.\n"
        "- recommend_query: 검색·이유생성에 쓸, 이번 요청을 독립적으로 이해할 수 "
        "있는 한국어 문장 하나로 재구성하세요. conversation 앞부분에서 이미 나온 "
        "분위기·상황을 이번 메시지가 그대로 이어받는 거라면 그 내용을 이번 문장에 "
        "합쳐서 쓰고(예: 이전에 \"퇴근길에 듣기 좋은 잔잔한 노래\"였고 이번 메시지가 "
        "\"좀 더 신나게\"면 → \"퇴근길에 듣기 좋은 신나는 노래\"), 이번 메시지 "
        "자체에 분위기·상황이 다 들어있다면 그걸 그대로 정리해서 쓰면 됩니다. "
        "\"방금 추천한 곡 빼고\", \"다른 곡으로\" 같은 제외·재요청 지시는 이미 "
        "따로 처리되니 이 문장에는 넣지 마세요. 다음도 이 문장에 넣지 마세요:\n"
        "  · 말투·호칭·응답 형식·응답 언어에 대한 지시(예: \"사투리로\", \"아빠 "
        "말투로\", \"일본어로 말해줘\"). 단 \"일본 노래\"처럼 곡 자체에 대한 조건은 "
        "남기세요.\n"
        "  · 음악과 무관한 요청 부분(예: \"갈비찜 레시피 알려주고\", \"마들렌이랑 "
        "크로와상 중에 골라주고\")\n"
        "  · 사용자가 주지 않은 조건을 새로 지어낸 것. 가리키는 대상이 정해지지 "
        "않았으면(예: \"오늘 먹을 음식에 어울리는 노래\"인데 음식이 정해지지 않음) "
        "임의로 채우지 말고 \"식사하면서 듣기 좋은 노래\"처럼 정해진 부분만 쓰세요.\n\n"
        "intent가 lookup일 때 추가로 채울 필드:\n"
        "- lookup_song: 조회 대상 곡 제목. 사용자가 말한 제목을 그대로 적으세요"
        "(한글이면 한글 그대로). 제목 언급이 없으면 null.\n"
        "- lookup_song_alt: lookup_song이 한국어 제목이고, 그 곡이 음원 사이트에 "
        "공식 영문/로마자 제목으로도 등록돼 있다는 걸 알면 그 제목(예: 악뮤 "
        "\"사람들이 움직이는 게\"→\"How People Move\"). 모르거나 lookup_song이 이미 "
        "영문이면 null.\n"
        "- lookup_artist: 조회 대상 아티스트명. 우리 카탈로그는 아티스트명이 영문/"
        "로마자 표기라, 한글로 적지 말고 공식 영문/로마자 표기로 적으세요(예: "
        "아이유→IU, 방탄소년단→BTS, 아이브→IVE, 잔나비→Jannabi, 볼빨간사춘기→BOL4). "
        "공식 표기를 모르면 로마자로 옮겨 적으세요. 아티스트 언급이 없으면 null.\n"
        "  (제목과 아티스트 중 언급된 것만 채우면 됩니다. 곡 제목·아티스트명이 "
        "하나도 없이 \"그 노래\"처럼 지시어만 있으면 둘 다 null로 두세요.)\n"
        "  사용자가 말하지 않은 아티스트·제목을 알고 있는 지식으로 추측해 채우지 "
        "마세요. 곡 제목만 말했으면 그 곡을 부른 아티스트를 알더라도 lookup_artist는 "
        "null로 두고, 그 제목을 아티스트로 바꿔 적지도 마세요(예: \"dynamite 노래 "
        "추천해줘\" → lookup_song=\"dynamite\", lookup_artist=null). \"X 노래\"에서 X가 "
        "곡 제목인지 아티스트인지 헷갈리면, 둘 중 더 그럴듯한 칸에 X를 그대로 "
        "적으세요.\n\n"
        "intent와 상관없이 항상 채울 필드:\n"
        "- has_non_music_request: 음악 추천·조회 외에 다른 작업(레시피, 메뉴 "
        "고르기, 일반 질문 등)을 함께 요청하면 true, 아니면 false. 단, \"냉면 "
        "먹으면서 들을 노래\"처럼 음악을 고르기 위한 상황 설명은 해당하지 않습니다.\n"
        "- response_style: 답변 말투·호칭·관계 설정에 대한 요청이 있으면 짧게 "
        "적으세요(예: \"경상도 사투리\", \"딸에게 말하는 아빠 말투\", \"친구처럼 "
        "반말\"). 응답 언어를 바꿔 달라는 요청(예: \"일본어로 말해줘\")은 넣지 "
        "마세요. 없으면 null.\n\n"
        "intent가 recommend/lookup이 아니면 recommend_has_enough_info/"
        "recommend_unsupported_condition/recommend_genres/recommend_min_year/"
        "lookup_song/lookup_song_alt/lookup_artist는 각각 "
        "false/false/null/null/null/null/null로 "
        "채우세요. recommend_query는 intent와 상관없이 위 방식대로 항상 채우거나, "
        "재구성할 필요가 없으면 이번 메시지 원문을 그대로 넣으세요."
    )


def classify(client: OpenAI, messages: list[str]) -> dict:
    """대화 기록을 참고해 사용자 메시지의 의도·대상·조건을 한 번에 추출한다."""

    try:
        response = client.responses.create(
            model=os.environ["LLM_MODEL"],
            instructions=classify_instructions(known_genres()),
            input=json.dumps({"conversation": messages}, ensure_ascii=False),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "intent_classification",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "intent": {
                                "type": "string",
                                "enum": [
                                    "recommend",
                                    "guide",
                                    "clarify",
                                    "lookup",
                                    "out_of_scope",
                                ],
                            },
                            "recommend_has_enough_info": {"type": "boolean"},
                            "recommend_unsupported_condition": {"type": "boolean"},
                            "recommend_genres": {
                                "type": ["array", "null"],
                                "items": {"type": "string"},
                            },
                            "recommend_min_year": {"type": ["integer", "null"]},
                            "recommend_query": {"type": "string"},
                            "lookup_song": {"type": ["string", "null"]},
                            "lookup_song_alt": {"type": ["string", "null"]},
                            "lookup_artist": {"type": ["string", "null"]},
                            "has_non_music_request": {"type": "boolean"},
                            "response_style": {"type": ["string", "null"]},
                        },
                        "required": [
                            "intent",
                            "recommend_has_enough_info",
                            "recommend_unsupported_condition",
                            "recommend_genres",
                            "recommend_min_year",
                            "recommend_query",
                            "lookup_song",
                            "lookup_song_alt",
                            "lookup_artist",
                            "has_non_music_request",
                            "response_style",
                        ],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            },
        )
        data = json.loads(response.output_text or "{}")
    except Exception as error:
        raise _model_unavailable() from error

    if not isinstance(data, dict) or data.get("intent") not in (
        "recommend", "guide", "clarify", "lookup", "out_of_scope"
    ):
        raise _model_unavailable()
    return data
