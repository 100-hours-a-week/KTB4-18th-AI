"""곡별 추천 이유."""

import json
import os
from openai import OpenAI
from db.types import Track


def reason_instructions() -> str:
    """곡별 추천 이유 생성을 위한 지시를 반환한다."""

    return (
        "You explain why each song fits the listener's request.\n\n"
        "Rules:\n"
        "- Write in Korean, one sentence per song, under 40 characters.\n"
        "- Base the explanation ONLY on the mood tags given. Do not invent "
        "facts about lyrics, artists, or chart performance.\n"
        "- Never mention songs outside the provided list.\n"
        '- Return JSON only: {"<track_id>": "<설명>", ...}'
    )


def assign_reasons(
    client: OpenAI,
    message: str,
    tracks: list[Track],
    mood_tags_by_id: dict[str, dict],
) -> None:
    """추천곡마다 곡별 추천 이유를 채운다.
    실패하면 db.search.search()가 채운 공통 문구를 그대로 둔다."""

    if not tracks:
        return

    payload = {
        "request": message,
        "songs": [
            {
                "track_id": t.track_id,
                "title": t.title,
                "artist": t.artist,
                "mood_tags": list(mood_tags_by_id.get(t.track_id, {}))[:5],
            }
            for t in tracks
        ],
    }
    # NOTE: schema 없이 "JSON만 반환해라" 지시문에만 의존했을 때, 모델이 track_id
    # 몇 개를 응답에서 그냥 빼먹는 일이 실측으로 5곡 중 2곡꼴로 있었다(파싱 실패가
    # 아니라 누락이라 예외로 안 잡힘). classify()처럼 track_id를 required로 강제하는
    # strict json_schema를 써서 전부 채우도록 만든다.
    schema = {
        "type": "object",
        "properties": {t.track_id: {"type": "string"} for t in tracks},
        "required": [t.track_id for t in tracks],
        "additionalProperties": False,
    }
    try:
        response = client.responses.create(
            model=os.environ["LLM_MODEL"],
            instructions=reason_instructions(),
            input=json.dumps(payload, ensure_ascii=False),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "song_reasons",
                    "schema": schema,
                    "strict": True,
                }
            },
        )
        data = json.loads(response.output_text or "{}")
    except Exception as error:
        # NOTE: 실패해도 추천 자체는 이어가되(폴백 문구 유지), 원인은 남긴다.
        # 여기서 실패 원인을 못 찾은 적이 있었다.
        print(f"[assign_reasons] 이유 생성 실패: {type(error).__name__}")
        return

    if not isinstance(data, dict):
        return
    for t in tracks:
        reason = data.get(t.track_id)
        if isinstance(reason, str) and reason.strip():
            t.reason = reason.strip()
