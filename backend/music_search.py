"""음악 데이터베이스 구축 전 사용하는 임시 Last.fm·iTunes 검색."""

import os
import re
import unicodedata

import httpx
from fastapi import HTTPException
from openai import OpenAI

from backend.schemas import Track


def _catalog_unavailable() -> HTTPException:
    """V1 음악 카탈로그 장애 응답을 반환한다."""

    return HTTPException(
        status_code=503,
        detail={
            "code": "SERVICE_UNAVAILABLE",
            "message": "일시적으로 음악 정보를 조회할 수 없습니다.",
            "details": {"reason": "MUSIC_CATALOG_UNAVAILABLE"},
        },
    )


def _model_unavailable() -> HTTPException:
    """임시 검색 태그 생성 모델의 V1 장애 응답을 반환한다."""

    return HTTPException(
        status_code=503,
        detail={
            "code": "SERVICE_UNAVAILABLE",
            "message": "일시적으로 AI 기능을 사용할 수 없습니다.",
            "details": {"reason": "MODEL_UNAVAILABLE"},
        },
    )


def normalize(value: str) -> str:
    """검색과 비교를 위해 문자열 표기를 정규화한다.

    NFKC 정규화와 대소문자 통일을 적용한 뒤,
    문자·숫자·밑줄 이외의 공백과 문장부호를 제거한다.
    """
    return re.sub(r"[^\w]", "", unicodedata.normalize("NFKC", value).casefold())


def find_itunes_track(candidate: dict[str, str], reason: str) -> Track | None:
    """Last.fm 후보와 제목·아티스트가 일치하는 iTunes 곡을 반환한다."""

    try:
        response = httpx.get(
            "https://itunes.apple.com/search",
            params={
                "term": f"{candidate['artist']} {candidate['title']}",
                "country": "US",
                "media": "music",
                "entity": "song",
                "limit": 5,
            },
            timeout=10,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
    except (httpx.HTTPError, ValueError, AttributeError) as error:
        raise _catalog_unavailable() from error

    expected_title = normalize(candidate["title"])
    expected_artist = normalize(candidate["artist"])
    for result in results:
        preview_url = result.get("previewUrl")
        store_url = result.get("trackViewUrl")
        if (
            preview_url
            and store_url
            and normalize(result.get("trackName", "")) == expected_title
            and normalize(result.get("artistName", "")) == expected_artist
        ):
            artwork_url = result.get("artworkUrl100")
            if artwork_url:
                artwork_url = artwork_url.replace("100x100bb", "600x600bb")
            return Track(
                track_id=str(result["trackId"]),
                title=result["trackName"],
                artist=result["artistName"],
                artwork_url=artwork_url,
                preview_url=preview_url,
                store_url=store_url,
                reason=reason,
            )
    return None


def search_tracks(client: OpenAI, message: str) -> tuple[str, list[Track]]:
    """임시 외부 API 조합으로 검색 태그와 검증된 추천곡을 반환한다.

    PostgreSQL·pgvector 음악 데이터베이스가 준비되면 이 함수의 구현을
    벡터 검색으로 교체한다.
    """

    lastfm_api_key = os.getenv("LASTFM_API_KEY")
    if not lastfm_api_key:
        raise _catalog_unavailable()

    try:
        tag_response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
            instructions=(
                "사용자의 음악 요청을 Last.fm tag.getTopTracks 검색에 사용할 영어 태그 "
                "하나로 바꾸세요. 사용자가 밝힌 취향을 최우선으로 반영하고, 별도 취향이 "
                "없으면 20~30대가 부담 없이 들을 만한 현대적인 음악을 고려하세요. "
                "설명, 따옴표, 문장부호 없이 태그 하나만 출력하세요."
            ),
            input=message,
        )
    except Exception as error:
        raise _model_unavailable() from error

    raw_tag = (tag_response.output_text or "").strip()
    tag = raw_tag.splitlines()[0].strip(" '\"") if raw_tag else ""
    if not tag or len(tag) > 80:
        raise _model_unavailable()

    try:
        lastfm_response = httpx.get(
            "https://ws.audioscrobbler.com/2.0/",
            params={
                "method": "tag.gettoptracks",
                "tag": tag,
                "api_key": lastfm_api_key,
                "format": "json",
                "limit": 5,
            },
            timeout=10,
        )
        lastfm_response.raise_for_status()
        payload = lastfm_response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise _catalog_unavailable() from error

    if payload.get("error"):
        raise _catalog_unavailable()

    candidates = [
        {
            "title": track.get("name"),
            "artist": track.get("artist", {}).get("name"),
        }
        for track in payload.get("tracks", {}).get("track", [])
        if track.get("name") and track.get("artist", {}).get("name")
    ]
    reason = f"'{tag}' 검색 조건과 어울리는 곡입니다."
    tracks = [
        track
        for candidate in candidates
        if (track := find_itunes_track(candidate, reason))
    ]
    return tag, tracks[:5]
