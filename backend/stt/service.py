"""업로드한 음성을 검증하고 OpenRouter STT로 전사한다."""

import base64
import os

import httpx
from fastapi import UploadFile

from backend.core.errors import _error, _unavailable
from backend.stt.audio import MAX_AUDIO_BYTES, _audio_format, _normalize_audio

STT_URL = "https://openrouter.ai/api/v1/audio/transcriptions"
DEFAULT_STT_MODEL = "openai/whisper-large-v3-turbo"


def transcribe_audio(audio: UploadFile) -> str:
    """검증된 음성의 전사 초안만 반환하며 추천 요청은 실행하지 않는다."""
    data = audio.file.read(MAX_AUDIO_BYTES + 1)
    if len(data) > MAX_AUDIO_BYTES:
        raise _error(413, "AUDIO_TOO_LARGE", "음성 파일은 최대 10 MiB까지 업로드할 수 있습니다.")
    if not data:
        raise _error(400, "EMPTY_AUDIO", "녹음된 음성이 없습니다.")
    audio_format = _audio_format(data, audio.content_type)
    api_key = os.getenv("OPENROUTER_STT_API_KEY", "").strip()
    if not api_key or api_key.startswith("<"):
        raise _unavailable()
    model = os.getenv("STT_MODEL", "").strip() or DEFAULT_STT_MODEL
    wav = _normalize_audio(data, audio_format)
    try:
        # 자동 재시도는 중복 과금 가능성이 있어 사용하지 않는다.
        response = httpx.post(
            STT_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "input_audio": {
                "data": base64.b64encode(wav).decode("ascii"), "format": "wav",
            }, "language": "ko", "response_format": "json"},
            timeout=httpx.Timeout(30.0, connect=3.0),
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        raise _unavailable() from None
    if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
        raise _unavailable()
    transcript = payload["text"].strip()
    if not transcript:
        raise _error(400, "NO_SPEECH_DETECTED", "음성을 인식하지 못했습니다. 다시 녹음해 주세요.")
    return transcript
