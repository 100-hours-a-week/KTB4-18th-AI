"""업로드한 음성을 검증하고 OpenRouter STT로 전사한다."""

import base64
import io
import os
import subprocess
import tempfile
import wave
from pathlib import Path

import httpx

from fastapi import HTTPException, UploadFile
from imageio_ffmpeg import get_ffmpeg_exe

MAX_AUDIO_BYTES = 10 * 1024 * 1024
MAX_AUDIO_SECONDS = 60
SAMPLE_RATE = 16000
STT_URL = "https://openrouter.ai/api/v1/audio/transcriptions"
# 비용 검증 없이 환경변수로 다른 모델이 선택되지 않도록 고정한다.
STT_MODEL = "openai/whisper-large-v3-turbo"


def _error(status: int, reason: str, message: str) -> HTTPException:
    """V1 공통 오류 형식으로 전사 실패를 표현한다."""
    codes = {400: "INVALID_REQUEST", 413: "PAYLOAD_TOO_LARGE",
             415: "UNSUPPORTED_MEDIA_TYPE", 503: "SERVICE_UNAVAILABLE"}
    return HTTPException(status_code=status, detail={
        "code": codes[status], "message": message, "details": {"reason": reason},
    })


def _unavailable() -> HTTPException:
    return _error(503, "TRANSCRIPTION_SERVICE_UNAVAILABLE", "일시적으로 음성 전사를 사용할 수 없습니다.")


def _audio_format(data: bytes, content_type: str | None) -> str:
    """선언된 MIME과 컨테이너 시그니처가 일치하는 파일만 허용한다."""
    formats = {
        "audio/wav": "wav", "audio/x-wav": "wav", "audio/wave": "wav",
        "audio/webm": "matroska", "audio/mp4": "mov", "audio/x-m4a": "mov",
        "audio/ogg": "ogg", "audio/mpeg": "mp3", "audio/flac": "flac",
        "audio/x-flac": "flac",
    }
    declared = formats.get((content_type or "").split(";", 1)[0].strip().lower())
    detected = None
    if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        detected = "wav"
    elif data.startswith(b"\x1a\x45\xdf\xa3"):
        detected = "matroska"
    elif data[4:8] == b"ftyp":
        detected = "mov"
    elif data.startswith(b"OggS"):
        detected = "ogg"
    elif data.startswith(b"fLaC"):
        detected = "flac"
    elif data.startswith(b"ID3") or (len(data) >= 2 and data[0] == 255 and data[1] & 224 == 224):
        detected = "mp3"
    if declared is None or detected is None:
        raise _error(415, "UNSUPPORTED_AUDIO_FORMAT", "지원하지 않는 음성 형식입니다.")
    if declared != detected:
        raise _error(415, "MIME_TYPE_MISMATCH", "음성 파일 형식과 Content-Type이 일치하지 않습니다.")
    return detected


def _normalize_audio(data: bytes, audio_format: str) -> bytes:
    """실제 디코딩 길이를 검사하고 16 kHz 모노 WAV로 변환한다."""
    # 길이 메타데이터가 없는 브라우저 녹음도 샘플 수로 검사한다.
    # 1초를 더 읽어 긴 입력을 잘라서 성공 처리하는 일을 방지한다.
    try:
        # MP4/M4A는 파일 끝의 메타데이터를 읽은 뒤 되감기가 필요할 수 있다.
        with tempfile.TemporaryDirectory(prefix="stt-") as directory:
            source = Path(directory) / "audio"
            source.write_bytes(data)
            decoded = subprocess.run(
                [get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
                 "-xerror", "-protocol_whitelist", "file,pipe", "-f", audio_format,
                 "-i", str(source), "-map", "0:a:0", "-vn", "-t", str(MAX_AUDIO_SECONDS + 1),
                 "-ac", "1", "-ar", str(SAMPLE_RATE), "-threads", "1",
                 "-f", "s16le", "pipe:1"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                timeout=15, check=True,
            ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        raise _error(415, "UNSUPPORTED_AUDIO_FORMAT", "음성 파일을 정상적으로 디코딩할 수 없습니다.") from None
    except (OSError, RuntimeError):
        raise _unavailable() from None
    if len(decoded) > MAX_AUDIO_SECONDS * SAMPLE_RATE * 2:
        raise _error(400, "AUDIO_TOO_LONG", "음성은 최대 60초까지 전사할 수 있습니다.")
    if not decoded or not any(decoded):
        raise _error(400, "EMPTY_AUDIO", "녹음된 음성이 없습니다.")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(decoded)
    return output.getvalue()


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
    wav = _normalize_audio(data, audio_format)
    try:
        # 자동 재시도는 중복 과금 가능성이 있어 사용하지 않는다.
        response = httpx.post(
            STT_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": STT_MODEL, "input_audio": {
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
