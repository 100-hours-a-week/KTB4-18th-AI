"""음성 MIME·FFmpeg·길이 검사."""

import io
import logging
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe
from silero_vad_lite import SileroVAD
from backend.core.errors import _error, _unavailable

MAX_AUDIO_BYTES = 10 * 1024 * 1024
MAX_AUDIO_SECONDS = 60
SAMPLE_RATE = 16000
VAD_FRAME_SAMPLES = 512  # Silero의 16kHz 입력은 32ms 단위다.
VAD_SPEECH_THRESHOLD = 0.5
MIN_SPEECH_MS = 300
logger = logging.getLogger("uvicorn.error")


def _has_speech(pcm: bytes) -> bool:
    """요청마다 독립된 VAD로 최소 발화량을 검사한다."""
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
    speech_ms = 0
    max_probability = 0
    # 요청 간 모델 상태를 공유하지 않는다. 모델·CPU 런타임은 패키지에 포함된다.
    # ponytail: 누적 300ms 기준은 짧은 발화를 거부할 수 있어 실녹음으로 조정한다.
    try:
        vad = SileroVAD(SAMPLE_RATE)
        for offset in range(0, len(samples), VAD_FRAME_SAMPLES):
            frame = samples[offset:offset + VAD_FRAME_SAMPLES]
            length = len(frame)
            if length < VAD_FRAME_SAMPLES:
                frame = np.pad(frame, (0, VAD_FRAME_SAMPLES - length))
            probability = vad.process(frame)
            max_probability = max(max_probability, probability)
            if probability >= VAD_SPEECH_THRESHOLD:
                speech_ms += length * 1000 / SAMPLE_RATE
    except (OSError, RuntimeError, ValueError):
        raise _unavailable() from None
    detected = speech_ms >= MIN_SPEECH_MS
    logger.info(
        "stt_vad detector=silero duration_ms=%.0f voiced_ms=%.0f max_probability=%.3f decision=%s",
        len(pcm) * 1000 / (SAMPLE_RATE * 2), speech_ms, max_probability,
        "allow" if detected else "reject",
    )
    return detected


def _audio_format(data: bytes, content_type: str | None) -> str:
    """선언된 MIME과 컨테이너 시그니처가 일치하는 파일만 허용한다."""
    # NOTE: 현재 녹음 UI는 WebM·MP4·Ogg 중 지원 형식을 선택한다. 아래는 업로드 허용 범위다.
    # TODO: 풀스택 연동 시 FE 녹음 MIME·BE 전달 Content-Type·파일 업로드 지원 범위를 합의하고,
    # 허용 목록·시그니처 검사·포맷 테스트·API 문서를 함께 조정한다.
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
        logger.info("stt_vad duration_ms=%.0f decision=reject reason=EMPTY_AUDIO",
                    len(decoded) * 1000 / (SAMPLE_RATE * 2))
        raise _error(400, "EMPTY_AUDIO", "녹음된 음성이 없습니다.")
    if not _has_speech(decoded):
        raise _error(400, "NO_SPEECH_DETECTED", "음성이 감지되지 않았습니다. 다시 녹음해 주세요.")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(decoded)
    return output.getvalue()
