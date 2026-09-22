# 음성 전사 API

`POST /v1/transcriptions`는 multipart `audio` 파일을 받아 `{"transcript":"전사문"}`을 반환한다. 전사문은 사용자 확인·수정용 초안이며 추천·채팅 전송을 실행하지 않는다. 추천 입력의 200자 제한은 전사문을 사용자가 수정한 뒤 적용하며 서버가 전사문을 임의로 자르지 않는다.

## 실행

```sh
uv sync --locked
# .env에 OPENROUTER_STT_API_KEY를 설정한다.
uv run uvicorn backend.main:app --host 127.0.0.1 --port 8001
```

```sh
curl http://localhost:8001/v1/transcriptions \
  -F 'audio=@recording.webm;type=audio/webm'
```

- `OPENROUTER_STT_API_KEY`: OpenRouter에서 발급한 STT 전용 키. 추천·임베딩 키와 별도다.
- 모델은 코드에서 `openai/whisper-large-v3-turbo`로 고정한다. 기존 `.env`의 `STT_MODEL`은 사용하지 않는다.
- STT용 키에 OpenRouter 사용 한도를 설정한다. 잔액 부족·한도 초과도 503으로 처리하며 다른 키나 모델로 우회하지 않는다.
- 외부 호출: `https://openrouter.ai/api/v1/audio/transcriptions`, 한국어 `ko`, JSON 응답.
- 연결 타임아웃 3초, 읽기·쓰기 등 HTTP 단계별 타임아웃 30초. 자동 재시도 없음.
- FFmpeg는 `imageio-ffmpeg` 패키지의 실행 파일을 사용한다. 지원 휠이 없는 배포 환경에서는 시스템 FFmpeg 설치 또는 `IMAGEIO_FFMPEG_EXE` 설정이 필요하다.

## 입력과 검증

최대 **10 MiB / 60초**. WAV, WebM, MP4/M4A, Ogg, MP3, FLAC의 오디오 MIME을 받는다. MIME의 `;codecs=...` 매개변수는 허용한다. 파일명 확장자는 신뢰하지 않으며 컨테이너 시그니처와 MIME을 확인한 뒤 실제 디코딩한다.

FFmpeg로 16 kHz 모노 PCM을 최대 61초까지 디코딩해 샘플 수로 길이를 검증한다. 60초 초과 입력을 잘라서 성공 처리하지 않는다. 디코딩 제한 시간은 15초다. 임시 파일은 성공·실패 시 모두 삭제하고, 정규화한 WAV를 Base64 JSON으로 OpenRouter에 전송한다.

빈 파일·빈 디코딩 결과·모든 샘플이 0인 무음을 거부한다. 잡음과 발화를 구분하는 VAD는 포함하지 않으므로 일반적인 무음/잡음에서 모델 환각이 없음을 보장하지 않는다. 공급자가 빈 전사문을 반환하면 다시 녹음하도록 안내한다.

파일 크기 검사는 multipart 파싱 후 수행한다. 배포 프록시에서도 multipart 오버헤드를 고려한 요청 본문 크기 상한을 설정해야 수신 단계의 디스크 사용량을 제한할 수 있다.

| HTTP | code | details.reason |
|---|---|---|
| 400 | INVALID_REQUEST | EMPTY_AUDIO, AUDIO_TOO_LONG, NO_SPEECH_DETECTED |
| 413 | PAYLOAD_TOO_LARGE | AUDIO_TOO_LARGE |
| 415 | UNSUPPORTED_MEDIA_TYPE | UNSUPPORTED_AUDIO_FORMAT, MIME_TYPE_MISMATCH |
| 422 | INVALID_REQUEST | 파일 누락 등 스키마 검증 실패 시 details는 null |
| 503 | SERVICE_UNAVAILABLE | TRANSCRIPTION_SERVICE_UNAVAILABLE |

설정 누락, FFmpeg 실행 불가, 공급자 오류·타임아웃·비정상 응답은 503으로 반환한다. 공급자 오류 본문, 키, 음성, 전사문은 로그나 오류 응답에 기록하지 않는다.

## 검증

```sh
uv run pytest backend/tests/test_transcriptions.py -q
```

실제 FFmpeg를 사용하는 포맷·길이 검증과 HTTP 모의 응답 테스트다. 유료 API와 한국어 인식 품질 검증을 대체하지 않는다. 실제 발화가 담긴 한국어 녹음으로 전사 내용·지연·과금을 별도 확인해야 한다.

기존 Wiki의 Base64 요청·30초 설계와 달리 이 구현은 현재 코드의 multipart·60초 계약을 따른다. 10 MiB는 기존 설계에서 가져온 초기 구현 상한이다. 정책 변경 시 코드·OpenAPI·이 문서를 함께 수정한다.

참고: [OpenRouter STT API](https://openrouter.ai/docs/guides/overview/multimodal/stt), [고정 모델](https://openrouter.ai/openai/whisper-large-v3-turbo).

## UI로 수동 테스트

서버 실행 후 `http://localhost:8001/app/`에 접속한다. HTML 파일을 직접 열지 않는다.

1. 전송 버튼 왼쪽의 **마이크 버튼**을 누르고 마이크 권한을 허용한다.
2. 다시 누르거나 60초가 지나면 녹음을 종료하고 multipart `audio` 필드로 전사 API에 전송한다. 별도의 파일 선택·업로드 UI는 제공하지 않는다.
3. 인스펙터에서 요청·HTTP 상태·JSON 응답을 확인한다. 모바일에서는 입력창 위 상태 문구를 확인한다.
4. 성공하면 입력창에 전사문이 채워진다. 자동으로 추천을 요청하지 않으며, 확인·수정 후 전송 버튼을 누른다. 200자 초과 전사문은 수정해야 한다.

모바일 마이크는 HTTPS 접속이 필요하다(localhost 제외). 파일 샘플은 위 curl 명령 또는 `/docs`의 전사 엔드포인트에서 테스트한다.

[samples/silence.wav](samples/silence.wav)는 1초 디지털 무음으로 **오류 확인용**이다. 키 설정 후에는 `400 EMPTY_AUDIO`, 키가 없으면 설정 검사에서 `503`이 반환된다. 이 파일은 외부 STT 호출 없이 거부된다. 한국어 인식 성공 테스트에는 별도의 실제 발화 녹음이 필요하다.

API 키는 서버의 `.env`에서 설정한다. 모델 변경은 코드 수정과 검증을 거쳐야 한다. API 키를 UI에 입력하지 않는다.
