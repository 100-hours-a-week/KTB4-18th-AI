# V1 API 호출 확인

현재 코드 기준 확인일: 2026-09-18. 실제 요청 형식의 기준은 `backend/main.py`, `backend/schemas.py`와 실행 중인 `/openapi.json`이다.

## 서버 실행과 Swagger 주소

저장소 루트에서 실행한다.

```bash
uv sync --locked
./run.sh
```

이 프로젝트의 로컬 기본 포트는 **8001**이다. 다른 포트가 필요하면 실행할 때만
`AI_SERVER_PORT=8011 ./run.sh`처럼 덮어쓴다. 공용 이름인 `PORT` 대신 프로젝트 전용
환경 변수 `AI_SERVER_PORT`를 사용해 다른 프로젝트 설정과 섞이지 않게 한다.

| 용도 | 로컬 주소 |
|---|---|
| Swagger UI | http://localhost:8001/docs |
| OpenAPI JSON | http://localhost:8001/openapi.json |
| ReDoc | http://localhost:8001/redoc |
| 생존 확인 | http://localhost:8001/health |
| 기존 채팅 테스트 화면 | http://localhost:8001/app/ |

Swagger에서 엔드포인트를 펼치고 **Try it out → Execute**로 호출한다. 채팅 요청에는 아래 JSON 샘플을 붙여 넣는다. 실제 SSE 수신 확인에는 `curl -N` 또는 Spring의 스트림 클라이언트를 사용한다.

`run.sh`는 로컬 개발용으로 `127.0.0.1`에 바인딩한다. 다른 PC나 컨테이너에서
접속해야 한다면 배포 설정에서 `0.0.0.0` 바인딩과 접근 제어를 별도로 구성한다.
이 문서의 주소는 배포된 공용 서버 주소가 아니다. Swagger 상단 제목이
`머문음 AI 채팅`인지 확인하면 이 프로젝트 서버인지 바로 구분할 수 있다.

포트가 헷갈릴 때는 다음 명령으로 점유 프로세스와 프로젝트 경로를 확인한다.

```bash
lsof -nP -iTCP:8001 -sTCP:LISTEN
```

이미 다른 프로세스가 8001을 사용한다면 그 프로세스를 임의로 종료하지 말고,
`AI_SERVER_PORT=8011 ./run.sh`로 임시 포트를 선택한 뒤 Spring의 AI 기본 주소와
`requests.http`의 `baseUrl`도 같은 값으로 맞춘다.

## 현재 가능한 검증

| 요청 | 현재 예상 결과 | 확인할 수 있는 것 |
|---|---|---|
| `GET /health` | 200, `{"status":"ok"}` | 서버 연결과 JSON 수신 |
| 유효한 `POST /v1/chat/messages` | 503, `MUSIC_CATALOG_UNAVAILABLE` | 요청 전달, 오류 상태·본문 처리 |
| 잘못된 채팅 요청 | 422, `INVALID_REQUEST` | 입력 검증과 오류 매핑 |
| `POST /v1/transcriptions`에 `audio` 파일 업로드 | 501, `REQUEST_FAILED` | multipart 전달과 미구현 응답 처리 |
| `audio` 누락 | 422, `INVALID_REQUEST` | 필수 파일 검증 |

**엔드포인트 정의는 존재하지만 실제 추천·전사 기능이 완료된 상태는 아니다.** DB·임베딩 검색은 연결 전이며 STT는 미구현이다. 현재 유효한 채팅 요청은 모델 호출 전에 503으로 종료하므로 이 샘플로 실제 추천 SSE 성공을 검증할 수는 없다. LangGraph 의도 분류도 현재 API에 연결되어 있지 않다.

요청 검증 실패는 HTTP 422와 공통 ErrorResponse 본문으로 반환하며 OpenAPI에도 동일하게 명시한다. `GET /health` 성공은 DB·모델 준비 완료를 뜻하지 않는다.

## 요청 샘플

[requests.http](./requests.http)는 IntelliJ HTTP Client 또는 VS Code REST Client에서 실행할 수 있다. `baseUrl`만 환경에 맞게 바꾼다.

| 파일 | 목적 |
|---|---|
| [chat-minimal.json](./samples/chat-minimal.json) | 필수 UUID 두 개와 메시지 |
| [chat-with-context.json](./samples/chat-with-context.json) | 선택적인 사용자 정보 포함. 모든 값은 가상 데이터 |
| [chat-invalid-blank.json](./samples/chat-invalid-blank.json) | 공백 메시지 |
| [chat-invalid-uuid.json](./samples/chat-invalid-uuid.json) | 잘못된 요청 UUID |
| [chat-invalid-too-long.json](./samples/chat-invalid-too-long.json) | 200자 상한을 넘는 201자 메시지 |
| [chat-invalid-missing-message.json](./samples/chat-invalid-missing-message.json) | 필수 메시지 누락 |
| [upload-silence.wav](./samples/upload-silence.wav) | 합성한 1초·16kHz·모노·16비트 무음 WAV. 업로드 확인 전용 |

실제 연동에서는 대화별 `thread_id`를 유지하고 새 요청마다 새 `request_id`를 만든다. 샘플의 고정 UUID는 수동 호출 확인용이다. 현재 서버는 중복 요청 방지와 대화 이력 복원을 구현하지 않았다.

## curl로 직접 호출

아래 명령은 저장소 루트 기준이다. Spring이 실행되는 호스트나 컨테이너에서 실행하면 해당 환경에서 AI 서버에 도달하는지도 확인할 수 있다. 다만 curl 성공만으로 Spring 애플리케이션의 호출 코드가 검증되는 것은 아니다.

```bash
export AI_BASE_URL=http://localhost:8001

curl -i "$AI_BASE_URL/health"

curl -i -N "$AI_BASE_URL/v1/chat/messages" \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  --data-binary @docs/api/v1/samples/chat-minimal.json

curl -i "$AI_BASE_URL/v1/chat/messages" \
  -H 'Content-Type: application/json' \
  --data-binary @docs/api/v1/samples/chat-invalid-blank.json

curl -i "$AI_BASE_URL/v1/transcriptions" \
  -H 'Accept: application/json' \
  -F 'audio=@docs/api/v1/samples/upload-silence.wav;type=audio/wav'
```

multipart의 `Content-Type` 헤더는 curl 또는 Spring 클라이언트가 경계 문자열과 함께 생성하게 한다. 파일 필드 이름은 반드시 `audio`다. STT 연결 후 의미 있는 전사 테스트에는 실제 발화가 있는 파일을 사용한다. 현재는 길이·코덱·용량 검증도 구현 전이므로 무음 파일 수신 결과로 음성 검증 완료를 판정하지 않는다.

## Spring에서 호출할 주소와 응답 처리

| Spring 실행 환경 | AI 기본 주소 예시 |
|---|---|
| FastAPI와 같은 PC에서 직접 실행 | `http://localhost:8001` |
| Docker Desktop 컨테이너 → 호스트 PC FastAPI | `http://host.docker.internal:8001`* |
| 동일 Docker 네트워크의 별도 컨테이너 | `http://<FastAPI 서비스명>:8001`* |
| 서로 다른 서버 | 배포 환경에서 정한 HTTPS 주소 |

\* 다른 컨테이너에서 접근하려면 FastAPI가 `0.0.0.0`에 바인딩되어 있고 해당 포트가
노출되어 있어야 한다. 기본 `run.sh`는 같은 PC에서만 접근할 수 있는 로컬 실행용이다.

컨테이너 안의 `localhost`는 그 컨테이너 자신이다. 서비스명·포트·방화벽은 실제 배포 구성에 맞춘다. Spring 서버 간 HTTP 호출에는 브라우저 CORS 설정이 필요하지 않다. 현재 FastAPI에는 인증 헤더 검증이 구현되어 있지 않으며, 인증이 있는 중계 서버를 거치는 환경은 별도 설정을 따른다.

Spring 호출 코드에서는 다음을 확인한다.

1. 위 JSON을 snake_case 그대로 직렬화하고 `POST /v1/chat/messages`에 전달한다.
2. HTTP 상태와 `Content-Type`을 먼저 구분한다. `Accept: text/event-stream`을 보냈어도 422·503 응답은 JSON이다.
3. 현재 503 본문을 읽어 `details.reason`까지 보존한다. Spring 클라이언트가 오류 상태를 예외로 바꾸면 예외 경로에서도 본문을 읽는다.
4. 검색·모델 연결 후 200 SSE는 단일 JSON으로 역직렬화하지 않는다. 이벤트별로 `text`, `tracks`, `done`, `error`를 처리한다. 임의의 네트워크 청크를 이벤트 하나로 간주하지 않는다.
5. 전사는 `audio`라는 multipart 파일 파트로 전달하고 JSON으로 받는다. 완성된 전사문을 사용자가 확인한 뒤 별도의 채팅 요청으로 보낸다.

현재 채팅 요청의 실제 오류 본문:

```json
{
  "code": "SERVICE_UNAVAILABLE",
  "message": "음악 DB 검색 연결을 준비 중입니다.",
  "details": {"reason": "MUSIC_CATALOG_UNAVAILABLE"}
}
```

현재 입력 검증 오류 본문:

```json
{"code":"INVALID_REQUEST","message":"요청값이 올바르지 않습니다.","details":null}
```

현재 전사 업로드 응답 본문:

```json
{"code":"REQUEST_FAILED","message":"Not Implemented","details":null}
```

연결 후 SSE 정상 응답 형식 예시(현재 실서버 성공 결과가 아님):

```text
event: text
data: {"delta":"조건에 맞는 곡을 찾지 못했어요."}

event: tracks
data: {"tracks":[]}

event: done
data: {}

```

정상 완료는 `done`으로 판정한다. 생성 중 `error` 이벤트가 오거나 `done` 없이 끊기면 미완료다. 추천 카드가 있는 경우 `tracks` 배열 요소는 Swagger의 모델 또는 `backend/schemas.py`의 `Track` 정의를 따른다.

## 검증 범위

기존 API 테스트는 다음 명령으로 실행한다.

```bash
uv run pytest backend/tests/test_main.py -q
```

이 테스트의 SSE 성공 경로는 가짜 검색·모델 응답으로 검증한다. 실제 Spring 연동, DB 검색, 모델 추론, STT 성공을 검증한 결과로 해석하지 않는다. Spring에서는 먼저 200 헬스체크 → 422 입력 오류 → 503 추천 준비 중 → 501 전사 미구현 순서로 상태와 본문을 확인한다. 이후 DB·모델·STT 연결이 완료되면 실제 성공 경로를 추가 검증한다.
