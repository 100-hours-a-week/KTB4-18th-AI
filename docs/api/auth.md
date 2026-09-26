# Spring → AI 서버 간 인증

`POST /v1/chat/messages`, `POST /v1/transcriptions`에 동일한 전용 API 키 인증을 적용한다. 사용자 로그인 JWT와 OpenRouter 모델 호출 키는 별개다.

## 설정과 요청

- AI 서버는 비밀값을 `AI_SERVICE_API_KEY`로 읽는다. Spring의 채팅 `RECOMMENDATION_AI_AUTH_TOKEN`과 전사 `SPEECH_TRANSCRIPTION_AI_AUTH_TOKEN`에는 같은 값을 설정한다. 배포 환경에서는 서버 환경변수 또는 비밀값 저장소를 사용한다.
- `.env.example`에는 이름만 기록한다. 실제 값은 Git에 올리지 않는 `.env`에 저장한다.
- Spring은 두 요청에 HTTP 헤더 `Authorization: Bearer <비밀값>`을 보낸다. 연동 본문 규격은 채팅 JSON과 전사 multipart `audio`다.
- 충분히 긴 암호학적 난수를 사용한다. 권장 생성 규격은 32바이트 난수를 hex로 인코딩한 64자리 문자열이다. 코드에서는 이 길이를 강제하지 않는다.
- AI는 키 누락·공백 설정 시 서버 시작을 실패시킨다. 인증 우회 모드는 제공하지 않는다.

```http
Authorization: Bearer <AI_SERVICE_API_KEY 값>
```

## 인증 실패

헤더 누락·Bearer 형식 오류·값 불일치는 모두 HTTP `401 Unauthorized`와 아래 본문을 반환한다. Spring은 이 오류를 자동 재시도하지 않는다.

```json
{
  "code": "UNAUTHORIZED",
  "message": "서버 인증에 실패했습니다.",
  "details": null
}
```

유효한 인증을 받은 뒤 기존 입력 검증과 AI 처리를 진행한다. `/health`, `/readiness`는 인증 대상이 아니며 `/docs`에서는 헤더 매개변수와 401 응답을 확인할 수 있다.

## 연동과 배포

1. 양쪽 서버에 동일한 키를 설정한다. FE에 키를 배포하거나 OpenRouter 키를 재사용하지 않는다.
2. Spring에서 두 API의 헤더 전송을 준비한다.
3. AI 인증을 배포하고 정상 키·잘못된 키 요청을 확인한다.

배포 환경에서는 HTTPS 또는 구간 암호화를 사용하고 네트워크에서도 Spring의 접근만 허용한다. 키는 요청 로그·소스·이슈·PR에 기록하지 않는다. 키 교체 시 양쪽 설정을 함께 반영해야 한다.

## 로컬 검증

`/app/`은 개발용 수동 테스트 콘솔이다. 개발용 서버 인증 키를 비밀번호 입력란에 직접 넣으면 두 요청에 헤더로 전달하며, 브라우저 저장소와 요청 본문 인스펙터에는 기록하지 않는다. 새로고침 후 다시 입력한다. 서버가 비밀값을 HTML에 삽입하지 않는다. 운영 키를 콘솔에 입력하지 않는다.

```sh
uv run pytest backend/tests -q
```

테스트는 실제 키·유료 API 없이 인증 차단, 정상 요청 형식, 설정 누락 시 시작 실패 및 상태 확인 API를 검증한다. 실제 Spring·배포 네트워크 연동 검증은 별도로 진행한다.
