# 머문음 추적과 회귀 평가

LangSmith의 `머문음` Agent 하나에서 Local·Development·Staging·Production 환경을 선택한다.
실제 실행 목적지는 Agent 환경에 연결된 프로젝트이며, 설치된 LangGraph와의 주소 충돌을 피하도록
코드는 `LANGSMITH_PROJECT=meomuneum-{APP_ENV}`를 설정한다. Agent 주소 환경변수는 사용하지 않는다.

## 최초 설정

```sh
uv sync --locked
uv run python scripts/setup_langsmith.py --execute
```

`.env`와 서버 비밀값 저장소에 `LANGSMITH_API_KEY`를 넣는다. 키를 Git·문서·로그에 넣지 않는다.

| 실행 환경 | APP_ENV | LANGSMITH_TRACING | APP_REVISION |
| --- | --- | --- | --- |
| 로컬 실제 테스트 | local | true | 실행한 Git SHA |
| 스테이징(dev 배포) | staging | true | 실제 배포 Git SHA |
| 운영(main 배포) | production | true | 실제 배포 Git SHA |
| 모의 테스트 | local | false | 해당 없음 |

`APP_ENV`는 Git 브랜치를 추측해 정하지 않는다. 배포 환경에서 명시한다.
`LANGSMITH_TRACING_V2`, `LANGCHAIN_TRACING_V2`, `LANGCHAIN_PROJECT`, `LANGSMITH_AGENT_ID`,
`LANGSMITH_AGENT_ENVIRONMENT`는 별도로 설정하지 않는다. 앱 시작 시 추적 플래그와 프로젝트가 통일된다.
SDK와 앱은 시작할 때 설정을 읽으므로 서버 환경변수 변경 뒤 재시작한다.

## 클라우드 팀 반영 사항

Cloud 저장소 `compose.yml`의 AI 서비스는 `.runtime/ai.env`를 읽는다. 확인한 main 배포 스크립트는
AWS SSM의 `/meomuneum/v1/ai` 아래 설정을 이 파일로 변환한다. 운영 SSM에는
`APP_ENV=production`, `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`를 설정한다.
스테이징은 해당 서버 전용 설정 경로에 같은 키를 넣되 `APP_ENV=staging`으로 설정한다.
현재 확인한 main에는 운영 배포만 확인되므로 스테이징 경로는 클라우드 팀에서 확인한다.

`APP_REVISION`에는 AI 이미지의 소스 Git SHA를 넣는다. 이미지 교체와 함께 값도 갱신해야 한다.
기존 모델·DB 설정을 유지한 채 이 추적 설정을 추가하고 컨테이너를 다시 생성한다.
운영 SSM 경로를 스테이징과 공유하면 두 서버가 같은 환경으로 기록될 수 있으므로 구분한다.
API 키는 팀의 SSM SecureString으로 전달하며 채팅·GitHub 문서에 붙이지 않는다.

배포 후 각 서버에서 실제 요청 1건을 보내고, LangSmith `머문음`의 해당 환경에
`머문음 요청` 루트와 모델 단계가 나타나는지 확인한다. 로컬의 연결 확인은 서버 배포 확인을 대체하지 않는다.

## 검증

로컬 요청은 Local, dev 서버 요청은 Staging, main 서버 요청은 Production에 나타나야 한다.
요청의 루트 이름은 `머문음 요청`이며 `environment`, `revision_id`, HTTP 상태, 처리 결과를 확인한다.
텍스트는 `thread_id`와 `request_id`도 루트에 기록한다. 텍스트·카드·완료 시간을 별도로 기록한다.
SSE 연결 중단과 스트림 오류는 정상 완료로 채우지 않는다. 정상 사진 거절과 공급자 차단도 구분한다.

모델 단계는 의도 분류·사진 분석·검색어 임베딩·안내 생성·음성 전사로 구분한다.
공급자가 반환한 토큰과 달러 비용을 사용한다. 비용이 없는 응답은 `cost_available=false`이며
0달러로 간주하지 않는다. 폴백의 실제 GPU 비용은 모델 API 비용과 별도로 관리한다.
사진·음성 원본 및 인증 헤더는 요청 트레이스에 올리지 않는다. 사용자 텍스트·출력·장면 분석은 기록된다.

```sh
uv run python -m pytest -q
```

일반 테스트는 외부 추적을 끈다. 테스트 때문에 `default`나 운영 환경에 새 기록이 생기면 실패로 본다.
사진·음성·SSE 계측 검증은 모의 공급자로 실행할 수 있으며 실제 추론 비용이 필요하지 않다.

## 회귀 평가

운영/개발 요청 트레이스와 고정 데이터셋 Experiment는 구분한다. 회귀 평가 실행기·데이터·실측 결과는
현재 로컬 전용 `lab/evaluation/`·`lab/results/`에 있으며 이 추적 연동 변경에 포함하지 않는다.
로컬 평가 실행기는 clean 서비스 checkout을 지정하고, 환경별 요청 추적과 분리된 Dataset Experiment에 기록한다.
실행 방법은 해당 로컬 폴더의 README를 따른다. 이 문서만으로 다른 개발자가 평가 실행기를 재현할 수 있는 상태는 아니다.

새 평가셋은 텍스트 30건·이미지 16건·멀티턴 6대화(각 3턴)·음성 8건·장애 20건으로 구성한다.
이전 로컬 데이터는 후보 자료로 보존하고, 기대값을 검수한 새 버전만 등록한다.
현재 서버 내부 실행기와 실제 HTTP/SSE 측정은 측정 범위를 구분해서 기록한다.
기준선과 비교 버전은 데이터·모델·DB·평가기 버전을 같게 유지하며 실제 코드 SHA를 기록한다.
한글 지표명은 짧게 고정하고, 계산식·판정 근거·평가 건수는 이름 밖에 보관한다.

초기화 전에 만든 `lab/evaluation/langsmith-resources.json`과 `remote-verification.json`의 원격 ID는
더 이상 유효하지 않다. 새 데이터셋 등록 전 이전 ID를 재사용하지 않는다.
