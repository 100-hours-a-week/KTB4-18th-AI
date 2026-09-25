# 머문음 - AI Service

사용자가 텍스트·음성·사진으로 남긴 순간과 분위기를 이해하고, 실제 재생 가능한 음악을 추천하는 **머문음**의 AI 전용 저장소입니다. AI 기능을 독립적으로 개발·검증한 뒤 합의된 API 계약으로 메인 서비스와 연동합니다.

설계 문서는 [AI 설계 Wiki](docs/wiki/)에서 관리합니다. [설계 주차 과제](docs/wiki/99-설계주차-과제내용.md), [1주차](docs/wiki/week1/), [2주차](docs/wiki/week2/) 문서를 함께 수정하며 최신 상태를 공유합니다. 외부 API 응답은 [iTunes Search API 곡 응답 필드](docs/reference/itunes-search-api-song-response.md)처럼 `docs/reference/`에 기록합니다.

## 개발 환경 준비

Python 버전 조건은 `pyproject.toml`, 패키지의 설치 버전은 `uv.lock`을 기준으로 합니다.
uv 설치 후 저장소 루트에서 다음 명령을 실행합니다.

```bash
uv sync --locked
```

환경변수가 필요하면 `.env.example`을 `.env`로 복사하고 개인 API 키를 입력합니다.
예시의 모델·외부 API·타임아웃 설정은 로컬 실험용이며 팀의 확정 설정이 아닙니다.
음성 전사 실행 방법과 API 계약은 [STT API 안내](docs/api/stt.md)를 참고합니다.

모델 API는 OpenRouter를 사용하며 기능별 키에 각각 비용 한도를 설정합니다.

| 기능 | 키 환경변수 | 모델 |
| --- | --- | --- |
| 추천 답변·이유 | `OPENROUTER_LLM_API_KEY` | `LLM_MODEL` |
| 음성 전사 | `OPENROUTER_STT_API_KEY` | `openai/whisper-large-v3-turbo` 고정 |
| 검색 임베딩 | `OPENROUTER_EMBEDDING_API_KEY` | `EMBEDDING_MODEL=google/gemini-embedding-2`, 3072차원 |

기존 `.env`의 `OPENAI_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`는 사용하지 않습니다.
새 키는 로컬 `.env`와 배포 비밀값 저장소에 각각 설정합니다. 모델명에는 `openai/`, `google/` 접두사가 필요합니다.
LLM과 임베딩 SDK의 자동 재시도는 중복 과금을 피하기 위해 끕니다.
임베딩 전환 후 기존 DB 벡터와의 검색 결과를 검증해야 합니다. 현재 배치의 CLAP은 로컬 모델이며,
팀원이 별도로 실행하는 Gemini 음성 적재 코드는 이 저장소에 없으므로 별도 전환이 필요합니다.

현재 채팅은 LangGraph에서 의도를 분류한 뒤 추천 요청만 임베딩·DB 검색·추천 이유 생성으로 연결합니다.
그 외 요청은 임시 안내문과 빈 곡 목록을 반환하며, STT는 별도 API로 유지합니다.
사용자 메시지는 `thread_id`별 서버 메모리에 누적됩니다. 재시작 시 사라지고 여러 워커 간에 공유되지 않으며,
기록 삭제·상한과 이전 추천곡을 활용한 후속 대화는 아직 구현되지 않았습니다.
배포 전 QA에서 실제 OpenRouter·DB 연결, 브라우저 녹음, 대화 상태 처리와 오류 대응을 확인합니다.

## 배포 상태 확인

- `GET /health`: 프로세스가 응답하면 `200 {"status":"ok"}`를 반환합니다.
- `GET /readiness`: PostgreSQL 연결과 `tracks` 테이블 조회를 통해
  `emb_gemini`와 `store_url`이 모두 있는 곡이 최소 1개 존재하는지 확인합니다.
  준비되면 `200 {"status":"ready"}`, 설정 누락·DB 오류·추천 가능 곡 없음은
  `503 {"status":"not_ready"}`를 반환합니다.
- 상태 확인은 읽기 전용이며 외부 모델 API를 호출하지 않습니다. 추천 품질이나 전체 연동 성공을 보장하지는 않습니다.


## 협업 원칙

- 중요한 결정과 변경 이유는 Issue 또는 Pull Request에 남깁니다.
- API 키, `.env`, 개인 환경 설정, 데이터셋과 모델 파일은 커밋하지 않습니다.
- API 계약은 코드와 OpenAPI를 기준으로 관리합니다.
- Issue·PR 템플릿과 라벨 사용법은 [GitHub 협업 가이드](.github/github-협업-가이드.md)에서 관리합니다.



## 브랜치 전략

초기 저장소 구성과 팀 동기화 단계에서는 설정·설계 문서를 `main`에 직접 반영합니다.
실제 개발을 시작한 뒤에는 `main`과 작업별 단기 브랜치를 사용하는 **GitHub Flow**를 따릅니다.

- `main`에는 직접 푸시하지 않습니다.
- 작업 브랜치는 최신 `main`에서 생성하고 Pull Request로 병합합니다.
- 상대 팀원의 리뷰와 검증 후 **Squash merge**합니다.
- 병합된 작업 브랜치는 삭제합니다.

여러 기능을 함께 검증하는 실험·통합 단계부터 `develop`을 도입을 검토합니다.

- 기능 및 실험 브랜치는 최신 `develop`에서 생성.
- 검증된 변경은 `develop`에 병합하고, 통합 검증 후 `develop`에서 `main`으로 Pull Request를 생성합니다.
- `main`의 긴급 수정은 `develop`에도 반영합니다.

**브랜치 이름은 작업 종류와 내용을 짧게 표현합니다.**

```text
feat/recommendation-api
fix/model-timeout
experiment/retrieval-tuning
docs/api-contract
```



## 커밋 메세지 컨벤션

| Type | 설명 | 예시 |
| :--- | :--- | :--- |
| **feat** | 새로운 기능 추가 | `feat: 음악 추천 결과 미리듣기 기능 추가` |
| **fix** | 버그 수정 | `fix: 추천곡 미리듣기 URL 누락 처리` |
| **refactor** | 기능 변경 없는 코드 구조 개선 | `refactor: 음악 추천 파이프라인 모듈 분리` |
| **chore** | 빌드, 패키지 설정 및 기타 변경 | `chore: FastAPI 의존성 버전 업데이트` |
| **docs** | README, API 명세 등 문서 수정 | `docs: AI 챗봇 SSE 응답 명세 갱신` |
| **style** | 동작 변화 없는 코드 형식 수정 | `style: FastAPI 코드 들여쓰기 정리` |
| **test** | 테스트 코드 추가 및 수정 | `test: 음악 추천 API 응답 테스트 추가` |
| **perf** | 성능 최적화 | `perf: pgvector 추천 검색 속도 개선` |


## 코드 주석 컨벤션

주석은 코드만 읽어도 알 수 있는 동작보다 **설계 이유, 제약 조건, 임시 처리의 종료 조건**을 설명합니다. 작업 추적이 필요하면 `# 태그(#이슈번호): 내용` 형식으로 작성하고, 이슈가 아직 없거나 짧게 끝날 작업은 이슈 번호를 생략할 수 있습니다.

| 태그 | 용도 | 프로젝트 예시 |
| :--- | :--- | :--- |
| **TODO** | 현재 동작에는 문제가 없지만 후속 구현이나 정책 확정이 필요한 작업 | `# TODO(#42): 음악 DB 구축 후 pgvector 검색으로 교체한다.` |
| **FIXME** | 현재 알려진 버그나 잘못된 동작으로 수정이 필요한 부분 | `# FIXME(#51): 동일한 request_id의 중복 처리를 방지한다.` |
| **HACK** | 일정이나 외부 의존성 때문에 사용한 임시 우회 구현 | `# HACK(#63): 음악 DB 구축 전까지 Last.fm 후보를 iTunes에서 검증한다.` |
| **NOTE** | 코드만으로 알기 어려운 설계 의도나 외부 API 제약 | `# NOTE: iTunes는 일부 곡에 previewUrl을 제공하지 않는다.` |
| **DEPRECATED** | 제거 예정이므로 새 코드에서 사용하지 않아야 하는 기능 | `# DEPRECATED(#71): pgvector 전환 후 임시 외부 API 검색과 함께 제거한다.` |

`XXX`는 의미가 모호하고 `OPTIMIZE`는 성능 개선의 근거가 불분명해지기 쉬우므로 사용하지 않습니다. 위험한 코드는 `FIXME`, 측정된 성능 문제는 GitHub Issue와 `perf` 커밋으로 관리합니다.

### 위치와 줄바꿈

- 클래스·함수 Docstring은 선언 바로 아래에 작성합니다. 클래스 Docstring 다음에는 한 줄을 띄우고, 함수 Docstring 다음에는 함수 본문을 바로 작성합니다.
- `TODO`처럼 코드 블록 전체에 적용되는 주석은 대상 바로 위에 같은 깊이로 작성하며, 대상 코드와 사이에 빈 줄을 넣지 않습니다.
- 인라인 주석은 단위, 값의 출처, 짧은 도메인 의미를 설명할 때만 사용하고 코드 뒤에 공백을 두 칸 이상 둡니다.
- 여러 필드나 여러 줄에 같은 설명이 적용되면 인라인 주석을 반복하지 않고 바로 위에 한 번만 작성합니다.
- 변수명이나 코드 동작을 그대로 번역하는 주석은 작성하지 않습니다.

```python
class Track(BaseModel):
    """추천곡 메타데이터 응답 모델."""

    # TODO(#42): 음악 DB 구축 후 식별자와 URL 필드를 최종 API 계약에 맞게 갱신한다.
    track_id: str  # 현재 iTunes track ID 사용
    title: str
    artist: str
    preview_url: str | None  # 30초 미리듣기 URL
```
