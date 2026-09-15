# 머문음 - AI Service

사용자가 텍스트·음성·사진으로 남긴 순간과 분위기를 이해하고, 실제 재생 가능한 음악을 추천하는 **머문음**의 AI 전용 저장소입니다. AI 기능을 독립적으로 개발·검증한 뒤 합의된 API 계약으로 메인 서비스와 연동합니다.

설계 문서는 [AI 설계 Wiki](docs/wiki/)에서 관리합니다. [설계 주차 과제](docs/wiki/99-설계주차-과제내용.md), [1주차](docs/wiki/week1/), [2주차](docs/wiki/week2/) 문서를 함께 수정하며 최신 상태를 공유합니다.

## 개발 환경 준비

Python 버전 조건은 `pyproject.toml`, 패키지의 설치 버전은 `uv.lock`을 기준으로 합니다.
uv 설치 후 저장소 루트에서 다음 명령을 실행합니다.

```bash
uv sync --locked
```

환경변수가 필요하면 `.env.example`을 `.env`로 복사하고 개인 API 키를 입력합니다.
예시의 모델·외부 API·타임아웃 설정은 로컬 실험용이며 팀의 확정 설정이 아닙니다.
현재 공유 범위는 의존성·문서·협업 템플릿이며 서비스 구현 코드는 포함하지 않습니다.


## 협업 원칙

- 중요한 결정과 변경 이유는 Issue 또는 Pull Request에 남깁니다.
- API 키, `.env`, 개인 환경 설정, 데이터셋과 모델 파일은 커밋하지 않습니다.
- API 계약은 코드와 OpenAPI를 기준으로 관리합니다.


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
