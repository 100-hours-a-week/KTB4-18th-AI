# GitHub 협업 설정

이 폴더는 저장소에 커밋·푸시하는 공유 설정이다. Git 내부 이력을 보관하는 `.git`과 다르다.

## 템플릿 사용

- Issue: 작업, 버그, 조사·실험·논의 중 하나를 선택한다.
- 담당자는 Assignee로 지정하고 완료 조건을 적는다. 작은 오탈자 수정은 Issue 없이 PR을 작성해도 된다.
- PR: 변경 목적, 실제 검증 결과, 연동·배포 영향을 기록한다. 상대 팀원 리뷰 후 기존 README 규칙대로 Squash merge한다.
- 기본 브랜치에 템플릿이 반영되면 GitHub에서 사용할 수 있다.
- 자동 라벨 지정은 비워 두었다. 아래 라벨을 GitHub에 생성한 뒤 직접 선택한다. 파일을 푸시한다고 라벨이 생성되지는 않는다.

## 권장 라벨

현재 화면의 한국어 명명 방식을 유지한 제안이다. 원격 라벨은 이 작업에서 변경하지 않았다.

| 이름 | 설명 |
| --- | --- |
| 기능·개선 | 새로운 기능, 기존 기능 개선 및 리팩터링 |
| 버그 | 기대와 다르게 동작하는 문제 수정 |
| 문서 | README, API 계약, 설계 문서 수정 |
| 배포·설정 | Docker, CI, 의존성 및 실행 설정 변경 |
| 조사·실험 | 데이터, 모델, 검색 방식 조사와 비교 실험 |
| 논의 | 구현 전에 팀의 결정이나 합의가 필요한 안건 |
| 긴급 | 서비스 장애나 배포 차단으로 우선 처리가 필요한 작업 |
| 진행 막힘 | 선행 작업이나 외부 답변을 기다리는 작업 |

작업 종류는 앞의 다섯 개 중 하나를 기본으로 선택하고 필요하면 논의·긴급·진행 막힘을 추가한다.
기존 버그-심각은 버그+긴급, 버그-일반은 버그로 정리할 수 있다.
질문·제안은 논의로 통합하고, 도움 요청은 담당자를 지정하거나 리뷰를 요청한다.
중복 Issue는 원본 링크를 남기고 닫는다. 기존 라벨을 정리할 때는 연결된 Issue를 먼저 확인한다.
버전은 Milestone(V1·V2·V3), 진행 상태는 Project 또는 Issue 열림·닫힘으로 관리해 라벨과 중복하지 않는다.

## Python lint 도입 제안 — 팀 합의 전

현재 `pyproject.toml`에는 pytest가 있고 Ruff 및 CI 설정은 없다.
Ruff는 코드를 실행하지 않고 미정의 이름, 불필요한 import 등 선택한 규칙 위반을 찾는 도구다.
추천 품질, DB 접속 성공, SSE 중계 정상 여부는 테스트나 통합 검증으로 별도 확인한다.

도입 시 공유할 설정:

1. Ruff를 개발 의존성에 추가하고 `pyproject.toml`과 `uv.lock`을 함께 커밋한다.
2. `pyproject.toml`에 팀의 검사 규칙을 명시한다. 초기 제안은 `E4`, `E7`, `E9`, `F`이다.
3. 팀원은 `uv sync --locked` 후 `uv run ruff check backend`로 같은 범위를 검사한다.
4. `.github/workflows/ci.yml`에서 PR마다 같은 명령과 `uv run pytest backend/tests`를 실행한다.
5. CI가 정상 실행되는 것을 확인한 뒤 GitHub의 main 보호 규칙에 필수 검사로 지정한다.

제안 설정 예시(현재 적용하지 않음):

```toml
[tool.ruff]
target-version = "py312"

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F"]
```

Formatter는 코드 모양을 통일하는 별도 기능이다. `ruff format --check backend`의 필수 적용은
기존 코드 정리와 팀 합의 후 진행한다. CI에서는 자동 수정하지 않고 검사 결과만 보고한다.
Ruff는 개발·CI에 필요하며 운영 서버가 시작할 때 실행할 필요는 없다.

참고: [Ruff 설정](https://docs.astral.sh/ruff/configuration/),
[Ruff lint](https://docs.astral.sh/ruff/linter/),
[GitHub 템플릿](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/about-issue-and-pull-request-templates).
