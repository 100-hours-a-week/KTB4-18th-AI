# 머문음 - AI Service

사용자가 텍스트·음성·사진으로 남긴 순간과 분위기를 이해하고, 실제 재생 가능한 음악을 추천하는 **머문음**의 AI 전용 저장소입니다. AI 기능을 독립적으로 개발·검증한 뒤 합의된 API 계약으로 메인 서비스와 연동합니다.

설계 문서는 [AI 설계 Wiki](docs/wiki/Home.md)에서 관리합니다.

## 협업 원칙

- 중요한 결정과 변경 이유는 Issue 또는 Pull Request에 남깁니다.
- API 키, `.env`, 개인 환경 설정, 데이터셋과 모델 파일은 커밋하지 않습니다.
- API 계약은 코드와 OpenAPI를 기준으로 관리합니다.

## 브랜치 전략

초기에는 `main`과 작업별 단기 브랜치를 사용하는 **GitHub Flow**를 따릅니다.

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

## 작업 규칙

- 하나의 Issue와 Pull Request는 하나의 독립적인 작업 목표를 다룹니다.
- Pull Request에는 변경 이유, 핵심 내용, 검증 방법을 간략히 작성합니다.
- 기능과 무관한 변경을 같은 Pull Request에 섞지 않습니다.
- 커밋과 Pull Request 제목은 `type: 한국어 설명` 형식을 사용합니다.

```text
feat: 음악 추천 API 추가
fix: 모델 응답 시간 초과 처리
docs: API 계약 문서 보완
chore: AI 협업 저장소 초기화
```
