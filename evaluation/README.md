# dev 기준 회귀 평가

개인 PC에서 고정 데이터셋으로 dev 소스를 평가한다. 서버 배포 시 자동 실행되지 않는다.
모델과 DB를 실제로 호출하므로 유료 실행은 사용자가 수행한다.

## 이 브랜치의 핵심

1. 고정 데이터셋을 한 명령으로 실행한다. 텍스트, 사진, 멀티턴, 음성, 장애 검사를 선택할 수 있다.
2. 같은 응답에 코드 검사와 선택적 LLM 채점을 적용한다. 기존 데이터와 판정 기준을 유지한다.
3. 실행 코드 SHA와 비교 조건을 기록한다. 중단되면 저장 응답으로 재개한다.
4. 두 dev 실행 결과를 비교한다. 바뀐 사례, 시간, 비용과 검사 분모를 보여준다.
5. 운영 요청 추적과 평가 Experiment를 구분한다. 배포 서버에서 평가를 자동 실행하지 않는다.

## 파일을 읽는 순서

| 파일 | 확인할 내용 |
| --- | --- |
| run.py | 시작점. 인자와 실행 조건을 확인하고 기능별 평가를 시작 |
| execution/experiments.py | LangSmith 데이터셋 연결과 기능별 실험 진행 |
| execution/runner.py | 텍스트·사진·음성·장애 호출과 반환 카드 DB 확인 |
| grading/scoring.py / grading/metrics.py | 기대값 통과/실패 판정 / 점수와 비율 계산 |
| grading/judging.py | 기존 LLM 채점, 저장 판정 재사용, 채점 예산 정산 |
| results.py / execution/telemetry.py | 결과 저장과 완료 판정 / 호출과 비용 기록 |
| compare.py / report.py | 두 실행 결과 비교 / Markdown 보고서 출력 |
| tests/evaluation/ | 채점, 재개, 비교, 전체 실행의 모의 검증 |

[지표와 계산식](METRICS.md)에 분모, 점수 범위와 미확인 값 처리까지 정리했다.
데이터 생성기와 과거 실험 보고서는 포함하지 않는다.

## 준비

저장소 루트에서 `uv sync --locked`를 실행한다. `.env`에 기존 서비스용 모델/DB 키,
`LANGSMITH_API_KEY`, `OPENROUTER_EVAL_API_KEY`를 설정한다. 평가키는 서버 추적에 필요하지 않다.
LangSmith 머문음 Agent 연결이 없다면 `uv run python scripts/setup_langsmith.py --execute`로 준비한다.
현재 평가기는 기존 정책대로 평가키 총한도 $20과 채점 모델/가격 조건을 검사한다.

데이터는 기존 개인 보관 경로를 그대로 지정한다. 새 브랜치에 복사하거나 커밋하지 않는다.

| 묶음 | 데이터 버전 | 선택 |
| --- | --- | --- |
| 회귀 80건 | validation-20261102 | `--split validation` |
| 빠른 확인 15건 | 위 회귀의 부분집합 | `--split smoke --no-judge` |
| 도전 47건 | validation-challenge-20261102 | `--kinds text --no-judge` |
| 최종 23건 | test-20261102 | `--kinds text,photo,dialogue --split test` |

최종 23건의 내용은 개선 작업에 사용하지 않는다. 회귀와 도전 점수를 합치지 않는다.
데이터는 사람 정답 검수 완료를 뜻하지 않는다. 고속 모드(Jev)는 평가하지 않는다.

## 실행

`--service-root`에는 평가할 dev SHA의 깨끗한 checkout을 지정한다. 실행 전 정확한 SHA를 확인한다.
아래 경로는 사용자 환경에 맞게 변경한다. CLI는 이 저장소 루트에서 실행한다.

```sh
uv run python -m evaluation.run \
  --env-file /개인/환경/.env \
  --service-root /깨끗한/dev-checkout \
  --dataset-dir /개인/데이터/validation-20261102 \
  --dataset-version validation-20261102 --split validation \
  --output /개인/결과/dev-SHA --name dev-SHA --execute
```

재개: 같은 명령과 출력 폴더에 `--resume`을 추가한다. 저장 응답을 재사용해 미완료 채점만 수행한다.
서비스가 실패한 사례는 다시 호출한다. 모델 채점 없이 확인하려면 `--no-judge`를 붙인다.
환경·데이터·평가기·모델·DB 조건이 바뀌면 같은 실행으로 재개할 수 없다.
일반 서버 요청 추적은 끄고 평가 Experiment에 기록한다.

## 비교

```sh
uv run python -m evaluation.compare /개인/결과/이전-dev-SHA /개인/결과/새-dev-SHA
```

비교는 모델을 호출하지 않는다. 의도적으로 바꾼 서비스 조건만 `--vary search_embedding`처럼 명시한다.
기존 정책대로 서비스 실패나 채점 누락이 있으면 기본 비교를 차단한다.
`--allow-incomplete`는 공통 사례의 참고 비교이며 확정 결과가 아니다.

## 결과와 한계

서비스 응답은 `*.service.jsonl`, 판정은 `*.jsonl`, 채점 비용은 `*.judge.jsonl`,
실행 조건은 `manifest.json`, 호출 기록은 `telemetry*.json`에 남긴다. 원본 결과를 덮어쓰지 않는다.
서비스 Git SHA, 데이터/첨부/평가기 해시, 모델, 검색 설정, DB 지문을 함께 기록한다.

이번 파일 분리로 평가 코드 해시가 달라진다. 기존 실행 폴더를 새 코드로 재개하거나
해시 검사를 우회하지 않는다. 과거 결과는 보존하고 새 코드에서 새 결과 폴더로 실행한다.
계산식·프롬프트·기대값은 유지하지만 서로 다른 평가기 해시는 기본 비교에서 차단된다.
서버 내부 라우터 평가이며 Spring/FE/네트워크/스테이징 배포 검증을 대신하지 않는다.
카드 없는 고정 안내는 기존 정책대로 LLM 품질 채점에서 제외된다.

검증: `uv run python -m pytest -q` (서비스와 평가 모의 테스트, 유료 호출 없음).

## dev 반복 평가 검토 결과

실행기는 특정 브랜치를 강제하지 않고 `--service-root`의 실제 Git SHA와 clean 상태를 기록한다.
dev를 평가하려면 사용자가 최신 dev를 받은 깨끗한 checkout을 지정해야 한다.
동일 평가기에서 이전 dev 결과와 새 dev 결과를 각각 새 폴더에 남기면 반복 비교할 수 있다.

기존 정책상 서비스 실패는 기본 비교를 차단하고, 카드 없는 질문은 LLM 품질 채점에서 빠진다.
비용 보충 조회도 저장된 사례 비용을 소급 수정하지 않는다. 이 제한은 METRICS.md에 명시했다.
자동 PR 게이트나 통계적 판정은 없으며, 실제 서버의 HTTP/FE 회귀 평가는 별도다.

평가 피드백은 SDK의 보관 연장 요청을 끈다. 프로젝트 기본 보관 등급이 shortlived이면 14일을 유지한다.
프로젝트 자체가 extended로 설정되거나 별도 자동화가 연장하면 이 코드만으로 보관 기간을 강제하지는 않는다.

manifest의 complete는 로컬 서비스와 채점 완료를 뜻하며 SDK의 비동기 업로드 성공을 보장하지 않는다.
원격 기록은 LangSmith에서 별도 확인한다. 스모크에서는 14일 기본 등급이어도 작업공간 한도 0으로 저장이 차단됐다.
토큰 미제공 응답은 0토큰으로 채우지 않는다. 음성 전사는 공급자 응답의 처리 시간과 비용을 사용한다.
