# 고정 dev 베이스라인과 다음 회귀 평가

현재 최종 기준선·1차 개선 결과는 `../results/dev-vs-fix-report.md`를 우선한다.
dev `3f7733f`와 fix `8cb6594`를 사용자 승인 보완 기준으로 비교했다. 8개 목적표본 판정은 승인받았으나 전체 의미 채점 검증을 뜻하지 않는다.
서비스정책상 최대5곡 맥락은 초과요청에 전달한다. T-025의 최초 판정과 보정 이력은 feedback.correction·하위LLMtrace·로컬ledger에 남는다.
현재 .env의 v4와 기준선의 gemini가 달라 실행 프로세스에 SEARCH_EMBEDDING=gemini를 고정했다. 원본 .env는 수정하지 않았다.
새 실험에는 마이크로달러를 발행하지 않는다. dev–fix 원격대조·비용대조·기준선지정 파일을 함께 보존한다.


승인된 Sonnet 기준선은 `../results/sonnet-baseline-report.md`, 지표 정의는 `initial-metrics.md`에 있다.
`../archive/cleanup-20261031/results-initial-gemini/initial-baseline-report.md`의 Gemini 의미 점수는 이전 기록이며 최종 품질 기준선이 아니다.
서비스는 dev `3f7733f`, 평가 데이터 버전은 `initial-20261027`이다.
데이터와 결과는 Git에서 제외되는 lab 폴더에 있으므로 원격 LangSmith 결과와 로컬 파일을 함께 보존한다.

## 언제 무엇을 돌리나

브랜치가 데이터를 정하지 않는다. 평가 시점이 데이터를 정하고, 비교 대상은 코드 SHA다.

| 시점 | 데이터 | 옵션 | 채점 |
|---|---|---|---|
| `feat/*` 코드 변경 | `validation-20261102` | `--split smoke --no-judge` | 코드 검사 |
| PR 후보, 회귀 층 | `validation-20261102` | `--split validation` | 코드 검사 + LLM. 1회 약 $1 |
| PR 후보, 도전 층 | `validation-challenge-20261102` | `--kinds text --no-judge` | 코드 검사 |
| 배포 후보 SHA | `test-20261102` | `--kinds text,photo,dialogue --split test` | 코드 검사 + LLM. 점수만 확인 |

- 회귀 층은 거의 다 통과해야 하는 유지 확인용이고, 도전 층은 이상적인 동작을 기대값으로 둔 개선 측정용이다. 두 점수를 합치지 않는다.
- 도전·test의 기대값은 2026-11-02 카탈로그 기준이다. 폴더의 `catalog_sha.txt`와 현재 DB가 다르면 실행을 멈춘다. 기대값을 다시 확인해 새 버전을 만든다.
- 음성·장애는 `validation-20261102`에 들어 있다. 관련 코드를 바꿨을 때와 배포 후보에서 돌린다.
- `test-20261102`는 Git 밖 메인 폴더의 `lab/datasets/`에만 둔다. 개선 작업과 AI 세션에 내용을 주지 않는다. `compare_runs.py`는 test 실행의 사례 ID를 가리고 건수만 보여 준다. LangSmith에서도 test 실험의 사례별 화면을 열지 않는다. test 내용을 보고 코드를 고쳤다면 그 사례를 validation으로 옮기고 새 사례를 보충한다.

깨끗한 비교 브랜치 checkout을 준비하고, 아래에서 서비스 경로·출력 폴더·실험명을 바꾼다. 출력 폴더는 실행마다 새로 쓴다.
원래 작업 폴더의 Python 환경과 `.env`를 재사용하며 비밀값은 복사하거나 출력하지 않는다.

```sh
.venv/bin/python lab/evaluation/baseline_suite.py \
  --env-file .env \
  --service-root /실제/비교할/깨끗한/checkout \
  --dataset-dir lab/datasets/validation-20261102 --dataset-version validation-20261102 \
  --split smoke --no-judge \
  --output lab/results/새-출력-폴더 \
  --name 실험-이름 \
  --execute
```

평가 대상은 일반 모드다. 고속 모드(Jev)는 평가하지 않는다.
`--no-judge`를 붙이면 LLM 채점 없이 서비스 실행과 코드 검사만 한다. 전체 1회 약 $0.03이며 LangSmith 실험은 만들어진다.
`--kinds` 전체는 `text,photo,dialogue,stt,faults`다. 데이터 폴더에 없는 기능은 고르지 않는다.
데이터는 `prepare_validation.py`, `prepare_challenge.py`로 만든다. `baseline-`으로 시작하는 실험명은 전체 사례를 채점했을 때만 데이터셋 기준 실험이 된다.

| 옵션 | 쓰임 |
|---|---|
| `--dataset-dir 폴더 --dataset-version 버전` | 평가 데이터를 고른다. 필수다 |
| `--split validation` | 해당 split만 실행한다. 사례의 `split`은 `["validation", "smoke"]`처럼 목록일 수 있다 |
| `--resume` | 같은 출력 폴더에서 이어서 실행한다. 조건이 바뀌었으면 멈춘다 |

서비스 응답은 채점 전에 `{kind}.service.jsonl`에 저장한다. 최종 결과는 `{kind}.jsonl`이다.
`--resume`은 끝난 사례를 건너뛰고, 서비스가 성공한 사례는 저장 응답으로 채점만 하며, 서비스가 실패한 사례만 다시 호출한다.
재개하면 LangSmith에 `시도 N` 실험을 새로 만들어 사례당 최종 결과 하나만 올린다. 완료되면 이전 실험 이름에 `[이전 시도]`를 붙인다.
턴 일부만 채점이 실패했으면 그 턴만 다시 채점한다. `--no-judge` 실행을 `--resume`으로 이어 나중에 채점할 수도 있다.
실패 기록은 지우지 않고 파일에 남긴다. 읽을 때는 사례별 마지막 기록을 쓴다.
채점 호출은 `{kind}.judge.jsonl`에 한 건씩 남긴다. manifest의 `judge_spent_usd`는 이 기록의 합계라 실패한 호출과 재개 전 시도의 비용도 들어간다.

두 결과 폴더는 아래 명령으로 비교한다. 모델을 호출하지 않으며, 코드 검사는 두 폴더 모두 현재 규칙으로 다시 계산한다.

```sh
cd lab/evaluation
../../.venv/bin/python compare_runs.py ../results/기준폴더 ../results/비교폴더
```

- 측정 기준(데이터·첨부 파일·평가 코드·채점 규칙·채점 모델·카탈로그·실행 범위)이 다르면 비교할 수 없다. 새 시리즈로 다시 측정한다.
- 서비스 조건(`models`, `search_embedding`, `embedding`, `agentic_chat`)을 일부러 바꿨다면 `--vary search_embedding`처럼 밝힌다.
- 끝나지 않은 실행, 실행 중 DB 변경, 예정 건수 미달, 서비스 실패, 채점 누락, 한쪽에만 있는 기능·사례, 양쪽 모두 기록이 없는 조건은 비교를 멈춘다. 카탈로그에서 확인하지 못한 카드는 서비스 실패로 센다. 기능을 일부만 볼 때는 `--kinds text,dialogue`로 고른다.
- `--allow-incomplete`는 기록이 없거나 미완료인 실행을 공통 사례만 참고용으로 보여 준다. 측정 기준이 실제로 다른 경우와 고속 모드는 이 옵션으로도 비교하지 않는다.
- `--kinds`로 고르면 비용 합계·경고·실패 목록도 고른 기능만 집계한다.
- 한쪽이라도 `--no-judge` 실행이면 의미 채점 지표는 `비교 안 함`으로 표시하고, 기능·시간·비용만 비교한다.
- 재시도가 있으면 시도 횟수를 따로 보여 주고, 비용 합계는 재시도까지 더한다. 시간과 요청당 비용은 최종 시도 기준이다.
- 시간은 두 실행 모두 값이 있는 같은 턴만 기능별로 비교한다. 비용은 기능별 요청당 값을 함께 보여 준다.
- 저장 당시 판정과 현재 규칙의 판정이 다르면 사례를 따로 표시한다.

테스트는 `cd lab/evaluation && ../../.venv/bin/python -m pytest -q`로 실행한다.

LangSmith의 동일 데이터셋에서 기준 실험과 새 실험을 비교한다. 기준 실험을 덮어쓰지 않는다.
현재 재사용 실행기는 비용 반올림 보완과 외부 `.env` 경로 옵션을 추가했다.
최초 실행 소스는 결과 폴더의 `evaluator.py`로 보존했으며 채점 프롬프트·품질 계산은 같다.
원격 피드백의 영어 부분은 소문자로 정규화된다. 새 비용 피드백은 USD만 발행하며 score는 반올림될 수 있어 value.exact_usd와 저장 원본으로 합산한다.

## 현재 채점 설정과 재개 기록

사용자 확정: Claude Sonnet 5.5, high, 최대 8,192토큰, 제한 300초, 동시 2건,
자동 재시도 0회, temperature 미전송, JSON 출력이다. 평가 키 총한도는 2026-10-31 실측 $20이며 잔여 한도 안에서 실행한다.
추가 모델·루브릭·주요 실행 설정은 `grilling`으로 먼저 확인한다.

실험 이름은 `dev 기준선 · 사진 · Sonnet 5.5 · 완료 16/16`처럼 기능·모델·채점 완료 상태를 표시한다.
`resumed`는 서비스 개선 버전이 아니라 평가 중단 후 재개 이력이다. 원래 데이터셋 구조를 유지하며 통합 배치는 만들지 않는다.
화면의 `16/16 runs`는 결과 행 수이며 의미 채점 완료 수가 아니다. 실제 채점 완료율과 실험 이름을 함께 확인한다.
중간 이력 대신 완료 실험을 회귀 비교의 기준으로 사용한다. 채점 규칙을 바꾸면 새 시리즈로 다시 측정한다.

P-035는 정책상 유효한 실내 사진이나 Azure 경유 Sonnet이 JPEG를 `Could not process image`로 거절했다.
허용된 수동 JPEG 재시도도 실패했고, RGB 픽셀이 완전히 같은 PNG 전송본으로 채점에 성공했다.
원본 데이터는 보존했으며 이 전송 보정과 해시를 최종 manifest 및 해당 판정에 기록했다.
다음 전체 평가에서도 이 사례의 전송 규칙을 동일하게 반영해야 비교 조건이 맞는다.

현재 폴더의 서비스 코드는 dev 원본이며 추적 패치를 적용하지 않았다.
평가 실행기는 일반 Local/Production 추적을 끄고 각 Dataset의 Experiment에 기록한다.
브라우저 QA용 기존 8001 서버는 원래 개선 브랜치에서 계속 실행 중이다. 평가 대상과 혼동하지 않는다.
