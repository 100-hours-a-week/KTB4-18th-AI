# 정리 작업 기록: 원래 실험 폴더 → 이 폴더

`~/Desktop/emb_test/adapter`(원래 실험)를 포트폴리오용으로 정리한 기록이다 (2026-10-20 ~ 10-23).
원칙은 하나다: **계산 방식은 그대로, 구조와 이름만 바꾼다.** 원래 폴더는 한 번도 수정하지 않았다.

## 1. 원래 파일 → 새 파일

| 원래 (`adapter/`) | 새 파일 | 바뀐 것 |
| --- | --- | --- |
| `evaluation/evaluate.py` (로더) | `core/data.py` | 데이터를 `Bench` 하나로 묶음. test는 `allow_test=True`일 때만 열림 |
| `evaluation/evaluate.py` (지표) + 각 스크립트의 응집도 | `core/metrics.py` | 지표 묶음 `evaluate_all()` 하나로. 응집도가 여러 파일에 복붙돼 있던 것을 합침 |
| `evaluation/systems.py` + 6개 스크립트에 복붙된 두 타워 로더 | `core/systems.py`, `core/evaluator.py` | 평균 제거 → 어댑터 → α 혼합 → γ 섞기 → 허브 벌점을 `AdapterSystem.space(α, β, γ)` 하나로 |
| `experiments/stage4_adapter.py`의 `LowRankAdapter`, `multi_positive_infonce` | `core/adapters.py` | 그대로 옮김 + 체크포인트 불러오기 |
| `experiments/v4_sdd.py`, `v4_song_mix.py`의 SDD 채점 | `core/sdd.py` | `sdd_eval/common.py`를 import하지 않고 csv만 읽음 (import하면 폴더를 만들려 해서) |
| `experiments/stage6_grid.py`의 격자·규칙 | `core/selection.py` | |
| (없음) | `core/safe_io.py`, `core/report.py` | 새로 만듦: 쓰기 제한, 보고서 저장 |
| `experiments/stage4_adapter.py` | `train/train_query_adapter.py` | 기본값을 최종 설정으로 (서비스 형식, rank 512, 5에폭, 정답 3곡 이상). 저장은 `runs/`에 |
| `experiments/stage14_two_tower.py` | `train/train_two_tower.py` | 저장은 `runs/`에 |
| `experiments/v4_stage2_center.py` | `studies/5_1_mean_removal.py` | |
| `experiments/stage6_grid.py` | `studies/5_5_alpha_beta_grid.py` | |
| `experiments/v4_compare.py` (+ rank별 격자) | `studies/5_6_rank.py` | 격자 → 규칙 지점 → 이웃 rank 비교까지 한 번에 |
| stage4 `--tail_pos_only` + `v4_compare.py` | `studies/5_7_tail_labels.py` | |
| `stage13_service_adapter.py musvc` + `v4_compare.py` | `studies/5_8_query_format.py` | μ_q를 맞춘 사본이 규칙대로 만든 것과 같은지 실행마다 확인 |
| `experiments/v4_compare_twotower.py` | `studies/5_9a_two_tower.py` | |
| `experiments/v4_song_mix.py` + 즉석 코드로 뽑은 응집도 로그 2개 | `studies/5_9b_song_side_mix.py` | 응집도 표를 같은 파일에서 냄 |
| `experiments/v4_sdd.py` | `studies/6_sdd.py` | γ0.25 행 추가, 원래 대비 부트스트랩 추가 |
| `evaluation/final_report_v4.py`, `final_report_v4_posthoc.py` | `studies/6_final_test.py` | 원래 test 결과 파일이 있으면 바로 멈춤 (다시 열지 않음) |
| `prep/export_catalog.py` | `dataset/1_export_catalog.py` | `--version v3` → `--with_tail` |
| `prep/melon_match.py`, `match_rules.py` | `dataset/2_match_melon.py`, `match_rules.py` | `--version v3` → `--title_aliases` |
| `prep/select_tail.py`, `itunes_tail.py`, `embed_tail_batch.py` | `dataset/3_tail_songs/a_select.py`, `b_itunes_search.py`, `c_embed_batch.py` | 캐시는 `runs/caches/`, DB 쓰기는 `--write_db` + 확인, 배치 제출은 비용 확인 |
| `prep/merge_tail_mapping.py` + `build_dataset_v2.py`, `overlap_groups.py` | `dataset/4_build_dataset.py`, `overlap.py` | `--version v2/v3` → `--recipe first/final`. 파일 안 config 값은 원래 이름 그대로 (바이트 같게) |
| `prep/convert_service_query.py` | `dataset/5_service_queries.py` | 호출 수를 보여주고 확인 |
| `prep/build_dataset_service.py` | `dataset/6_service_dataset.py` | |
| `prep/embed_queries.py` | `dataset/7_embed_queries.py` | 원래 캐시에 있는 문장은 건너뜀, 확인 |
| `evaluation/pairwise_export.py`, `prep/export_store_urls.py` | `human_eval/export_lists.py`, `export_track_extras.py` | `--gamma` 옵션 추가 |
| `evaluation/pairwise_tally.py` | `human_eval/tally.py` | 평가자 이름을 기본으로 가림. 설명문에 있던 평가자 이름 삭제 |
| `evaluation/pairwise_page/serve.py`, `index.html` | `human_eval/serve.py`, `index.html` | 결과 저장은 `runs/human_eval_results/` |

옮기지 않은 것: 멜론 원문 형식 검색어 시절의 실험(stage5·6_debias·7·9·10·11), 폐기한 실험(stage12, stage13 중 musvc 외), 개인화(stage8), 일회성 분석(itunes_pilot, peek, topk 등), v1 데이터셋 빌드(`build_dataset.py`).

## 2. 이름표

코드에서는 뜻 있는 이름을 쓰고, 디스크의 원래 파일명은 `config.py`에서만 연결한다 (원래 파일명은 로그·문서가 가리키므로 바꾸지 않았다).

| 코드 이름 | 원래 파일 |
| --- | --- |
| `service_format` / `melon_format` | `dataset_v4` / `dataset_v3` |
| `query_adapter_r{32,128,512,3072}` | `stage4_r{…}_ds4_q3_base_e5` |
| `query_adapter_r512_tail_pos_only` | `stage4_r512_ds4_q3_tailpos_e5` |
| `query_adapter_r512_melon_trained` | `stage4_r512_ds3_q3_base_e5_mutitle` |
| `two_tower_lam{0,025,05,1}`, `_init` | `stage14_q512_s512_v4_lam{…}` |

## 3. 바꾸지 않은 것

계산식, 시드(학습 0, 부트스트랩 0, 분할 해시 seed), α·β·γ 격자, 선택 규칙, 동점 처리(격자 순서에서 먼저 나온 것), 허브 벌점의 기준 검색어, 학습 설정, 데이터 규칙(매칭·분할·cold·중복 제거) 전부. 두 타워 변형의 비교 지점은 원래처럼 학습 스크립트가 고른 지점을 상수로 옮겼다 (학습 스크립트의 격자는 r(s)를 학습 검색어 16,939개로 계산해 비교 격자와 끝자리가 조금 다르다).

## 4. 안전장치

- 쓰기는 `runs/` 안에만 (`core/safe_io.py`가 밖이면 에러). 실행마다 새 폴더라 이전 결과를 덮지 않는다
- 원래 SQLite는 `mode=ro`로만 연다 (실수로 쓰려 해도 DB가 거부)
- 유료 호출(LLM, 임베딩, Batch API)과 DB 쓰기는 실행 직전에 개수·비용을 보여주고 y를 받는다
- test는 `allow_test=True`일 때만 열리고, `6_final_test.py`는 원래 결과가 있으면 멈춘다
- 작업 중 원래 폴더(`adapter/data`, `sdd_eval`, `~/Desktop/melon`)를 sha256·수정 시각으로 여러 번 확인했다 — 바뀐 파일 0개

## 5. 확인한 것

| 대상 | 방법 | 결과 |
| --- | --- | --- |
| `core/` | 원래 로그 8개의 수치 77개를 같은 자릿수로 대조 | 77/77 |
| `studies/` 실험 8개 | 원래 로그와 대조 (격자 표 49줄은 글자 그대로 diff) | 전부 일치 |
| `train/train_query_adapter.py` | r32를 새로 학습해 원래 학습 로그와 대조 | 에폭별 손실·val·‖AB‖까지 일치 |
| `train/train_two_tower.py` | 3스텝 동작 확인 (저장 안 함) | 동작 |
| `dataset/` 2·3a·4·6 | 다시 만들어 원래 파일과 md5 비교 (매칭 v2·v3, 비인기곡 후보, 매칭+비인기곡, 데이터셋 v2·v3·v4) | 전부 md5 일치 |
| `human_eval/export_lists.py` | 본 평가 목록·대응표 md5 | 일치 |
| `human_eval/tally.py` | `tally_final.log`와 대조 | 일치 |
| `human_eval/serve.py` | 페이지 200, 대응표 404, 경로 조작 이름 차단 | 동작 |
| 실행 안 함 | 1·3b·3c·5·7, export_track_extras (팀 DB·외부 API·비용) | 코드만 |

## 6. 작업 중 발견한 것

- **사람 평가는 γ0(곡 어댑터 100%)으로 했다.** γ0.25는 그 뒤에 정했다 → 출시는 γ0으로 결정 (README (6-1))
- `임베딩 고도화 2.md` (6) 표의 "λ=0.25"는 λ=0.5의 오타 (숫자는 λ0.5 값)
- 원래 폴더의 `docs/handoff_serving.md`는 아직 γ0.25로 적혀 있다 → 서비스 반영 때 γ0(곡 쪽 섞기 없음)으로
- rank가 클수록 SDD가 낮아진다 (r32 39.28 → r3072 35.39). λ0 두 타워는 SDD 19.57로 크게 무너진다
- 5-7에서 C와 평균 제거의 SDD가 38.47로 같은 건 우연 (맞힌 캡션 수가 287개로 같을 뿐, 같은 등수인 캡션은 7.6%)
- 원래 `pairwise_tally.py` 설명문에 평가자 이름이 적혀 있었다 → 새 파일에서는 뺐다

## 7. 팀 저장소로 옮길 때

1. 팀 저장소에서 새 브랜치
2. 이 폴더를 `experiments/embedding_adapter/`로 복사 (`runs/`, `__pycache__/`는 빼도 되고 어차피 `.gitignore`로 빠진다)
3. 팀 저장소 안에서는 `config.KTB4_ROOT`가 자동으로 그 저장소를 가리킨다
4. 배포 이미지에서 빼고 싶으면 팀 `.dockerignore`에 `experiments/` 한 줄
