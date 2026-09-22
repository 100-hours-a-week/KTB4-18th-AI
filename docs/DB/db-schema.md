# DB 스키마

AI 파트가 사용하는 테이블 두 개. 백엔드 도메인 테이블(사용자, 플레이리스트 등)과는 겹치지 않는다.

- PostgreSQL + **pgvector 확장 필요** (`CREATE EXTENSION vector`)
- 스키마는 `db/models.py`에 정의되어 있고 `scripts/load_db.py` 실행 시 자동 생성된다

---

## `tracks`

곡 하나가 한 행. 메타데이터와 임베딩 벡터가 같은 행에 있어 서로 어긋날 수 없다.

| 컬럼 | 타입 | NULL | 설명 |
|---|---|---|---|
| `track_id` | BIGINT PK | X | **iTunes trackId.** 곡을 식별하는 유일한 키. 프론트·백엔드·AI가 모두 이 값으로 곡을 지칭한다 |
| `title` | TEXT | X | 곡 제목 |
| `artist` | TEXT | X | 아티스트명. 협업곡은 `A & B` 형태로 들어온다 |
| `album` | TEXT | O | 앨범명 |
| `genre` | VARCHAR(64) | O | iTunes `primaryGenreName`. 아래 주의 참고 |
| `release_date` | DATE | O | 발매일. 컴필레이션 수록곡은 원곡이 아닌 컴필레이션 발매일일 수 있다 |
| `preview_url` | TEXT | X | **30초 미리듣기 주소 (Apple CDN).** 재생에 사용 |
| `artwork_url` | TEXT | O | 앨범 아트 주소 (100x100) |
| `store_url` | TEXT | O | **Apple Music 곡 페이지 주소.** 프로모션 자산 사용 시 스토어 링크 표시가 필요할 수 있어 함께 보관한다 |
| `duration_ms` | INTEGER | O | 곡 전체 길이(밀리초). 임베딩에 쓰는 미리듣기 30초와는 별개다 |
| `seed_artist` | TEXT | O | 수집 시 사용한 아티스트명. 디버깅용이며 서비스 로직에서는 쓰지 않는다 |
| `bucket` | VARCHAR(32) | O | 수집 분류 (`kpop`, `pop_hiphop_rnb`, `electronic_rock`). 코퍼스 구성 점검용 |
| `mood_tags` | JSONB | O | **분위기 태그.** 아래 참고 |
| `emb_gemini` | vector(3072) | **O** | **현재 추천 검색에 쓰는 임베딩** (gemini-embedding-2, 오디오 30초 미리듣기). NULL이면 아직 처리되지 않은 곡 |
| `emb_clap` | vector(512) | **O** | 이전 세대 임베딩. 더 이상 검색에 쓰이지 않고 보관용/롤백용으로만 남아 있다. NULL이면 아직 처리되지 않은 곡 |
| `model_version` | VARCHAR(128) | O | `emb_clap`을 생성한 모델. 모델 교체 시 재임베딩 대상을 골라내는 데 쓴다(`emb_gemini`는 별도 관리, 아래 참고) |
| `embedded_at` | TIMESTAMP | O | 임베딩 시각 |
| `fail_count` | INT | X | 임베딩 실패 횟수. 3회 이상이면 배치가 더 시도하지 않는다 |
| `fail_reason` | TEXT | O | 마지막 실패 사유 |

---

## 백엔드가 알아야 할 것

### 1. `emb_gemini IS NULL` 이 곧 처리 대기열이다

사용자가 카탈로그에 없는 곡을 검색하면 **메타데이터만 먼저 INSERT**하고 벡터는 비워둔다. 배치가 채운다.

```sql
-- 신규 곡 등록 (벡터 없이)
INSERT INTO tracks (track_id, title, artist, album, genre, release_date,
                    preview_url, artwork_url, store_url, duration_ms)
VALUES (...)
ON CONFLICT (track_id) DO NOTHING;
```

`ON CONFLICT DO NOTHING`을 반드시 넣는다. 같은 곡을 여러 사용자가 검색할 수 있다.

**추천 대상 조회 시에는 반드시 emb_gemini의 NULL을 제외한다** (emb_clap이 아니다 — 현재 검색은 emb_gemini 기준이다).

```sql
SELECT ... FROM tracks WHERE emb_gemini IS NOT NULL ...
```

이 조건이 "오늘 등록된 곡은 임베딩된 다음부터 추천에 나온다"를 구현한다. 별도 큐 테이블이 없다.

`emb_clap`도 같은 NULL=대기열 패턴을 쓰지만, 지금은 어떤 조회도 이 컬럼을 기준으로 필터링하지 않는다 — 검색 경로에서 완전히 빠졌다.

### 2. 곡 상세 조회

```sql
SELECT track_id, title, artist, album, preview_url, artwork_url, mood_tags
FROM tracks WHERE track_id = ANY($1);
```

추천 API는 `track_id` 목록과 점수를 반환한다. 곡 정보의 소유자는 이 테이블 하나이므로, 다른 곳에 제목·아티스트를 복사해 두지 않는다.

### 3. 오디오와 이미지는 우리가 서빙하지 않는다

`preview_url`과 `artwork_url`은 Apple CDN 주소다. 프론트가 직접 로드한다.

```html
<audio src={preview_url} />
<img src={artwork_url} />
```

우리 서버를 거치지 않으므로 스토리지·전송 비용이 발생하지 않는다. 오디오 파일은 어디에도 저장하지 않는다.

**주의**: `preview_url`은 영구 주소가 아니다. 실측에서 15,000곡 중 5곡이 HTTP 403을 반환했다(0.03%). 재생 실패 처리가 필요하다.

### 4. `mood_tags` 형식

곡의 분위기를 나타내는 태그와 확률. 확률 내림차순, 최대 10개, 0.05 미만은 제외된다.

```json
{"relaxing": 0.71, "melancholic": 0.64, "calm": 0.58}
```

오디오에서 추출한 값이며 사람이 붙인 라벨이 아니다. 용도는 두 가지다.

- **필터**: `WHERE mood_tags ? 'relaxing'` 또는 `WHERE (mood_tags->>'calm')::float > 0.5`
- **추천 이유 생성**: LLM에 넘겨 설명 문장을 만든다

태그 어휘는 56개로 고정되어 있다(MTG-Jamendo mood/theme). `sad`, `energetic`, `dark`, `epic`, `dream`, `film` 등.

### 5. `genre` 는 신뢰도가 낮다

iTunes가 붙인 라벨이며 일관성이 없다. 한국 곡 대부분이 `K-Pop`으로 뭉뚱그려지고, 밴드 음악이 `Rock`이나 `Worldwide`로 찍히기도 한다.

**장르로 곡의 성격을 판단하지 않는다.** 표시용, 그리고 사용자가 명시적으로 장르를 지정했을 때의 필터로만 쓴다. 곡의 실제 성격은 `emb_gemini`와 `mood_tags`가 담는다.

물론 자연어 쿼리가 장르를 요구한다면 이용할 여지도 있다.

### 6. 같은 곡이 여러 `track_id` 로 존재할 수 있다

컴필레이션 앨범 재수록, 싱글/앨범 중복 등. 수집 단계에서 대부분 제거했으나 협업 표기가 다른 경우(`ROSÉ` vs `ROSÉ & Bruno Mars`)는 남는다.

추천 API는 응답 전에 `(artist, title)` 기준으로 중복을 제거한다. 백엔드에서 직접 곡을 조회해 목록을 만든다면 같은 처리가 필요하다.

---

## `corpus_stats`

전역 상태 저장용. 현재는 벡터 보정에 쓰는 평균값 한 행뿐이다.

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `key` | VARCHAR(32) PK | 현재 `clap_mean` 하나 |
| `value` | vector(512) | 코퍼스 평균 벡터 |
| `updated_at` | TIMESTAMP | 갱신 시각 |

임베딩 공간 중앙에 위치한 곡이 아무 질의에나 상위로 올라오는 현상을 완화하기 위해, 저장 시점에 이 평균을 뺀 벡터를 기록한다(결과 다양성 85.6% → 92.2%).

**AI 파트 내부용이므로 백엔드가 직접 읽거나 쓸 일은 없다.** 코퍼스가 크게 늘면 재계산 후 전량 갱신이 필요하다.

**주의**: 이 중심화는 `emb_clap` 적재 시점에만 적용되고, 지금 검색이 쓰는 `emb_gemini`에는 해당하지 않는다. `emb_clap`이 검색 경로에서 빠지면서 이 테이블도 사실상 휴면 상태다.

---

## 규모

| 항목 | 현재 (약 15,000곡) | 100,000곡 기준 |
|---|---:|---:|
| `emb_clap` (512차원) | 31MB, 15,226/15,226곡 채워짐 | 205MB |
| `emb_gemini` (3072차원) | 60MB, **4,879/15,226곡만 채워짐** (15,226곡 전량이면 약 187MB) | 약 1.2GB |
| 메타데이터 + 태그 | 약 20MB | 130MB |
| 오디오·이미지 | **0** | **0** |

현 규모에서는 **벡터 인덱스(HNSW 등)가 불필요하다.** 전수 스캔이 밀리초 단위이며 ANN 인덱스는 근사값을 반환하므로 오히려 정확도 손해다. 게다가 `emb_gemini`는 3072차원이라 pgvector의 ivfflat/hnsw 인덱스(최대 2000차원)를 애초에 걸 수도 없다. 100,000곡을 넘어가면 인덱스든 차원 축소든 그때 검토한다.

쓰기는 초기 적재와 배치뿐이고, 이후는 읽기 전용에 가깝다.

---

## 결정: 검색 임베딩을 gemini-embedding-2로 교체

과거엔 자연어 질의를 CLAP 텍스트 인코더로 임베딩할지, CLAP을 API 서버에 넣을지 야간 배치에 남길지가 미해결이었다. 아래 두 선택지 중 2번으로 결정됐다.

1. ~~CLAP 텍스트 인코더를 API 서버에 직접 넣는다~~ (미채택)
2. **오디오/텍스트를 함께 받는 외부 임베딩 API를 쓴다** — `gemini-embedding-2` 채택

`gemini-embedding-2`는 오디오(30초 미리듣기)와 텍스트가 같은 벡터 공간에 놓이고 다국어를 지원해서, 한국어 질의 원문을 변환 없이 그대로 임베딩해도 검색이 된다(기존엔 CLAP이 영어 전용이라 "한국어 → 영어 소리 서술" LLM 변환 단계가 필수였는데, 이 단계가 없어졌다). API 서버는 `google-genai` SDK로 `embed_content` 호출 한 번만 하면 되므로, 우려했던 `torch`/`laion-clap` 등 무거운 ML 라이브러리를 API 이미지에 넣을 필요가 없어졌다 — `worker` 그룹 분리 구성을 그대로 유지할 수 있다.

CLAP(`emb_clap`, `batch/embedder.py`)은 폐기되지 않았다 — `mood_tags`(분위기 태그)를 뽑는 EffNet 백본이 여기 있어서, 신곡이 들어올 때 태그를 채우는 배치가 여전히 이 경로를 쓴다. 다만 검색(`db/search.py`)은 더 이상 `emb_clap`을 보지 않는다.

현재 `emb_gemini`는 예산 문제로 15,226곡 중 4,879곡만 채워져 있다(위 "규모" 표 참고). 나머지는 `emb_gemini IS NULL` 상태로 대기열에 있으며, 예산이 허락하는 대로 점진적으로 채워나갈 예정이다.