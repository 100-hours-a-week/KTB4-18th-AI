"""
경로·이름표·최종 설정값을 한 곳에 모은다

경로 (환경 변수로 바꿀 수 있다)
  EMB_DATA_DIR  실험 데이터 폴더: 카탈로그(.npy/.json), 데이터셋(.json), query_cache.db, adapters/*.pt  — 읽기 전용
  SDD_DIR       Song Describer Dataset 평가 폴더 (data/song_describer.csv, emb/gemini_h30.*.npy)       — 읽기 전용
  MELON_DIR     멜론 Playlist Dataset 원본 (train.json, song_meta.json)                               — 읽기 전용
  KTB4_ROOT     서비스 저장소. classify()·DB를 빌려 쓰는 데이터 준비 스크립트만 필요
  RUNS_DIR      이 저장소가 결과를 쓰는 유일한 곳 (core/safe_io.py가 다른 곳에 쓰는 것을 막는다)

이름표
  코드에서는 뜻 있는 이름(service_format, query_adapter_r512 …)을 쓰고,
  디스크의 원래 파일명(dataset_v4, stage4_r512_ds4_q3_base_e5 …)은 여기서만 연결한다.
  원래 파일명을 바꾸지 않는 이유: 실험 로그·문서가 그 이름을 가리킨다.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_EMB_TEST = Path.home() / "Desktop" / "emb_test"

EMB_DATA_DIR = Path(os.environ.get("EMB_DATA_DIR", _EMB_TEST / "adapter" / "data"))
SDD_DIR = Path(os.environ.get("SDD_DIR", _EMB_TEST / "sdd_eval"))
MELON_DIR = Path(os.environ.get("MELON_DIR", Path.home() / "Desktop" / "melon"))
# 팀 저장소 안(experiments/embedding_adapter/)에 있으면 그 저장소, 아니면 바탕화면의 팀 저장소
_TEAM_REPO = ROOT.parents[1] if (ROOT.parents[1] / "backend").is_dir() else \
    Path.home() / "Desktop" / "team_project_18th" / "KTB4-18th-AI"
KTB4_ROOT = Path(os.environ.get("KTB4_ROOT", _TEAM_REPO))
RUNS_DIR = Path(os.environ.get("RUNS_DIR", ROOT / "runs"))

ADAPTER_DIR = EMB_DATA_DIR / "adapters"
QUERY_CACHE = EMB_DATA_DIR / "query_cache.db"          # 검색어 임베딩 캐시 (키 = sha256(모델\n문장))
SERVICE_QUERY_DB = EMB_DATA_DIR / "service_query.db"   # 멜론 제목 → classify() 결과, eval.txt 변환 결과
EMBED_MODEL = "google/gemini-embedding-2"
DIM = 3072

# ── 데이터셋 ──────────────────────────────────────────────────
# 이름 → (데이터셋 파일, 카탈로그 파일). 둘 다 EMB_DATA_DIR 아래, 확장자 없이
DATASETS = {
    # 검색어 = 멜론 제목 → 서비스 classify() → recommend_query (추천 분기만). 문서 (2)의 최종 데이터셋
    "service_format": ("dataset_v4", "catalog_v3_2026-10-15"),
    # 같은 재생목록·분할·정답, 검색어만 멜론 원문 "제목 | 태그". 5-8 비교용
    "melon_format": ("dataset_v3", "catalog_v3_2026-10-15"),
}
DEFAULT_DATASET = "service_format"

# ── 데이터 준비 (dataset/)의 원래 산출물. 스크립트 입력 기본값이자 재현 확인 대상 ──
SERVICE_CATALOG = EMB_DATA_DIR / "catalog_2026-10-13"            # 서비스 DB tracks만 (15,102곡). 멜론 매칭 기준
MAPPING_NO_ALIAS = EMB_DATA_DIR / "melon_to_track_v2.json"       # 매칭 (제목 별칭 없이). 비인기곡 후보 고르기에 썼다
MAPPING = EMB_DATA_DIR / "melon_to_track_v3.json"                # 매칭 (제목 별칭 + HTML 디코딩)
MAPPING_WITH_TAIL = EMB_DATA_DIR / "melon_to_track_v3_tail.json"  # 매칭 + 비인기곡 3,925곡
TAIL_CANDIDATES = EMB_DATA_DIR / "tail_candidates.json"
PREVIOUS_DATASET = EMB_DATA_DIR / "dataset_v2.json"              # 이전 분할. 이 val·test 제목은 test에 넣지 않는다

# ── 체크포인트 ────────────────────────────────────────────────
# 이름 → ADAPTER_DIR 아래 .pt 파일명. 모두 service_format train으로 5에폭 학습, 마지막 시점
CHECKPOINTS = {
    # 쿼리 어댑터 (train/train_query_adapter.py)
    "query_adapter_r32": "stage4_r32_ds4_q3_base_e5",
    "query_adapter_r128": "stage4_r128_ds4_q3_base_e5",
    "query_adapter_r512": "stage4_r512_ds4_q3_base_e5",
    "query_adapter_r3072": "stage4_r3072_ds4_q3_base_e5",
    "query_adapter_r512_tail_pos_only": "stage4_r512_ds4_q3_tailpos_e5",   # 5-7 C: 비인기곡 = 정답 전용
    # 5-8: melon_format train으로 학습한 r512 (μ_q도 멜론 형식 검색어 평균)
    "query_adapter_r512_melon_format": "stage4_r512_ds3_q3_base_e5",
    # 위의 A·B + service_format의 μ_q. 서비스 형식 검색어로 평가할 때 μ_q 차이를 빼고 어댑터만 비교하려고
    "query_adapter_r512_melon_trained": "stage4_r512_ds3_q3_base_e5_mutitle",
    # 쿼리·곡 어댑터 (train/train_two_tower.py). lam = 곡-곡 손실 가중치 λ, init = 쿼리 어댑터를 r512에서 시작
    "two_tower_lam0": "stage14_q512_s512_v4_lam0",
    "two_tower_lam0_init": "stage14_q512_s512_v4_lam0_init",
    "two_tower_lam025": "stage14_q512_s512_v4_lam025",
    "two_tower_lam05": "stage14_q512_s512_v4_lam05",
    "two_tower_lam05_init": "stage14_q512_s512_v4_lam05_init",
    "two_tower_lam1": "stage14_q512_s512_v4_lam1",
}

# ── 선택 절차 (결과를 보기 전에 고정한 것) ─────────────────────
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)   # 평균 제거 쿼리를 섞는 비율 (5-2)
BETAS = (0.0, 0.25, 0.5, 0.75)         # 허브 벌점 세기 (5-5)
HUB_K = 10                             # r(s) = 가장 가까운 train 검색어 k개의 평균 코사인
GAMMAS = (0.0, 0.25, 0.5, 0.75, 1.0)   # 곡 쪽에서 평균 제거 곡을 섞는 비율 (5-9)
N_BOOT, BOOT_SEED = 1000, 0            # paired bootstrap

# ── 최종 시스템 (문서 (6)) ────────────────────────────────────
FINAL_QUERY_ADAPTER = {"checkpoint": "query_adapter_r512", "alpha": 0.25, "beta": 0.5}
FINAL_TWO_TOWER = {"checkpoint": "two_tower_lam05", "alpha": 0.5, "beta": 0.5, "gamma": 0.25}


def dataset_paths(name: str) -> tuple[Path, Path]:
    """(데이터셋 .json, 카탈로그 경로 접두사 — .npy/.json을 붙여 쓴다)"""
    ds, cat = DATASETS[name]
    return EMB_DATA_DIR / f"{ds}.json", EMB_DATA_DIR / cat


def checkpoint_path(name: str) -> Path:
    """이름표의 이름, 또는 .pt 파일 경로 (새로 학습한 runs/…/checkpoint.pt)"""
    if name.endswith(".pt"):
        return Path(name)
    return ADAPTER_DIR / f"{CHECKPOINTS[name]}.pt"
