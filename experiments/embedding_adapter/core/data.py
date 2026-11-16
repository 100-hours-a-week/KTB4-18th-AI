"""
실험 데이터 불러오기 (모두 읽기 전용)

  카탈로그   곡 벡터 (N, 3072) + 곡 메타. 행 번호 = 곡 번호
  데이터셋   {"config", "cold_tracks", "records"}, record = {pid, split, query, title_key, pos_warm, pos_cold}
  검색어 벡터  query_cache.db에서 sha256(모델\n문장)으로 찾는다. 캐시에 없는 검색어는 건너뛴다

Bench = 실험 하나에 필요한 것 묶음: 카탈로그, train 검색어(μ_q·허브 벌점용), 평가 split, 정답 목록, 곡 성질(kpop·비인기곡·cold)
"""
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config
from core.safe_io import connect_readonly


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def cache_key(text: str) -> str:
    return hashlib.sha256(f"{config.EMBED_MODEL}\n{text.strip()}".encode()).hexdigest()


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_catalog(dataset: str) -> tuple[np.ndarray, list[dict], dict]:
    """(곡 벡터 (N, D), 곡 메타, {track_id: 행 번호})"""
    _, cat = config.dataset_paths(dataset)
    vecs = np.load(f"{cat}.npy")
    meta = load_json(Path(f"{cat}.json"))
    return vecs, meta, {t["track_id"]: i for i, t in enumerate(meta)}


def load_dataset(dataset: str) -> dict:
    ds, _ = config.dataset_paths(dataset)
    return load_json(ds)


def embed_lookup(texts: list[str]) -> list[np.ndarray | None]:
    """문장마다 캐시된 임베딩 (없으면 None). 원래 실험의 캐시를 먼저, 없으면 7_embed_queries가 새로 만든 캐시에서"""
    caches = [config.QUERY_CACHE] + [p for p in [config.RUNS_DIR / "caches" / "query_cache.db"] if p.exists()]
    cons = [connect_readonly(p) for p in caches]
    out = []
    for t in texts:
        row = None
        for con in cons:
            row = con.execute("SELECT vec FROM emb WHERE key = ?", (cache_key(t),)).fetchone()
            if row is not None:
                break
        out.append(None if row is None else np.frombuffer(row[0], dtype=np.float32))
    for con in cons:
        con.close()
    return out


def load_split(ds: dict, split: str, row_of: dict) -> tuple[np.ndarray, list[dict]]:
    """(검색어 벡터 (Q, D), 레코드). 정답 track_id를 카탈로그 행 번호(warm_rows, cold_rows)로 바꿔 둔다"""
    records = [r for r in ds["records"] if r["split"] == split]
    vecs = embed_lookup([r["query"] for r in records])
    kept = [({**r, "warm_rows": [row_of[t] for t in r["pos_warm"]], "cold_rows": [row_of[t] for t in r["pos_cold"]]}, v)
            for r, v in zip(records, vecs) if v is not None]
    if len(kept) < len(records):
        print(f"  [{split}] 캐시에 없는 검색어 {len(records) - len(kept):,}개는 건너뜀")
    return np.stack([v for _, v in kept]), [r for r, _ in kept]


def load_evaltxt_queries() -> tuple[list[str], np.ndarray]:
    """eval.txt 사용자 말투 요청 중 추천 분기로 분류된 것의 recommend_query와 임베딩 (라벨 없음)"""
    con = connect_readonly(config.SERVICE_QUERY_DB)
    texts = [q for q, res in con.execute("SELECT recommend_query, result FROM evaltxt")
             if json.loads(res)["intent"] == "recommend"]
    con.close()
    vecs = embed_lookup(texts)
    assert all(v is not None for v in vecs), "eval.txt 검색어 중 임베딩 캐시에 없는 것이 있다"
    return texts, np.stack(vecs)


@dataclass
class Bench:
    """한 데이터셋·한 평가 split에서 시스템을 채점하는 데 필요한 모든 것"""
    dataset: str
    split: str
    S_raw: np.ndarray            # 카탈로그 곡 벡터 (원래 Gemini, 19,027 × 3072)
    meta: list[dict]
    row_of: dict
    Qt_raw: np.ndarray           # train 검색어 전부 (μ_q, 허브 벌점 r(s)의 기준)
    train_records: list[dict]
    Q_raw: np.ndarray            # 평가 split 검색어
    records: list[dict]
    cold: set                    # cold 곡 행 번호 (학습에서 숨긴 곡)
    is_kpop: np.ndarray          # 행마다 kpop 여부 (seed 아티스트 기준 근사)
    tail: set                    # 비인기곡 행 번호 (5-7에서 추가 임베딩한 곡)
    melon_rows: set              # 멜론과 매칭된 행 번호
    all_pos: list = field(init=False)    # 검색어별 정답 (warm + cold)
    cold_pos: list = field(init=False)   # 검색어별 cold 정답
    warm_ign: list = field(init=False)   # cold 채점 때 순위에서 뺄 warm 정답

    def __post_init__(self):
        self.all_pos = [r["warm_rows"] + r["cold_rows"] for r in self.records]
        self.cold_pos = [r["cold_rows"] for r in self.records]
        self.warm_ign = [r["warm_rows"] for r in self.records]

    @property
    def label_kpop(self) -> float:
        """정답 곡의 kpop 비율 (top-10 kpop 비율을 비교할 기준)"""
        return float(np.mean([self.is_kpop[j] for p in self.all_pos for j in p]))


def load_bench(dataset: str = config.DEFAULT_DATASET, split: str = "val", allow_test: bool = False) -> Bench:
    """test는 마지막에 한 번만 연다 → allow_test=True를 명시해야 열린다"""
    if split == "test" and not allow_test:
        raise SystemExit("test는 studies/6_final_test.py에서만, allow_test=True로 연다")
    S_raw, meta, row_of = load_catalog(dataset)
    ds = load_dataset(dataset)
    Qt_raw, train_records = load_split(ds, "train", row_of)
    Q_raw, records = load_split(ds, split, row_of)
    mapping = load_json(config.EMB_DATA_DIR / ds["config"]["mapping"])
    return Bench(
        dataset=dataset, split=split, S_raw=S_raw, meta=meta, row_of=row_of,
        Qt_raw=Qt_raw, train_records=train_records, Q_raw=Q_raw, records=records,
        cold={row_of[t] for t in ds["cold_tracks"]},
        # NOTE: 비인기곡은 bucket이 melon_tail이라 kpop 아님으로 센다
        is_kpop=np.array([t["bucket"] == "kpop" for t in meta]),
        tail={i for i, t in enumerate(meta) if t.get("source") == "candidates"},
        melon_rows={row_of[t] for t in mapping.values()},
    )
