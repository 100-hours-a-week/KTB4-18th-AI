"""
SDD 회귀 테스트: 어댑터가 범용 음악 검색(영어 캡션 → 곡)을 망가뜨리지 않았나

Song Describer Dataset — 캡션 1,106개 → 곡 706개, valid: 캡션 746개 → 547곡. 멜론·학습과 무관한 곡과 문장
SDD 곡은 학습에 한 번도 안 나온 곡이라 곡 어댑터에게는 전부 처음 보는 곡(cold)이다

임베딩은 SDD_DIR/emb/gemini_h30.{audio,text}.npy (이미 계산해 둔 것, 비용 0)
NOTE: sdd_eval/common.evaluate()는 안에서 다시 정규화해서 허브 벌점 칸을 망가뜨린다 → 같은 규칙으로 직접 채점
      (텍스트 → 오디오, 등수 = 정답보다 점수가 높은 곡 수 + 1, 동점 유리)
NOTE: sdd_eval/common.py는 import하면 emb/ 폴더를 만들려 해서 import하지 않고 csv만 읽는다
"""
import csv
from dataclasses import dataclass

import numpy as np

import config
from core.systems import Space

EMB_NAME = "gemini_h30"


@dataclass
class SDD:
    audio: np.ndarray   # (706, 3072) 곡, track_id 오름차순
    text: np.ndarray    # (1106, 3072) 캡션, csv 순서
    captions: list[dict]
    tracks: list[str]


def load_sdd() -> SDD:
    with open(config.SDD_DIR / "data" / "song_describer.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    caps = [{"track_id": r["track_id"], "valid": r["is_valid_subset"] == "True"} for r in rows]
    tracks = sorted({c["track_id"] for c in caps}, key=int)
    emb = config.SDD_DIR / "emb"
    return SDD(np.load(emb / f"{EMB_NAME}.audio.npy"), np.load(emb / f"{EMB_NAME}.text.npy"), caps, tracks)


def ranks(space: Space, sdd: SDD, subset: str = "valid") -> np.ndarray:
    """캡션마다 정답 곡의 등수. space.Q = 캡션, space.S = SDD 곡 (이미 시스템 공간, 다시 정규화하지 않는다)"""
    tidx = {t: i for i, t in enumerate(sdd.tracks)}
    if subset == "valid":
        qi = [i for i, c in enumerate(sdd.captions) if c["valid"]]
        gal = sorted({tidx[sdd.captions[i]["track_id"]] for i in qi})
    else:
        qi, gal = list(range(len(sdd.captions))), list(range(len(sdd.tracks)))
    gpos = {g: k for k, g in enumerate(gal)}
    S = space.Q[qi] @ space.S[gal].T
    gold = np.array([gpos[tidx[sdd.captions[i]["track_id"]]] for i in qi])
    return (S > S[np.arange(len(qi)), gold][:, None]).sum(axis=1) + 1


def score(space: Space, sdd: SDD, subset: str = "valid") -> dict:
    r = ranks(space, sdd, subset)
    return {"R@1": 100 * np.mean(r <= 1), "R@10": 100 * np.mean(r <= 10), "MedR": float(np.median(r)),
            "per_query": (r <= 10).astype(float) * 100}
