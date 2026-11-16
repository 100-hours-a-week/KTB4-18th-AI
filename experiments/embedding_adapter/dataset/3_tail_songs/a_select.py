"""
3a. 비인기곡 추가 후보 고르기 (로컬 계산, 외부 호출 없음)

목적      정답의 95%가 멜론 인기 상위 10% 곡이라, 어댑터가 "인기곡 쪽으로 당기기"만 배운다 → 정답을 비인기곡 쪽으로 넓힌다 (5-7)
입력      MELON_DIR/train.json, song_meta.json, 매칭 (기본 config.MAPPING_NO_ALIAS = 2_match_melon 옵션 없이)
출력      runs/{날짜}_3a_tail_select/tail_candidates.json  [{melon_id, tier, n_playlists, title, artist, album}, ...]
규칙      후보 = 쓸 수 있는 플레이리스트(매칭 3곡 이상) 안에 있는데 카탈로그에 없는 멜론 곡
          인기도 구간 (멜론 train.json에서 곡이 등장한 플레이리스트 수의 순위)
            10~50%   학습 신호 (곡당 플레이리스트 7~13개)
            하위50%  가장 검색어 특유한 신호 (곡당 1~2개)
          구간마다 쓸 수 있는 플레이리스트에 많이 나오는 순서로, iTunes 성공률(약 50%)을 감안해 목표의 2배를 뽑는다
          버전 표기 곡, 아티스트 없는 곡, 서비스에 넣지 않기로 한 아티스트는 뺀다
재현      원래 tail_candidates.json과 같은 파일

사용:
  python -m dataset.3_tail_songs.a_select
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import config
from core.safe_io import new_run_dir, write_json
from dataset.match_rules import artist_keys, title_key

MIN_POSITIVES = 3
EXCLUDED_ARTISTS = {"승리", "seungri"}   # 서비스에 넣지 않기로 한 아티스트 (artist_keys 기준)
TIERS = [  # (이름, 순위 하한, 순위 상한, 뽑을 곡 수 = 목표의 2배)
    ("10~50%", 0.10, 0.50, 6000),
    ("하위50%", 0.50, 1.01, 2000),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", type=Path, default=config.MAPPING_NO_ALIAS)
    args = ap.parse_args()

    playlists = json.load(open(config.MELON_DIR / "train.json", encoding="utf-8"))
    songs = {s["id"]: s for s in json.load(open(config.MELON_DIR / "song_meta.json", encoding="utf-8"))}
    melon_to_track = {int(k): v for k, v in json.load(open(args.mapping)).items()}

    # 멜론 인기도: 곡이 등장한 플레이리스트 수의 순위 (0 = 가장 인기, 1 = 가장 비인기)
    freq = Counter(s for p in playlists for s in set(p["songs"]))
    rank = {s: i / len(freq) for i, (s, _) in enumerate(freq.most_common())}

    usable = [p for p in playlists
              if len({melon_to_track[s] for s in p["songs"] if s in melon_to_track}) >= MIN_POSITIVES]
    in_usable = Counter(s for p in usable for s in set(p["songs"]) if s not in melon_to_track)
    print(f"쓸 수 있는 플레이리스트 {len(usable):,}개, 그 안의 미매칭 곡 {len(in_usable):,}개")

    def eligible(s: int) -> bool:
        basket = songs[s]["artist_name_basket"]
        return bool(basket) and bool(title_key(songs[s]["song_name"])) \
            and not set(artist_keys(basket[0])) & EXCLUDED_ARTISTS

    candidates = []
    for tier, lo, hi, n in TIERS:
        picked = [s for s, _ in in_usable.most_common() if lo <= rank[s] < hi and eligible(s)][:n]
        for s in picked:
            m = songs[s]
            candidates.append({"melon_id": s, "tier": tier, "n_playlists": in_usable[s], "title": m["song_name"],
                               "artist": m["artist_name_basket"][0], "album": m["album_name"]})
        print(f"  {tier:7s} {len(picked):,}곡 (플레이리스트 등장 {in_usable[picked[0]]}~{in_usable[picked[-1]]}회)")

    out = write_json(new_run_dir("3a_tail_select") / "tail_candidates.json", candidates, ensure_ascii=False, indent=1)
    print(f"저장: {out} ({len(candidates):,}곡)")


if __name__ == "__main__":
    main()
