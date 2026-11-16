"""
2. 멜론 곡 ↔ 서비스 카탈로그 곡 매칭 (정규화 후 정확 매칭, 로컬 계산)

입력      MELON_DIR/song_meta.json, train.json (멜론 Playlist Dataset)
          서비스 카탈로그 메타 (기본 config.SERVICE_CATALOG = 1_export_catalog 결과, 15,102곡)
출력      runs/{날짜}_2_match_melon/melon_to_track.json   {멜론 곡 id: track_id}
          화면: 매칭률(버킷별), 플레이리스트별 매칭 곡 수, 사람 검수용 무작위 샘플 40개
규칙      키 = (정규화한 제목, 정규화한 첫 번째 아티스트)
          제목: 괄호 안에 버전 표기(Remix, Inst. 등)가 있으면 소리가 다른 곡이라 매칭하지 않는다.
                feat., Explicit 같은 표기만 있으면 괄호를 지우고 비교한다. 라이브 앨범 수록곡도 뺀다
          아티스트: 멜론은 "에픽하이 (EPIK HIGH)"처럼 쓰므로 괄호 밖/안 이름을 모두 키로.
                    이름이 달라도(SEVENTEEN ↔ 세븐틴) 드문 제목이 3곡 이상 겹치면 같은 아티스트로 본다
          같은 키에 카탈로그 곡이 여럿이면 괄호까지 제목이 같은 곡, 없으면 track_id가 가장 낮은 곡
          --title_aliases (최종 데이터셋에 쓴 것):
            + 괄호 속 번역 제목 별칭 "으르렁 (Growl)" ↔ "Growl" (멜론은 "한글 (English)", iTunes는 영어만 적는 K-pop이 많다)
              원래 제목 키로 먼저 찾고, 없을 때만 별칭으로 (아티스트는 그대로 일치해야 한다)
            + 멜론 제목·아티스트의 HTML 코드 디코딩 ("Can&#39;t" → "Can't")
재현      --title_aliases → 원래 melon_to_track_v3.json, 옵션 없이 → melon_to_track_v2.json 과 같은 파일

사용:
  python -m dataset.2_match_melon --title_aliases
"""
import argparse
import html
import json
import random
import re
from collections import Counter
from pathlib import Path

import config
from core.safe_io import new_run_dir, write_json
from dataset.match_rules import HANGUL, artist_keys, norm, title_aliases, title_key

ALIAS_MIN_TITLES = 3            # 아티스트 별칭: 겹치는 제목이 이 이상
RARE_TITLE_MAX_ARTISTS = 5      # "Love"처럼 흔한 제목은 세지 않는다 (멜론 아티스트 5명 이하가 쓰는 제목만)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title_aliases", action="store_true", help="괄호 속 번역 제목 별칭 + HTML 디코딩 (최종 데이터셋)")
    ap.add_argument("--catalog", type=Path, default=Path(f"{config.SERVICE_CATALOG}.json"), help="카탈로그 메타 json")
    args = ap.parse_args()
    use_aliases = args.title_aliases

    # 1. 카탈로그 (Gemini 임베딩 있는 곡만, track_id 오름차순)
    # NOTE: 같은 키에 곡이 여럿일 수 있다 ("One Call Away"와 "One Call Away (Piana-pella)").
    #   dict에 덮어쓰면 track_id가 가장 큰 곡이 남으므로 후보를 모두 모아 두고 pick()에서 고른다
    catalog = json.load(open(args.catalog, encoding="utf-8"))
    catalog_by_key = {}
    for t in catalog:
        tk = title_key(t["title"])
        if tk is None:
            continue
        for ak in artist_keys(t["artist"]):
            catalog_by_key.setdefault((tk, ak), []).append(t)
    n_collide = sum(len(v) > 1 for v in catalog_by_key.values())
    print(f"카탈로그: {len(catalog):,}곡 / 키 {len(catalog_by_key):,}개 (곡이 2개 이상인 키 {n_collide}개)")

    # 1-0. 카탈로그 제목의 괄호 속 별칭 키. 원래 키와 따로 둬서 원래 키를 먼저 찾게 한다
    catalog_by_alias = {}
    if use_aliases:
        for t in catalog:
            if title_key(t["title"]) is None:
                continue
            for alias in title_aliases(t["title"]):
                for ak in artist_keys(t["artist"]):
                    catalog_by_alias.setdefault((norm(alias), ak), []).append(t)
        print(f"카탈로그 제목 별칭 키 {len(catalog_by_alias):,}개")

    def pick(candidates: list[dict], melon_title: str) -> str:
        """같은 키의 카탈로그 곡 중: 괄호까지 멜론 제목과 같은 곡 → 없으면 track_id가 가장 낮은 곡"""
        exact = [t for t in candidates if norm(t["title"]) == norm(melon_title)]
        return (exact or candidates)[0]["track_id"]

    songs = json.load(open(config.MELON_DIR / "song_meta.json", encoding="utf-8"))
    # NOTE: 멜론 제목·아티스트에는 HTML 코드가 남아 있다 ("Can&#39;t" → norm 하면 "can39t" ≠ "cant").
    #   제목 11,567개, 아티스트 1,696개. 아포스트로피가 든 영어 제목이 전부 매칭에 실패했다
    if use_aliases:
        for s in songs:
            s["song_name"] = html.unescape(s["song_name"])
            s["artist_name_basket"] = [html.unescape(a) for a in s["artist_name_basket"]]

    # 1-1. 아티스트 별칭: 이름이 달라도 드문 제목이 3곡 이상 겹치면 같은 아티스트로 본다
    # NOTE: 목적은 한글/영어 표기 차이를 메우는 것. 겹침 1등이 한글 이름일 때만 인정한다 (영어면 이미 직접 매칭된다)
    melon_titles_by_artist = {}
    for s in songs:
        tk = title_key(s["song_name"])
        if tk is None or not s["artist_name_basket"]:
            continue
        for ak in artist_keys(s["artist_name_basket"][0]):
            melon_titles_by_artist.setdefault(ak, set()).add(tk)

    catalog_titles_by_artist = {}
    for t in catalog:
        tk = title_key(t["title"])
        if tk is not None:
            catalog_titles_by_artist.setdefault(t["artist"], set()).add(tk)

    melon_artists_by_title = {}
    for ak, tks in melon_titles_by_artist.items():
        for tk in tks:
            melon_artists_by_title.setdefault(tk, set()).add(ak)

    aliases = {}
    for artist, tks in catalog_titles_by_artist.items():
        # NOTE: set 순서는 실행마다 달라서(해시 무작위화) 동점 1등이 바뀐다. 정렬해서 매번 같게 한다
        rare_tks = [tk for tk in sorted(tks) if len(melon_artists_by_title.get(tk, ())) <= RARE_TITLE_MAX_ARTISTS]
        overlap = Counter(ak for tk in rare_tks for ak in sorted(melon_artists_by_title.get(tk, ())))
        if not overlap:
            continue
        best, n = overlap.most_common(1)[0]
        if n >= ALIAS_MIN_TITLES and HANGUL.search(best):
            aliases[artist] = (best, n)

    for t in catalog:
        tk = title_key(t["title"])
        if tk is None or t["artist"] not in aliases:
            continue
        candidates = catalog_by_key.setdefault((tk, aliases[t["artist"]][0]), [])
        if t not in candidates:
            candidates.append(t)
    print(f"제목 겹침으로 찾은 아티스트 별칭 {len(aliases)}개")

    # 2. 멜론 곡 → 카탈로그 track_id
    melon_to_track = {}
    via_alias = set()
    version_skipped = []
    for s in songs:
        if not s["artist_name_basket"]:
            continue
        tk = title_key(s["song_name"])
        # NOTE: 라이브 앨범 수록곡은 제목에 표기가 없어도 라이브 버전
        if tk is None or re.search(r"\blive\b", (s["album_name"] or "").lower()):
            version_skipped.append(s)
            continue
        aks = artist_keys(s["artist_name_basket"][0])
        # 찾는 순서: 원래 제목 키 → 멜론 별칭 ↔ 카탈로그 원래 키 → 멜론 원래 키 ↔ 카탈로그 별칭 → 별칭끼리
        lookups = [(catalog_by_key, [tk])]
        if use_aliases:
            melon_aliases = [norm(a) for a in title_aliases(s["song_name"])]
            lookups += [(catalog_by_key, melon_aliases), (catalog_by_alias, [tk]), (catalog_by_alias, melon_aliases)]
        for step, (table, title_keys) in enumerate(lookups):
            hit = next(((k, ak) for k in title_keys for ak in aks if (k, ak) in table), None)
            if hit:
                melon_to_track[s["id"]] = pick(table[hit], s["song_name"])
                if step > 0:
                    via_alias.add(s["id"])
                break

    out = write_json(new_run_dir("2_match_melon") / "melon_to_track.json", melon_to_track)
    matched_tracks = set(melon_to_track.values())
    print(f"멜론 곡 {len(songs):,}개 중 매칭 {len(melon_to_track):,}개 (버전 표기로 제외 {len(version_skipped):,}개, "
          f"제목 별칭으로만 매칭 {len(via_alias):,}개)")
    print(f"→ 카탈로그 {len(matched_tracks):,}곡이 멜론에 있음 ({len(matched_tracks) / len(catalog):.1%})")

    # 3. 버킷별 매칭률 (kpop이 해외 곡보다 크게 낮지 않은지)
    total = Counter(t["bucket"] for t in catalog)
    hit = Counter(t["bucket"] for t in catalog if t["track_id"] in matched_tracks)
    for b, n in total.most_common():
        print(f"  {str(b):18s} {hit[b]:5,d} / {n:5,d} ({hit[b] / n:.0%})")

    # 4. 플레이리스트별 매칭 곡 수
    playlists = json.load(open(config.MELON_DIR / "train.json", encoding="utf-8"))
    counts = [len({melon_to_track[x] for x in p["songs"] if x in melon_to_track}) for p in playlists]
    for k in (1, 3, 5, 10):
        n = sum(c >= k for c in counts)
        print(f"매칭 곡 {k:2d}개 이상 플레이리스트: {n:7,d}개 ({n / len(playlists):.1%})")

    # 5. 쓸 수 있는 플레이리스트(매칭 3곡 이상): 제목 다양성, 정답 곡 쏠림
    usable = [p for p, c in zip(playlists, counts) if c >= 3]
    print(f"\n쓸 수 있는 플레이리스트 {len(usable):,}개 → 서로 다른 제목 {len({norm(p['plylst_title']) for p in usable}):,}개")
    track_freq = Counter(tid for p in usable for tid in {melon_to_track[x] for x in p["songs"] if x in melon_to_track})
    freqs = sorted(track_freq.values(), reverse=True)
    print(f"정답으로 등장하는 곡 {len(freqs):,}개, 상위 100곡이 정답 등장의 {sum(freqs[:100]) / sum(freqs):.1%} 차지")

    # 6. 사람 검수용 무작위 샘플: kpop 15개 + 그 외 15개, 버전 표기로 제외된 멜론 곡 10개
    random.seed(0)
    melon_by_id = {s["id"]: s for s in songs}
    catalog_by_id = {t["track_id"]: t for t in catalog}
    kpop_ids = sorted(m for m, t in melon_to_track.items() if catalog_by_id[t]["bucket"] == "kpop")
    other_ids = sorted(m for m, t in melon_to_track.items() if catalog_by_id[t]["bucket"] != "kpop")
    print("\n[검수 샘플]  멜론 (제목 / 아티스트 / 앨범)  ↔  카탈로그 (제목 / 아티스트 / 앨범)")
    for i, melon_id in enumerate(random.sample(kpop_ids, 15) + random.sample(other_ids, 15), 1):
        m, t = melon_by_id[melon_id], catalog_by_id[melon_to_track[melon_id]]
        print(f"{i:2d}. {m['song_name']} / {', '.join(m['artist_name_basket'])} / {m['album_name']}")
        print(f"    {t['title']} / {t['artist']} / {t['album']}  [{t['bucket']}]")
    print(f"\n[아티스트 별칭 전체 {len(aliases)}개]  카탈로그 → 멜론 (겹친 제목 수)")
    print(", ".join(f"{a}→{b}({n})" for a, (b, n) in sorted(aliases.items(), key=lambda x: x[1][1])))
    print("\n[버전 표기로 제외된 멜론 곡 샘플]")
    for s in random.sample(version_skipped, 10):
        print(f"  {s['song_name']} / {', '.join(s['artist_name_basket'])}")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
