"""
3b. 비인기곡 후보(3a) → iTunes에서 미리듣기 있는 곡 찾기 (무료 API, 임베딩 없음, 비용 0)

입력      tail_candidates.json (기본 config.TAIL_CANDIDATES), 서비스 카탈로그 메타, 매칭(아티스트 별칭용)
출력      runs/caches/itunes_tail.db  (requests: URL별 응답 캐시, verdicts: 곡별 판정 + iTunes 결과) → 3c가 읽는다
          원래 실험의 캐시(EMB_DATA_DIR/itunes_tail.db)는 건드리지 않는다
시간      iTunes는 분당 약 20회 제한 → 요청마다 3초. 원래 실험은 약 4,000곡을 찾는 데 몇 시간 걸렸다

언제 꺼도 된다: 응답과 판정을 곡마다 sqlite에 저장하므로 다시 실행하면 이어서 한다.
목표에 닿으면 멈춘다: 구간별로 찾은 곡(strict + roman)이 TARGETS에 닿으면 그 구간은 더 검색하지 않는다.

검색 (iTunes는 분당 약 20회 제한 → 요청마다 SLEEP초 대기, 요청 수 = 시간)
  후보가 2곡 이상인 아티스트: 아티스트 id를 찾아 곡 목록(최대 200곡)을 한 번 받고 그 안에서 찾는다.
    같은 아티스트의 다른 후보는 캐시된 목록을 다시 쓴다. 목록에 없으면 곡 검색으로 넘어간다.
  나머지: "아티스트 + 제목" 곡 검색 (멜론 아티스트 이름의 괄호 밖/안을 각각)

판정
  strict      제목 키 일치 + 미리듣기 있음 (곡 검색이면 아티스트 키도 일치)
  roman       한글 제목을 로마자로 바꾼 것과 iTunes 제목이 비슷함 (iTunes는 한국 곡을 "Dduk"처럼 적는다)
  in_catalog  찾은 곡이 이미 우리 카탈로그에 있음 (멜론 매칭이 놓친 곡 → 임베딩 없이 라벨로 쓸 수 있음)
  none        못 찾음. 번역된 제목(비 → RAINING)은 문자열로 못 잡는다

사용:
  python -m dataset.3_tail_songs.b_itunes_search --limit 30     # 먼저 소수로 성공률·속도 확인
  caffeinate -i python -m dataset.3_tail_songs.b_itunes_search  # 전체 (맥이 잠들지 않게). Ctrl+C 후 다시 실행하면 이어서
  python -m dataset.3_tail_songs.b_itunes_search --status       # 진행 상황만
"""
import argparse
import difflib
import json
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import config
from core.safe_io import cache_db
from dataset.match_rules import HANGUL, PARENS, artist_keys, norm, title_aliases, title_key

MELON = config.MELON_DIR
CATALOG = Path(f"{config.SERVICE_CATALOG}.json")
MAPPING = config.MAPPING_NO_ALIAS
DB_NAME = "itunes_tail.db"
TARGETS = {"10~50%": 3000, "하위50%": 1000}   # 구간별 찾을 곡 수 (strict + roman)
ARTIST_MIN_SONGS = 2                          # 후보가 이 이상인 아티스트는 아티스트 목록으로 찾는다
ROMAN_MIN_RATIO = 0.8                         # 로마자 제목 유사도 하한
SLEEP = 3
BACKOFF, RETRIES = 60, 5                      # 403/429/5xx면 BACKOFF초 쉬고 다시, RETRIES번까지
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
FOUND = ("strict", "roman")


# ── 로마자 변환 (국어의 로마자 표기법 단순판: 음운 변화는 무시) ──────────────
INITIALS = "g kk n d tt r m b pp s ss  j jj ch k t p h".split(" ")
VOWELS = "a ae ya yae eo e yeo ye o wa wae oe yo u wo we wi yu eu ui i".split(" ")
FINALS = " k k k n n n t l k m l l l p l m p p t t ng t t k t p t".split(" ")


def romanize(text: str) -> str:
    out = []
    for ch in text:
        code = ord(ch) - 0xAC00
        if 0 <= code < 11172:   # 한글 음절 = (초성 × 21 + 중성) × 28 + 종성
            out.append(INITIALS[code // 588] + VOWELS[code % 588 // 28] + FINALS[code % 28])
        else:
            out.append(ch)
    return "".join(out)


# ── iTunes 요청 (URL별 캐시) ──────────────────────────────────────
def open_db() -> sqlite3.Connection:
    con = cache_db(DB_NAME)
    con.execute("CREATE TABLE IF NOT EXISTS requests (url TEXT PRIMARY KEY, body TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS verdicts (melon_id INTEGER PRIMARY KEY, tier TEXT, verdict TEXT, "
                "via TEXT, itunes TEXT)")
    return con


last_request = 0.0   # 지난 요청을 시작한 시각


def wait_turn() -> None:
    """지난 요청을 시작하고 SLEEP초가 지날 때까지만 기다린다
    NOTE: 응답을 기다린 시간(1~2초)도 간격에 들어가므로 분당 제한은 지키면서 곡당 시간이 준다
    """
    global last_request
    time.sleep(max(0.0, SLEEP - (time.time() - last_request)))
    last_request = time.time()


def fetch(con: sqlite3.Connection, url: str) -> list[dict] | None:
    """iTunes 결과 목록. 캐시에 있으면 요청하지 않는다. RETRIES번 실패하면 None (판정을 미루고 다음 실행 때 다시)"""
    row = con.execute("SELECT body FROM requests WHERE url = ?", (url,)).fetchone()
    if row:
        return json.loads(row[0])
    for attempt in range(RETRIES):
        wait_turn()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
                results = json.loads(r.read().decode("utf-8", "replace") or "{}").get("results", [])
            con.execute("INSERT OR REPLACE INTO requests VALUES (?, ?)", (url, json.dumps(results)))
            return results
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            print(f"\n    요청 실패 ({e}), {BACKOFF}초 쉬고 다시 ({attempt + 1}/{RETRIES})")
            time.sleep(BACKOFF)
    return None


def search_url(term: str, entity: str, limit: int) -> str:
    params = {"term": term, "media": "music", "entity": entity, "limit": limit, "country": "US"}
    return f"https://itunes.apple.com/search?{urllib.parse.urlencode(params)}"


def load_artist_aliases() -> dict[str, str]:
    """멜론 아티스트 이름 → 카탈로그(iTunes) 아티스트 이름. 이미 매칭된 멜론 곡에서 가장 많이 짝지어진 이름
    멜론은 "방탄소년단"처럼 한글로만 적는 아티스트가 있어 US 스토어 검색에 안 걸린다 (→ BTS)
    """
    songs = {s["id"]: s for s in json.load(open(MELON / "song_meta.json", encoding="utf-8"))}
    artist_of = {t["track_id"]: t["artist"] for t in json.load(open(CATALOG, encoding="utf-8"))}
    pairs = Counter()
    for melon_id, track_id in json.load(open(MAPPING)).items():
        basket = songs[int(melon_id)]["artist_name_basket"]
        if basket:
            pairs[basket[0], artist_of[track_id]] += 1
    aliases = {}
    for (melon_artist, catalog_artist), _ in pairs.most_common():
        aliases.setdefault(melon_artist, catalog_artist)
    return aliases


def melon_artist_names(cand: dict) -> list[str]:
    """검색어용 아티스트 이름: 카탈로그 별칭, 멜론 괄호 밖/안 이름 ("에픽하이 (EPIK HIGH)" → EPIK HIGH, 에픽하이)
    NOTE: US 스토어는 한글 이름으로 거의 안 나오므로 한글이 없는 이름을 먼저 (요청 수 = 시간)
    """
    artist = cand["artist"]
    names = [cand.get("alias")] + [re.sub(PARENS, "", artist).strip()] + [x[1:-1] for x in re.findall(PARENS, artist)]
    unique = list(dict.fromkeys(n for n in names if n))
    return sorted(unique, key=lambda n: bool(HANGUL.search(n)))


def candidate_artist_keys(cand: dict) -> set[str]:
    return set(artist_keys(cand["artist"])) | (set(artist_keys(cand["alias"])) if cand.get("alias") else set())


def artist_tracks(con: sqlite3.Connection, cand: dict) -> list[dict] | None:
    """후보 아티스트의 곡 목록 (최대 200곡). 이름이 맞는 아티스트가 없으면 []"""
    keys = candidate_artist_keys(cand)
    for name in melon_artist_names(cand):
        found = fetch(con, search_url(name, "musicArtist", 5))
        if found is None:
            return None
        ids = [a["artistId"] for a in found if keys & set(artist_keys(a.get("artistName", "")))]
        if ids:
            tracks = fetch(con, f"https://itunes.apple.com/lookup?id={ids[0]}&entity=song&limit=200&country=US")
            return None if tracks is None else [t for t in tracks if t.get("wrapperType") == "track"]
    return []


# ── 판정 ───────────────────────────────────────────────────────
def match(cand: dict, tracks: list[dict], check_artist: bool) -> tuple[str, dict | None]:
    """tracks 중 후보 곡 → (strict | roman | none, iTunes 결과)"""
    title_keys = {title_key(cand["title"])} | {norm(a) for a in title_aliases(cand["title"])}
    keys = candidate_artist_keys(cand)
    title_plain = re.sub(PARENS, "", cand["title"]).strip()
    roman = norm(romanize(title_plain)) if HANGUL.search(title_plain) else None

    best_roman, best_ratio = None, ROMAN_MIN_RATIO
    for t in tracks:
        if not t.get("previewUrl"):
            continue
        if check_artist and not keys & set(artist_keys(t.get("artistName", ""))):
            continue
        if title_key(t.get("trackName", "")) in title_keys:
            return "strict", t
        if roman:
            t_key = title_key(t.get("trackName", ""))   # 버전 표기 곡(라이브, JP Ver. 등)은 None → 건너뜀
            ratio = difflib.SequenceMatcher(None, roman, t_key).ratio() if t_key else 0
            if ratio >= best_ratio:
                best_roman, best_ratio = t, ratio
    return ("roman", best_roman) if best_roman else ("none", None)


def find(con: sqlite3.Connection, cand: dict, by_artist: bool) -> tuple[str, str, dict | None] | None:
    """(판정, 찾은 경로, iTunes 결과). 요청이 계속 실패하면 None"""
    if by_artist:
        tracks = artist_tracks(con, cand)
        if tracks is None:
            return None
        verdict, t = match(cand, tracks, check_artist=False)
        if t:
            return verdict, "artist", t
    # 검색어: (아티스트 이름 × 제목) — 괄호 속 영어 제목이 있으면 그것부터
    titles = title_aliases(cand["title"]) + [re.sub(PARENS, "", cand["title"]).strip()]
    for name in melon_artist_names(cand):
        for title in titles:
            tracks = fetch(con, search_url(f"{name} {title}", "song", 25))
            if tracks is None:
                return None
            verdict, t = match(cand, tracks, check_artist=True)
            if t:
                return verdict, "song", t
    return "none", "song", None


def status(con: sqlite3.Connection, candidates: list[dict]) -> Counter:
    counts = Counter(con.execute("SELECT tier, verdict FROM verdicts").fetchall())
    for tier, target in TARGETS.items():
        n_tier = sum(1 for c in candidates if c["tier"] == tier)
        done = sum(n for (t, _), n in counts.items() if t == tier)
        found = sum(counts[tier, v] for v in FOUND)
        print(f"  {tier:7s} 찾음 {found:,}/{target:,} | 처리 {done:,}/{n_tier:,} | "
              + ", ".join(f"{v} {counts[tier, v]:,}" for v in ("strict", "roman", "in_catalog", "none")))
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, default=config.TAIL_CANDIDATES, help="3a의 tail_candidates.json")
    ap.add_argument("--limit", type=int, help="이번 실행에서 처리할 최대 곡 수")
    ap.add_argument("--status", action="store_true", help="진행 상황만 출력")
    args = ap.parse_args()

    candidates = json.load(open(args.candidates, encoding="utf-8"))
    con = open_db()
    if args.status:
        status(con, candidates)
        return

    catalog_ids = {t["track_id"] for t in json.load(open(CATALOG, encoding="utf-8"))}
    aliases = load_artist_aliases()
    for c in candidates:
        c["alias"] = aliases.get(c["artist"])
    print(f"후보 {len(candidates):,}곡 중 카탈로그 아티스트 별칭이 있는 곡 {sum(1 for c in candidates if c['alias']):,}")
    n_by_artist = Counter(c["artist"] for c in candidates)
    done = {m for (m,) in con.execute("SELECT melon_id FROM verdicts")}

    # NOTE: 구간 안 순서의 비율(0~1)로 정렬하면 두 구간이 크기에 비례해 번갈아 나온다 (6,000 : 2,000 = 3 : 1)
    position = Counter()
    order = []
    for c in candidates:
        order.append((position[c["tier"]] / sum(1 for x in candidates if x["tier"] == c["tier"]), c))
        position[c["tier"]] += 1
    order = [c for _, c in sorted(order, key=lambda x: x[0])]

    counts = status(con, candidates)
    found = Counter({tier: sum(counts[tier, v] for v in FOUND) for tier in TARGETS})
    processed, t0 = 0, time.time()
    for cand in order:
        if cand["melon_id"] in done or found[cand["tier"]] >= TARGETS[cand["tier"]]:
            continue
        if all(found[t] >= TARGETS[t] for t in TARGETS) or (args.limit and processed >= args.limit):
            break

        by_artist = n_by_artist[cand["artist"]] >= ARTIST_MIN_SONGS and cand["artist"] != "Various Artists"
        result = find(con, cand, by_artist)
        if result is None:
            con.commit()   # 받아 둔 응답은 저장. 판정은 미뤄서 다음 실행 때 다시
            continue
        verdict, via, t = result
        if t and t["trackId"] in catalog_ids:
            verdict = "in_catalog"
        con.execute("INSERT INTO verdicts VALUES (?, ?, ?, ?, ?)",
                    (cand["melon_id"], cand["tier"], verdict, via, json.dumps(t, ensure_ascii=False) if t else None))
        con.commit()
        processed += 1
        found[cand["tier"]] += verdict in FOUND

        rate = (time.time() - t0) / processed
        print(f"  [{processed:5,d}] {verdict:10s} {via:6s} {cand['title'][:24]} / {cand['artist'][:16]}"
              f"  →  {(t['trackName'] + ' / ' + t['artistName'])[:40] if t else ''}"
              f"   ({rate:.1f}초/곡, 찾음 {found['10~50%']:,}+{found['하위50%']:,})")

    print("\n[진행 상황]")
    status(con, candidates)


if __name__ == "__main__":
    main()
