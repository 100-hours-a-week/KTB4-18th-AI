"""
7. 사람 평가 목록 만들기: 요청마다 시스템 3개의 추천 목록 5곡 (서비스 search()와 같은 규칙), 블라인드

입력      카탈로그(service_format, 19,027곡), 곡 부가 정보 track_extras.json (export_track_extras.py, 장르·발매일·store_url·미리듣기)
          요청 문장과 classify() 결과: service_query.db (5_service_queries)
            test = eval.txt 사용자 말투 요청 (구역별 해시 순 앞 n개, 외국어 구역 제외) — 본 평가
            val  = 멜론 val 재생목록 제목 (질문 유형 키워드별 해시 순 앞 n개) — 시범
          세 번째 목록의 체크포인트와 지점 (--checkpoint, --alpha, --beta, --gamma)
출력      runs/{날짜}_human_eval_lists/pairs_{set}.json  페이지가 읽는 데이터. 시스템 이름 대신 코드만 있다
                                           key_{set}.json    코드 → 시스템 대응표. 평가자에게 보이면 안 된다 (게시 금지)
목록 규칙  서비스 db/search.py search(k=5, overfetch=6)를 그대로 따라 한다
          연도 필터(발매일 NULL은 빠짐) → 장르 필터 → 점수 순 30곡 → (가수, 제목) 소문자 중복 제거 → 5곡
          장르를 걸었는데 5곡이 안 되면 장르만 풀고 다시 찾아 채운다 (연도는 풀지 않는다)
          검색은 recommend_query 임베딩 + recommend_genres·연도 필터, 화면에는 변환 전 요청을 보여준다
          항목마다 세 목록의 배치 순서를 해시로 섞는다
시스템     원래 공간 / 평균 제거(검색어·곡 둘 다) / 체크포인트 시스템. μ_q·μ_s는 체크포인트에 저장된 값 (평균 제거도 같은 값)
재현      --set test --per_group 3 --checkpoint two_tower_lam05 --alpha 0.5 --beta 0.5
          → 원래 본 평가의 pairs_test.json, key_test.json과 같은 파일 (본 평가는 곡 어댑터 100% = γ 없음으로 했다)

사용:
  python -m human_eval.export_lists --set test --per_group 3 --checkpoint two_tower_lam05 --alpha 0.5 --beta 0.5
  python -m human_eval.export_lists --set test --per_group 3 --checkpoint two_tower_lam05 --alpha 0.5 --beta 0.5 --gamma 0.25
"""
import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path

import numpy as np

import config
from core.adapters import load_checkpoint
from core.data import embed_lookup, l2norm, load_bench, load_json
from core.safe_io import connect_readonly, new_run_dir, write_json
from core.systems import AdapterSystem

K, OVERFETCH = 5, 6           # db/search.py search() 기본값

# val: 먼저 맞는 유형 하나에만 넣는다 (순서가 우선순위)
VAL_GROUPS = [
    # NOTE: "비 "는 "알앤비 노래"에도 걸린다 → "비 오/비가/빗/장마"처럼 날씨로만 읽히는 꼴만
    ("날씨·계절", r"비 오|비가|비오|빗|장마|눈 오|눈이|봄|여름|가을|겨울|햇살"),
    ("감정", r"슬픈|우울|이별|설레|위로|외로|그리움|행복|짝사랑"),
    ("상황·활동", r"출근|퇴근|운동|드라이브|공부|카페|잠들|자기 전|여행|산책|파티"),
    ("장르 언급", r"힙합|재즈|팝|록|발라드|EDM|일렉|R&B|인디|클래식"),
    ("분위기 일반", r"."),
]


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def pick_queries(svc, which: str, per_group: int) -> list[dict]:
    """[{key, group, query, prompt, genres, min_year, max_year}] — 그룹별 해시 순 앞 per_group개"""
    if which == "val":
        # 그룹 = 문장 키워드로 정한 질문 유형. 변환 전 요청 = 멜론 제목 (classify()에 넣은 그대로)
        rows = [(f"pid{pid}", next(name for name, pat in VAL_GROUPS if re.search(pat, q)), title, q, json.loads(res))
                for pid, s, title, q, res in
                svc.execute("SELECT pid, split, source, recommend_query, result FROM conv_title") if s == "val"]
    else:
        # 그룹 = eval.txt의 구역 주석. 외국어 입력 구역은 분기가 한국어로 바꾸는지 보는 구역이라 뺀다
        #   (recommend_query가 한국어 요청과 겹친다: "推荐...下雨天" → "비 오는 날 듣기 좋은 노래")
        rows = [(line, section, line, q, json.loads(res))
                for line, section, q, res in svc.execute("SELECT line, section, recommend_query, result FROM evaltxt")
                if not section.startswith("외국어")]
    picked, counts = [], {}
    for key, group, prompt, q, res in sorted(rows, key=lambda r: h(f"rank:{which}:{r[0]}")):
        if res["intent"] != "recommend" or not q.strip():
            continue
        if counts.get(group, 0) >= per_group:
            continue
        counts[group] = counts.get(group, 0) + 1
        picked.append({"key": key, "group": group, "query": q.strip(), "prompt": prompt,
                       "genres": res["recommend_genres"],
                       "min_year": res["recommend_min_year"], "max_year": res["recommend_max_year"]})
    return sorted(picked, key=lambda p: (list(counts).index(p["group"]), p["key"]))


def service_search(scores: np.ndarray, meta: list[dict], extras: dict, q: dict) -> list[int]:
    """db/search.py search()를 그대로 따라 한 5곡 (카탈로그 행 번호)"""
    ok = np.array([bool(extras.get(str(m["track_id"]), {}).get("store_url")) for m in meta])
    if q["min_year"] or q["max_year"]:
        lo = date(q["min_year"], 1, 1).isoformat() if q["min_year"] else "0000"
        hi = date(q["max_year"], 12, 31).isoformat() if q["max_year"] else "9999"
        rd = [extras.get(str(m["track_id"]), {}).get("release_date") for m in meta]
        ok &= np.array([d is not None and lo <= d <= hi for d in rd])

    def candidates(genres):
        mask = ok if not genres else ok & np.array(
            [extras.get(str(m["track_id"]), {}).get("genre") in genres for m in meta])
        rows = np.flatnonzero(mask)
        return rows[np.argsort(-scores[rows])][:K * OVERFETCH].tolist()

    def dedup(rows):
        seen, out = set(), []
        for j in rows:
            key = (meta[j]["artist"].lower(), meta[j]["title"].lower())
            if key not in seen:
                seen.add(key)
                out.append(j)
            if len(out) >= K:
                break
        return out

    hits = dedup(candidates(q["genres"]))
    if q["genres"] and len(hits) < K:
        hits = dedup(hits + [j for j in candidates(None) if j not in hits])
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["val", "test"], required=True)
    ap.add_argument("--per_group", type=int, default=3)
    ap.add_argument("--checkpoint", required=True, help="세 번째 목록의 체크포인트 (이름표 이름 또는 .pt)")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--gamma", type=float, help="곡 쪽 섞기 (곡 어댑터가 있을 때만, 없으면 곡 어댑터 100%)")
    ap.add_argument("--extras", type=Path, default=config.EMB_DATA_DIR / "track_extras.json")
    ap.add_argument("--service_db", type=Path, default=config.SERVICE_QUERY_DB)
    args = ap.parse_args()

    # 허브 벌점 기준 = service_format train 검색어. 카탈로그는 같은 19,027곡
    bench = load_bench("service_format", "val")
    meta, extras = bench.meta, load_json(args.extras)
    queries = pick_queries(connect_readonly(args.service_db), args.set, args.per_group)
    vecs = embed_lookup([q["query"] for q in queries])
    assert all(v is not None for v in vecs), "임베딩 캐시에 없는 요청이 있다 (dataset.7_embed_queries --service 먼저)"
    Q_raw = np.stack(vecs)

    system = AdapterSystem(load_checkpoint(args.checkpoint), Q_raw, bench.S_raw, bench.Qt_raw)
    space = system.space(args.alpha, args.beta, args.gamma)
    # NOTE: 이름은 원래 본 평가의 대응표와 같게 (코드 = 이름의 해시라 이름이 다르면 코드도 바뀐다)
    third = f"{config.CHECKPOINTS.get(args.checkpoint, Path(args.checkpoint).stem)} α{args.alpha} β{args.beta}" \
            + (f" γ{args.gamma}" if args.gamma is not None else "")
    systems = {"원래 공간": l2norm(Q_raw) @ l2norm(bench.S_raw).T,
               "평균 제거 (둘 다)": system.Q_c @ system.S_c.T,
               third: space.Q @ space.S.T}
    code = {name: "s" + h(f"{args.set}:{name}")[:6] for name in systems}

    items, same = [], 0
    for k, q in enumerate(queries):
        lists = {name: service_search(sc[k], meta, extras, q) for name, sc in systems.items()}
        same += len({tuple(v) for v in lists.values()}) < len(lists)
        item_id = f"{args.set}-{h(q['key'])[:10]}"
        order = sorted(systems, key=lambda name: h(f"{item_id}:{name}"))   # 항목마다 다른 배치
        track = lambda j: {"title": meta[j]["title"], "artist": meta[j]["artist"],
                           "preview": extras[str(meta[j]["track_id"])]["preview_url"],
                           "store": extras[str(meta[j]["track_id"])]["store_url"]}
        items.append({"id": item_id, "set": args.set, "group": q["group"], "prompt": q["prompt"], "query": q["query"],
                      "filters": {"genres": q["genres"], "min_year": q["min_year"], "max_year": q["max_year"]},
                      "lists": [{"code": code[name], "tracks": [track(j) for j in lists[name]]} for name in order]})

    run = new_run_dir("human_eval_lists")
    write_json(run / f"pairs_{args.set}.json", {"name": args.set, "items": items}, ensure_ascii=False, indent=1)
    write_json(run / f"key_{args.set}.json", {v: k for k, v in code.items()}, ensure_ascii=False, indent=1)
    print(f"{args.set}: {len(items)}개 요청 (목록 두 개 이상이 똑같은 요청 {same}개) → {run}")
    for it in items:
        f = it["filters"]
        flt = ", ".join(x for x in [f"장르 {f['genres']}" if f["genres"] else "",
                                    f"연도 {f['min_year']}~{f['max_year']}" if f["min_year"] or f["max_year"] else ""] if x)
        shown = it["prompt"] if it["prompt"] == it["query"] else f"{it['prompt']}  → {it['query']}"
        print(f"  [{it['group']}] {shown}" + (f"  ({flt})" if flt else ""))


if __name__ == "__main__":
    main()
