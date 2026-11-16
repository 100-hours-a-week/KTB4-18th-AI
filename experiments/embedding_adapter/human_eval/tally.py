"""
7. 사람 평가 집계와 채택 판정: 평가자별 순위 → 시스템 쌍별 승·무·패, 부호 검정

입력      평가 결과 폴더 (기본 원래 실험의 EMB_DATA_DIR/pairwise/results, 읽기만): {set}__{평가자}.json
            {"who", "answers": {항목 id: {"group", "ranks": {목록 코드: 등수}}}}  등수는 1~3, 같은 등수 허용
          대응표 key_{set}.json, 평가 데이터 pairs_{set}.json (--pairs_dir, export_lists.py 결과)
출력      runs/{날짜}_human_eval_tally/report.md — 쌍별 승·무·패, 승률, 부호 검정 p, 사람별 방향, 질문 유형별, 판정
          평가자 이름은 "평가자 1, 2, …"로 가린다 (--show_names로 끌 수 있다. 게시할 보고서에는 쓰지 말 것)
방법      요청 하나의 순위 → 세 쌍의 승·무·패 (같은 등수 = 비김). 부호 검정은 비김을 빼고 양측
채택 규칙  (2026-10-19 확정. 본 평가 판정이 일부 들어온 상태에서, 결과를 다 보기 전에 정했다)
          "X가 Y를 분명히 이김" = 부호 검정 p < 0.05 이고 X 쪽 사람이 3명 이상
          1. 원래가 평균 제거를 분명히 이김                   → 아무것도 배포 안 함 (지금 서비스 유지)
          2. 원래 또는 평균 제거가 쿼리·곡 어댑터를 분명히 이김 → 평균 제거만 배포
          3. 그 밖                                           → 쿼리·곡 어댑터 배포
          평가자 5명으로는 작은 차이를 가려낼 수 없다 → "사람이 분명하게 거부하는 것을 막는" 안전장치로 쓴다
결과      (본 평가: eval.txt 15문장 × 5명 = 75개) 3. 쿼리·곡 어댑터 배포.
          쿼리·곡 어댑터가 원래를 44승 10무 21패 (p = 0.006, 4명), 평균 제거를 45승 8무 22패 (p = 0.007, 5명).
          원래 vs 평균 제거는 구분 불가 (p = 0.81). 원본 로그: pairwise/tally_final.log
          NOTE: 본 평가의 쿼리·곡 어댑터는 곡 어댑터 100%(γ 없음)다. γ0.25(5-9b)는 사람 평가를 따로 받지 않았다

사용:
  python -m human_eval.tally              # 본 평가 (test)
  python -m human_eval.tally --set val    # 시범
"""
import argparse
import itertools
import json
from math import comb
from pathlib import Path

import config
from core.report import Report

SHORT = {"원래 공간": "원래", "평균 제거": "평균 제거"}
ORIG, MEAN, TOWER = "원래", "평균 제거", "쿼리·곡 어댑터"


def short(name: str) -> str:
    return next((v for k, v in SHORT.items() if name.startswith(k)), TOWER if "stage14" in name else name)


def sign_test(w: int, l: int) -> float:
    """양측 부호 검정: 비김을 뺀 n = w + l번이 반반이라면 이만큼 치우칠 확률"""
    n = w + l
    if n == 0:
        return 1.0
    k = max(w, l)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="test", choices=["test", "val"])
    ap.add_argument("--results", type=Path, default=config.EMB_DATA_DIR / "pairwise" / "results")
    ap.add_argument("--pairs_dir", type=Path, default=config.EMB_DATA_DIR / "pairwise", help="key·pairs json 폴더")
    ap.add_argument("--show_names", action="store_true", help="평가자 이름을 가리지 않는다 (게시 금지)")
    args = ap.parse_args()

    key = {c: short(n) for c, n in json.load(open(args.pairs_dir / f"key_{args.set}.json")).items()}
    files = sorted(args.results.glob(f"{args.set}__*.json"))
    n_items = len(json.load(open(args.pairs_dir / f"pairs_{args.set}.json"))["items"])
    systems = sorted(set(key.values()))
    rep = Report("human_eval_tally", f"7. 사람 평가 집계 ({args.set})")

    total, by_person, by_group = {}, {}, {}
    for i, f in enumerate(files, 1):
        d = json.load(open(f))
        who = d["who"] if args.show_names else f"평가자 {i}"
        rep.add(f"- {who}: {len(d['answers'])}/{n_items}개 판정")
        for ans in d["answers"].values():
            r = {key[c]: v for c, v in ans["ranks"].items()}
            for x, y in itertools.combinations(systems, 2):
                res = 0 if r[x] == r[y] else (1 if r[x] < r[y] else -1)   # x 기준: 1 이김, 0 비김, -1 짐
                for table, k in ((total, (x, y)), (by_person, (x, y, who)), (by_group, (x, y, ans["group"]))):
                    table.setdefault(k, [0, 0, 0])[1 - res] += 1          # [이김, 비김, 짐]

    def clearly_beats(x: str, y: str) -> bool:
        a, b = (x, y) if (x, y) in total else (y, x)
        w, _, l = total.get((a, b), [0, 0, 0])
        xw, yw = (w, l) if a == x else (l, w)
        people = {k[2] for k in by_person if k[:2] == (a, b)}
        x_side = sum(1 for p in people if (by_person[a, b, p][0] > by_person[a, b, p][2]) == (a == x)
                     and by_person[a, b, p][0] != by_person[a, b, p][2])
        return xw > yw and sign_test(xw, yw) < 0.05 and x_side >= 3

    n_answers = sum(len(json.load(open(f))["answers"]) for f in files)
    rep.add(f"\n## 쌍별 결과 (앞 시스템 기준, 판정 {n_answers}개)")
    rep.add("| 비교 | 승 | 무 | 패 | 승률(비김 제외) | 부호 검정 p | 사람별 (앞 쪽 / 뒤 쪽 / 동률) |\n|---|---|---|---|---|---|---|")
    for x, y in itertools.combinations(systems, 2):
        w, t, l = total.get((x, y), [0, 0, 0])
        people = [k[2] for k in by_person if k[:2] == (x, y)]
        pw = sum(1 for p in people if by_person[x, y, p][0] > by_person[x, y, p][2])
        pl = sum(1 for p in people if by_person[x, y, p][0] < by_person[x, y, p][2])
        rate = f"{w / (w + l):.0%}" if w + l else "-"
        rep.add(f"| {x} vs {y} | {w} | {t} | {l} | {rate} | {sign_test(w, l):.3f} | {pw} / {pl} / {len(people) - pw - pl} |")

    rep.add("\n## 질문 유형별 (앞 시스템 기준 승·무·패)")
    for g in sorted({k[2] for k in by_group}):
        cells = [f"{x} vs {y} {'·'.join(map(str, by_group.get((x, y, g), [0, 0, 0])))}"
                 for x, y in itertools.combinations(systems, 2)]
        rep.add(f"- [{g}] " + " | ".join(cells))

    if clearly_beats(ORIG, MEAN):
        verdict = "1. 원래가 평균 제거를 분명히 이김 → 아무것도 배포 안 함"
    elif clearly_beats(ORIG, TOWER) or clearly_beats(MEAN, TOWER):
        verdict = "2. 원래 또는 평균 제거가 쿼리·곡 어댑터를 분명히 이김 → 평균 제거만 배포"
    else:
        verdict = "3. 분명히 진 비교 없음 → 쿼리·곡 어댑터 배포"
    rep.add(f"\n## 채택 규칙 판정 (평가자 {len(files)}명)\n**{verdict}**")
    rep.save()


if __name__ == "__main__":
    main()
