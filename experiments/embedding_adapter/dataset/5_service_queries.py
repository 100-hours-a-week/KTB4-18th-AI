"""
5. 멜론 플레이리스트 제목 → 서비스 형식 검색어 (서비스 classify()를 그대로 호출)  ⚠ LLM API 비용

왜       초기 데이터셋의 검색어는 멜론 원문 "제목 | 태그 나열"이었다. 서비스가 실제로 임베딩하는 문장은 의도 분류 LLM
          (classify())이 사용자 메시지를 다시 쓴 명사구 한 문장이고, 장르·연도는 문장에서 빠져 검색 필터로 간다.
          → 플레이리스트 제목을 사용자 메시지처럼 넣어, 서비스와 같은 함수·프롬프트·모델·장르 목록으로 변환한다 (문서 (2), 5-8)
입력      4_build_dataset의 dataset.json (기본 원래 dataset_v3.json), KTB4_ROOT의 classify()·장르 목록(DB)·.env 키
          --eval_txt: KTB4_ROOT/scripts/eval/eval.txt (사용자 말투 요청, 라벨 없음)
출력      runs/caches/service_query.db
            conv_title  (pid, split, source, result(JSON 전체), recommend_query, model, created)   ← --input title
            evaltxt     (line, section, result, recommend_query, model, created)                  ← --eval_txt
          이미 변환한 pid는 건너뛴다 → 멈춰도 다시 실행하면 이어서. 원래 실험의 변환 결과(EMB_DATA_DIR/service_query.db)는 건드리지 않는다
대상 순서  val 전체 → train 해시 순 앞 3,000개 (μ_q 계산용) → --all: train 나머지 + test
기록      처음 변환(--input full)은 "제목 | 태그"를 통째로 넣어 recommend_query에 태그 나열이 섞였다 (평균 31.7자 → 제목만 22.4자).
          사용자는 태그 나열을 치지 않는다 → 전부 제목만으로 다시 변환했다. full은 기록으로만 남긴다
안전장치   --pilot 외에는 호출 수를 보여주고 y를 받아야 시작한다

사용 (팀 저장소 폴더에서, 팀 가상환경으로 — openai·서비스 코드·DB 접근이 그쪽에 있다):
  cd <KTB4_ROOT>
  PYTHONPATH=<이 폴더> uv run python -m dataset.5_service_queries --pilot 20    # 원문과 나란히 20개
  PYTHONPATH=<이 폴더> uv run python -m dataset.5_service_queries --run
  PYTHONPATH=<이 폴더> uv run python -m dataset.5_service_queries --all
  PYTHONPATH=<이 폴더> uv run python -m dataset.5_service_queries --eval_txt
"""
import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import config
from core.safe_io import cache_db, confirm

TRAIN_N = 3000
WORKERS = 8
TABLE_OF = {"full": "conv", "title": "conv_title"}

sys.path.insert(0, str(config.KTB4_ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(config.KTB4_ROOT / ".env")
from openai import OpenAI  # noqa: E402

from backend.providers.openrouter import OPENROUTER_BASE_URL  # noqa: E402
from backend.recommendation.intent import classify, known_genres  # noqa: E402


def hash_order(pid) -> str:
    return hashlib.sha256(f"service:{pid}".encode()).hexdigest()


def targets(dataset: Path, everything: bool) -> list[dict]:
    """(val 전체, train 해시 순 앞 TRAIN_N개) [+ train 나머지, test]. 언제 돌려도 같은 목록"""
    records = json.load(open(dataset, encoding="utf-8"))["records"]
    by_split = {s: sorted([r for r in records if r["split"] == s], key=lambda r: hash_order(r["pid"]))
                for s in ("train", "val", "test")}
    first = by_split["val"] + by_split["train"][:TRAIN_N]
    return first + by_split["train"][TRAIN_N:] + by_split["test"] if everything else first


def open_db(table: str):
    con = cache_db("service_query.db")
    con.execute(f"CREATE TABLE IF NOT EXISTS {table} (pid INTEGER PRIMARY KEY, split TEXT, source TEXT, "
                "result TEXT, recommend_query TEXT, model TEXT, created REAL)")
    return con


def message_of(rec: dict, how: str) -> str:
    """classify()에 넣을 사용자 메시지. title = " | " 앞 제목만 (태그 나열은 사용자가 치는 말이 아니다)"""
    return rec["query"].split(" | ")[0].strip() if how == "title" else rec["query"]


def convert(client: OpenAI, rec: dict, how: str) -> tuple[dict, dict]:
    # NOTE: 서비스는 conversation = [이번 메시지]로 부른다 (prepare_recommendation의 messages=[message])
    return rec, classify(client, [message_of(rec, how)])


def convert_eval_txt(client: OpenAI, con, model: str, workers: int) -> None:
    """eval.txt 한 줄 = 사용자 메시지 하나. 서비스와 같이 classify([메시지])로 분기·recommend_query·필터를 얻는다"""
    con.execute("CREATE TABLE IF NOT EXISTS evaltxt (line TEXT PRIMARY KEY, section TEXT, result TEXT, "
                "recommend_query TEXT, model TEXT, created REAL)")
    lines, section = [], ""
    for raw in open(config.KTB4_ROOT / "scripts/eval/eval.txt", encoding="utf-8"):
        s = raw.strip()
        if s.startswith("#"):
            section = s.strip("# ─").strip() or section
        elif s:
            lines.append((s, section))
    done = {line for (line,) in con.execute("SELECT line FROM evaltxt")}
    todo = [(s, sec) for s, sec in lines if s not in done]
    confirm(f"eval.txt {len(lines)}줄 중 {len(todo)}줄을 classify()로 변환한다 (LLM 호출 {len(todo)}번, {model})")
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(classify, client, [s]): (s, sec) for s, sec in todo}
        for fut in as_completed(futures):
            s, sec = futures[fut]
            if fut.exception() is not None:
                print(f"  실패: {s}")
                continue
            res = fut.result()
            con.execute("INSERT INTO evaltxt VALUES (?, ?, ?, ?, ?, ?)",
                        (s, sec, json.dumps(res, ensure_ascii=False), res["recommend_query"], model, time.time()))
            con.commit()
    for intent, n in con.execute("SELECT json_extract(result, '$.intent'), count(*) FROM evaltxt GROUP BY 1"):
        print(f"  {intent}: {n}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=config.EMB_DATA_DIR / "dataset_v3.json", help="4의 dataset.json")
    ap.add_argument("--pilot", type=int, help="val 앞 n개만 변환하고 원문과 나란히 출력")
    ap.add_argument("--run", action="store_true", help="val + train 3,000개 변환 (이어하기)")
    ap.add_argument("--all", action="store_true", help="--run 대상 + train 나머지 + test")
    ap.add_argument("--workers", type=int, default=WORKERS, help="동시 호출 수")
    ap.add_argument("--input", choices=["title", "full"], default="title",
                    help="title = 제목만 (기본), full = 제목 | 태그 (처음 변환, 기록용)")
    ap.add_argument("--eval_txt", action="store_true", help="KTB4 scripts/eval/eval.txt의 사용자 말투 요청 → evaltxt 표")
    args = ap.parse_args()
    assert args.pilot or args.run or args.all or args.eval_txt, "--pilot n, --run, --all 또는 --eval_txt"

    client = OpenAI(api_key=os.environ["OPENROUTER_LLM_API_KEY"].strip(), base_url=OPENROUTER_BASE_URL,
                    timeout=60.0, max_retries=2)
    model = os.environ["LLM_MODEL"]
    genres = known_genres()
    assert genres, "장르 목록을 못 가져왔다 (DB 연결 확인) — 서비스와 다른 프롬프트가 된다"
    print(f"모델 {model}, 장르 {len(genres)}개: {', '.join(genres)}")

    table = TABLE_OF[args.input]
    con = open_db(table)
    if args.eval_txt:
        convert_eval_txt(client, con, model, args.workers)
        return
    done = {pid for (pid,) in con.execute(f"SELECT pid FROM {table}")}
    todo = targets(args.dataset, False)[:args.pilot] if args.pilot else targets(args.dataset, args.all)
    todo = [r for r in todo if r["pid"] not in done]
    print(f"변환할 검색어 {len(todo):,}개 (이미 {len(done):,}개 있음)")
    if not args.pilot:
        confirm(f"classify()를 {len(todo):,}번 호출한다 ({model}, LLM API 비용)")

    t0, n, failed = time.time(), 0, []
    with ThreadPoolExecutor(args.workers) as pool:
        # NOTE: 끝난 순서대로 저장한다. pool.map(순서 유지)은 응답 없는 요청 하나가 뒤의 결과 저장을 전부 막았다
        futures = {pool.submit(convert, client, r, args.input): r for r in todo}
        for fut in as_completed(futures):
            if fut.exception() is not None:
                failed.append(futures[fut]["pid"])   # 다시 실행하면 이것만 재시도된다
                continue
            rec, res = fut.result()
            con.execute(f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (rec["pid"], rec["split"], message_of(rec, args.input), json.dumps(res, ensure_ascii=False),
                         res["recommend_query"], model, time.time()))
            con.commit()
            n += 1
            if args.pilot:
                print(f"\n[{rec['split']}] 입력: {message_of(rec, args.input)}   (원문: {rec['query']})\n"
                      f"  → {res['recommend_query']}\n"
                      f"    intent {res['intent']}, 장르 {res['recommend_genres']}, "
                      f"연도 {res['recommend_min_year']}~{res['recommend_max_year']}")
            elif n % 200 == 0:
                print(f"  {n:,}/{len(todo):,}  {time.time() - t0:.0f}s")
    print(f"\n완료 {n:,}개, 실패 {len(failed):,}개 {failed[:10]}, {time.time() - t0:.0f}s → {config.RUNS_DIR / 'caches'}")


if __name__ == "__main__":
    main()
