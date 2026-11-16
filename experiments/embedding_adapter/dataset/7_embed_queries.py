"""
7. 검색어 문장 → Gemini Embedding 2 벡터 (캐시)  ⚠ 임베딩 API 비용 (원래 실험 전체 약 $0.13)

서비스(backend/recommendation/embedding.py embed_query)와 같은 방식으로 호출한다: 문자열 입력, dimensions=3072.
같아야 여기서 학습한 어댑터를 서비스 검색어에 그대로 쓸 수 있다.

입력      데이터셋 json의 query (--dataset), 또는 --service: 5의 변환 결과(service_query.db)의 recommend_query (추천 분기만)
출력      runs/caches/query_cache.db  emb(key, model, text, vec)   key = sha256(모델명 + "\n" + 문장.strip())
          원래 실험의 캐시(EMB_DATA_DIR/query_cache.db)에 이미 있는 문장은 건너뛴다 (읽기만) → 같은 문장에 두 번 돈을 쓰지 않는다
          실험 코드(core/data.py)는 원래 캐시 → 이 캐시 순서로 찾는다
안전장치   --limit 없이 실행하면 남은 문장 수를 보여주고 y를 받아야 시작한다. --budget(기본 $0.5)에 닿으면 멈춘다
          NOTE: 대소문자는 바꾸지 않는다. 모델이 구분할 수 있다

사용 (OPENROUTER_EMBEDDING_API_KEY는 환경 변수 또는 이 저장소의 .env):
  python -m dataset.7_embed_queries --service --limit 10    # 먼저 10개로 실제 단가 확인
  python -m dataset.7_embed_queries --service               # 남은 전부
  python -m dataset.7_embed_queries --dataset runs/…/dataset.json
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

import config
from core.data import cache_key
from core.safe_io import cache_db, confirm, connect_readonly

BATCH = 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, help="query를 임베딩할 데이터셋 json")
    ap.add_argument("--service", action="store_true", help="service_query.db의 recommend_query (intent=recommend만)")
    ap.add_argument("--service_db", type=Path, default=config.SERVICE_QUERY_DB)
    ap.add_argument("--limit", type=int, help="이번 실행에서 임베딩할 최대 문장 수")
    # NOTE: 예상이 틀렸을 때(라우팅 변경, 버그 반복 호출)를 막는 안전장치
    ap.add_argument("--budget", type=float, default=0.5, help="이번 실행 비용 상한 (USD)")
    args = ap.parse_args()
    assert args.dataset or args.service, "--dataset 또는 --service"

    if args.service:
        svc = connect_readonly(args.service_db)
        rows = []
        for table in ("conv", "conv_title", "evaltxt"):
            if svc.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (table,)).fetchone():
                rows += svc.execute(f"SELECT recommend_query, result FROM {table}").fetchall()
        texts = sorted({q.strip() for q, res in rows if json.loads(res)["intent"] == "recommend" and q.strip()})
    else:
        texts = sorted({r["query"].strip() for r in json.load(open(args.dataset, encoding="utf-8"))["records"]})

    con = cache_db("query_cache.db")
    con.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, model TEXT, text TEXT, vec BLOB)")
    cached = {k for (k,) in con.execute("SELECT key FROM emb")}
    cached |= {k for (k,) in connect_readonly(config.QUERY_CACHE).execute("SELECT key FROM emb")}
    todo = [t for t in texts if cache_key(t) not in cached]
    print(f"문장 {len(texts):,}개 중 캐시 {len(texts) - len(todo):,}개, 남은 {len(todo):,}개")
    if args.limit:
        todo = todo[:args.limit]
    elif todo:
        confirm(f"{len(todo):,}문장을 {config.EMBED_MODEL}로 임베딩한다 (유료, 상한 ${args.budget})")
    if not todo:
        return

    load_dotenv(config.ROOT / ".env")
    client = OpenAI(api_key=os.environ["OPENROUTER_EMBEDDING_API_KEY"], base_url="https://openrouter.ai/api/v1")
    spent, tokens, done = 0.0, 0, 0
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        # NOTE: eu 리전은 10% 비싸다. 벡터는 리전과 무관하게 같다
        resp = client.embeddings.create(model=config.EMBED_MODEL, input=batch, dimensions=config.DIM,
                                        encoding_format="float", extra_body={"provider": {"ignore": ["google-vertex/eu"]}})
        vecs = [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
        assert len(vecs) == len(batch) and all(len(v) == config.DIM for v in vecs), "응답 개수/차원 불일치"
        con.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?, ?, ?)",
                        [(cache_key(t), config.EMBED_MODEL, t, np.asarray(v, dtype=np.float32).tobytes())
                         for t, v in zip(batch, vecs)])
        con.commit()
        done += len(batch)
        # NOTE: 토큰은 SDK 필드, cost는 오픈라우터가 덧붙인 필드라 model_extra에 있다
        tokens += resp.usage.prompt_tokens
        spent += (resp.usage.model_extra or {}).get("cost") or 0.0
        print(f"  {done:,}/{len(todo):,}  누적 {tokens:,}토큰 ${spent:.5f}", end="\r")
        if spent >= args.budget:
            print(f"\n예산 ${args.budget} 도달, 중단. 다시 실행하면 캐시 다음부터 이어서 한다")
            break
    print(f"\n이번 {done}개: {tokens:,}토큰, ${spent:.5f} (문장당 ${spent / done:.7f})")


if __name__ == "__main__":
    main()
