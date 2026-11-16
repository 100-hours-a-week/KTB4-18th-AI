"""
7. 같은 와이파이용 사람 평가 서버 (표준 라이브러리만). 미리듣기를 페이지 안에서 재생하고, 판정을 파일로 저장한다

  GET  /                 index.html (평가 페이지. claude.ai 아티팩트로 게시한 것과 같은 파일)
  GET  /pairs_val.json   평가 데이터 (이 두 파일만 내보낸다. key_*.json 대응표는 절대 안 내보낸다)
  GET  /pairs_test.json
  POST /save             {name, who, answers} → runs/human_eval_results/{name}__{who}.json (사람마다 덮어쓰기 = 자동 저장)

입력      --pairs_dir: export_lists.py가 만든 pairs_{val,test}.json 폴더
출력      runs/human_eval_results/ — 평가자 이름이 파일 이름에 들어간다 (게시 금지, .gitignore)
주소      test 세트 http://<노트북 IP>:8765/   val 세트 http://<노트북 IP>:8765/#val
          맥에서 처음 실행하면 "들어오는 연결 허용" 창이 뜰 수 있다 → 허용. 평가가 끝나면 Ctrl+C

사용:
  python -m human_eval.serve --pairs_dir runs/<날짜>_human_eval_lists
"""
import argparse
import json
import re
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import config

PORT = 8765
PAGE = Path(__file__).resolve().parent / "index.html"
RESULTS = config.RUNS_DIR / "human_eval_results"
PUBLIC = {"/pairs_val.json", "/pairs_test.json"}
MAX_BODY = 1_000_000


def make_handler(pairs_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path in PUBLIC and (pairs_dir / path.lstrip("/")).exists():
                self.send(200, (pairs_dir / path.lstrip("/")).read_bytes(), "application/json; charset=utf-8")
            else:
                self.send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            if self.path != "/save":
                return self.send(404, b"not found", "text/plain")
            n = int(self.headers.get("Content-Length", 0))
            if not 0 < n <= MAX_BODY:
                return self.send(413, b"too large", "text/plain")
            body = json.loads(self.rfile.read(n))
            name, who = str(body.get("name", "")), str(body.get("who", "")).strip()
            if name not in ("val", "test") or not who or not isinstance(body.get("answers"), dict):
                return self.send(400, b"bad request", "text/plain")
            # NOTE: 이름은 파일 이름에 들어가므로 글자·숫자·한글·-_만 남긴다 (경로 조작 방지)
            safe = re.sub(r"[^0-9A-Za-z가-힣_-]", "_", who)[:40]
            RESULTS.mkdir(parents=True, exist_ok=True)
            (RESULTS / f"{name}__{safe}.json").write_text(json.dumps(body, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"  저장: {name} — {len(body['answers'])}개")
            self.send(200, b"ok", "text/plain")

        def log_message(self, fmt, *args) -> None:   # 요청마다 찍히는 로그는 끈다 (저장만 출력)
            pass
    return Handler


def lan_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("10.255.255.255", 1))   # 실제로 보내지 않는다. 바깥으로 나갈 때 쓰는 내 주소만 알아낸다
        return s.getsockname()[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_dir", type=Path, required=True, help="export_lists.py 결과 폴더")
    args = ap.parse_args()
    found = [f for f in PUBLIC if (args.pairs_dir / f.lstrip("/")).exists()]
    assert found, f"{args.pairs_dir}에 pairs_*.json이 없다 (export_lists.py 먼저)"
    ip = lan_ip()
    print(f"test: http://{ip}:{PORT}/\nval:  http://{ip}:{PORT}/#val\n저장: {RESULTS}\n종료: Ctrl+C")
    ThreadingHTTPServer(("0.0.0.0", PORT), make_handler(args.pairs_dir)).serve_forever()


if __name__ == "__main__":
    main()
