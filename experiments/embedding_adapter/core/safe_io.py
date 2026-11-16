"""
덮어쓰기 방지: 이 저장소의 코드가 파일을 읽고 쓰는 길을 여기로만 낸다

  읽기   원본 데이터(EMB_DATA_DIR, SDD_DIR, MELON_DIR)는 읽기만 한다. SQLite는 mode=ro로 열어 DB가 쓰기를 거부한다
  쓰기   RUNS_DIR 아래에만, 실행마다 새 폴더(runs/날짜_시각_이름/)에. 이미 있는 파일은 덮어쓰지 않는다
"""
import sqlite3
from datetime import datetime
from pathlib import Path

import config


def connect_readonly(path: Path) -> sqlite3.Connection:
    """읽기 전용 SQLite 연결. 파일이 없으면 만들지 않고 에러 (sqlite3.connect는 없으면 빈 DB를 만든다)"""
    if not path.exists():
        raise FileNotFoundError(path)
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _inside_runs(path: Path) -> bool:
    try:
        path.resolve().relative_to(config.RUNS_DIR.resolve())
        return True
    except ValueError:
        return False


def new_run_dir(name: str) -> Path:
    """runs/{날짜_시각}_{name}/ 를 새로 만든다. 실행마다 다른 폴더라 이전 결과를 덮지 않는다"""
    out = config.RUNS_DIR / f"{datetime.now():%Y-%m-%d_%H%M%S}_{name}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def write_text(path: Path, text: str) -> Path:
    """RUNS_DIR 안, 아직 없는 파일에만 쓴다"""
    path = Path(path)
    if not _inside_runs(path):
        raise PermissionError(f"RUNS_DIR({config.RUNS_DIR}) 밖에는 쓰지 않는다: {path}")
    if path.exists():
        raise FileExistsError(f"이미 있다 (덮어쓰지 않는다): {path}")
    path.write_text(text, encoding="utf-8")
    return path


def cache_db(name: str) -> sqlite3.Connection:
    """이어하기용 캐시 SQLite (API 응답 등). RUNS_DIR/caches/ 아래에 두고 실행끼리 공유한다 (원본 폴더의 캐시는 건드리지 않는다)"""
    path = config.RUNS_DIR / "caches" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path, check_same_thread=False)


def write_json(path: Path, obj, **kwargs) -> Path:
    """RUNS_DIR 안, 아직 없는 파일에만. kwargs는 json.dump에 그대로 (원래 파일과 같은 바이트로 쓰려고)"""
    import json
    path = guarded_output(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, **kwargs)
    return path


def confirm(message: str) -> None:
    """돈이 들거나 DB를 바꾸는 단계 직전에 사람에게 묻는다. y가 아니면 멈춘다"""
    if input(f"{message}\n계속할까? [y/N] ").strip().lower() != "y":
        raise SystemExit("멈춤 (아무것도 바꾸지 않았다)")


def guarded_output(path: Path) -> Path:
    """학습 결과처럼 RUNS_DIR에 새로 만들 파일 경로를 검사해서 돌려준다 (torch.save 등에 넘기기 전에)"""
    path = Path(path)
    if not _inside_runs(path):
        raise PermissionError(f"RUNS_DIR({config.RUNS_DIR}) 밖에는 쓰지 않는다: {path}")
    if path.exists():
        raise FileExistsError(f"이미 있다 (덮어쓰지 않는다): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
