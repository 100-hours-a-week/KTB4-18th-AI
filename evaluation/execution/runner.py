"""기존 라우터를 프로세스 안에서 실행한다. 네트워크/FE 지연은 포함하지 않는다."""

from pathlib import Path

import copy
import os
import subprocess
import sys
import asyncio
import io
import json
import time
from uuid import uuid4
from starlette.datastructures import Headers, UploadFile
from ..grading.scoring import route_of, score_case


def merge_catalog(tracks, rows):
    """DB 정보를 카드에 붙인다. DB에 없는 곡은 확인할 수 없으므로 오류로 둔다."""
    if any(t["track_id"] not in rows for t in tracks):
        raise LookupError("track_not_in_catalog")
    return [
        dict(
            t,
            genre=rows[t["track_id"]].genre,
            release_year=rows[t["track_id"]].release_date.year
            if rows[t["track_id"]].release_date
            else None,
            mood_tags=rows[t["track_id"]].mood_tags,
        )
        for t in tracks
    ]


def catalog(tracks):
    if not tracks:
        return tracks
    from sqlalchemy import select
    from db.models import SessionLocal, TrackRow

    with SessionLocal() as session:
        rows = {
            str(r.track_id): r
            for r in session.scalars(
                select(TrackRow).where(TrackRow.track_id.in_([int(t["track_id"]) for t in tracks]))
            )
        }
    return merge_catalog(tracks, rows)


def attach_catalog(out):
    """카드를 DB 정보로 채운다. 확인하지 못한 카드가 통과로 세어지지 않게 턴을 실패로 둔다."""
    try:
        out["tracks"] = catalog(out["tracks"])
    except Exception as error:
        out.setdefault("error", f"catalog_{type(error).__name__}")


async def run_turn(step, thread, root, attachments):
    from backend.chat.router import chat
    from backend.chat.photo.router import photo_chat
    from backend.chat.schemas import ChatRequest
    from backend.chat.graph.workflow import build_graph

    started = time.perf_counter()
    out = {
        "text": "",
        "tracks": [],
        "done": False,
        "first_text_ms": None,
        "cards_ms": None,
        "verdict_ms": None,
        "input": step,
    }
    elapsed = lambda: round((time.perf_counter() - started) * 1000, 2)
    try:
        if "image" in step:
            path = (root / attachments[step["image"]]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("image_outside_dataset")
            response = await photo_chat(
                UploadFile(io.BytesIO(path.read_bytes()), filename=path.name), thread, uuid4()
            )
        else:
            body = ChatRequest(thread_id=thread, request_id=uuid4(), **step)
            response = await asyncio.to_thread(chat, body)
        async for chunk in response.body_iterator:
            chunk = chunk.decode() if isinstance(chunk, bytes) else chunk
            for block in chunk.strip().split("\n\n"):
                fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
                event, data = fields.get("event"), json.loads(fields.get("data", "{}"))
                if event == "text":
                    if out["first_text_ms"] is None:
                        out["first_text_ms"] = elapsed()
                    out["text"] += data.get("delta", "")
                elif event == "tracks":
                    out.update(tracks=data["tracks"], cards_ms=elapsed())
                elif event == "scene":
                    out.update(scene=data["scene"], provider=data.get("provider"))
                elif event == "verdict":
                    out.update(scene=data, verdict_ms=elapsed())
                elif event == "rejected":
                    out.update(text=data.get("message", ""), rejected=data.get("reason"))
                elif event == "error":
                    out["error"] = data.get("code") or data.get("reason", "stream_error")
                elif event == "done":
                    out["done"] = True
        if response.background:
            await response.background()
        state = build_graph().get_state({"configurable": {"thread_id": str(thread)}}).values
        out["state"] = {
            k: v for k, v in state.items() if k not in ("query_vector", "tracks", "lookup_tracks")
        }
        # 이전 턴의 카드가 상태에 남을 수 있어 이번 응답 카드를 기준으로 outcome을 만든다.
        out["route"] = route_of(dict(state, tracks=out["tracks"]))
    except Exception as error:
        # 공급자 오류 문자열에는 URL/키가 들어갈 수 있어 유형만 저장한다.
        out["error"] = type(error).__name__
    out["total_ms"] = elapsed()
    attach_catalog(out)
    return out


async def run_case(row, root):
    thread = uuid4()
    inputs = row["inputs"]
    if "turns" in inputs:
        steps, expected = inputs["turns"], row["outputs"]["turns"]
    else:
        steps = [{"image": "image"}] + inputs.get("followup", [])
        expected = [row["outputs"]] + row["outputs"].get("followup_turns", [])
    if len(steps) != len(expected):
        raise ValueError("turn_count_mismatch")
    turns = []
    for step, gold in zip(steps, expected):
        actual = await run_turn(step, thread, root, row.get("attachments", {}))
        turns.append({"expected": gold, "actual": actual})
        if actual.get("error") or not actual.get("done"):
            break  # 실패한 대화의 뒤 턴을 정상 세션처럼 평가하지 않는다.
    from backend.chat.graph.workflow import _checkpointer

    _checkpointer.delete_thread(str(thread))
    return score_case(
        {
            "id": row["metadata"]["id"],
            "split": row["split"],
            "metadata": row["metadata"],
            "expected_turns": len(steps),
            "turns": turns,
            "outputs": row["outputs"],
            "attachments": row.get("attachments", {}),
        }
    )


def calls_cost(calls):
    costs = [(c.get("usage") or {}).get("cost") for c in calls]
    return sum(costs) if costs and all(c is not None for c in costs) else None


async def execute_case(kind, inputs, row, dataset_root, service_root, capture):
    start = time.perf_counter()
    if kind == "stt":
        from backend.stt.service import transcribe_audio

        upload = UploadFile(
            io.BytesIO((dataset_root / inputs["audio"]).read_bytes()),
            filename=Path(inputs["audio"]).name,
            headers=Headers({"content-type": "audio/wav"}),
        )
        case = {"reference": row["outputs"]["transcript"], "transcript": "", "success": False}
        try:
            case["transcript"] = await asyncio.to_thread(transcribe_audio, upload)
            case["success"] = True
        except Exception as exc:
            case["error"] = type(exc).__name__
        case["total_ms"] = (time.perf_counter() - start) * 1000
    elif kind == "faults":
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--disable-warnings",
            "--tb=short",
            inputs["pytest_node"],
        ]
        env = dict(
            os.environ,
            LANGSMITH_TRACING="false",
            LANGSMITH_TRACING_V2="false",
            LANGCHAIN_TRACING_V2="false",
        )
        done = await asyncio.to_thread(
            subprocess.run,
            command,
            cwd=service_root,
            env=env,
            text=True,
            capture_output=True,
            timeout=120,
        )
        case = {
            "success": done.returncode == 0,
            "returncode": done.returncode,
            "test_output": (done.stdout + done.stderr)[-6000:],
            "total_ms": (time.perf_counter() - start) * 1000,
        }
    else:
        case = await run_case(row, dataset_root)
    service_calls = [c for c in capture.records if c["case_id"] == row["metadata"]["id"]]
    case.update(
        id=row["metadata"]["id"],
        kind=kind,
        service_cost_usd=calls_cost(service_calls),
        service_calls=copy.deepcopy(service_calls),
    )
    return case
