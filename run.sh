#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
exec uv run uvicorn backend.main:app --reload --host 127.0.0.1 --port "${AI_SERVER_PORT:-8001}"
