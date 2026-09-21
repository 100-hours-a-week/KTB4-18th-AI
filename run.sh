#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
exec uv run uvicorn backend.main:app --reload --host 127.0.0.1 --port "${PORT:-8001}"