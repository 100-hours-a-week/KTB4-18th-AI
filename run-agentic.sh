#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export AGENTIC_CHAT=1 LOCAL_INSPECTOR=1 SEARCH_EMBEDDING=v4
export LANGSMITH_TRACING=false LANGSMITH_TRACING_V2=false LANGCHAIN_TRACING_V2=false
exec .venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port "${PORT:-8002}"
