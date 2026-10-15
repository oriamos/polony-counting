#!/usr/bin/env bash
# Build the frontend (when node/npm are available) and start the app server.
# Env: POLONY_DATA_DIR (default ./data), APP_PY (default python), PORT (default 8000)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${APP_PY:-python}"

if [ -f frontend/package.json ] && command -v npm >/dev/null 2>&1; then
  if [ ! -d frontend/node_modules ]; then (cd frontend && npm install --no-audit --no-fund); fi
  (cd frontend && npm run build) || echo "WARNING: frontend build failed; serving API only" >&2
elif [ ! -d frontend/dist ]; then
  echo "NOTE: node/npm not found and frontend/dist missing: serving the API only (docs at /docs)." >&2
fi
export POLONY_DATA_DIR="${POLONY_DATA_DIR:-$ROOT/data}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" -m polony_app --data-dir "$POLONY_DATA_DIR" --port "${PORT:-8000}"
