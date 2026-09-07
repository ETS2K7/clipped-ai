#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

if [ -f "$ROOT_DIR/venv/bin/python3" ]; then
    PYTHON_BIN="$ROOT_DIR/venv/bin/python3"
elif [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/bin/python3" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python3"
else
    PYTHON_BIN="python3"
fi

export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/fast-asd/talknet:$PYTHONPATH"

echo "Starting ClippedAI Engine on http://localhost:8000 using $PYTHON_BIN..."
exec "$PYTHON_BIN" backend/server.py
