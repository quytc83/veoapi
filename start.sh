#!/bin/sh
set -e

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"

if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
  python -m pip install -r "$SCRIPT_DIR/requirements.txt"
fi

exec python -m uvicorn main:app --host 0.0.0.0 --port 8080 --proxy-headers --log-level debug
