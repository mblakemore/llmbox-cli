#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python3"

# Use venv python if available, fall back to system python
if [ -x "$VENV_PYTHON" ]; then
    exec "$VENV_PYTHON" "$SCRIPT_DIR/llmbox.py" "$@"
else
    exec python3 "$SCRIPT_DIR/llmbox.py" "$@"
fi
