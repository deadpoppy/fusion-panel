#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
CONFIG="${FUSION_CONFIG:-config.yaml}"

if [ ! -f "$CONFIG" ]; then
  cp config.yaml.example "$CONFIG"
  echo "Created $CONFIG. Fill in your model API keys, then run ./start-fusion-panel.sh again."
  exit 1
fi

if [ ! -d ".venv" ]; then
  "$PYTHON" -m venv .venv
fi

. .venv/bin/activate
python -m pip install -q -e .

exec fusion-panel --config "$CONFIG" "$@"
