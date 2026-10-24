#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

if [ ! -x .venv/bin/python ]; then
  echo "未找到 .venv，请先运行 scripts/setup_env.sh" >&2
  exit 1
fi

exec .venv/bin/python -m app
