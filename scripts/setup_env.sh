#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3.11}"

if [ ! -x .venv/bin/python ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python "$PYTHON" .venv
  else
    "$PYTHON" -m venv .venv
  fi
fi

if command -v uv >/dev/null 2>&1; then
  uv pip install --python .venv/bin/python -e ".[dev]"
else
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -e ".[dev]"
fi

echo
echo "依赖安装完成。下一步："
echo "  1) scripts/download_models.sh   # 若 .models 还没有权重"
echo "  2) scripts/run_web.sh"
echo
.venv/bin/python -m app --check || true
