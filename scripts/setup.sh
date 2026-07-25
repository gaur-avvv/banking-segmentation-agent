#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

if command -v uv >/dev/null 2>&1; then
  uv sync --extra dev --extra adk
  RUNNER=(uv run)
else
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -e ".[dev,adk]"
  RUNNER=()
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example. Add GEMINI_API_KEY before cloud-model use."
else
  echo ".env already exists; leaving it unchanged."
fi

if [[ -t 0 ]]; then
  read -r -p "Dataset CSV, ZIP, or folder path (Enter for automatic demo): " DATASET_PATH
  if [[ -n "${DATASET_PATH}" ]]; then
    sed -i '/^BANKING_DATA_PATH=/d' .env
    printf '\nBANKING_DATA_PATH=%s\n' "$DATASET_PATH" >> .env
    echo "Saved dataset path in .env"
  fi
fi

echo "Setup complete. Examples:"
echo "  ${RUNNER[*]} banking-agent setup"
echo "  ${RUNNER[*]} banking-agent chat"
echo "  ${RUNNER[*]} banking-agent api --port 8000"
echo "  ${RUNNER[*]} banking-agent adk-web --port 8001"
