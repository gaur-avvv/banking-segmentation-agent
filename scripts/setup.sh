#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: bash scripts/setup.sh [--help]"
  echo "Installs the project, asks for a dataset path, and offers a start menu."
  echo "Examples:"
  echo "  bash scripts/setup.sh"
  echo "  uv run banking-agent chat --data-path /path/to/data.csv"
  exit 0
fi

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
  echo "Choose provider: 1) Gemini  2) OpenRouter  3) Groq  4) OpenAI  5) Ollama  6) None"
  read -r -p "Provider [1]: " PROVIDER_CHOICE
  case "${PROVIDER_CHOICE:-1}" in
    2) PROVIDER=openrouter; KEY_NAME=OPENROUTER_API_KEY ;;
    3) PROVIDER=groq; KEY_NAME=GROQ_API_KEY ;;
    4) PROVIDER=openai; KEY_NAME=OPENAI_API_KEY ;;
    5) PROVIDER=ollama; KEY_NAME= ;;
    6) PROVIDER=none; KEY_NAME= ;;
    *) PROVIDER=gemini; KEY_NAME=GEMINI_API_KEY ;;
  esac
  sed -i '/^LLM_PROVIDER=/d' .env
  printf 'LLM_PROVIDER=%s\n' "$PROVIDER" >> .env
  if [[ -n "$KEY_NAME" ]]; then
    read -r -s -p "Enter $KEY_NAME (hidden): " API_KEY; echo
    if [[ -n "$API_KEY" ]]; then
      sed -i "/^${KEY_NAME}=/d" .env
      printf '%s=%s\n' "$KEY_NAME" "$API_KEY" >> .env
    fi
  fi
fi

echo "Setup complete. Examples:"
echo "  ${RUNNER[*]} banking-agent setup"
echo "  ${RUNNER[*]} banking-agent chat"
echo "  ${RUNNER[*]} banking-agent api --port 8000"
echo "  ${RUNNER[*]} banking-agent adk-web --port 8001"

if [[ -t 0 ]]; then
  echo
  echo "Choose how to start:"
  echo "  1) Terminal chat"
  echo "  2) Local API + browser UI"
  echo "  3) Google ADK Web"
  echo "  4) One-shot segmentation query"
  echo "  5) Exit"
  read -r -p "Selection [5]: " choice
  case "${choice:-5}" in
    1) "${RUNNER[@]}" banking-agent chat ;;
    2) "${RUNNER[@]}" banking-agent api --port 8000 ;;
    3) "${RUNNER[@]}" banking-agent adk-web --port 8001 ;;
    4) "${RUNNER[@]}" banking-agent run --query "Segment customers into priority, regular, and dormant groups" ;;
    *) echo "Setup complete. Run a command above when ready." ;;
  esac
fi
