#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/setup_git.sh"
bash "$SCRIPT_DIR/setup_env.sh"
# bash "$SCRIPT_DIR/setup_codex.sh"

source "$HOME/llm-aws/.venv/bin/activate"

echo "Setup completed. Virtual environment activated."