#!/usr/bin/env bash
# Verification: confirms tools, auth, venv, and key Python imports.
# Safe to re-run anytime.

set -euo pipefail

REPO_DIR="$HOME/llm-aws"
VENV_DIR="$REPO_DIR/code/llm-env"

echo "==> Tool versions"
gh --version | head -n1
git --version
python3 -V

echo ""
echo "==> gh auth status"
gh auth status || echo "    (not authenticated — run: gh auth login)"

echo ""
echo "==> git global config"
git config --global --list | grep -E '^(user\.|credential\.)' || true

echo ""
echo "==> Repo check"
if [ -d "$REPO_DIR/.git" ]; then
    echo "    OK: $REPO_DIR"
else
    echo "    MISSING: $REPO_DIR  (run setup.sh)"
    exit 1
fi

echo ""
echo "==> Python venv check"
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "    MISSING venv: $VENV_DIR  (run setup.sh)"
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "    python: $(which python)"
python -V

echo ""
echo "==> hf auth status"
if hf auth whoami >/dev/null 2>&1; then
    echo "    user   : $(hf auth whoami | head -n1)"
    echo "    tokens :"
    hf auth list 2>/dev/null || true
else
    echo "    (not authenticated — run: hf auth login, or set HF_TOKEN and re-run setup.sh)"
fi

echo ""
echo "==> Python imports"
python - <<'PY'
import torch, transformers, sentencepiece, accelerate
print("torch        :", torch.__version__)
print("transformers :", transformers.__version__)
print("sentencepiece:", sentencepiece.__version__)
print("cuda available:", torch.cuda.is_available())
PY

echo ""
echo "==> System resources"
free -h
echo ""
df -kh / "$HOME"

echo ""
echo "==> All checks passed."