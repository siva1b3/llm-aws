#!/usr/bin/env bash
# One-time setup: gh, git, Python env, repo clone, ML deps, HF auth.
# Re-running is safe (idempotent where possible).

set -euo pipefail

GITHUB_USER="siva1b3"
GITHUB_EMAIL="siva1b3@gmail.com"
REPO_URL="https://github.com/siva1b3/llm-aws"
REPO_DIR="$HOME/llm-aws"
VENV_DIR="$REPO_DIR/code/llm-env"

# Hugging Face token (replace with real token, or export HF_TOKEN before running)
HF_TOKEN="${HF_TOKEN:-EvIzcelvrcrIvDNNaThxBoSQywByWccAM}"

echo "==> [1/8] apt update + base packages"
sudo apt update
sudo apt install -y \
    gh git curl wget build-essential \
    python3-pip python3-venv python3-dev
    
echo "==> [2/8] Tool versions"
gh --version | head -n1
git --version
python3 -V

echo "==> [3/8] GitHub auth"
if gh auth status >/dev/null 2>&1; then
    echo "    gh already authenticated, skipping login."
else
    echo "    Launching 'gh auth login' (interactive)..."
    gh auth login
fi

echo "==> [4/8] git global config"
git config --global user.name  "$GITHUB_USER"
git config --global user.email "$GITHUB_EMAIL"
git config --global --list

echo "==> [5/8] Clone repo (if missing)"
if [ -d "$REPO_DIR/.git" ]; then
    echo "    Repo already cloned at $REPO_DIR, skipping."
else
    git clone "$REPO_URL" "$REPO_DIR"
fi

echo "==> [6/8] Python venv + dependencies"
mkdir -p "$REPO_DIR/code"
cd "$REPO_DIR/code"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --upgrade pip

# CPU-only PyTorch (change index-url for CUDA, e.g. .../whl/cu121)
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install \
    transformers accelerate huggingface_hub \
    sentencepiece protobuf \
    jupyter ipykernel ipywidgets \
    pandas matplotlib psutil

python -m ipykernel install --user \
    --name llm-env --display-name "Python (llm-env)"

echo "==> [7/8] Hugging Face auth"
if hf auth whoami >/dev/null 2>&1; then
    echo "    hf already authenticated as: $(hf auth whoami 2>/dev/null | head -n1)"
else
    if [ "$HF_TOKEN" = "hf_REPLACE_ME_WITH_REAL_TOKEN" ]; then
        echo "    WARNING: HF_TOKEN is a placeholder."
        echo "    Set a real token and re-run:  export HF_TOKEN=hf_xxx && ./setup.sh"
        echo "    Or login interactively now:   hf auth login"
    else
        hf auth login --token "$HF_TOKEN" --add-to-git-credential
    fi
fi

echo "==> [8/8] Done."
echo "    Activate later with: source $VENV_DIR/bin/activate"