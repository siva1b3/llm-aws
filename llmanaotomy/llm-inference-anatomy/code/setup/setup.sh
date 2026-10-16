
#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/siva1b3/llm-aws.git"
REPO_DIR="$HOME/llm-aws"

echo "[INFO] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y git curl wget ca-certificates

# 1. Install GitHub CLI
echo "[INFO] Installing GitHub CLI..."

sudo mkdir -p -m 755 /etc/apt/keyrings

wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg \
  | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg >/dev/null

sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
  | sudo tee /etc/apt/sources.list.d/github-cli.list >/dev/null

sudo apt-get update
sudo apt-get install -y gh

# 2. Install Codex CLI
echo "[INFO] Installing Codex CLI..."

curl -fsSL https://chatgpt.com/codex/install.sh \
  -o /tmp/install-codex.sh

sh /tmp/install-codex.sh

export PATH="$HOME/.local/bin:$PATH"
hash -r

# 3. Verify installations
echo "[INFO] Installed versions:"
git --version
gh --version | head -n 1

# 4. Authenticate GitHub using PAT
echo "[INFO] Authenticating GitHub..."

if gh auth status >/dev/null 2>&1; then
    echo "[INFO] GitHub already authenticated."
else
    read -r -s -p "Enter GitHub PAT: " GITHUB_PAT
    echo

    if [[ -z "$GITHUB_PAT" ]]; then
        echo "[ERROR] GitHub PAT is empty."
        exit 1
    fi

    printf '%s' "$GITHUB_PAT" | gh auth login \
        --hostname github.com \
        --git-protocol https \
        --with-token

    unset GITHUB_PAT
fi

gh auth setup-git
gh auth status

# 5. Clone GitHub repository
echo "[INFO] Cloning repository..."

if [[ -d "$REPO_DIR/.git" ]]; then
    echo "[INFO] Repository already exists. Skipping clone."
elif [[ -e "$REPO_DIR" ]]; then
    echo "[ERROR] Directory exists but is not a Git repository."
    exit 1
else
    git clone "$REPO_URL" "$REPO_DIR"
fi

# 6. Enter repository
cd "$REPO_DIR"

echo "[INFO] Working directory: $(pwd)"

# 7. Launch Codex
echo "[INFO] Launching Codex..."

codex --version
