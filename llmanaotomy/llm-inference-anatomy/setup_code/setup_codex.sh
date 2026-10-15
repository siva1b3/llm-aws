#!/usr/bin/env bash
set -euo pipefail


sudo apt-get update
sudo apt-get install -y curl


# Install Codex CLI
curl -fsSL https://chatgpt.com/codex/install.sh | sh

# Verify installation
codex --version

echo "Codex CLI installation completed."