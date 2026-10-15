#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y git gh

if ! gh auth status --hostname github.com >/dev/null 2>&1; then
    read -rsp "GitHub PAT (hidden): " github_pat
    printf '\n'
    printf '%s\n' "$github_pat" |
        gh auth login --hostname github.com \
            --git-protocol https --with-token
    unset github_pat
fi

gh auth setup-git

read -rp "Git commit author name: " author_name
read -rp "Git commit email: " author_email
git config --global user.name "$author_name"
git config --global user.email "$author_email"

gh auth status
printf '\nGit setup complete.\n'

cd "$HOME"
git clone https://github.com/siva1b3/llm-aws.git
