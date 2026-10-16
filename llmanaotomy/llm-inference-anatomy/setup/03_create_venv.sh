#!/usr/bin/env bash
set -euo pipefail

setup_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
env_dir="$(dirname -- "$setup_dir")/.venv"

if [[ -e "$env_dir" || -L "$env_dir" ]]; then
    echo "STOP: $env_dir already exists. Inspect it before retrying."
    exit 1
fi

python3 -m venv "$env_dir"
echo "Created: $env_dir"
