#!/usr/bin/env bash
set -euo pipefail

# Read-only check. Does not install packages or create an environment.
python3 --version
python3 -m ensurepip --version
python3 -c 'import venv; print("venv module is available")'
