#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
python=code/.venv/bin/python
"$python" -m pip --isolated install --disable-pip-version-check --no-cache-dir --only-binary=:all: --retries 0 --timeout 30 --index-url https://download.pytorch.org/whl/cpu 'torch==2.12.1'
"$python" -m pip --isolated install --disable-pip-version-check --no-cache-dir --only-binary=:all: --retries 0 --timeout 30 --index-url https://pypi.org/simple 'torch==2.12.1' 'transformers==5.20.0' pandas numpy
