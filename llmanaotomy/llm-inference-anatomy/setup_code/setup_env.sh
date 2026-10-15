#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y python3-venv

mkdir -p "$HOME/llm-inference-anatomy"
cd "$HOME/llm-inference-anatomy"

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install transformers
python -m pip check

python - <<'PY'
import sys
import torch
import transformers

print("Python:", sys.version)
print("Environment:", sys.prefix)
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)

result = torch.tensor([1.0, 2.0]) + 1
assert result.tolist() == [2.0, 3.0]
print("CPU tensor check:", result)
print("Environment setup verified.")
PY

python -m pip freeze > requirements-installed.txt
