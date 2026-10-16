#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
code/.venv/bin/python -m pip --isolated check
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 code/.venv/bin/python - <<'PY'
import sys
from pathlib import Path
import torch
import transformers
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

root = Path('code/.venv').absolute()
assert Path(sys.prefix) == root
for module in (torch, transformers, pd, np):
    assert Path(module.__file__).is_relative_to(root)
    print(f'{module.__name__}: {module.__version__}')
assert torch.__version__.split('+')[0] == '2.12.1'
assert transformers.__version__ == '5.20.0'
assert torch.version.cuda is None
assert torch.version.hip is None
assert not torch.cuda.is_available()
x = torch.tensor([[1., 2.], [3., 4.]], device='cpu')
result = x @ torch.tensor([2., 1.], device='cpu')
assert result.device.type == 'cpu'
torch.testing.assert_close(result, torch.tensor([4., 10.]))
assert np.array([1, 2, 3]).sum() == 6
assert pd.Series([1, 2, 3]).sum() == 6
print('CPU matrix-vector result:', result.tolist())
print('NumPy/pandas sums: 6')
print('Model/tokenizer class imports: passed (offline; no model loaded)')
print('Dependency checks passed.')
PY
