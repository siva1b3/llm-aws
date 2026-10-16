# Pass-001 evidence

Checkpoint: 2026-10-16 (Asia/Calcutta). Evidence owner: assistant.

## Final observed check

Command from repository root: `bash code/setup/04_check_venv.sh`.
Exit status: 0.

```text
Python: 3.12.3
Executable: /home/ubuntu/llm-aws/llmanaotomy/llm-inference-anatomy/code/.venv/bin/python
Environment: /home/ubuntu/llm-aws/llmanaotomy/llm-inference-anatomy/code/.venv
Isolation: passed
pip 24.0 from /home/ubuntu/llm-aws/llmanaotomy/llm-inference-anatomy/code/.venv/lib/python3.12/site-packages/pip (python 3.12)
.venv/pyvenv.cfg
Environment checks passed.
```

`git ls-files -- code/.venv` returned no paths.

## Recoverable output identity

SHA-256 observed with sha256sum:

| Artifact | SHA-256 |
| --- | --- |
| [Creation script](../../code/setup/03_create_venv.sh) | 2d342feee5d89dc2ddd0d5f86f85b53dee6732545a92250a4326f31bd5c2c902 |
| [Verification script](../../code/setup/04_check_venv.sh) | 545e28c87fced938f94fe226cf05bf64fad63bba0a4d41c47244686a9ac462a3 |
| code/.venv/pyvenv.cfg (local ignored configuration) | 2b296284d0ab7bcec64beb6e699d12c24de79a9c2c4214d4855847792ff0e1f8 |

Creation script execution is not claimed. Runtime check covers the environment present at this checkpoint. Earlier failures and support installation observations are retained in [verification](verification.md).
