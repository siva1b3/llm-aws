# Pass-002 evidence

Checkpoint: 2026-10-16, Asia/Calcutta. Assistant owns local evidence.

## Commands and observed results

- A1: `code/.venv/bin/python -m pip list --format=json --disable-pip-version-check`; exit 0, only pip 24.0.
- V1: separate Python executable/prefix, pip origin and isolation assertions; exit 0, Python 3.12.3, pip 24.0, all inside code/.venv.
- Syntax: `bash -n code/setup/05_install_cpu_dependencies.sh code/setup/06_check_cpu_dependencies.sh`; exit 0.
- A3: `bash code/setup/05_install_cpu_dependencies.sh`; exit 0. Installed CPU torch from official PyTorch CPU index, other direct packages from PyPI with required dependencies. No retry.
- V3/V4: `bash code/setup/06_check_cpu_dependencies.sh`; exit 0.

```text
No broken requirements found.
torch: 2.12.1+cpu
transformers: 5.20.0
pandas: 3.0.6
numpy: 2.5.4
CPU matrix-vector result: [4.0, 10.0]
NumPy/pandas sums: 6
Model/tokenizer class imports: passed (offline; no model loaded)
Dependency checks passed.
```

Full package versions: [snapshot](installed-requirements.txt), captured by `code/.venv/bin/python -m pip --isolated freeze`.

## Recoverable identity

SHA-256:

| Output | Digest |
| --- | --- |
| [Installation script](../../code/setup/05_install_cpu_dependencies.sh) | 57019e3299b5df93488dba6ebb6101e56a23aa3455561ef80e2d3ab32ac4dbf9 |
| [Verification script](../../code/setup/06_check_cpu_dependencies.sh) | 6d0ff0230cd5147728bcc061b524e0b839b0a4022906286f1d51d99a5dd74405 |
| [Version snapshot](installed-requirements.txt) | c9b3fa75c58d8d00b296cb74f186bed752a1962a83b40bd62edf68e5b33a44ae |

## Proposal sources

Consulted 2026-10-16:
- [PyTorch CPU installation instructions](https://pytorch.org/get-started/previous-versions/).
- [Transformers published metadata](https://pypi.org/pypi/transformers/json): version 5.20.0, Python >=3.10.0.
- [Selected model usage](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct): ordinary CPU loading documented.

No model downloads, inference, infrastructure changes, commit, or push occurred. Package installation changed only the project environment; imports may create ordinary local library caches. No assistant-started process remains running. No model-specific compatibility or learner confirmation is claimed.


## Closure recheck — 2026-10-16

Under the learner's final-check request, `bash code/setup/06_check_cpu_dependencies.sh` ran again and exited 0, with the same versions and successful output shown above. A separate Python comparison of sorted current `pip --isolated freeze` lines with installed-requirements.txt passed: `Installed version snapshot: matches`. This did not reinstall dependencies. No new model execution or download occurred. No assistant-started operation remains running.

Documentation closure check: action outcomes, approvals, limitations, next learning boundary, and evidence retained; stale pass disposition references corrected. No commit or push performed. See [findings](findings.md) for usage and reproducibility limits.
