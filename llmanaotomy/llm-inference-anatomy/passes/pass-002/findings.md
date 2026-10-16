# Pass-002 findings

Closed successfully: 2026-10-16. Scope: CPU dependency readiness only.

- Observed environment: Ubuntu 24.04.4 LTS x86_64, Python 3.12.3, pip 24.0, isolated code/.venv.
- Installed and checked: torch 2.12.1+cpu, transformers 5.20.0, pandas 3.0.6, numpy 2.5.4; full supporting versions in [snapshot](installed-requirements.txt).
- Installation and verification are separate: [script 05](../../code/setup/05_install_cpu_dependencies.sh) and [script 06](../../code/setup/06_check_cpu_dependencies.sh). Both are invoked with Bash from any working directory; each locates the repository root itself.
- Original installation command from repository root: `bash code/setup/05_install_cpu_dependencies.sh`. It uses the official PyTorch CPU index and PyPI with binary wheels, no cache, and zero network retries. Installation was successful; no rerun is needed.
- Recheck command from repository root: `bash code/setup/06_check_cpu_dependencies.sh`. It performs pip consistency checks and offline import/CPU calculations; it downloads no model assets.
- Original and closure checks passed. Exact package snapshot matches at closure. Evidence and artifact hashes are retained in [evidence](evidence.md).
- Reproduction limit: the installation script leaves pandas/NumPy and supporting versions unpinned. The retained version snapshot identifies tested packages, including the CPU torch build; it is not a hash-locked installer. A fresh-environment replay was not performed. If replay is later requested, agree on it separately and use the correct CPU package source.
- Model loading and inference remain untested, as intended for this pass. No internal model operation has been opened; learner confirmation remains pending for future lessons. No other pass-002 requirement remains unfinished.

Next action: agree on the model-assets pass plan and download limits. No next-pass execution or model download is authorized by closure.
