# Verification checkpoint — 2026-10-16

Context: pass-001 r04; local EC2 repository as previously confirmed by learner; Python 3.12.3. Checker: assistant.

- V1 (supports G1–G6 prerequisites): `python3 --version` returned Python 3.12.3. Passed for version only. Repository-change inspection was explicitly skipped by learner; venv support was reported by learner, not verified before creation.
- V2 (supports G1–G6 prerequisites): expected a complete isolated environment with working Python and pip. Creation returned exit 1: ensurepip unavailable; command recommended python3.12-venv. Failed. No pip or isolation success is claimed.
- Git exclusion check: `git check-ignore code/.venv/bin/python3` returned that path. Passed for the partial environment path; does not establish absence of previously tracked files.
- Partial-state check: code/.venv contains bin, include, lib, lib64 and pyvenv.cfg. Its Python reports 3.12.3. This does not establish a usable complete environment.

No dependencies, downloads, model runs, or system package changes occurred. No learner understanding or acceptance is inferred.

## Later installation attempt — 2026-10-16

The learner requested numbered scripts, then explicitly said “now run them”. The assistant ran `bash code/setup/01_install_venv_support.sh`. apt proposed three new packages and five related Python upgrades, with 10.7 MB download and 2790 kB extra disk use. It reached its confirmation prompt, received no interactive input, printed `Abort.`, and exited 1. No package installation was reported. The check script was not run because installation failed. No operation remains running. A retry has not occurred.

Scripts: code/setup/01_install_venv_support.sh (installation) and code/setup/02_check_venv_support.sh (separate check). Both passed `bash -n` before execution; syntax validity does not establish runtime success.
