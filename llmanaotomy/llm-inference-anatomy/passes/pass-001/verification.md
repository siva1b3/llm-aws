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


## 2026-10-16 — successful recovery and final V2

Earlier failures remain historical evidence. This checkpoint supersedes their blocked environment-readiness outcome without erasing those attempts.

- **Support check:** following explicitly approved interactive recovery, script 01 exited 0 and script 02 exited 0. Observed Python 3.12.3, pip 24.0, venv available. This checks support, not project package compatibility.
- **Identifier:** V2-final; supports prerequisites for G1–G6, not satisfaction of those learning criteria.
- **Method / checker:** assistant ran `bash code/setup/04_check_venv.sh` against the existing code/.venv, under the learner's latest request.
- **Expectation:** Python and pip belong to code/.venv; system and user packages excluded; Git ignores the environment.
- **Observation:** exit 0; Python 3.12.3; executable and prefix inside code/.venv; prefix differs from base prefix; user site disabled; include-system-site-packages false; pip 24.0 imported from environment site-packages; Git ignore check passed.
- **Additional check:** `git ls-files -- code/.venv` returned no paths, establishing no environment files tracked at this checkpoint.
- **Status:** passed.
- **Context:** 2026-10-16, closure checkpoint r05, learner-confirmed EC2 repository. Tested artifact digests in [evidence](evidence.md).
- **Limitations:** does not establish how the environment was created, project dependency readiness, model compatibility, inference, or learner confirmation. No creation or package installation was performed during this final check.
