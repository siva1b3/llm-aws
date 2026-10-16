# Pass-001 — Create the Python environment

Current disposition: closed, successful — 2026-10-16. See the closure checkpoint below.

## Preserved r03 plan

Revision: r03 · 2026-10-16 · Reserved and inactive at that checkpoint.
Plan direction confirmed by the learner. Execution approval is pending.

**Outcome:** an isolated Python environment in this EC2 repository, supporting G1–G6. No model operation is opened.

| Action | Separate check | Status |
| --- | --- | --- |
| A1: Inspect the repository changes, Python version, venv support, and any existing environment. | V1: Confirm the prerequisites and a safe environment location; report missing or unknown items. | not started |
| A2: Create the environment at the agreed location after V1. | V2: Confirm its Python and pip run from that environment and are isolated from system packages. | not started |

Assistant owns execution and checks. Explain exact commands and obtain approval for each action/check pair before running it. Stop and report after each pair. Preserve existing files; do not overwrite an existing environment.

No dependency installation, system changes, downloads, inference, or external resource changes are included. If prerequisites are missing, stop and agree on recovery.

Overall workflow: [roadmap](../planned_roadmap.md).

## 2026-10-16 — r04 scope and failure checkpoint

The learner narrowed inspection to Python version, reported environment support and no existing environment, and explicitly skipped repository-change inspection. Python 3.12.3 was observed using `python3 --version`.

The learner then instructed: “we have a code folder use it to create the env , care ful we have git so we need proper gitignore files”. This authorized creation under code with Git exclusions. The assistant announced code/.venv, creation, and separate Git/Python/pip checks before execution. No separate r04 plan-approval exchange or pre-execution checkpoint occurred; this is a process deviation, not a claim that those gates were completed.

Pass is now open and blocked. Creation was attempted with `python3 -m venv code/.venv` and failed because ensurepip is unavailable. code/.gitignore was created; code/.venv is partial. No installation or recovery is authorized or performed. Preserve r03 above as history.

Next proposed action: obtain authority to install the missing system venv support, with a separate availability check, before any retry. System package installation and environment retry require separately described approval.

## Later recovery attempt — 2026-10-16

Learner requested numbered scripts for later execution, then authorized running both: “now run them”. Installation script ran but apt aborted at its confirmation prompt in the noninteractive invocation. The separate check was not run. Stop before retry; proposed recovery is an interactive invocation of the unchanged installation script, answering apt's confirmation, followed by the separate check if installation succeeds. apt also proposes upgrading five related Python packages; disclose this effect in the recovery request.


## 2026-10-16 — closure checkpoint, r05

**Outcome:** closed, successful. Decision owner: human learner. The learner requested checking the existing environment, updating pass-001 and current state, and reserving pass-002 for the next part. This is authority for verification, records, closure after a passing check, and empty placeholders; it is not authority to install project dependencies.

**Approvals and intervening results:**
- Learner explicitly approved the interactive installation retry and subsequent check. Script 01 completed with exit 0, installing three packages and upgrading five related Python packages. Script 02 completed with exit 0: Python 3.12.3, pip 24.0, venv module available. apt reported deferred restarts for networkd-dispatcher and unattended-upgrades; no manual service restart was performed.
- Learner approved the proposed numbered environment creation/check plan with “yes”. Scripts 03 and 04 were prepared and passed Bash syntax checking. Their execution was then requested separately; no assistant invocation of script 03 is recorded.
- The partial environment was absent when scripts 03/04 were prepared. At this checkpoint a complete environment exists. Its creation command/operator is not independently established by the available conversation; no assistant creation run is invented.
- Latest request authorizes running the environment check and updating records. Script 04 passed. See [verification](verification.md) and [evidence](evidence.md).

| Action | Final status | Separate result |
| --- | --- | --- |
| A1: Python prerequisite inspection, narrowed by learner | done | Python observed; repository-change inspection explicitly skipped. |
| Recovery: install missing venv support | done | Installation and separate support check passed after approved retry. Earlier failures retained above. |
| A2: establish code/.venv | done, by observed resulting artifact | V2 passed; creator/invocation not independently established. |

**Boundary:** environment setup only. No model operation was exposed. G1–G6 remain outstanding. No dependency installation, model download, inference, release, or next-pass execution is authorized by closure.

**Reviews and limitations:** technical environment check passed; no separate learner understanding or whole-project acceptance is claimed. Existing broader repository changes were not inspected at the learner's direction. No process launched by the assistant remains running.

**Next action:** discuss the dependency setup plan in conversation. Pass-002 is reserved and inactive with empty required records. Obtain plan agreement and separate execution authority before installation.
