# Pass-002 — CPU dependency setup

Current disposition: **closed, successful** — 2026-10-16 (Asia/Calcutta). Final implemented revision: r02. See the closure checkpoint below.

## Preserved r01 plan

Revision: r01 · 2026-10-16 (Asia/Calcutta). The following plan and pending statuses are historical; later checkpoints record approvals and results.

## Outcome and scope

Prepare the existing isolated Python environment for CPU model work. This supports prerequisites for [G1–G6](../../definition/goal.md); it does not satisfy those learning criteria.

Work type: dependency setup in this EC2 repository, using code/.venv. Pass-001 verified Python 3.12.3 and pip 24.0. Package readiness is unknown. Preserve existing files and unrelated user changes.

| Action | Purpose and output | Dependency | Separate check | Status | Execution authority |
| --- | --- | --- | --- | --- | --- |
| A1 | Inspect existing packages in code/.venv; report the package baseline. | Verified environment from pass-001. | V1: confirm the inspection used code/.venv and report installed packages and inspection limits. | not started | pending |
| A2 | Agree on minimal required packages, versions, and installation limits. | A1/V1 results. | V2: review the proposal against CPU execution, selected model requirements, and agreed resource limits. | not started | pending |
| A3 | Install only the agreed dependencies. | A2/V2 and explicit installation approval. | V3: inspect installation outcome and resulting package versions; report failures and partial effects. | not started | pending |
| A4 | Verify imports, installed versions, and CPU operation. | Successful installation check. | V4: compare observations with the agreed dependency and CPU requirements. | not started | pending |
| A5 | Record accumulated results and present the pass for closure review. | Prior results and limitations. | V5: review evidence completeness and unresolved items; close only with applicable human agreement. | not started | pending |

Assistant owns authorized actions, checks, and records. Explain exact work, inputs, expected check result, limits, and stopping point before requesting each execution approval. Report each result and stop before the next action. All checks are not run.

## Design and limits

Use the existing environment and minimal dependencies under [constraints](../../definition/constraints.md). Exact packages, versions, commands, download limits, and recovery terms remain to be agreed after inspection. No stack change is selected here. Keep installation and verification procedures distinct; use separate executable artifacts when appropriate.

No model downloads, model loading, inference, system package changes, external resource creation, publication, or release are included. Stop on missing prerequisites, incompatibility, failed checks, or effects outside agreed limits. Inspect partial effects before proposing a retry; recovery outside approved scope needs new authority.

Completion requires agreed dependencies installed, import/version/CPU checks passed, reproducible setup and evidence retained, limitations reported, and human closure review. No learner understanding or project completion is inferred.

## Approvals

- **Plan scope r01 — approved:** on 2026-10-16 the learner responded to the proposed five-step scope: “this is a nice plan can you update this a scope for the plan-002”. This authorizes saving that scope and records plan agreement.
- **Execution — pending:** the preceding request explicitly asked for no execution. The latest request authorizes this documentation update only. No package inspection, installation, or runtime check is authorized or performed here.

Next action: explain A1/V1 and obtain execution approval. Pass remains reserved until the first execution approval checkpoint.


## First execution checkpoint — 2026-10-16

The learner said “do it, step by step” after the assistant described A1 as inspection only. This approves A1/V1 under r01; it does not authorize subsequent actions or installation. The pass is now active. Assistant owns execution, verification, and checkpoint records; no contributors are added.

A1: run `code/.venv/bin/python -m pip list --format=json --disable-pip-version-check` from the repository root. Output: installed package names and versions. No installation, network lookup, model work, or system change is included.

V1: separately inspect Python executable/prefix, isolation settings, and pip module path/version. Expect the executable, prefix, and pip module to belong to code/.venv, with system and user packages excluded. Status: not run. A1 status at this checkpoint: not started.

Stop after reporting A1/V1. Next action then requires A2/V2 approval. The repository was clean before this checkpoint; branch main. Bash on the learner-confirmed EC2 repository is the execution context. Exact OS and project package compatibility are not established by this inspection.


## r02 installation checkpoint — 2026-10-16

Preserve r01 above as history. Current revision: r02; open. A1 completed: only pip 24.0 was listed. V1 passed: Python 3.12.3 and pip are inside code/.venv with system/user packages excluded. The learner approved A2/V2 with “yes”. Official package metadata and vendor documentation were reviewed; proposed pins were torch 2.12.1 CPU and transformers 5.20.0. This is a documentation compatibility check, not runtime proof.

The learner then instructed “just install them and pandad numpy other basic also”. This approves adding pandas and NumPy to scope and executing installation and the previously described separate checks. Interpret “pandad” as pandas and “other basic” as required supporting dependencies, with no optional package bundle. This latest direct instruction supersedes the pending metadata-only approval request.

A3: prepare and run code/setup/05_install_cpu_dependencies.sh. Install torch==2.12.1 from the official PyTorch CPU index, then transformers==5.20.0, pandas, and numpy from PyPI; preserve the CPU torch pin. Only binary wheels and required dependencies; no sudo, global install, CUDA, model download, or automatic recovery. Use pip network timeouts and zero network retries. Stop on failure and inspect partial effects before recovery. No numerical byte or spending cap was supplied; do not claim one. Available disk observed as 25 GiB, available RAM 6.3 GiB; Ubuntu 24.04.4 LTS, Linux x86_64. No infrastructure changes or paid resources are created.

V3: separately inspect installed versions and pip dependency consistency after successful installation. A4/V4: run code/setup/06_check_cpu_dependencies.sh; check imports, CPU-only torch build, a tiny known-result CPU calculation, NumPy/pandas calculations, and model/tokenizer class import without downloading assets. Expect matching pinned versions and correct numerical results. All installation/runtime checks: not run at this checkpoint. Stop after reporting results; pass closure and model work remain unauthorized.

Record exact resulting versions in a retained dependency snapshot. Assistant owns outputs and checks. Existing uncommitted changes are the pass-002 plan, verification, and current-state checkpoint records; preserve them. No commit or push is authorized by this installation request.


## Review checkpoint — 2026-10-16

Current revision r02. A1 and A2 done; V1/V2 passed within their stated limits. A3 and A4 done; V3/V4 passed. Evidence and version snapshot retained. A5 evidence preparation is done; human closure review and V5 remain pending. See [verification](verification.md) and [evidence](evidence.md) for actual results; earlier not-run entries describe their original pre-execution checkpoints.

Pass remains open, waiting for review. No learner confirmation or acceptance is inferred. No operations remain running. G1–G6 remain outstanding; the current boundary is dependency setup with no model operation exposed. Next action: human review of dependency setup results and decision on pass closure. Do not download model assets or begin another pass without agreement and execution authority.


## Closure checkpoint — 2026-10-16

**Outcome: closed, successful.** Decision owner: human learner. The learner requested “check onec and conclude the pass-002 i think we completed it” and updates to relevant Markdown. This authorizes the final check, record review, and closure on success; no next-pass execution is authorized.

| Action | Final status | Separate check |
| --- | --- | --- |
| A1 — inspect package baseline | done | V1 passed. |
| A2 — agree dependencies and limits | done | V2 passed within documented limits; learner directly authorized installation and pandas/NumPy additions. |
| A3 — install selected dependencies | done | V3 passed. |
| A4 — check imports, versions, and CPU operation | done | V4 passed, including final rerun. |
| A5 — retain evidence and review closure | done | V5 passed: records, scripts, version snapshot, and limitations reviewed; learner authorized closure after successful recheck. |

Final check: script 06 exited 0 again; installed package snapshot matched actual pip freeze. No installation rerun, model download, or inference occurred. Outputs: scripts 05/06 and installed-requirements.txt, identified by digests in [evidence](evidence.md). Assistant retains record ownership; no external contributor or operator is added. No assistant-started operation remains running.

**Reviews and acceptance:** technical checks passed. The human requested pass closure, conditional on this review. This is acceptance of dependency setup for the pass, not learner confirmation of model mechanisms or whole-project acceptance. No separate reviewer was required or supplied.

**Limitations and risks:** script 05 leaves some packages unpinned; a future fresh installation can differ. Mitigation: use the retained tested version snapshot when planning reproduction and recheck it. Assistant owns proposing that check if reproduction is needed. The snapshot is not hash-locked and a fresh-environment replay was not run. No model compatibility claim extends beyond offline class imports. Missing numerical resource caps in the historical installation approval remain disclosed; no retroactive budget agreement is claimed. These limits do not leave an unmet dependency-setup criterion.

**Learning boundary:** no internal operation exposed or learner-confirmed; G1–G6 remain outstanding. **Next action:** discuss the model-assets pass plan and limits; obtain plan and execution approvals before downloads. No pass-003 is opened or reserved by this closure.

## Explicit human closure confirmation — 2026-10-16

The learner confirmed: “Pass-002 is complete and closed successfully.” They requested the relevant Markdown updates. Pass-002's dependency setup outcome is accepted and its successful closure is explicitly confirmed. This confirmation does not authorize the next pass, model downloads, or inference, and does not establish understanding of any model operation. Earlier pending and conditional statements are preserved as historical checkpoints.
