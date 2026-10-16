# Pass-003 — Model and tokenizer assets

Revision: r01 · 2026-10-16 (Asia/Calcutta).
Disposition: reserved and inactive. Plan prepared at the learner's request; execution approval pending.

## Outcome and scope

Download the selected pretrained model and tokenizer, verify the files, and check that they load locally on CPU. Stop before any model forward pass or next-token prediction.

This is model-asset setup in the existing repository and code/.venv. It supports prerequisites for [G1–G6](../../definition/goal.md); it does not satisfy those learning criteria. Follow [constraints](../../definition/constraints.md) and the [roadmap](../planned_roadmap.md).

## Inputs and design

- Selected model: HuggingFaceTB/SmolLM2-135M-Instruct, from its Hugging Face repository. Exact revision and required file set must be established before download.
- Existing dependencies: pass-002 verified CPU torch 2.12.1+cpu, transformers 5.20.0, pandas 3.0.6, and numpy 2.5.4. See its [evidence](../pass-002/evidence.md). These checks did not establish model loading compatibility.
- Use CPU and prefer FP32. Keep the model internals hidden. Do not enable arbitrary remote model code. Downloading data files and loading them are separate actions.
- Proposed responsibility: model files on disk → local tokenizer/model objects in RAM → loading check only. No forward call, generation, prompt tokenization lesson, or architecture dissection.
- Asset/cache location, exact executable artifacts, disk and memory allowance, download bytes, time limit, retry limit, and partial-download cleanup terms remain to be agreed before affected execution. Do not invent these values.
- Expected affected paths: this pass's records and later agreed setup scripts and asset/cache location. Inspect existing files and relevant user changes before edits. Preserve all unrelated changes, including prior pass records and setup files.

## Actions and separate checks

Assistant owns authorized actions, checks, and checkpoint records. The learner owns approvals and acceptance. No additional contributor is assigned.

| Action | Purpose and expected output | Inputs and dependency | Separate check and expected result | Status | Execution authority |
| --- | --- | --- | --- | --- | --- |
| A1 — Check storage and target | Inspect free disk space, memory, and the proposed storage location; report whether setup can proceed safely. | Existing repository and environment; proposed location must be explained before inspection. | V1: confirm the actual target, available resources, and existing files. Report unknowns; do not claim capacity is sufficient before file-size requirements are known. | not started | pending |
| A2 — Agree on the download | Identify the exact model revision, compatible tokenizer, required files, source, sizes, relevant use terms, destination, and download limits. | A1/V1; authorized read-only source/metadata review. | V2: compare the proposed download with the selected model, available resources, trust requirements, and agreed limits. Stop until blocking choices are resolved. | not started | pending |
| A3 — Download the files | Obtain only the agreed model, configuration, and tokenizer assets at the agreed location. | A2/V2; explicit approval naming source, destination, intended effect, and resource/retry limits. | V3: inspect download outcome and partial effects; confirm revision and expected files. Do not retry blindly. | not started | pending |
| A4 — Verify downloaded files | Prepare a bounded inventory with actual paths, sizes, purposes, revision, and recoverable file identifiers. | Successful A3/V3 and retained download information. | V4: compare the inventory against the approved file set and reliable source metadata where available. Disclose what integrity checks establish and any missing source hashes. | not started | pending |
| A5 — Check local CPU loading | Load the tokenizer and model from verified local assets, with network access disabled for the loading procedure. Record actual loading settings, CPU device, dtype, and outcome. | A4/V4; sufficient resources and separately approved loading command. | V5: confirm the selected assets load locally, model parameters are on CPU with the agreed dtype, and no forward/generation call occurs. Do not treat class imports alone as loading proof. | not started | pending |
| A6 — Record and review closure | Save commands, versions, provenance, results, limitations, approvals, and the learning boundary; present the pass for review. | Earlier action/check outcomes. | V6: review completion requirements, evidence completeness, and remaining operations. Close only with the learner's agreement. | not started | pending |

All checks are **not run**. Explain exact commands and check expectations before each execution approval. After each action/check pair, report and stop before the next action. Use separate action and verification artifacts when executable scripts are appropriate.

## Limits and stopping points

No package upgrades, system changes, new cloud resources, paid service setup, publication, or release are included. No inference, repeated generation, or new internal model lesson is included. Preserve the selected model and CPU choice.

Stop if files are missing, the revision cannot be established, source requirements conflict with scope, resource limits are unclear, a check fails, remote executable code appears necessary, or loading requires dependency changes. Report partial effects and save a failure checkpoint. Recovery outside the approved branch requires new authority; revise the plan if the approach changes materially.

Downloaded files persist at the agreed destination. The assistant must report partial downloads and running processes at handover. Cleanup or replacement of existing files requires applicable authority. A failed loading check does not justify redownloading or reinstalling without diagnosis and approval.

## Readiness, completion, and risks

Execution readiness is pending: exact destination, model revision, required files, and resource limits are unresolved. A1 can establish local facts within its separately approved scope; A2 resolves download decisions before A3.

Completion requires agreed assets obtained, provenance and inventory retained, file checks and local CPU loading passed, reproducible commands recorded, limitations disclosed, and learner closure review completed. Loading success does not establish inference correctness or understanding. G1–G6 remain outstanding.

| Risk trigger | Consequence | Mitigation and owner | Unresolved decision |
| --- | --- | --- | --- |
| Files or loaded model exceed available resources | Partial download or failed load | Assistant checks requirements before download/loading and stops on failure. | Destination and resource limits. |
| Published branch changes | Unclear reproduction identity | Assistant resolves and retains an exact source revision. | Revision to use. |
| Current packages cannot load selected files | Loading blocked | Assistant preserves evidence and proposes the smallest diagnosis; no automatic dependency change. | Resolve only if observed. |

## Approvals

- **Documentation preparation — approved:** learner requested “create pass-003 place holders and fill the plan.md file with action iteams”. This authorizes creating the four required records and populating this proposed plan. Empty creation was checked before population.
- **Plan r01 — pending:** the request to save the proposed actions is not recorded as separate approval of the complete detailed plan or unresolved download settings.
- **Execution — pending:** no storage investigation, metadata lookup, download, file verification, or model loading has been performed under this plan.

Next action: review and agree on r01, then explain A1/V1 and obtain execution approval. Keep pass-003 reserved until the first execution approval checkpoint.
