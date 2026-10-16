# Pass-003 — Understand and prepare the model assets

Revision: r02 · 2026-10-16 (Asia/Calcutta).
Disposition: proposed, reserved and inactive. Saved at the learner's explicit request. Plan approval and execution approval remain pending.
Previous proposal: [r01](plan-r01.md), preserved unchanged. This revision makes the teacher and learner roles explicit; the selected model and technical execution boundary remain unchanged.

## Outcome and scope

Prepare the selected model and tokenizer for our first working demonstration. Download the agreed files, verify them, and confirm they load locally on CPU. Help the learner understand what the assets are for and how files on disk differ from a model loaded into memory.

The assistant is the teacher and technical guide. Setup serves the learning goal. Begin with the whole model as a closed box:

```text
Input text → [ model: closed box ] → one predicted token
```

This pass prepares that demonstration; it does not run it. Explain only the assets and loading needed now. Keep tokenization mechanics and internal model computations closed until their later lessons.

This is teaching and model-asset setup in the existing repository and code/.venv. It supports prerequisites for [G1–G6](../../definition/goal.md); it does not satisfy those criteria. Follow [constraints](../../definition/constraints.md) and the [roadmap](../planned_roadmap.md).

## Teacher and learner responsibilities

- **Assistant — teacher, technical guide, and implementer:** explain the purpose before each action, make and explain proportionate technical choices, provide or run the authorized small commands, show bounded actual results, and check them separately. Adapt explanations to questions and stay with the current concept when it is unclear. Own the checkpoint records.
- **Human — learner and decision owner:** inspect the observations, ask questions, explain or explicitly confirm understanding, and decide when to advance. Own plan approval, each execution approval, acceptance, and pass closure. The learner is not expected to write setup code or choose unfamiliar settings without guidance.
- Technical checks, learner understanding, acceptance, and execution authority are separate. A successful command or silence does not establish understanding or permit advancement. No additional contributor is assigned.

## Inputs and design

- Selected model: HuggingFaceTB/SmolLM2-135M-Instruct, from its Hugging Face repository. Exact revision and required file set must be established before download.
- Existing dependencies: pass-002 verified CPU torch 2.12.1+cpu, transformers 5.20.0, pandas 3.0.6, and numpy 2.5.4. See its [evidence](../pass-002/evidence.md). These checks did not establish model loading compatibility.
- Use CPU and prefer FP32. Explain the agreed numerical precision briefly when loading becomes relevant. Do not enable arbitrary remote model code.
- Design boundary: model files on disk → local tokenizer/model objects and parameters in RAM → loading check only. Downloading and loading are separate actions. No forward call, generation, prompt tokenization lesson, or architecture dissection.
- Explain the broad purposes of weights, configuration, and tokenizer assets in A2. Use a bounded inventory of actual files in A4. Do not assume each category is a separate file or expose internal computations.
- Asset/cache location, exact executable artifacts, disk and memory allowance, download bytes, time limit, retry limit, and partial-download cleanup terms remain to be agreed before affected execution. The assistant proposes settings with reasons; do not invent agreed values.
- Expected affected paths: this pass's records and later agreed setup scripts and asset/cache location. Inspect existing files and relevant user changes before edits. Preserve all unrelated changes, including prior pass records and setup files.

## Actions and separate checks

All actions belong to this pass. Each requires its own execution approval covering the named action and separate check.

| Action | Purpose and expected output | Inputs and dependency | Separate technical check and expected result | Learner check and stopping point | Status | Execution authority |
| --- | --- | --- | --- | --- | --- | --- |
| A1 — Choose where the files will live | Explain the proposed storage location. Inspect existing files, free disk space, and memory. Explain why downloading uses disk and loading uses RAM. | Existing repository and environment; explain the proposed location and exact inspection scope before execution approval. | V1: confirm the actual target, available resources, and existing files. Report unknowns; capacity cannot be judged sufficient before file-size requirements are known. | Ask the learner to explain or confirm the disk/RAM distinction. Resolve questions, report V1, and stop before A2. | not started | pending |
| A2 — Understand and agree on the download | Identify the exact revision, compatible tokenizer, required files, source, sizes, and relevant use terms. Explain the broad purposes of the assets and propose a destination and resource limits. | A1/V1 and learner confirmation of its concept; separately authorized read-only source/metadata review. | V2: compare the proposed download with the selected model, available resources, trust requirements, and agreed limits. Stop until blocking choices are resolved. | Ask the learner to explain or confirm the basic asset purposes. Agree on download settings, report V2, and stop before A3. | not started | pending |
| A3 — Download the agreed files | Obtain only the agreed model, configuration, and tokenizer assets at the agreed location. Show the actual outcome. | A2/V2, its learner confirmation, and resolved settings; explicit execution approval naming source, destination, intended effect, and resource/retry limits. | V3: inspect download outcome and partial effects; confirm revision and expected files. Do not retry blindly. | Explain the observed result. Report V3 and stop before inspecting the assets further in A4. | not started | pending |
| A4 — Inspect what actually arrived | Show a short directory tree, actual paths, sizes, purposes, revision, and recoverable file identifiers. Connect real files to the broad purposes discussed in A2. | Successful A3/V3 and retained download information. | V4: compare the inventory against the approved file set and reliable source metadata where available. Disclose integrity-check limits and missing source hashes. | Ask the learner to connect the observed assets to their purposes. Resolve discrepancies and questions, report V4, and stop before A5. | not started | pending |
| A5 — Load the files into memory | Load the tokenizer and model from verified local assets with network access disabled for the loading procedure. Explain what loading places in RAM and what remains on disk. Record actual settings, CPU device, dtype, and outcome. | A4/V4 and learner confirmation; sufficient resources and separately approved loading command and limits. | V5: confirm local loading, parameters on CPU with the agreed dtype, and absence of forward/generation calls. Class imports alone are not loading proof. | Ask the learner to distinguish downloaded, loaded, and executed. Resolve questions, report V5, and stop before A6. | not started | pending |
| A6 — Review what we learned and record closure | Recap the asset purposes, locations, loading commands, and observed results. Save provenance, versions, checks, approvals, limitations, and the learning boundary. | Earlier action/check outcomes and learner confirmations. | V6: review completion requirements, evidence completeness, reproducible instructions, and remaining operations. | Review understanding with the learner. Present the pass for acceptance and closure; close only with the learner's agreement. Do not begin inference. | not started | pending |

All technical checks are **not run**. All learner confirmations for this pass are **pending**. State the exact commands, expected results, limits, and stopping point before each execution approval. Keep action and check procedures distinct, including separate executable artifacts when appropriate.

After each action/check pair, report actual results and limitations, then wait. If a concept is unclear, explain it differently within the same boundary; new execution still needs applicable approval. Keep routine exchanges in conversation until a defined checkpoint. Never record an expected result as observed.

## Limits and stopping points

No package upgrades, system changes, new cloud resources, paid service setup, publication, or release are included. No inference, repeated generation, or internal model lesson is included. Preserve the selected model and CPU choice.

Stop if files are missing, the revision cannot be established, source requirements conflict with scope, resource limits are unclear, a check fails, remote executable code appears necessary, or loading requires dependency changes. Report partial effects and save a failure checkpoint. Recovery outside the approved scope requires new authority; revise the plan if the approach changes materially.

Also stop at each learner check until explicit understanding is confirmed. Do not infer understanding from agreement to run a command. Do not advance merely because a technical check passes.

Downloaded files persist at the agreed destination. Report partial downloads and running processes at handover. Cleanup or replacement of existing files requires applicable authority. A failed loading check does not justify redownloading or reinstalling without diagnosis and approval.

## Readiness, completion, and risks

Execution readiness is pending: exact destination, model revision, required files, and resource limits are unresolved. A1 can establish local facts within its separately approved scope; A2 resolves download decisions before A3.

Completion requires agreed assets obtained, provenance and inventory retained, file checks and local CPU loading passed, reproducible commands recorded, limitations disclosed, and learner closure review completed. The learner must explicitly confirm the asset purposes and the distinction between files on disk, loaded objects in RAM, and execution. Loading success does not establish inference correctness or understanding of model computations. G1–G6 remain outstanding.

| Risk trigger | Consequence | Mitigation and owner | Unresolved decision |
| --- | --- | --- | --- |
| Files or loaded model exceed available resources | Partial download or failed load | Assistant checks requirements before download/loading and stops on failure. | Destination and resource limits. |
| Published branch changes | Unclear reproduction identity | Assistant resolves and retains an exact source revision. | Revision to use. |
| Current packages cannot load selected files | Loading blocked | Assistant preserves evidence and proposes the smallest diagnosis; no automatic dependency change. | Resolve only if observed. |
| Setup introduces too many concepts or technical success is mistaken for understanding | Learning boundary is crossed without understanding | Assistant teaches only the current asset or loading concept and asks for explicit learner confirmation. Learner controls advancement. | Actual understanding remains pending. |

## Revision history and approvals

- **r01 — preserved:** original proposed asset-setup plan is retained in [plan-r01.md](plan-r01.md). It was not approved for execution.
- **r02 — proposed:** reframed in conversation around the assistant as teacher and the human as learner. Adds explicit learning objectives and confirmation gates while preserving the technical scope.
- **Original documentation preparation — approved:** learner requested “create pass-003 place holders and fill the plan.md file with action iteams”. Empty creation was checked before population, as recorded in r01.
- **r02 documentation update — approved, 2026-10-16:** learner requested “update plan.md , planned_roadmap by considering role into the picture and consider me also leraner”. This authorizes saving this proposal and updating the roadmap, with related status references aligned. It does not authorize model-asset execution or establish learner understanding.
- **Plan r02 — pending:** the documentation request is not recorded as separate approval of the detailed plan and unresolved settings.
- **Execution — pending:** no storage investigation, metadata lookup, download, file verification, or model loading has been performed under this plan.
- **Learner confirmations and acceptance — pending:** none established for pass-003.

Next action: review and agree on r02, then explain A1/V1 and obtain execution approval. Keep pass-003 reserved until the first execution approval checkpoint.
