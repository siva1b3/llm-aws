# Pass-001 — Check environment readiness

- Revision: r01.
- Saved: 2026-10-15 (Asia/Calcutta), at the human's explicit session-handover request.
- State: reserved and inactive. Execution is deferred until a later session.
- Plan approval: approved. The human liked the proposed plan and explicitly asked to save it in pass-001 for later execution.
- Execution approval: pending for every action and check. No project action has run.

## Outcome and scope

Find out whether the selected AWS computer exists, can be accessed, and has the software and capacity needed for the learning exercise. Produce a short readiness summary showing what is ready, what is missing, and one recommended next action.

This is read-only environment discovery supporting [goal criteria G1–G6](../../definition/goal.md); it does not satisfy those criteria itself. Selected settings and teaching limits remain owned by [constraints](../../definition/constraints.md).

Repository: this project. Target: the selected EC2 environment, whose actual identity and access remain unknown. No application paths or existing user files will be changed. Existing remote files and configuration are unknown and will be inspected only within later approved scope. No branch, commit, or release policy is selected by this plan.

## Small actions and separate checks

| Action | Purpose and work | Inputs and dependency | Separate check and expected result | Status | Execution authority |
| --- | --- | --- | --- | --- | --- |
| A1 — Establish access information | Ask whether an EC2 instance already exists and how the learner normally connects. Identify missing information without collecting credentials. | Learner's account of their setup; no machine access assumed. | V1 — Review the supplied information for whether it is sufficient to propose a specific connection step; distinguish reported facts from verified access. | not started | pending |
| A2 — Inspect the environment | Inspect machine identity, OS, CPU, memory, disk, Python, relevant packages, and existing project files using read-only commands. | A1/V1 complete; access established and exact destination, commands, and limits explained before separate execution approval. | V2 — Compare observed output with selected settings; report ready, missing, and unknown items. | not started | pending |

Assistant owns the instructions, checks, readiness summary, and checkpoint updates. The learner supplies setup information and runs commands when direct authorized access is unavailable. No additional contributors are authorized.

V1 and V2 are not run. Retain actual commands, outputs, environment identity, and limitations in pass evidence when execution occurs. No executable artifacts are needed for the initial conversation; later inspection commands and their comparison procedure must remain distinct.

## Limits and stopping points

- After A1/V1, report the result and stop for the next execution approval.
- If no instance exists or access is unavailable, stop and agree on the needed setup separately. This plan does not authorize provisioning or access changes.
- Before A2, establish the specific access route and target. Do not guess credentials, machine identity, ownership, or capacity.
- Do not create resources, install packages, download model assets, or run inference.
- Do not alter existing files, configuration, or selected technical choices. No migration, rollout, release, or recovery mutation is included.
- No paid resource creation or spending commitment is authorized. Resource ownership and applicable cost limits remain unresolved before any action requiring them.
- On failed access or unexpected output, stop dependent work, preserve the result, and propose the smallest recovery separately. No automatic retries or repair branch is approved.
- After A2/V2, report readiness and propose one next action; obtain human review before closure or further work.

## Proportionate design and risk note

This pass gathers environment facts and compares them with existing requirements. It makes no model design changes and opens no internal model operation. Inputs are learner statements and, later, read-only command output; the output is a bounded readiness summary.

The main risk is inspecting an unintended machine or exposing sensitive information. Trigger: uncertain target or output containing credentials. Consequence: unreliable evidence or disclosure. Mitigation: establish destination before commands, keep secrets out of requests and records, and stop on uncertainty. Assistant owns this check; actual access and resource ownership remain unresolved.

## Approval and handover history

2026-10-15: r01 was proposed in conversation and restated in simple language without changing scope. The human then said: “us this in the pass-001 files i liked the idea we will execute it later for now I liked the idea and i am clossing the seession”. This is recorded as plan acceptance and explicit authority to save the handover, not execution approval.

Saving this approved plan before execution approval is a specific exception requested by the human to the usual empty-reserved-record rule in [pass instructions](../README.md). The pass remains inactive. Other pass records remain empty until needed at the execution checkpoint. This does not change the general workflow.

No runtime observations, learner confirmations, or project completion claims were added. The entire single-forward-pass model remains the current learning boundary.

Next action: explain A1 and V1 and request their execution approval. Do not repeat plan approval for unchanged r01. Once approved, record that authority and activate the pass before performing A1/V1.
