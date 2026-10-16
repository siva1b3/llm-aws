# Pass-001 — Check environment readiness

## Current revision r02 — 2026-10-16 (Asia/Calcutta)

This revision supersedes r01's starting assumptions and next action. The original revision and its approval are preserved below.

- State: reserved and inactive. No remote project execution has started.
- Authority: the human supplied the changed setup and explicitly requested an update to the state and pass-001 plan. This authorizes saving this route and checking the documentation; it does not authorize project execution.
- Outcome: establish the minimum local facts needed to choose the next setup step, using Codex CLI directly on the supplied EC2. Infrastructure discovery stays small.
- Scope: read-only readiness discovery supporting [G1–G6](../../definition/goal.md), without satisfying those learning criteria. [Constraints](../../definition/constraints.md) retain ownership of selected settings and teaching gates.
- Inputs: the human reports a running Ubuntu EC2 (`t3.latge`, understood as `t3.large` pending verification), Codex CLI installed, and the Git repository containing `llm-inference-anatomy`. They report no project environment, project dependencies, or model assets. These are reported facts, not runtime evidence.
- Target: the future Codex CLI session on that EC2. Exact instance identity, OS release, repository path, branch, existing changes, Python, packages, and capacity remain uninspected. This documentation session is in Windows and has not connected to EC2.

| Action | Purpose and output | Inputs and dependency | Separate check and expected result | Status | Execution authority |
| --- | --- | --- | --- | --- | --- |
| A2 — Check the local starting point | Inspect local identity/OS, CPU, memory, disk, Python, relevant packages, repository path, branch, and existing changes through bounded read-only commands. Produce a short readiness summary. | Human opens Codex CLI in the EC2 repository. Explain exact target, commands, and limits before execution approval. The earlier A1 question is superseded by the human's supplied information; no execution of A1/V1 is claimed. | V2 — Separately compare observations with selected requirements. Report ready, missing, and unknown items and one recommended next action. Expected: enough evidence to propose the smallest setup step. | not started | pending |

V2 is not run. Assistant owns the instructions, comparison, evidence, and checkpoint. The human supplies the EC2 session. No contributors are added. Retain exact commands, relevant output, environment context, and limitations after authorized execution.

Design note: inspect existing local state, then compare it separately with requirements. No model operation is opened. Main risk: checking the wrong machine or checkout. Trigger: unclear target; consequence: misleading evidence; mitigation: establish local identity and repository first and stop on uncertainty. Owner: assistant.

Limits: no provisioning, AWS configuration changes, environment creation, installation, model download, inference, commit, push, or release. Preserve existing files and configuration. Do not collect credentials. No paid resource creation, spending commitment, or instance lifecycle action is authorized. Future download/compute limits remain to be agreed when applicable.

Stop after A2/V2 and report before the next action. If access fails, the target is unclear, or observations conflict with requirements, stop and propose the smallest recovery. No automatic retry or repair is approved.

2026-10-16 approval record: the human reported the setup, said future work would happen directly in the AWS EC2 CLI, and asked to “update the state , plan-001”. This authorizes this documentation checkpoint. Remote execution approval remains pending. The other three pass records remain empty and the learning boundary is unchanged.

Next action: from the future EC2 Codex CLI session, explain A2/V2 and request their execution approval. Save applicable approval and activate the pass before execution.

## Preserved historical revision r01

The following record retains the earlier plan and its approval. Its starting assumptions and next-action directions are superseded by r02 above.

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
