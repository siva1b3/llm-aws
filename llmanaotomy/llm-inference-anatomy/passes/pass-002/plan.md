# Pass-002 — understand and prepare the CPU runtime

Updated: 2026-10-16 17:41:42 Asia/Calcutta (UTC+05:30)
Revision: r02.
Pass state: open, waiting for A1 execution approval.
Outcome: learner understands the Python runtime, library roles, installation boundaries, and a basic CPU tensor calculation; runtime readiness is checked.
Related criteria: prerequisite for G1 and learning method supporting G2; this pass does not establish a model forward pass.
Plan approval: approved in context by the human's request to save the revised pass-002 and current state after reviewing r02.
Execution approval: pending for every action. Saving records grants no runtime execution authority.
Record timing: saved before execution at the human's explicit request; this overrides the default first-execution-checkpoint timing for these records only.

## Parties, roles, responsibilities, and boundaries

- H: human in this chat, learner and approval/acceptance owner; inspects evidence and explicitly confirms understanding.
- C: Codex, teacher first, technical guide, implementer, verifier, and record keeper; teaches one concept at a time within approved execution scope.
- E: current Ubuntu CLI workspace `/home/ubuntu/llm-aws/llmanaotomy/llm-inference-anatomy`, identified by H as EC2; operated through C tools; hosts runtime and records. AWS specifications remain unverified.

C performs technical checks. H owns learning confirmation, action approval, acceptance, and closure.
Stop after each action's explanation and check until H confirms understanding and authorizes advancement.
No model download, model loading, inference, provisioning, or instance changes are in scope.

## Actions

| Action | Expected learning outcome and work | Assignments | Separate check and expected result | Dependency / limit | Status / execution authority |
| --- | --- | --- | --- | --- | --- |
| A1 | Understand virtual-environment isolation; inspect Python/pip paths and compare with system Python | H learns/confirms; C teaches/inspects/checks; E runs inspection | V1: paths identify project Python/pip; H explains which Python runs the project | Prior venv evidence available; no package changes | not started; execution pending |
| A2 | Understand PyTorch and Transformers roles; inspect availability inside .venv | H learns/confirms; C teaches/inspects/checks; E hosts packages | V2: actual availability and versions reported; H explains each library's purpose | A1 technical check and learner confirmation | not started; execution pending |
| A3 | Understand installation location and version reproducibility; install agreed missing CPU dependencies if necessary | H approves/learns/confirms; C teaches/installs/checks; E hosts .venv | V3: imports, versions, and locations verified; H explains where packages were installed | A2 confirmed; exact installation commands require approval; if unnecessary, explain and record not applicable | not started; execution pending |
| A4 | Understand tensor values, shape, dtype, and device through one small CPU calculation | H predicts/explains/confirms; C teaches/implements/checks; E executes calculation | V4: actual values match stated expected values and device is CPU; H explains properties | A3 readiness check and learning gate resolved; no model loading | not started; execution pending |
| A5 | Understand what runtime readiness proves; assemble evidence-backed explanation and update records | H explains/accepts; C teaches/reviews/records; E stores outputs | V5: records agree with evidence; H explains setup and explicitly accepts closure | A4 technical check and learner confirmation; no model download/inference | not started; execution pending |

Inputs: pass-001 verified .venv Python 3.12.3 and pip 24.0; package availability inside .venv remains unchecked.
Unexpected failures stop dependent work; proposed recovery requires applicable approval.
First next step: explain A1's inspection and check, obtain execution approval, then perform it. Do not recreate the verified venv.
