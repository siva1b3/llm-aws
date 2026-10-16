# Pass-002 — CPU dependency setup

Revision: r01 · 2026-10-16 (Asia/Calcutta).
Disposition: reserved and inactive. Scope approved; execution approval pending.

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
