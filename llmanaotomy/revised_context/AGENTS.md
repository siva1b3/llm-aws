# Revised project context

Use this file as the entry point for the revised context.
This package reorganizes the original documentation. It does not report new inference execution.

## Read in this order

1. Read [current progress](project-status/current-progress.md) for the checkpoint and next action.
2. Read [project guide](project-rules/project-guide.md) for the goal, scope, and success criteria.
3. Read [learning method](project-rules/learning-method.md) before teaching.
4. Read [implementation rules](project-rules/implementation-rules.md) before coding, inspection, or evidence capture.
5. Read [technical reference](project-rules/technical-reference.md) before environment, model, or dependency work.
6. Read [evidence navigation](project-evidence/README.md) before using earlier findings.

For design work, read the applicable HLD or LLD status first.
Then read its agreed design and supporting evidence using the table below.

For short follow-ups, read current progress and the applicable document only.
Inspect available code and terminal state before assuming a planned action occurred.
Resume at the last learner-confirmed boundary.

## Three adjacent folders

| Folder | Purpose | Update trigger |
| --- | --- | --- |
| [project-rules](project-rules/README.md) | Stable contract: goals, teaching agreements, implementation rules, and selected technical choices | An agreement or governing decision changes |
| [project-status](project-status/README.md) | Current state: active work, verified progress, questions, blockers, and next action | Meaningful progress or a handoff |
| [project-evidence](project-evidence/README.md) | Accumulated evidence: methods, observations, results, interpretations, and limitations | Actual inspection, experiment, or correction |

Choose one owning document for each rule or fact. Link to that owner elsewhere.
Keep READMEs for navigation. Update this file when routing or ownership changes.

## HLD and LLD ownership

HLD means high-level design: components, responsibilities, boundaries, and overall flow.
LLD means low-level design: interfaces, implementation details, and checks for the current operation.

| Design | Agreed contract | Current state | Supporting evidence |
| --- | --- | --- | --- |
| HLD | [Rules](project-rules/hld.md) | [Status](project-status/hld.md) | [Evidence](project-evidence/hld.md) |
| LLD | [Rules](project-rules/lld.md) | [Status](project-status/lld.md) | [Evidence](project-evidence/lld.md) |

Update design rules after an agreed decision changes boundaries or interfaces.
Update design status when work, unresolved choices, or verification states change.
Add evidence only after inspection or execution produces observations.
The initial evidence files contain source reviews, not inference results.
Keep the overall checkpoint in [current progress](project-status/current-progress.md).
Link design-specific details instead of copying them into that checkpoint.

## Ownership and decisions

Higher-priority instructions and the user's latest explicit request take precedence over project documents.
Each document owns its stated domain. This file routes readers to those documents.
Status notes do not authorize changes to scope, teaching agreements, or major technical choices.
Update the applicable rule document after an agreed change. Then update affected status and evidence.

Prefer direct, reproducible runtime evidence when documents disagree about execution.
Reconcile the conflicting records explicitly.
Present conflicts about objectives or teaching agreements to the learner.
Do not silently choose an incompatible interpretation.

Keep planned, implemented, executed, technically verified, and learner-confirmed states distinct.
Never invent runs, values, versions, assets, or learner confirmations.
Follow the confirmation gate in [learning method](project-rules/learning-method.md#stop-at-the-learning-boundary).

## Package boundaries and writing

The original [sources](../sources/AGENTS.md) remain unchanged.
Keep this project's new output inside `revised_context/`.
The user also authorized updating the root [project context guide](../project-context-guide.md) for HLD/LLD organization.
Preserve the original source files and root `asd-ste100.md`.
This guide update is the sole authorized exception outside `revised_context/` for the design extension.

Use [project-context-guide.md](../project-context-guide.md) for organization.
Use [asd-ste100.md](../asd-ste100.md) for direct sentences and consistent terms.
Limit instruction sentences to 20 words and description sentences to 25 words when meaning permits.
Keep conditions, uncertainty, and technical precision. Explain any necessary departure from the length limits.
Define necessary technical terms. Separate observations, expectations, examples, and instructions.
This package uses STE-inspired guidance. It does not claim official ASD-STE100 compliance.

See [context review](project-evidence/context-review.md) for source mapping and revision checks.
