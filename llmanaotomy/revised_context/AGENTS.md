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
The user limits this revision's output to `revised_context/`.
Do not edit original source files or the two root guidance files for this revision.

Use [project-context-guide.md](../project-context-guide.md) for organization.
Use [asd-ste100.md](../asd-ste100.md) for direct sentences and consistent terms.
Limit instruction sentences to 20 words and description sentences to 25 words when meaning permits.
Keep conditions, uncertainty, and technical precision. Explain any necessary departure from the length limits.
Define necessary technical terms. Separate observations, expectations, examples, and instructions.
This package uses STE-inspired guidance. It does not claim official ASD-STE100 compliance.

See [context review](project-evidence/context-review.md) for source mapping and revision checks.
