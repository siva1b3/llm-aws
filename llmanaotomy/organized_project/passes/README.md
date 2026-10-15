# Pass records

No pass is active. pass-001 is reserved.
Its plan, verification, findings, and evidence files remain at zero bytes.

Use sequential pass identifiers. Never reuse an identifier for unrelated work.
Keep all records and supporting evidence directly inside the pass folder.
Do not create pass subfolders.
Later passes require plan.md and verification.md; add findings.md and evidence.md when needed.
Code, tests, design, and infrastructure are deliverables with separately agreed locations.

## Responsibilities

- plan.md: outcome, revision, authority, actions, checks, limits, assumptions, and design references.
- verification.md: expected results, actual observations, context, limitations, reviews, and closing decision.
- findings.md: detailed observations and unresolved questions.
- evidence.md: index and context for retained supporting evidence.

## Lifecycle

Follow the [core workflow](../../software_files/core/software-ai-development-workflow.md).

1. Propose a bounded outcome without drafting its action sequence.
2. Obtain planning authority before activating the reserved pass or drafting its plan.
3. Record planning authority and update current state.
4. Draft the recoverable plan revision and checks; mark expected results as not run.
5. Present scope, actions, affected paths, limits, checks, and external effects.
6. Obtain separate execution authority for that revision.
7. Execute within authority and maintain records before dependent work.
8. Verify actual observations against expectations; preserve failures and limitations.
9. Record reviews, remaining criteria, decision owner, and closing outcome.
10. Close as successful, unsuccessful, deferred, or cancelled.
11. Set active pass to none and obtain human confirmation before starting another pass.

Paused or blocked passes remain active.
Record approval type, response, scope, revision, date, time zone, and recoverable reference.
Do not repeat approval requests for unchanged authorized work.
Changed actions or checks require revised execution authority.
Preserve earlier revisions and closed history; correct closed records with dated notes.

Link criterion IDs to design, planned actions, and check evidence during authorized stages.
Each check records identifier, method, expectation, observation, status, context, evidence, and limitation.
Use core check statuses and preserve superseded results with replacement links.
Keep engineering closure separate from learner confirmation.
Preparation checks belong to current-state, not this inactive pass.
