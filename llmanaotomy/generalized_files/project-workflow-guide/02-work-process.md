# Work process

Apply the [system laws](01-system-laws.md) before every pass and action.
Follow [record ownership](03-record-structure.md) and [update timing](04-checkpoints-and-history.md) throughout this process.

## Small actions and separate checks

Keep one pass open at a time. A pass may contain several actions.
An action does not need its own pass.
Divide every type of work into manageable actions with inspectable outcomes and appropriate checks.
Check a result before beginning dependent work.
Avoid splitting trivial internal operations that produce no useful independent result.

Each action identifies its roles, outcome, inputs, dependencies, work, separate check, expected result, limits, and stopping point.
Use concise fields. Include dependencies and limits only when applicable; do not fill records with irrelevant boilerplate.
An action and its stated check may share one execution approval, but their procedures and results remain distinct.
When executable artifacts are appropriate, separate the action artifact from the verification artifact.
Give them descriptive names and a clear execution order.
Use inspection, comparison, demonstration, or human review where scripts would not help.

Do not combine dependent stages into one unattended operation that crosses human review boundaries.
Do not generate a master procedure that bypasses approvals.

| Work | Example action outcome | Separate check | Later outcome requiring another execution approval |
| --- | --- | --- | --- |
| Research | Claims extracted from an agreed source | Compare claims with supporting passages | Comparison against another source |
| Writing | One agreed section drafted | Review requirements, support, and clarity | Dependent section drafted |
| Analysis | Input data prepared | Check completeness and data rules | Requested calculation completed |
| Design | Proposed layout created | Inspect against agreed requirements | Selected layout developed |
| Learning | One concept explored through an exercise | Check learner explanation or performance | Next concept introduced |
| Software | Required asset downloaded | Verify the asset | Asset installed or used |

## 1. Discuss roles, outcomes, and the plan

A request to draft a plan authorizes drafting within that request, including revisions during the same discussion.
Do not request redundant permission for the requested draft.
If only background material was supplied, establish the intended task before starting project work.

Present pass roles and outcome first, followed by scoped action items, checks, dependencies, and limits.
Assign a revision identifier, such as `r01`, so approvals have clear targets.
Keep draft plans and unapproved revisions in conversation.
Do not write draft content into Markdown or activate a reserved pass because a draft exists.
Preserve the current approved plan while discussing a replacement.
The snapshot may reflect changed pending status under the checkpoint rules; it must not contain the unapproved draft.

Use supplied information and current records to propose the plan.
Do not investigate or implement project work under the label of planning.
Propose additional investigation as its own action, with roles, outcome, approval, and check.

## 2. Obtain plan approval

Ask the human to approve the revision or request changes.
Interpret their response in context. Silence is not approval.
Approval covers the declared approach, participants, roles, boundaries, and expected outcomes.
Execution approval remains a separate decision.

## 3. Explain and approve the next action

After plan approval, explain the next action and check in simple terms.
Confirm its party-to-role assignments, intended outcome, changes, limits, and stopping point.
Name any external destination, intended effect, resource use, and required human review.

Use a compact request such as:

> Role/outcome: state the assigned role and intended result.
> Pass/revision/action: identify the exact scope.
> Parties: name each human, agent, and environment with its assigned role.
> Work and check: explain each separately, including the expected check result.
> Limits and stop: identify the boundary and what happens after reporting.
> Decision: approve this action and check, or request changes.

Obtain final execution approval before acting.
An approval for another action does not cover this action.
Retain valid approval across sessions when revision, scope, roles, and conditions remain unchanged.
Do not request the same unchanged approval again.

## 4. Commit and execute

After the first execution approval, save the agreed plan and approval at a plan checkpoint before execution.
This required record update needs no additional approval request.
Activate the reserved pass at this checkpoint. Record expected checks as `not run`.
Perform only the approved action and check, within the declared roles and boundaries.

Report actual outcomes, check results, and limitations.
Wait for execution approval before the next action, even when that action appears in the approved plan.
Successful verification grants neither acceptance nor permission to advance.
Keep required human or learner review pending until it actually occurs.

## 5. Change the agreed scope

Stop work that depends on the changed part.
Review changed outcomes, roles, environments, dependencies, checks, limits, and recovery options.
Material changes require a revised plan and new plan approval, followed by execution approval for the affected scope.
Discuss the revision in conversation, then save its checkpoint before changed execution.
Preserve earlier approved revisions using the history rules.
Do not apply old execution approval to changed work.

If approval is withdrawn, stop affected work and save a checkpoint.
Resume only with applicable authority.
Use the [failure and closure procedures](05-failures-and-closure.md) for recovery, handover, and ending work.
