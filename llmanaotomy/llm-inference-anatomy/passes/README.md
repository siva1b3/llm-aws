# Pass records and lifecycle

Read [operating instructions](../AGENTS.md) and [current state](../status/current-state.md).
Pass-001 and pass-002 are closed successfully. No pass is active. Pass-003 is reserved for model assets. Its proposed [r02 plan](pass-003/plan.md), saved at the learner's request, makes teacher/learner roles explicit. The [r01 proposal](pass-003/plan-r01.md) is preserved; the three supporting placeholders remain empty. No execution is approved. See [pass-002 closure](pass-002/plan.md) and [current state](../status/current-state.md). Reservation is not activation.
The plan may be saved before execution when explicitly requested by the learner. Other reserved records stay empty until needed at an approved checkpoint.
Use [planned roadmap](planned_roadmap.md) for the overall sequence; keep each pass plan limited to its own work.
Keep one pass open, including while blocked or paused. Close or defer it by agreement before opening another.
Use sequential identifiers and never reuse one for different work.
Keep records and optional supporting artifacts directly inside each pass; no nested folders.
Keep application deliverables in later agreed locations and link recoverable versions.

Required records: plan.md, verification.md, findings.md, and evidence.md.
Populate findings when needed for detailed observations/questions; use an evidence index when multiple artifacts need context.
Do not insert speculative templates into reserved records.

Discuss the plan, obtain plan approval, explain one action and separate check, then obtain execution approval.
Save the agreed plan and first execution approval before execution; activate the pass at this checkpoint.
Record expected checks as not run. Report results and obtain approval for every next action.
Follow the precise checkpoint event table in operating instructions; do not save routine exchanges after every success.

### Plan

Include the pass outcome, related goal criteria, work type, revision, inputs, environment, and assumptions.
Include dependencies, resource limits, stop conditions, and recovery options when relevant.
Use a table separating actions and checks:

| Action | Purpose and output | Dependency | Separate check | Status | Execution authority |
| --- | --- | --- | --- | --- | --- |
| A1 | Actual agreed work | Actual prerequisite | V1 | not started | Actual response and scope, or pending |

Use action states `not started`, `in progress`, `done`, and `blocked`.
An action marked `done` is not automatically verified or accepted.
Record plan approval separately from execution approvals.
Each approval entry identifies the revision, request, actual response, scope, status, and available date and reference.
Use approval statuses `pending`, `approved`, `changes requested`, `rejected`, or `withdrawn`.
Keep earlier approved revisions recoverable within the pass folder.

### Verification and evidence

For each check, record its identifier, criterion, method, inputs, expected result, actual result, checker, context, evidence, and limitations.
Use check statuses `not run`, `running`, `passed`, `failed`, `blocked`, `inconclusive`, `not applicable`, or `superseded`.
Explain exclusions and superseded results.
Record human review separately from AI verification.
Keep actual results as `not run` until observations exist.

An exit code, generated test, screenshot, or source inspection supports only what it actually establishes.
Tie claims to the tested output, relevant conditions, and recoverable evidence.
Recheck affected claims after relevant changes under applicable execution approval.
Do not rerun unrelated checks without a reason.

Use findings for detail that does not fit clearly in verification.
Use an evidence index when multiple supporting artifacts need context.
Keep supporting files directly inside the pass folder.
Retain evidence needed by later decisions, claims, or handovers.

## Closure and project report

Save accumulated results, approvals, decision owner, reason, reviews, unfinished criteria, and next action at closure.
Closed outcomes are successful, unsuccessful, deferred, or cancelled. Paused passes remain open.
Resolve or explicitly transfer control of running operations before closure.
Preserve failures and prior revisions. Correct historical mistakes with dated additions.
A successful pass does not establish whole-project completion.
Populate the final report only at completion or authorized early closure.
Include agreed goal, outputs/versions, criterion results, evidence, acceptance, ownership, and limitations.
At early closure, distinguish delivered, accepted, unfinished, and unverified work.
Do not remove unmet criteria without an authorized goal change.
