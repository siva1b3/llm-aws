# Workflow

## 1. Load Relevant Context

On startup, read `protected/laws.md` and `protected/file_structure.md`. For step planning, changes, execution, or retries, read the goal, this workflow, current state, action plan, and the relevant step history and implementation files. Reuse current contents already in context.

Read only the relevant state or plan for informational questions. Create required missing placeholders according to the file structure rules; an empty placeholder does not supply missing instructions.

## 2. Define the Steps

Use `action_plan.json` for the current plan. Give every step a stable ID, description, status, dependencies, verification checks, history file path, and revision number.

Before any execution, define every participating party's specific role and verifiable expected outcome in JSON. No participating party may have a missing, empty, or undefined role or expected outcome. This applies to the original step and every revised version, retry, or separately introduced action.

The initial plan has five steps. Attempts do not become extra steps. If five steps no longer fit the goal, propose a plan change for human approval instead of silently adding, removing, or combining steps.

## 3. Keep a Simple History for Each Step

Use `step_history/<step_id>.json`, outside the protected folder. Create the file before the step's first revision or execution; an unstarted step may have no history file yet. Its planned status remains available in the action plan.

Use this initial shape:

```json
{
  "step_id": "step_01",
  "status": "proposed",
  "revision": 1,
  "changes": [],
  "attempts": []
}
```

- `status` is the step's current state.
- `revision` identifies the current step definition.
- `changes` retains definition changes, including changes made before any attempt.
- `attempts` retains all execution attempts in order.

For each change, append the timestamp, reason, previous step definition, and new step definition. These snapshots retain the roles, expected outcomes, and verification checks as well as the description. Increase the revision number.

Before execution, append an attempt containing:
- Attempt number and start time.
- A snapshot of the exact step definition, including all participating parties and their expected outcomes.
- The approval that covers this attempt.
- What will be run or performed.
- Status `running`, with results and finish time initially null.

When execution ends, update that attempt with actual results, verification evidence, finish time, and status: `passed`, `failed`, or `interrupted`. Record each check's expected and actual result and whether it passed. Missing evidence is not a pass.

Use UTC ISO 8601 timestamps. Keep evidence concise; reference larger logs by path. Do not store credentials or secrets. Never erase previous failures or rewrite a completed attempt to make a later retry appear successful.

## 4. Approve and Execute

Propose the next step or a small group of steps. Record the human's approval and the exact approved revision and scope. Approval to prepare these documents does not imply approval to run project commands.

Before starting:
1. Confirm the required dependencies have passed.
2. Confirm the step's party definitions and verification checks are complete.
3. Confirm existing approval covers the work.
4. Create or load its history file and record the new attempt.
5. Mark the step `running` in both the plan and its history.

Execute only the approved scope. The AI writes code or instructions and uses direct EC2 access when available. Otherwise, the human runs the commands and supplies results. Do not claim that local workspace execution proves EC2 behavior.

Do not request approval again for work already approved. If the scope, participating roles, or expected outcomes change, record the revised step and obtain approval for the changed work before executing it.

## 5. Verify the Outcome

Compare evidence with every participating party's expected outcome and every verification check.

- Mark an attempt and step `passed` only when all required outcomes are verified.
- Mark an attempt and step `failed` when evidence shows a required outcome was not achieved.
- If execution stops or results are unavailable, record the attempt as `interrupted` and the step as `blocked`; describe the missing evidence or prerequisite.
- A command exiting successfully does not by itself prove the step passed.
- A partial success is not a completed step. Retain its evidence and record what remains.

The attempt history owns the detailed evidence. Keep the plan's status and latest-attempt summary consistent with it. If an update is interrupted, reconcile the plan from the recorded evidence before continuing.

## 6. Handle Changes, Failures, and Retries

| Situation | Required handling |
| --- | --- |
| Change before execution | Record the old and new definitions in `changes`, increase the revision, and update the plan. There is no execution attempt yet. Renew approval if the changed scope, roles, or outcomes require it. |
| Failure | Record the observed failure, evidence, and known cause; distinguish a confirmed cause from a guess. Inspect partial effects before deciding what to do next. |
| Retry unchanged | Keep the same step ID and revision. Confirm that rerunning is safe and covered by existing approval, then append a new attempt. |
| Modify and retry | Record the change and new revision; obtain approval where required, then append a new attempt with the revised definition. |
| Missing prerequisite or unavailable access | Mark the step `blocked` and record what is needed. Do not invent results. |
| Interrupted execution | Check the actual system state before rerunning; an interrupted command may have changed data. Preserve the interrupted attempt. |
| Resume after a pause | Read the latest step history, inspect uncertain effects, and continue only the remaining approved work. |
| Verification fails after execution | Record a failed attempt even if the command itself succeeded. |
| Reopen a previously passed step | Preserve the old passing evidence, explain why it no longer establishes the current result, and record a new revision or attempt as appropriate. Reassess dependent steps. |
| Cancel or replace a step | Record the human's decision and reason; retain the old step ID and history with status `cancelled`. A replacement receives a new ID. Cancellation does not count as completion. |

Do not loop indefinitely on an unchanged failure. After a repeated failure with no new evidence or corrective change, mark the step blocked and present the blocker and proposed next action.

Each retry must still satisfy Law 2 through its recorded step snapshot. Any separate diagnostic or repair action outside the approved step's scope needs its own JSON party definitions and approval before execution.

## 7. Update Progress

After each attempt or plan revision:
1. Save the step history.
2. Update the action plan's status, revision, approval where applicable, and latest-attempt summary.
3. Update `current_state.md` with a short factual paragraph describing the present system state, important blocker, and next action.

Use these step statuses consistently: `proposed`, `ready`, `running`, `passed`, `failed`, `blocked`, and `cancelled`. `ready` means approval and prerequisites are satisfied. A failed step may become ready for an approved retry without losing its failure history.

Recording proposals, changes, approvals, and results requires no separate execution approval. Protected-file editing remains governed by the laws and explicit human instructions.

## 8. Finish

Declare the goal achieved only when its completion criteria are verified. Passing all five steps is insufficient if the final system does not meet the goal. Preserve the complete attempt histories and update the current state with the verified result.

