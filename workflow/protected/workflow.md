# Workflow

## 1. Terms and Boundaries

- **Goal:** The overall required result and its acceptance criteria.
- **Action / step:** One planned piece of work with a stable `step_XX` ID.
- **Revision:** A version of an action's definition.
- **Pass:** One bounded execution attempt at one action revision, identified by `pass_XXX`.
- **Tool call:** An operation inside a pass. It is not automatically a separate action or pass.

Use one action per pass. A pass may include several related operations and checks. A retry uses a new pass ID and retains the same action ID. Preserve the five planned steps unless the human authorizes a plan-structure change.

## 2. Load and Plan

Follow the reading order in `AGENTS.md`. For execution, load the goal, this workflow, state, plan, contracts, relevant earlier passes, and relevant implementation files. Do not interpret example records as project history.

Every action in `action_plan.json` has a stable ID, revision, description, dependencies, parties, resources, scope, deliverables, acceptance criteria, relevant goal criteria, recovery limits, authorization reference, pass IDs, and latest-pass summary.

Each party has a stable ID, name, role, authority, and expected outcomes. Each outcome has an ID, condition, and verification method. Every separate action must satisfy Law 2. Do not add the human as an executor when the AI can execute directly; if the human must run commands, revise the action's responsibilities before starting and check whether existing authorization covers that arrangement.

Plan edits and progress reporting do not require their own execution pass. They are bookkeeping, not evidence that application work occurred. Actual diagnostic commands, implementation changes, and checks must be covered by a recorded action and pass.

## 3. Revisions and Authorization

The plan's `authorizations` array records actual human instructions: ID, source/reference, scope, and timestamp when known. The initial array is empty because this rewrite does not establish authorization to execute the task-tracker project.

Reference an existing authorization if it covers the next work. Ask only for uncovered work. A role wording change or a failed check does not automatically invalidate existing authority. New access, broader changes, or changed required outcomes may require additional authorization.

Before changing an action definition, append the old and new definitions, reason, and UTC timestamp to `revision_history`, then increment the revision. A definition contains the description, dependencies, parties, resources, scope, deliverables, acceptance criteria, goal links, and recovery limits. Do not recursively embed runtime state or revision history in these snapshots. Status or latest-pass updates do not change the definition revision.

Changing protected files requires explicit authorization. Keep the prior document versions through version history or a version-control checkpoint. Never change criteria just to make a failed attempt pass.

## 4. Prepare a Pass

Before creating a runnable pass:

1. Verify dependencies have passed and their evidence still applies.
2. Confirm the required authorization, resources, and access exist.
3. Check the attempt limit. The initial limit is three passes per action revision; repeated identical failures without progress must stop earlier. Do not bump a revision merely to reset this limit.
4. Allocate the next unused real pass ID.
5. Build `passes/<pass_id>/input.json` against the input schema.

The input snapshots the action definition and relevant starting state. Include source versions or hashes where available; use null for an unknown version, never an invented value. Unknown information that is necessary for safe or correct execution must be resolved before starting.

Record `previous_pass_id` for a retry and summarize its actual effects. Define the expected deliverables, output locations, party outcomes, acceptance criteria, and ordered operations. Set `record_kind` to `execution` for real work.

Use UTC ISO 8601 timestamps. The schemas check structure. Also inspect meaning: unique IDs, correct references, coherent scope, actual authority, and feasible verification. If no JSON Schema validator is available, report that limitation and manually inspect the contract; never claim automated validation occurred.

If preflight is blocked, mark the action blocked and record why. Do not create a fake started pass. A corrected, unstarted draft input can still be edited; freeze it on execution start.

## 5. Start and Execute

Create a schema-valid `output.json` with status `running` before operations begin. Set its start time, finish time null, empty results/evidence, and the matching pass/action/revision identifiers. Mark the action running and append the pass ID to its `pass_ids`.

Execute the planned operations within scope. Tool selection and harmless implementation details may vary without changing the contract. If the objective, participating responsibility, permitted scope, or required outcomes materially change, finalize the current pass honestly and prepare a new revision/pass as appropriate.

Record operations performed, real effects, command exit statuses where relevant, and evidence sources. Never store credentials or secrets in logs. Capture evidence from the actual execution environment and identify the delivered code/version to which it applies.

The AI uses direct EC2 access only if available and authorized. Otherwise the human runs commands under recorded responsibilities and supplies outputs. Clearly distinguish human-supplied evidence from direct observations. Local execution cannot prove EC2 behavior.

## 6. Verify and Finalize

For every acceptance criterion, write exactly one verification entry with `passed`, `failed`, or `not_checked`, an actual result, and evidence references. Do the same for every participating party's expected outcome. These are related but separate obligations; check both.

Evidence entries have stable IDs, locations, descriptions, and a source type. File evidence locations are relative to the pass folder. An `inline:` location refers to an observed result recorded directly in the evidence description. Synthetic evidence is permitted only in example records.

| Pass status | Meaning |
| --- | --- |
| running | Operations or verification are still active; finish time is null |
| passed | Every required outcome and criterion passed, evidence exists, deliverables are present, and no required unresolved work remains for this action |
| failed | Evidence shows at least one required criterion or party outcome failed |
| blocked | The pass cannot continue because a prerequisite, access, or required evidence is unavailable, without a demonstrated requirement failure |
| interrupted | Execution stopped unexpectedly; effects or results may be uncertain |

If a required check failed, keep status failed even when the next action is blocked. If results are incomplete with no demonstrated failure, use blocked or interrupted as appropriate. Partial success is recorded in results and effects; it is not a passed status.

Before finalization, check:

1. Input/output IDs, action revision, and record kind match.
2. Every input criterion and party outcome appears exactly once; no unknown or duplicate IDs exist.
3. Every evidence reference resolves. Passed and failed results have evidence; not-checked entries explain why.
4. All required deliverables exist when status is passed.
5. Start/finish timestamps are ordered and the status matches the evidence.
6. Results apply to the final artifact version being delivered.

Schema validation does not enforce all cross-file relationships; these semantic checks are mandatory as well. Once finalized, preserve input, output, and evidence. A correction goes in `corrections/correction_NNN.json` with timestamp, target, reason, original statement, corrected statement, and supporting evidence. Do not rewrite a completed pass.

## 7. Recover, Retry, or Resume

On failure, record observed symptoms, evidence, and confirmed or suspected cause. Inspect partial effects before deciding whether to keep, repair, or roll back changes. Any rollback must itself be within authorization.

- **Unchanged retry:** retain action ID and revision; create a new pass with a justified reason.
- **Changed definition:** retain action ID; preserve the change, increment revision, and use a new pass.
- **In-scope repair:** may be included in the retry's operations if its contract covers it.
- **Separate out-of-scope repair:** define its parties and outcomes in JSON and obtain any missing authorization; do not silently expand the five-step plan.
- **Repeated failure without progress:** mark the action blocked and report the proposed next decision.
- **Interrupted pass:** inspect the actual system first. Reconcile the running record to interrupted when the prior execution has ended; use a new pass to resume remaining work.
- **Reopened passed action:** preserve previous evidence, explain why it is no longer sufficient, and reassess dependent actions.

Never blindly repeat an operation that may already have changed data. If a late result arrives after finalization, record it as a correction or evidence in a later pass rather than silently changing history.

## 8. Update the Plan and Current State

Save the pass result first. Then update the action's status, pass IDs, authorization reference, and latest-pass summary. Finally update `current_state.md` with a short factual paragraph: verified progress, relevant unknowns/blockers, and next action.

Action statuses are `proposed`, `ready`, `running`, `passed`, `failed`, `blocked`, and `cancelled`. Ready means prerequisites and authorization are satisfied. An interrupted pass normally leaves its action blocked until effects are reconciled. A failed action may become ready for a justified retry without losing failed pass records. Cancelling an action requires the human's decision and never counts as completing it.

If bookkeeping is interrupted, reconcile the plan and summary from pass records and actual state before continuing. The latest recorded pass is historical evidence, not proof that an external system has remained unchanged.

## 9. Finish the Goal

In step_05, verify every `GOAL-XX` criterion against the final code on EC2. The final action's criteria use these same IDs, providing an explicit completion checklist. Existing earlier evidence may be reused only when it remains applicable to the delivered version and environment.

Deliver the application and usage instructions, identify supporting evidence, update the plan and current state, and stop. Do not declare completion from example passes, an unverified summary, or the number of steps marked passed.
