# Project workflow guide: one pass at a time

This guide organizes project work, human approvals, and project records.
It applies to coding, learning, research, writing, and other projects.

The workflow is:

**Define → Understand → Plan → Work → Verify → Decide**

The golden law is:

**Keep only one pass active. Update the records before advancing. Obtain human approval before planning and before execution.**

This guide uses [project-workflow-guide.md](project-workflow-guide.md) as its reference.
This version uses separate pass folders and preserves closed pass records.
It replaces the reference guide's reset procedure and permission for simultaneous passes.

This document describes the proposed project structure.
Reading it does not create folders, create instructions, or approve project actions.

## 1. Terms and responsibilities

| Term | Meaning |
| --- | --- |
| Goal | The agreed outcome for the whole project. |
| Completion criterion | An observable condition that the project must satisfy. |
| Pass | One agreed unit of work with an outcome, a plan, checks, and a closing decision. |
| Active pass | The single open pass, including a pass that waits for approval or remains paused. |
| Plan revision | An identified version of the actions and checks proposed for a pass. |
| Checkpoint | A saved description of the current position and next action. |
| Verification | A comparison between an expected result and an observed result. |
| Evidence | Recorded observations and supporting details that justify a claim or decision. |
| Human | The person who approves plans, execution, and pass changes. |
| AI | The assistant that proposes work, performs approved actions, and maintains records. |

The human controls the scope of approval.
The human can request a plan change or a pass change at any time.
The AI records the request before dependent work continues.

A pass can continue across several sessions.
A failed check does not automatically create a new pass.
A paused pass still occupies the single active position.

Reading records and updating records within the agreed scope need no separate approval.
These record updates do not authorize new work or new checks.

## 2. Folder structure

```text
project/
├── README.md                  # Project overview and folder map
├── AGENTS.md                  # Workflow rules and human approval requirements
│
├── definition/
│   ├── goal.md                # Outcome, scope, completion criteria
│   └── constraints.md         # Boundaries, responsibilities, review rules
│
├── status/
│   └── current-state.md       # Active pass, latest position, next action
│
├── decisions/
│   └── decision-log.md        # Significant decisions and their reasons
│
├── passes/
│   ├── README.md              # Pass naming, contents, and lifecycle
│   ├── pass-001/
│   │   ├── plan.md            # Proposed actions, approval, progress
│   │   ├── verification.md    # Checks, results, closing decision
│   │   ├── findings.md        # Optional detailed observations and unknowns
│   │   ├── evidence.md        # Optional index of supporting evidence
│   │   └── ...                # Supporting files directly inside the pass
│   └── pass-002/
│       ├── plan.md
│       └── verification.md
│
└── reports/
    └── final-report.md        # Outcome, verification, limitations
```

Create each pass folder only when the human approves its start and planning.
Put all pass files directly inside that folder.
Do not create folders inside an individual pass folder.

Use identifiers in sequence: `pass-001`, `pass-002`, and `pass-003`.
Do not reuse an identifier for different work.
The second pass folder in the example does not imply a second active pass.

Create `findings.md` and `evidence.md` only when their conditions apply.
Create `reports/final-report.md` when the project finishes.
Keep actual project outputs in their agreed project locations.
Code, documents, and other deliverables have their own structure.

The root `README.md` explains the project and its records.
The root `AGENTS.md` contains the rules for AI work.
The `passes/README.md` explains pass naming, contents, and lifecycle.
Additional README files are not required for this structure.

## 3. Rules for the root AGENTS.md

Use these rules in the project's root `AGENTS.md` when adopting this workflow.

1. **Keep only one pass active at a time.**
2. Follow **Define → Understand → Plan → Work → Verify → Decide**. Return to an earlier stage when findings require it.
3. Update the relevant Markdown records after each step. Complete these updates before advancing.
4. Obtain human approval before drafting a plan.
5. Present a short summary of the drafted plan and its checks. Obtain human approval before execution.
6. Record human requests for plan or pass changes. Obtain approval before drafting a revised plan. Obtain approval before executing revised work.
7. Apply approval only to its agreed scope. Do not treat silence as approval.
8. Record approvals and their scope in the pass records. Record pending approvals and the next action in `status/current-state.md`.
9. Record performed actions separately from verified outcomes. A completed action can have a failed or pending check.
10. Preserve observations and supporting evidence with the relevant pass. Create optional files only when needed.
11. Close the current pass before starting another pass. Obtain human confirmation before starting the next pass.
12. Preserve closed pass records. Keep `status/current-state.md` current. Link to supporting details.
13. Declare completion only when the goal criteria and required human reviews are satisfied.

The `AGENTS.md` file must also identify the reading order:

1. Read `definition/goal.md` and `definition/constraints.md`.
2. Read `status/current-state.md`.
3. Read the active pass records, if a pass exists.
4. Read linked decisions and evidence that affect the next action.

Use this section as the rule source for the guide.
Keep project-specific responsibilities and review requirements in `definition/constraints.md`.
Link to those requirements from `AGENTS.md`.

## 4. Human approval sequence

### Before drafting a plan

State the proposed pass outcome in one or two sentences.
Ask the human to approve the pass start and plan drafting.
This request identifies the proposed work without drafting its action sequence.

Record the request in `status/current-state.md` while approval is pending.
Keep the active pass as `none` if no pass is open.
Do not create a new pass folder while another pass remains open.

After approval, create the pass folder and its two required records.
Record the planning approval in that pass's `plan.md`.
Set the new pass as active in `status/current-state.md`.

Example request:

> Proposed Pass 2 outcome: create the requested folders. May I start Pass 2 and draft its plan?

### Before executing a plan

Identify the pass and plan revision.
Give a short summary of the actions, affected locations, and checks.
State any important limit or unresolved assumption.
Ask the human to approve execution of that revision.

Example request:

> Pass 2, revision 1: create folders with `mkdir` in the agreed location. Check that each requested directory exists. May I execute this plan?

Record the response in `plan.md`.
Record the pending approval or approved next action in `status/current-state.md`.
Execute only the approved actions and checks.

The two approvals serve different purposes.
Approval to draft permits planning.
Approval to execute permits the specified actions and checks.
Neither approval proves that the work passed verification.

### When the human requests a change

Record the requested change and its effect on the current work.
Stop work that depends on the changed part.
Treat an explicit request to revise the plan as approval to draft that revision within the requested scope.
Ask for drafting approval if the request leaves that authority unclear.

Assign a new revision identifier to the changed plan.
Preserve the earlier revision's scope and approval in the plan history.
Present the revised execution summary to the human.
Obtain execution approval for the new revision.

Do not apply an earlier execution approval to changed actions or checks.
Do not request the same approval again if the approved revision and scope remain valid.
Progress updates and recorded observations do not create a new plan revision.

### Approval record fields

| Field | Required information |
| --- | --- |
| Type | Pass start and planning, revised planning, or execution. |
| Pass | The pass identifier. |
| Revision | The applicable revision, or `not drafted` for the first planning approval. |
| Request | The proposed outcome or execution summary shown to the human. |
| Response | The human's actual response and its scope. |
| Status | `pending`, `approved`, `changes requested`, `rejected`, or `withdrawn`. |
| Context | Human identity, date, time zone, and message reference when available. |

Record approval only when the human's response clearly covers the request.
Keep the request pending if the response is unclear.
A message can approve both the next pass start and its plan drafting.
Execution approval follows the completed draft.

## 5. File responsibilities

Each fact has one main record.
Other files can contain short summaries and links.

| File | Required contents | Update condition |
| --- | --- | --- |
| `README.md` | Project purpose, folder map, reading order, current-state link. | The structure or reading route changes. |
| `AGENTS.md` | Workflow rules, approval requirements, reading order, instruction links. | The human agrees to change the workflow rules. |
| `definition/goal.md` | Outcome, included scope, excluded scope, numbered completion criteria. | The human agrees to change the goal. |
| `definition/constraints.md` | Boundaries, resources, responsibilities, required reviews, governing references. | The human agrees to change a constraint or responsibility. |
| `status/current-state.md` | Active pass, actual position, remaining work, pending approval, next action, supporting links. | A workflow step changes status or the next action. |
| `decisions/decision-log.md` | Significant decisions, reasons, human authority, dates, supporting links. | A choice affects later passes or project agreements. |
| `passes/README.md` | Naming convention, required files, optional files, approval sequence, closure rules. | The human agrees to change pass organization. |
| Each pass's `plan.md` | Outcome, revision, actions, progress, assumptions, check references, approval history. | Planning, approval, progress, or an agreed revision changes. |
| Each pass's `verification.md` | Expected results, actual results, check status, context, reviews, closing decision. | A check, review, observation, or decision occurs. |
| Each pass's `findings.md` | Detailed observations, open questions, assumptions, interpretations, supporting references. | An investigation needs more detail than the other records can clearly hold. |
| Each pass's `evidence.md` | Supporting file index, check references, source, context, limitations. | Several supporting files need an index or explanation. |
| `reports/final-report.md` | Achieved outcome, final outputs, criterion results, human reviews, limitations, evidence links. | The project reaches completion. |

Keep routine pass decisions in `verification.md`.
Keep decisions that affect later work in `decisions/decision-log.md`.
Link the pass decision to the significant decision when both records apply.

Observed environment details belong in the current state and verification records.
An observation does not automatically become an agreed constraint.

### Current-state fields

Include these fields in `status/current-state.md`:

- Updated date, time, and time zone.
- Goal reference.
- Active pass identifier, or `none`.
- Last closed pass identifier, or `none`.
- Workflow stage.
- Position: `active`, `waiting for approval`, `waiting for review`, `blocked`, `paused`, or `complete`.
- Existing outputs and performed actions.
- Verified results and links to their checks.
- Remaining criteria, unknowns, and blockers.
- Pending human request and its approval type.
- Last decision and its owner.
- Next action, its owner, and prerequisites.

Use `Next action: none. All completion criteria are satisfied.` for completed work.
Do not copy full logs into this file.

### Plan fields

Include the pass identifier, outcome, goal criteria, revision, environment, assumptions, actions, checks, stop conditions, and approvals.
Give each action a status: `not started`, `in progress`, `done`, or `blocked`.
Keep expected check results in `verification.md`.
Link to check identifiers from the plan.

The status `done` means that the action occurred.
It does not mean that verification passed.

### Verification fields

| Field | Required information |
| --- | --- |
| Check | Identifier, such as `V1`, and related goal criterion. |
| Checker | The AI, human, or named reviewer. |
| Method | The check procedure and relevant inputs. |
| Expected result | The condition required for a pass. |
| Actual result | The observation, or `not run`. |
| Status | The status defined in Section 7. |
| Context | Date, tested version, environment, and plan revision where relevant. |
| Evidence | Written observation or link to a supporting file. |
| Limitation | What the check does not establish. |

End the record with the pass decision, reason, decision owner, review status, and next action.
State whether the pass is `open` or `closed`.
For a closed pass, state its outcome: `successful`, `unsuccessful`, `deferred`, or `cancelled`.

## 6. Workflow and record updates

| Stage | Action | Records and approval |
| --- | --- | --- |
| Define | Establish the outcome, boundaries, and completion criteria. | Read the definition files. Record human-approved changes there. |
| Understand | Inspect the actual position. Identify gaps, assumptions, and unknowns. | Read current state and relevant evidence. Record findings before planning. |
| Plan | Obtain drafting approval. Draft actions and checks for one pass. | Create or revise `plan.md`. Create planned checks in `verification.md` as `not run`. Record approvals. |
| Work | Obtain execution approval. Perform the approved actions. | Update action progress and observations. Save the checkpoint before advancing. |
| Verify | Perform the approved checks. Compare actual and expected results. | Update `verification.md` and current state. Record required human reviews. |
| Decide | Assess the results. Select the next action within human authority. | Record the decision and checkpoint. Obtain the approval needed for the selected action. |

Every stage ends with the relevant record updates.
A stage does not require a separate file.
An unchanged agreement needs no rewrite.
Record its confirmation in the pass records or current state when relevant.

Return to Understand when the situation differs from an assumption.
Return to Plan when actions or checks need revision.
Apply the approval sequence before drafting or executing the revision.
Return to Define only when an agreement needs review.

Within an approved plan, record updates do not require approval at every stage.
New planning, changed execution, and a new pass require the specified human approval.

**Current state and evidence are information types. They are not additional workflow stages.**

Work produces outputs and observations.
Verify establishes what the checks show.
Decide uses those results to select the next action.
Current state preserves the latest position.
Evidence preserves the support for claims and decisions.

## 7. Verification and decision rules

| Check status | Meaning |
| --- | --- |
| `not run` | The check did not start. |
| `running` | The check started, but no final result is available. |
| `passed` | The actual result satisfies the expected result. |
| `failed` | The actual result does not satisfy the expected result. |
| `blocked` | A missing prerequisite prevents the check. |
| `inconclusive` | The available result cannot establish whether the expectation holds. |
| `not applicable` | The agreed scope excludes this check. Record the reason. |
| `superseded` | A changed version or corrected finding replaces this result. Link to the replacement. |

Record AI checks and human checks separately.
Pending human review is not a passed check.
Approval is permission. A check result is evidence.

| Decision | Required response |
| --- | --- |
| Continue within the plan | Continue only within the existing approved revision and scope. Update records before the next step. |
| Correct or replan | Preserve the failed result. Obtain the required drafting and execution approvals for changed work. |
| Seek input | Record the missing information or approval. Keep dependent work pending. |
| Close and continue | Save the closing decision and checkpoint. Obtain confirmation before starting and planning the next pass. |
| Defer or cancel | Record the human decision and unfinished criteria. Close the pass before starting another. |
| Finish | Check all goal criteria and required reviews. Save the final report and checkpoint. Preserve the pass records. |

A closed pass can be unsuccessful.
Closing that pass does not mark its criterion complete.
Only a human-approved goal change can remove a completion criterion.
Record that change before assessing the revised goal.

## 8. Pass lifecycle and evidence

### Start and maintain a pass

1. Read the current state and applicable agreements.
2. Confirm that no other pass remains open.
3. Obtain human confirmation for the proposed pass and its plan drafting.
4. Create the new pass folder.
5. Create `plan.md` and `verification.md`.
6. Record the planning approval in `plan.md`.
7. Set the active pass in `status/current-state.md`.
8. Draft the plan and expected checks.
9. Present the short execution summary.
10. Obtain execution approval for the plan revision.
11. Perform the approved work and checks.
12. Update records after each workflow step.

Do not create an empty future pass while the current pass remains open.
Keep future work as remaining work in current state until the human approves its planning.

### Close a pass

1. Record the actual check results and required human reviews.
2. Record the closing decision, its owner, and unfinished criteria.
3. Preserve the supporting evidence needed to explain those results.
4. Check the links to supporting files.
5. Mark the pass closed in `verification.md`.
6. Set the active pass to `none` in current state.
7. Record the last closed pass and the next proposed action.

Do not reset, replace, or delete a closed pass's plan and verification records.
Preserve their approval history and recorded findings.
Add a dated correction if a later finding changes an earlier claim.
Identify the superseded claim and its replacement.
Do not silently rewrite the earlier result.

For project completion, create `reports/final-report.md`.
Link its conclusions to the final checks.
Do not create a next pass when no work remains.

### Evidence placement

Use written results in `verification.md` for simple checks.
Create `findings.md` for a detailed investigation or several related unknowns.
Create `evidence.md` when supporting files need an index and context.

Store supporting files directly beside the pass records.
Use descriptive names, such as `V1-directory-check.txt` or `V2-output.png`.
Reference the check identifier, tested version, and relevant conditions.

Retain supporting material when a claim, decision, later action, or project agreement depends on it.
Retain results that cannot be reproduced under the same conditions.
Raw logs are not required when the written result contains sufficient evidence.
Avoid saving unrelated output.
Do not remove retained support while a claim or agreement still depends on it.

Keep credentials out of all project records.
Reference the agreed access method instead.

## 9. Three-pass example: folders and files

This example is a simulation.
Its commands, observations, and human approvals are hypothetical.
It does not report actual execution.

Assume the fixed project records already exist.
The goal has two criteria:

- **G1:** The agreed output location contains directories `alpha` and `beta`.
- **G2:** Each directory contains a regular file named `note.txt`.

Assume the output location starts empty.
Assume the shell is Bash on Ubuntu.
Assume the shell cannot resolve the PowerShell command `New-Item`.
PowerShell can run on Ubuntu when installed. This example concerns the command used in Bash.

Each main action has its own pass for this exercise.
The first pass deliberately uses the wrong command.
Its approved plan includes environment inspection after a failure.
That inspection uses `cat /etc/os-release` and `ps -p $$ -o comm=`.
The simulated results identify Ubuntu and Bash.

### Pass 1: attempt folder creation

| Stage | Simulated action | Record activity |
| --- | --- | --- |
| Define | Read G1 and G2. | Read `definition/goal.md` and applicable constraints. |
| Understand | Confirm the empty output location. Miss the environment check. | Record the starting position and unverified environment in current state. |
| Plan | Obtain approval to start Pass 1 and draft its plan. Plan `New-Item -ItemType Directory -Path alpha,beta`. | Create `pass-001/plan.md` and `verification.md`. Record drafting approval, expected directories, and conditional environment checks. |
| Work | Present the summary. Obtain execution approval. Attempt the command in Bash. | Record execution approval in `plan.md`. Record the simulated `command not found` error in `verification.md`. |
| Verify | Confirm that neither directory exists. Perform the approved environment checks. Discover Ubuntu and Bash. | Record the failed G1 check and the environment results. Update current state. |
| Decide | The human agrees to close Pass 1 as unsuccessful. Propose folder creation with `mkdir` next. | Record the closing decision. Mark Pass 1 closed. Save the checkpoint before requesting Pass 2 approval. |

| Information | Checkpoint after Pass 1 |
| --- | --- |
| Current state | No output folders or files exist. Ubuntu and Bash are confirmed. Active pass: none. Next: request Pass 2 planning approval. |
| Evidence | Pass 1 verification contains the command, error, directory check, and environment observations. |
| New records | `passes/pass-001/plan.md` and `passes/pass-001/verification.md`. |

The failure returns the reasoning to Understand.
The approved diagnostic checks fit within Pass 1.
Without that approval, the AI would request approval before performing those new checks.
No separate findings or evidence file is needed for these short observations.

### Pass 2: create the folders

| Stage | Simulated action | Record activity |
| --- | --- | --- |
| Define | Confirm the same goal. This pass addresses G1. | Read `definition/goal.md`. |
| Understand | Read the Ubuntu/Bash finding and the empty output state. | Read current state and Pass 1 verification. |
| Plan | Obtain approval to start Pass 2 and draft its plan. Plan `mkdir alpha beta` and directory checks. | Create `pass-002/plan.md` and `verification.md`. Record drafting approval. Set Pass 2 active. |
| Work | Present the summary. Obtain execution approval. Create both directories. | Record approval and action progress in `plan.md`. Record relevant observations in `verification.md`. |
| Verify | Check each path with `test -d`. Both checks succeed in this simulation. | Record the methods and actual results. G1 passes. G2 remains incomplete. |
| Decide | The human agrees to close Pass 2 as successful. Propose file creation next. | Record the decision. Mark Pass 2 closed. Save current state before requesting Pass 3 approval. |

| Information | Checkpoint after Pass 2 |
| --- | --- |
| Current state | Both directories exist. G1 passed. G2 remains incomplete. Active pass: none. Next: request Pass 3 planning approval. |
| Evidence | Pass 2 verification contains the directory checks. Pass 1 retains the earlier failure and environment findings. |
| New records | `passes/pass-002/plan.md` and `passes/pass-002/verification.md`. |
| New project outputs | The directories `alpha` and `beta`. |

### Pass 3: create the files

| Stage | Simulated action | Record activity |
| --- | --- | --- |
| Define | Confirm that completion requires G1 and G2. | Read `definition/goal.md`. |
| Understand | Confirm the existing directories and missing files. | Read current state and relevant checks. |
| Plan | Obtain approval to start Pass 3 and draft its plan. Plan `touch alpha/note.txt beta/note.txt` and final checks. | Create `pass-003/plan.md` and `verification.md`. Record drafting approval. Set Pass 3 active. |
| Work | Present the summary. Obtain execution approval. Create both files. | Record approval and action progress. Record relevant observations. |
| Verify | Use `test -d` for both directories and `test -f` for both files. All checks succeed in this simulation. | Record final results for G1 and G2. Record the required human review. |
| Decide | The human confirms completion after review. Close Pass 3. | Save the closing decision and final checkpoint. Create `reports/final-report.md` with links to the final checks. |

| Information | Checkpoint after Pass 3 |
| --- | --- |
| Current state | G1 and G2 passed. Required review is complete. Active pass: none. Next action: none. |
| Evidence | Pass 3 verification contains the final directory and file checks. Earlier pass records remain available. |
| New records | The two Pass 3 records and `reports/final-report.md`. |
| New project outputs | `alpha/note.txt` and `beta/note.txt`. |

The goal requires file existence. It does not require file content.
The checks support only the stated criteria under the recorded conditions.

The root README, AGENTS, definition files, and pass README need no change during these passes.
The routine pass decisions stay in each verification record.
The significant decision log needs no entry for this command correction.

### Resulting records

```text
project/
├── README.md
├── AGENTS.md
├── definition/
│   ├── goal.md
│   └── constraints.md
├── status/
│   └── current-state.md
├── decisions/
│   └── decision-log.md
├── passes/
│   ├── README.md
│   ├── pass-001/
│   │   ├── plan.md
│   │   └── verification.md
│   ├── pass-002/
│   │   ├── plan.md
│   │   └── verification.md
│   └── pass-003/
│       ├── plan.md
│       └── verification.md
└── reports/
    └── final-report.md
```

The output directories remain at the agreed output location.
They are project outputs, separate from the records shown above.

## 10. Interruptions and exceptional conditions

| Condition | Required response |
| --- | --- |
| The session ends during a pass. | Save progress, approvals, running operations, pending checks, and the next action. Resume the same pass. |
| The human pauses the pass. | Record the pause. Keep the pass open. Do not start another pass. |
| The human requests different work. | Record the request. Obtain a decision to revise, defer, or cancel the current pass before starting another. |
| A remote operation might still run. | Record its identifier. Inspect its state before repeating it within the approved scope. |
| A connection fails before results arrive. | Record the outcome as unknown. Check actual state before retrying an action with possible duplicate effects. |
| A check fails. | Preserve the failure. Request the required approvals for changed actions or checks. |
| Human review remains pending. | Record the exact review request. Do not mark the check passed or the goal complete. |
| The human approves execution without reporting a check. | Record execution approval only. Keep the human check pending. |
| AI and human observations differ. | Preserve both observations and their conditions. Obtain approval for additional investigation when needed. |
| Code, inputs, or environment change. | Identify affected results. Mark them superseded. Obtain approval for changed checks when needed. |
| A local check passes for a remote goal. | Keep remote verification pending. Record the local result with its actual scope. |
| Source inspection suggests correct behavior. | Label the result as source inspection. Keep required execution checks pending. |
| A goal criterion changes. | Record human approval and the revised criterion. Preserve earlier failures with their original context. |
| Two records conflict. | Inspect the underlying work and agreements within the approved scope. Correct the records before dependent work. |
| More than one pass appears open. | Stop pass execution. Resolve the conflict with the human. Record one active pass before continuing. |
| A closed pass contains a mistaken claim. | Add a dated correction and supporting evidence. Update current state and dependent claims. |
| Existing approval covers the unchanged plan. | Retain that approval. Resume within its scope after checking the actual state. |

For remote checks, record the resource identifier, location, working directory, and tested version when relevant.
Record uncommitted differences when a commit identifier does not fully identify the tested work.

## 11. Controlled language for Markdown records

Use [asd-ste100.md](asd-ste100.md) as the local writing reference.
Use ASD-STE100 as the controlled-language reference for project Markdown files.
Apply the following structural rules and consistent project terms.

This guide applies ASD-STE100 writing principles.
It does not claim full compliance with the official dictionary or all standard requirements.
Use the [official ASD-STE100 source](https://www.asd-ste100.org/) for the standard and vocabulary guidance.

1. Limit instruction sentences to 20 words.
2. Limit description sentences to 25 words.
3. Give one instruction per sentence.
4. Use active voice when the actor is known.
5. Use the same term for the same concept.
6. Prefer simple verbs and simple tenses.
7. Avoid semicolons and unnecessary phrasal verbs.
8. Keep each paragraph on one topic.
9. Use numbered lists for procedures.
10. Define technical terms before readers need them.
11. State the actor when responsibility matters.
12. Preserve uncertainty, conditions, and limitations.
13. Separate proposed work, performed work, and observed results.
14. Preserve commands, identifiers, paths, and recorded error text exactly.

Use the terms defined in Section 1 throughout the records.
Use `approval` for permission and `verification` for checks of results.
Do not replace an uncertain result with a definite claim to shorten a sentence.
Check vocabulary against the official standard before claiming full ASD-STE100 compliance.

| Avoid | Use |
| --- | --- |
| Everything works. | G1 and G2 passed for the recorded version and environment. |
| Approved. | The human approved execution of Pass 2, revision 1. |
| Mostly done. | The directories exist. The files remain absent. |
| Continue later. | Next: request approval to draft Pass 3. |
| The command worked. | The command ended successfully. Both directory checks passed. |
| We need to look into this. | The check failed. Next: request approval for an environment check. |

## 12. Record checks before a transition

Before advancing to another workflow stage, check the applicable items:

- The records describe the actual completed step.
- The next action stays within the approved scope.
- The plan revision matches its execution approval.
- Pending checks and approvals remain explicit.
- Current state identifies the single active pass.
- Observations support each completion claim.
- Supporting links identify the correct pass and tested work.

Before starting another pass, check these additional items:

- The previous pass is closed with its outcome recorded.
- The previous pass records and required evidence remain available.
- Current state identifies the last closed pass.
- The human approved the new pass start and plan drafting.

At project completion, check all goal criteria and required human reviews.
Save the final report and its evidence links.
Preserve all closed pass records.
Keep documentation proportional to the work and its claims.
