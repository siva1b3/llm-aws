# Project workflow guide

Use this guide to organize work across repeated passes. It applies to coding, learning, research, writing, and other projects.

The workflow is:

**Define → Understand → Plan → Work → Verify → Decide**

The goal gives direction. The plan describes the next work. Verification establishes what the work achieved. The decision determines the next action.

The documents preserve enough information to resume work and explain important conclusions.

This guide describes a process. Reading this guide does not create the project files or authorize actions on an external system.

Reading route:

- [The two types of tables](#1-the-two-types-of-tables)
- [The four core files](#2-project-records-table-the-four-core-files)
- [Work flow table](#3-work-flow-table-actions-and-file-changes)
- [Required file contents](#4-what-each-core-file-contains)
- [Verification and decisions](#5-verification-and-decisions)
- [File lifecycle and reset procedure](#6-file-lifecycle-what-stays-changes-or-disappears)
- [Evidence retention](#7-evidence-what-to-retain-and-what-to-discard)
- [Optional files and their triggers](#8-optional-files-and-the-conditions-that-require-them)
- [Edge cases](#9-edge-cases-and-exact-responses)
- [Three coding passes](#10-worked-example-three-coding-passes)
- [Writing rules](#11-writing-rules-for-these-records)
- [Handover checks](#12-check-the-records-before-handover)

## 1. The two types of tables

| Table name | What it shows | Question it answers |
| --- | --- | --- |
| **Work flow table** | The actions in Define → Understand → Plan → Work → Verify → Decide. | What do we do? |
| **Project records table** | The documents that preserve agreements, plans, status, checks, and decisions. | What information do we keep? |

The tables describe two views of the same work. A workflow step can read or update several documents.

For example, Verify produces observations for `verification.md`. Decide uses those observations to choose an action. The resulting checkpoint belongs in `current-state.md`.

Current state and evidence are types of information. They are not additional workflow steps.

### Terms used in this guide

| Term | Meaning |
| --- | --- |
| **Goal** | The agreed outcome for the whole project or task. |
| **Completion criterion** | A condition that must be satisfied before you call the goal complete. |
| **Pass** | A unit of work with one stated outcome, checks, and a decision. A pass can include several corrections. |
| **Checkpoint** | A saved description of the current position and the next action. |
| **Verification** | A check that compares an actual result with an expected result. |
| **Evidence** | Observations and supporting details used to assess a claim or choose an action. |
| **Persistent record** | Information kept across passes. Its contents can change when the facts or agreements change. |
| **Active record** | Information for the current pass. Its contents can reset after the pass closes. |
| **Reset** | Replace the previous pass contents with contents for the next pass. |
| **Retention** | Keeping information after the active pass ends. |
| **Decision owner** | The person or agent authorized to choose the next action. |

A pass does not equal a chat session. One pass can continue across several sessions.

A new message, a failed check, or a change of AI session does not automatically start a new pass.

## 2. Project records table: the four core files

This workflow uses four core Markdown files. Required means required when you adopt this four-file workflow.

Use an existing document when it already serves one of these purposes. Record the path once. Do not maintain duplicate owners.

| Document | Requirement | Purpose | What happens across passes |
| --- | --- | --- | --- |
| **`goal.md`** | Required before planned work starts. | Defines the outcome, scope, constraints, responsibilities, review points, and completion criteria. | Keep it. Update it only after an agreed change. |
| **`current-state.md`** | Required before the first pass. | Describes the actual starting point, completed work, verified results, remaining work, and next action. | Keep the file. Replace outdated status with the latest checkpoint. |
| **`plan.md`** | Required for an active pass. | Defines the pass outcome, actions, dependencies, and intended checks. | Keep it while the pass is open. Reset it when the next pass starts. |
| **`verification.md`** | Required for an active pass. | Records expected results, actual results, AI checks, user checks, limitations, and the decision. | Keep it while the pass is open. Reset it only after required information survives elsewhere. |

Use these filenames unless existing project instructions specify other names.

The four files can stay in one folder. Separate folders are not required.

```text
goal.md
current-state.md
plan.md
verification.md
```

Before the first pass, only the first two files need to exist. Plan creates the two active files.

An existing starting-point document can supply the initial current state. Check its claims against the actual situation before relying on them.

After that transfer, identify `current-state.md` as the owner of live status. Keep the earlier document only if its original snapshot has a defined use.

### How these files relate to the earlier categories

| Information category | Where it usually belongs in this workflow |
| --- | --- |
| **Stable contract** | The agreements in `goal.md` and any linked governing references. |
| **Current state** | The checkpoint in `current-state.md`, plus the active intentions in `plan.md`. |
| **Accumulated evidence** | Actual findings in `verification.md` and the selected findings retained across passes. |

One document can contain several types of information. For example, `verification.md` can contain pending checks, actual observations, and a decision.

A planned check is an intention. It becomes a source of evidence only when someone performs it and records observations.

## 3. Work flow table: actions and file changes

Enter the workflow where the work actually stands. Reuse an established goal and plan after checking that they still apply.

| Step and its purpose | What to do | Documents to read or change |
| --- | --- | --- |
| **1. Define: establish the outcome and the boundaries.** Decide what counts as complete. Identify who can make decisions. | State the goal. Define the scope and constraints. Define completion criteria. Record the agreed review points and decision authority. | Create `goal.md`, or read its existing equivalent. Change agreements only through the agreed decision process. |
| **2. Understand: establish the actual starting point.** Find the gap between the current position and the goal. | Inspect existing work, instructions, resources, and dependencies. Compare observations with recorded status. Identify unresolved questions and blockers. | Read `goal.md` and `current-state.md`. Create or correct `current-state.md`. Read open pass files when resuming. |
| **3. Plan: choose a checkable portion of work.** Define both the action and the way to judge its result. | State one pass outcome. List the actions and dependencies. Assign checks to the AI or user. Specify expected results. | Create or update `plan.md`. Create `verification.md` with the planned checks marked `not run`. Set the pass identifier in all active records. |
| **4. Work: perform the planned actions.** Produce something that can be inspected or checked. | Perform authorized actions within the agreed scope. Record actual changes and observations. Revise the plan when new information changes the approach. | Update progress in `plan.md`. Record observations in `verification.md`. Save a checkpoint in `current-state.md` at the events defined below. |
| **5. Verify: establish what the work achieved.** Compare the actual result with each expected result. | Perform the planned checks. Record actual results. Request the agreed user checks. Preserve failures and incomplete checks accurately. | Update `verification.md`. Update `current-state.md` when a result changes the next action or the completion status. |
| **6. Decide: choose the next action from the results.** Determine whether to continue, correct, replan, seek input, or finish. | Compare the findings with the pass outcome and overall goal. Record the selected action, its reason, and its decision owner. | Record the decision in `verification.md`. Save the resulting checkpoint in `current-state.md`. Apply the retention rules before any reset. |

The repeated working cycle is **Work → Verify → Decide → next action**.

Return to Understand when the actual situation differs from your assumptions. Return to Plan when the approach changes.

Return to Define only when an agreement needs review. A failed check alone does not change the goal.

### When to update the current state

Update `current-state.md` at each of these events:

1. The active pass starts.
2. A completed action changes what the next person must do.
3. A check changes a completion claim or the next action.
4. Work stops for a question, interruption, blocker, or user review.
5. The pass closes.

An update can be one sentence. Do not copy every command or observation into the checkpoint.

## 4. What each core file contains

### `goal.md`: the agreed reference

Include these fields:

| Field | Required content |
| --- | --- |
| Outcome | The result the project must achieve. |
| Scope | What the project includes and excludes. |
| Constraints | Limits that govern the work, such as environment, resources, or permitted operations. |
| Completion criteria | Observable conditions for completion. Assign identifiers such as `G1` and `G2`. |
| Responsibilities | Who performs work, checks results, and maintains records. |
| Review and decision rules | Which user checks are required. Which decisions the AI can make independently. Who can change agreements. |
| Governing references | Links to existing requirements, technical choices, or project instructions. |

For joint AI and user work, record the user's agreed review points explicitly.

If the user requires review after each pass, keep that requirement. If the user delegates a decision, record its scope.

Do not request the same authorization again when it still applies. Do not treat authorization to execute as proof of successful verification.

### `current-state.md`: the place to resume

Include these fields:

| Field | Required content |
| --- | --- |
| Updated | The date and time of the checkpoint, including the time zone. |
| Goal reference | The path to the agreed goal. |
| Pass | The active or last closed pass identifier, with `open` or `closed`. Use `none` before the first pass. |
| Position | `active`, `waiting for review`, `blocked`, `paused`, or `complete`. State the reason. |
| Completed work | What exists now. Identify the relevant files, artifacts, or remote resources. |
| Verified results | What checks established. Identify the tested version and conditions when they affect the claim. |
| User review | The result of required user checks, or the exact check still pending. |
| Remaining work | Unfinished goal criteria, unresolved questions, limitations, and dependencies. |
| Last decision | The selected action, its reason, and the decision owner. |
| Next action | One concrete action, its owner, and any prerequisite. |
| Supporting records | Links to retained details that the next pass needs. Use `none` when no extra record exists. |

For a completed project, write `Next action: none. All completion criteria are satisfied.`

For a blocked task, name the missing information or dependency. State what must happen before work can resume.

Avoid percentage estimates unless the project defines how to calculate them. A list of satisfied and remaining criteria is usually clearer.

### `plan.md`: the active pass

Include these fields:

- Pass identifier, such as `pass-001`.
- Goal criteria served by the pass.
- The pass outcome and its completion conditions.
- Assumptions that still need checks.
- Actions in dependency order.
- The target environment and required resources.
- Planned checks, their identifiers, and the assigned checker.
- Conditions that require a new decision before further work.

Give each action a state: `not started`, `in progress`, `done`, or `blocked`.

In this file, `done` means the action occurred. It does not mean the outcome passed verification.

For example, `Write the script: done` can coexist with `Run the script: blocked`.

Link check identifiers to `verification.md`. Keep expected results in that file to avoid conflicting copies.

### `verification.md`: checks and the resulting decision

Identify the pass, tested artifact, and target environment at the start of the file.

Use this check structure:

| Field | What to record |
| --- | --- |
| Check identifier | A short identifier such as `V1`. |
| Criterion | The pass condition or goal criterion that this check addresses. |
| Checker | AI, user, or another named reviewer. |
| Method and inputs | The command, inspection, exercise, or other check. Include inputs that affect the result. |
| Expected result | What must happen for this check to pass. |
| Actual result | What the checker observed. Use `not run` before execution. |
| Status | One of the statuses defined in Section 5. |
| Context | Date, version, and environment details needed to interpret the result. |
| Supporting record | A retained output, screenshot, report, or log when the written result alone is insufficient. |
| Limitations | What the check did not establish. |

At the end, record the decision, supporting check identifiers, decision owner, and next action.

For a small task, several fields can fit in one row. The information matters more than the layout.

## 5. Verification and decisions

### Check statuses

| Status | Exact meaning | Required response |
| --- | --- | --- |
| **Not run** | Nobody performed this check yet. | Keep it pending. |
| **Running** | The check started, but no final result is available. | Inspect its progress or wait for completion. |
| **Passed** | The observed result satisfied the stated expectation. | Record the observation and its scope. |
| **Failed** | The observed result did not satisfy the stated expectation. | Record the mismatch. Decide how to respond. |
| **Blocked** | A missing prerequisite prevents the check. | Name the prerequisite and the action needed. |
| **Inconclusive** | The check ran, but the result cannot establish whether the expectation holds. | Record the uncertainty. Choose a check that resolves it. |
| **Not applicable** | The criterion does not apply under the agreed scope. | Record the reason and applicable agreement. |
| **Superseded** | A later version or correction replaces this result. | Identify the replacement. Do not use this result as proof for the changed work. |

Pending user feedback is not a passed check. Record the user's actual response and what it covers.

Approval to continue is a decision. An observation from a user check is evidence. A user can provide one without the other.

A test file describes a check. A recorded test execution establishes what happened during that execution.

An exit code of zero establishes successful termination according to that program. It does not establish every goal criterion.

### Decision choices

| Choice | Use it when | What happens to the pass and documents |
| --- | --- | --- |
| **Continue** | The pass met its conditions, required reviews are complete, and more goal work remains. | Close the pass. Save its checkpoint. Prepare the next pass after the retention check. |
| **Correct and recheck** | The current outcome remains appropriate, but the work or check needs correction. | Keep the pass open. Update the plan. Preserve the failed result before recording the recheck. |
| **Replan** | New information changes the approach, dependencies, or proposed pass outcome. | Revise the same pass if its outcome stays the same. Otherwise, record a decision to defer it before starting another pass. |
| **Seek input** | A required fact, user check, or decision is missing. | Record the exact request. Keep the affected work pending. Preserve the active files. |
| **Finish** | All overall completion criteria have current support, and all required reviews are complete. | Save the final checkpoint and report. Remove active files only after the retention check. |

If the goal has several criteria, check all of them before Finish. One successful pass can satisfy only part of the goal.

If required verification cannot occur, report the limitation. Do not call the original criterion complete.

Only the authorized decision owner can change a completion criterion. Record that change in `goal.md` before using the revised criterion.

For unresolved input, continue independent work only when the existing plan and authority permit it. Do not guess the missing answer.

## 6. File lifecycle: what stays, changes, or disappears

### File lifecycle table

| Point in the workflow | `goal.md` | `current-state.md` | `plan.md` | `verification.md` |
| --- | --- | --- | --- | --- |
| Before the first pass | Create or reuse. | Create from the checked starting point. | Not required yet. | Not required yet. |
| Define and Understand | Read applicable agreements. | Read and correct actual status. | Read if a pass is already open. | Read if a pass is already open. |
| Plan a new pass | Keep the agreed reference. | Record the active pass and next action. | Create or reset for the new pass. | Create or reset. Mark planned checks `not run`. |
| Work | Read constraints that apply to each action. | Update at checkpoint events. | Update action progress and plan changes. | Add observations as they occur. |
| Verify | Read the applicable criteria. | Update confirmed results and pending review. | Mark completed actions accurately. | Record AI and user checks separately. |
| Correct or replan within the pass | Change only by agreement. | Record the revised position. | Revise the current plan. | Keep earlier results. Add new checks or mark superseded results. |
| Wait, pause, or interruption | Keep. | Save the exact resumption point. | Keep. | Keep. |
| Close the pass | Keep. | Save the outcome, decision, and retained support. | Keep until the next pass starts. | Keep until required findings survive elsewhere. |
| Start the next pass | Read again. | Read the closed checkpoint, then record the new pass. | Replace previous pass contents. | Replace previous pass contents after the retention check. |
| Finish the project | Keep. | Keep the final checkpoint and verification summary. | Remove after the retention check, or archive when required. | Remove after the retention check, or archive when required. |

The active files do not appear and disappear at every step. Their contents serve the whole pass.

Resetting their contents is the default. This keeps their paths stable for links and reading instructions.

If you prefer deletion and recreation, use the same reset sequence. Never delete the only copy of required information first.

### Close a pass before resetting its records

A pass is closed when its decision and resulting checkpoint are saved. Required review must be complete or explicitly deferred by its owner.

Deferral closes the pass as unfinished. It does not establish success.

Before any reset:

1. Record the decision and its owner in `verification.md`.
2. Identify findings that meet a retention condition in Section 7.
3. Preserve those findings in the current state or a linked persistent record.
4. Update `current-state.md` with completed work, verified results, remaining work, and the next action.
5. Check that retained links still lead to the intended information.
6. Mark the pass closed in the checkpoint.
7. At the next pass, assign a new identifier.
8. Replace both active files with records bearing the new identifier.
9. Update `current-state.md` with the new open pass and its next action.
10. Check that `current-state.md` and both active files name the same pass.

When replacing a checkpoint, save its replacement to a temporary file first. Read the saved replacement before replacing the previous checkpoint.

After the replacement succeeds, remove any leftover temporary copy. These temporary files are not additional project records.

If an interruption leaves different pass identifiers, stop dependent work. Recover the last consistent checkpoint before proceeding.

Do not reset project outputs with the active documents. Code, tests, generated deliverables, and deployed resources have separate lifecycles.

## 7. Evidence: what to retain and what to discard

Evidence is part of ordinary work. It can be a checked output, an observation, a measurement, or an inspected source.

For example, an AI checks an uploaded file by downloading it and comparing its contents with the original.

The matching contents support a specific conclusion: this upload preserved this file under the tested conditions.

You can preserve that finding in two sentences inside `current-state.md`. A separate evidence document is optional.

### Retention conditions

Keep a finding across passes when at least one condition below applies.

| Condition | Minimum information to preserve |
| --- | --- |
| A current completion claim depends on the finding. | The checked condition, actual result, tested version, and any limitation. |
| A later action or design choice depends on the finding. | The observation, the decision it supports, and the conditions under which it applies. |
| Repeating the check needs unavailable data, access, equipment, or a past system state. | The relevant input, method, result, and original context. |
| Repeating the check would exceed an agreed time or resource limit. | The inputs, method, result, and limits needed to reuse the finding. |
| The result changes between runs, such as timing or model output. | The observed result and conditions. Preserve repeated measurements when the conclusion depends on them. |
| An active plan or proposal includes an approach that previously failed. | The failed approach, observed failure, and the reason for avoiding or changing it. |
| A project agreement requires retained checks, reviews, or reports. | The records and retention period required by that agreement. |

After a pass closes, discard routine detail only when none of these conditions requires it.

For a simple check, a written result can satisfy retention. Raw logs are not automatically required.

For a detailed investigation, preserve the supporting record. Link to it from the current state instead of copying the investigation.

If a later valid check replaces an earlier result, preserve the result needed for the current claim. Retain earlier findings only when another condition applies.

### When a separate evidence document is needed

Use an existing test report, issue, or retained run record when it contains the required information.

Create a separate evidence document when a retention condition applies and the existing records lack the required detail.

Use a descriptive name, such as `upload-failure-investigation.md`. Include the question, inputs, method, observations, interpretation, and limitations.

Do not create an empty evidence document for future work. Put planned checks in the active pass records.

Stateless code does not remove the need for verification. Stateless means an operation does not depend on stored state from earlier calls.

The need for retained detail depends on the claims and future decisions. It does not depend only on whether code is stateless.

### A small coding task

Suppose a script correctly handles the agreed inputs. Its automated checks run successfully for the recorded version.

The current state can retain the version, check command, result, and unchecked cases. The test files remain with the code.

After that transfer, the previous `verification.md` can reset. No separate evidence document is needed if no other retention condition applies.

## 8. Optional files and the conditions that require them

These files are optional at project start. Create one only when its stated condition applies and no existing document serves the purpose.

| Optional record | When to create it | What belongs there | When it can disappear |
| --- | --- | --- | --- |
| **`decisions.md`** | Later work needs the reasons for a choice after current status changes. | The choice, alternatives considered, supporting findings, owner, and date. | After no active choice relies on it and no retention agreement requires it. Preserve still-applicable reasons elsewhere first. |
| **A named evidence document** | A Section 7 retention condition applies, and the summary or existing records omit required detail. | Actual findings, their context, interpretation, and limitations. | After no current claim or future action relies on it and no retention agreement requires it. |
| **`technical-reference.md`** | Several passes need the same environment details, interfaces, settings, or technical agreements. | Shared technical information and its owner. Label observed settings separately from agreed constraints. | After all readers use a replacement or the information no longer applies. |
| **`design.md`** | Work across components requires an agreed description of boundaries, responsibilities, or interfaces. | The agreed design and links to supporting findings. Put unresolved proposals in the active plan or current state. | After an agreed replacement or removal of the affected work. Update links first. |
| **`questions.md`** | Independent unresolved questions continue across passes and need separate owners or investigation records. | Each question, its owner, its effect on work, and links to answers. | After all questions are resolved or withdrawn and required answers survive elsewhere. |
| **`AGENTS.md`** | An AI uses a project entry point to locate instructions and active records. | Reading order, document paths, ownership, and links to governing instructions. | Keep while the AI relies on it. Update the entry point before replacement. |
| **`README.md`** | A folder needs a reading route or an explanation of its contents. | What belongs there, which document to read first, and where related information lives. | After the folder disappears or another entry point replaces its purpose. |
| **An archive of a pass** | An agreement requires full pass history, or the full sequence is needed to explain a retained conclusion. | Copies of the closed plan and verification records under the pass identifier. | Only when the stated retention condition or period ends. |

For ordinary choices, keep the decision and reason in `current-state.md`. A separate decision document is not compulsory.

For one unresolved question, use `current-state.md`. A separate question file is not compulsory.

If a project separates high-level design and low-level design, use `hld.md` and `lld.md`.

High-level design describes components and their responsibilities. Low-level design describes interfaces, behavior, and implementation details.

Their agreed content belongs with governing references. Their progress belongs in the current state. Supporting observations belong in verification or retained evidence.

## 9. Edge cases and exact responses

EC2 means Amazon Elastic Compute Cloud. In this guide, an EC2 instance is the remote computer used for the task.

Git stores file versions in commits. A commit identifier refers to one recorded version.

| Situation | Required response |
| --- | --- |
| The task is small. | Keep the four records short. A plan can have one action. Verification can have one check. |
| An existing file already defines the goal or starting point. | Read it. Confirm that it applies. Reuse its role or transfer ownership explicitly. Avoid competing live copies. |
| The AI session ends during work. | Save completed actions, incomplete operations, pending checks, and the next action. Keep the same pass open. |
| A remote operation may still be running. | Record its process or job identifier when available. Inspect its actual status before repeating the operation. |
| A connection fails before the result arrives. | Mark the outcome unknown. Check remote state before retrying an action that could create duplicate effects. |
| The plan says done, but verification did not run. | Report the action as performed and the outcome as unverified. Keep the check pending. |
| AI checks pass, but user review is pending. | Set current state to `waiting for review` if that review blocks progress. Keep the active records. |
| The user accepts one result but gives no next-action decision. | Record only the acceptance received. Follow existing decision authority or request the missing decision. |
| AI and user observations disagree. | Record both observations and their conditions. Investigate the difference before claiming a shared conclusion. |
| Code or inputs change after a passed check. | Identify the affected checks. Mark those results superseded. Recheck the changed behavior and dependent behavior. |
| The environment changes. | Record the difference. Recheck claims that depend on the changed environment. Preserve unaffected results with their original scope. |
| A local check passes, but the goal concerns EC2. | Verify the relevant behavior on the agreed EC2 instance. Do not present local results as remote verification. |
| Source inspection supports an expected behavior, but no execution occurred. | Label the finding as source inspection. Keep runtime verification pending if the criterion requires execution. |
| A check fails repeatedly. | Compare failure details before another retry. Change the approach or request missing information. Preserve each distinct finding. |
| A finding later proves wrong. | Correct the claim. Identify the earlier conclusion as superseded. Update dependent status, decisions, and plans. |
| A goal change removes an unfinished criterion. | Record the authorized change and its reason. Report completion against the revised goal. Do not rewrite the earlier check as passed. |
| A learning task runs successfully, but understanding is untested. | Record execution as verified. Keep understanding pending until the agreed explanation or exercise demonstrates it. |
| Two people or agents work on different passes. | Give each pass separate active files. Assign one owner to reconcile the shared current state before either reset. |
| Two records conflict. | Inspect the underlying work and applicable agreement. Correct the records. Do not select a claim only because its timestamp is newer. |
| A record links to a file that will reset. | Move required details to a persistent location. Update the link before resetting the active file. |
| Git history is the proposed archive. | Confirm that a commit contains the required records. Retain its identifier and file path. Uncommitted files are not an archive. |
| The user pauses or cancels the work. | Save the actual position and the user's decision. Preserve any information required for recovery or retained conclusions. |
| The work is complete. | Check all goal criteria. Save the final report and retained support. End the active pass without creating an empty next pass. |

For remote checks, record the instance identifier, region, working directory, and tested code version when they identify the result.

If the code has no commit identifier, record another recoverable version reference. State when local edits differ from the recorded commit.

Keep credentials out of project records. Reference the approved access method instead.

## 10. Worked example: three coding passes

This example is illustrative. It does not report an actual execution or a real user's approval.

Assume the user and AI agree to review each pass together. The AI works on one agreed EC2 instance.

The goal is to write a script that adds whole numbers from a text file.

The goal has three criteria:

- **G1:** For valid whole numbers, print their sum.
- **G2:** Ignore blank lines without changing the sum.
- **G3:** For invalid text, report its line number and return a nonzero exit code. Print no total.

An exit code is a number that a program returns when it stops. This example uses zero for success and nonzero for failure.

The starting point has an accessible instance and an empty project folder. No script exists yet.

The labels `version-1`, `version-2`, and `version-3` identify example revisions. Real records need recoverable version references.

### Pass 1: implement addition for valid input

**Work flow table: Pass 1**

| Step and its purpose | What happens in this pass |
| --- | --- |
| **Define: establish the outcome and completion criteria.** | Read G1, G2, and G3. The overall goal includes all three. This pass addresses G1. |
| **Understand: inspect the starting point.** | Confirm the target instance and folder. Confirm that no implementation exists. |
| **Plan: choose actions and checks.** | Plan the first script version. Use a file with `2`, `3`, and `-1` on separate lines. Expect `4`. |
| **Work: perform the planned action.** | Write `version-1` on the instance. |
| **Verify: compare actual and expected results.** | The AI runs the script and observes `4` with exit code zero. The user checks the same result. |
| **Decide: choose the next action.** | The user and AI accept this pass result. They choose blank-line handling next. G2 and G3 remain incomplete. |

**Project records table: after Pass 1**

| Document | Contents or action |
| --- | --- |
| **`goal.md`** | Keep G1, G2, and G3. |
| **`current-state.md`** | `version-1` exists. The recorded valid-input check passed on EC2. The user confirmed the result. Next: implement blank-line handling. |
| **`plan.md`** | Mark Pass 1 actions done. Keep this plan until Pass 2 starts. |
| **`verification.md`** | Record the input, expected `4`, actual `4`, exit code, version, environment, user observation, and Continue decision. |
| **Persistent evidence file** | None required. The retained checkpoint contains this pass result. No other retention condition applies. |

Before Pass 2, save the checkpoint and required result details. Then reset the two active files for `pass-002`.

### Pass 2: handle blank lines

**Work flow table: Pass 2**

| Step and its purpose | What happens in this pass |
| --- | --- |
| **Define: confirm the agreed goal.** | Read the same three criteria. This pass addresses G2 while preserving G1. |
| **Understand: inspect the current position.** | Read the Pass 1 checkpoint. Inspect `version-1`. A blank-line check fails, which identifies the missing behavior. |
| **Plan: specify the change and checks.** | Plan to ignore blank lines. Expect `4` for both the original input and a copy with blank lines. |
| **Work: implement the change.** | Write `version-2` with blank-line handling. |
| **Verify: check the new and existing behavior.** | Both checks return `4` with exit code zero. The user checks the blank-line result. |
| **Decide: assess remaining work.** | The user and AI accept the result. G3 remains incomplete. They choose invalid-input handling next. |

**Project records table: after Pass 2**

| Document | Contents or action |
| --- | --- |
| **`goal.md`** | Keep all three criteria. |
| **`current-state.md`** | `version-2` passes the valid-input and blank-line checks. User review is complete for this pass. Next: implement G3. |
| **`plan.md`** | Mark Pass 2 actions done. Keep the plan until Pass 3 starts. |
| **`verification.md`** | Keep the initial blank-line failure and the later passing checks while this pass is active. Identify which version each result concerns. |
| **Persistent evidence file** | None required under these assumptions. Retain the current successful checks in the checkpoint before resetting. |

The checkpoint now describes `version-2`. The full Pass 1 account is not needed for the next action.

If a later investigation needs the original failure, preserve it under Section 7. Otherwise, the reset can remove routine attempt details.

### Pass 3: handle invalid input and finish

**Work flow table: Pass 3**

| Step and its purpose | What happens in this pass |
| --- | --- |
| **Define: confirm what overall completion requires.** | All three criteria must hold for the final version. |
| **Understand: identify the remaining gap.** | G1 and G2 passed for the prior version. G3 still needs implementation and verification. |
| **Plan: specify final work and all required checks.** | Plan invalid-input handling. Add an input with `2`, `hello`, and `3`. Expect a line-2 error, a nonzero exit code, and no total. |
| **Work: complete the planned change.** | Write `version-3`. |
| **Verify: check the final version.** | The original and blank-line checks return `4`. Invalid text produces the expected error, exit code, and absence of a total. The user completes the agreed checks. |
| **Decide: compare the results with the whole goal.** | The user and AI confirm that the agreed checks support G1, G2, and G3. They choose Finish. |

**Project records table: after Pass 3**

| Document | Contents or action |
| --- | --- |
| **`goal.md`** | Keep the agreed goal and all three criteria. |
| **`current-state.md`** | Record the final version, completed criteria, check inputs and results, user review, limitations, and Finish decision. Next action: none. |
| **`plan.md`** | Remove after the final checkpoint and retention check. No next pass is created. |
| **`verification.md`** | Preserve the final check details in the checkpoint or a retained report. Then remove the active file. |
| **Code and tests** | Keep them as project outputs. Active document cleanup does not remove them. |

The final report states what the script does, which version was checked, and which cases passed.

It also states the limits of those checks. Passing these cases does not prove correct behavior for every possible input or environment.

## 11. Writing rules for these records

Use [asd-ste100.md](asd-ste100.md) as the local writing reference.

This guide applies ASD-STE100 writing principles, including short sentences, direct instructions, and consistent terms. The document does not claim full dictionary compliance.

The [official ASD-STE100 standard](https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf) provides the writing rules and approved vocabulary.

Apply these rules when maintaining project records:

1. Use a maximum of 20 words in an instruction sentence.
2. Use a maximum of 25 words in a description sentence.
3. Give one instruction per sentence.
4. Use the same term for the same concept.
5. State who acts when responsibility matters.
6. Separate planned actions from actual observations.
7. Preserve uncertainty and limitations.
8. Define technical terms before readers need them.
9. Replace vague status with an action, result, or unresolved condition.
10. Keep each paragraph on one topic.

| Vague wording | Specific replacement |
| --- | --- |
| Everything works. | The three listed checks passed for version-3 on the recorded instance. |
| Verified. | The downloaded file matched the original contents. The user confirmed the destination. |
| Mostly done. | G1 and G2 passed. G3 remains incomplete. |
| Continue later. | Next: test invalid input on the agreed instance. The AI owns this action. |
| Needs investigation. | The command ended without an output file. Next: inspect its error output and exit code. |
| Save evidence when useful. | Retain findings that meet a condition in Section 7. |

The [official guidance on AI and STE](https://www.asd-ste100.org/STE_downloads.html) distinguishes plausible AI wording from verified compliance.

Use the official standard and an appropriate language review before claiming full ASD-STE100 compliance.

## 12. Check the records before handover

Confirm each item before ending a pass or handing work to another person or AI session:

- The goal and completion criteria have an identified owner.
- The current state matches the actual work.
- The active files name the same pass.
- Performed work and verified outcomes remain distinguishable.
- Required user checks show their actual status.
- Each completion claim has supporting observations or a working link.
- Remaining criteria and blockers are explicit.
- The next action has an owner and any prerequisite.
- Findings required after reset have a persistent location.
- Required links do not depend on contents that will disappear during reset.

At completion, report the achieved outcome, verification results, and remaining limitations.

Keep the amount of documentation proportional to the information needed for these checks. Short records can satisfy the same rules.
