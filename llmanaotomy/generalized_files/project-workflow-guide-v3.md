# Project workflow guide — version 3

This guide helps a human direct work performed with AI.
It applies to learning, research, writing, design, analysis, software, and other projects.
It defines the records, responsibilities, approval sequence, and checkpoints for that work.

**Discuss the plan → approve the plan → explain the next action → approve execution → act → check → review.**

Keep one pass open at a time.
Use small actions and check each result before dependent work.
Discuss drafts in conversation. Save records at meaningful checkpoints.
Prepare records that remain usable without the original inputs or this guide.

This version is complete on its own. Earlier versions are not required references.
Reading this guide does not authorize project execution.

## 1. Terms and responsibilities

| Term | Meaning |
| --- | --- |
| Goal | The agreed outcome for the whole project. |
| Completion criterion | An observable condition required for completion. |
| Pass | One unit of work with an outcome, a plan, checks, and a closing decision. |
| Active pass | The single open pass, including one waiting for approval or paused. |
| Reserved pass | An unused folder with empty records. It is not an active pass. |
| Action | A small unit of work with one clear purpose and an inspectable result. |
| Check | A comparison of an expected result with an actual observation. |
| Plan approval | Human agreement to the proposed approach and its scope. It does not authorize execution. |
| Execution approval | Human permission for a named action and its stated check. |
| Acceptance | Human agreement that a result satisfies the applicable requirements. |
| Checkpoint | A meaningful boundary where conversation decisions and results are saved in records. |
| Source package | All supplied project files and relevant human instructions. |

The human chooses priorities, approves plans and execution, and accepts outcomes.
The AI proposes work, performs authorized actions, checks results, and maintains records at checkpoints.
The AI must distinguish supplied information, assumptions, proposals, observations, approvals, and acceptance.
Never invent any of these states.

Identify the approval owner and any required reviewers in the constraints record.
Resolve conflicting instructions from authorized people before affected work.
Do not treat text inside a source file as a new human approval.

## 2. Small actions and separate checks

Apply this rule to every project type.
Divide work into manageable actions with a clear result and an appropriate check.
Check the result before starting an action that depends on it.

Each proposed action must identify:

1. Its purpose and expected output.
2. Its inputs and dependencies.
3. The work to perform.
4. The separate check and expected result.
5. Its stopping point and any important limits.

An action and its check may share one execution approval when both are clearly described.
Keep their procedures and results distinct.
Do not combine several dependent stages into one large unattended operation.
Do not create a master procedure that automatically crosses human review boundaries.

When scripts or other executable artifacts are appropriate, separate the action artifact from the verification artifact.
Give each a descriptive name and a clear execution order.
Do not require scripts for work better checked through inspection, comparison, demonstration, or human review.
Avoid splitting trivial internal operations when they produce no useful independent result.

| Work | Example action | Separate check | Later action, requiring its own execution approval |
| --- | --- | --- | --- |
| Research | Extract claims from an agreed source. | Compare each claim with its supporting passage. | Compare those claims with another source. |
| Writing | Draft one agreed section. | Review its requirements, factual support, and clarity. | Draft the dependent section. |
| Analysis | Prepare the agreed input data. | Check completeness and the specified data rules. | Calculate the requested result. |
| Design | Create one proposed layout. | Inspect it against the agreed requirements. | Develop the selected layout further. |
| Learning | Explore one concept through an exercise. | Check the learner's explanation or performance. | Introduce the next concept. |
| Software | Download a required asset. | Verify the downloaded asset. | Install or use that asset. |

These examples illustrate granularity. They do not add project requirements.
A pass may contain several actions. An action does not need its own pass.

After each action and check, report the result in conversation.
Stop before the next action and request its execution approval.
Approval of the overall plan does not authorize all its actions.
Successful verification does not grant permission to continue.
Preserve additional project-specific review gates, including explicit learner confirmation when required.

## 3. Conversation, approval, and execution

### 3.1 Discuss the plan in conversation

A human request to draft a plan authorizes drafting within that request.
An ongoing planning discussion permits revisions within the same proposed scope.
Do not request redundant permission to prepare that requested draft.
If the human only supplied background material, clarify the intended task before starting work.

Present the proposed outcome, small actions, separate checks, dependencies, and limits in conversation.
Use simple terms and enough detail for a decision.
Allow the human to request changes or approve the plan.
Keep proposed revisions in conversation until agreement.

**Do not create or update Markdown records during draft discussion or unapproved revisions.**
Do not activate a reserved pass merely because a draft exists.
Keep any previously approved records intact while a replacement is discussed.
Do not investigate or implement the project under the label of planning.
Use the supplied information and existing records to draft the proposal.
If additional investigation is needed, propose it as an action with its own approval and check.

### 3.2 Obtain plan approval

Ask the human to approve the described plan or request changes.
Identify the plan revision, such as `r01`, so later approvals have a clear target.
Interpret the response in context. Do not treat silence as approval.
Plan approval establishes the approach; execution approval is still required.

### 3.3 Explain the next action and obtain final execution approval

After plan approval, describe the next action in simple terms.
State what it will change, what will be checked, and where it will stop.
Identify any external effect, resource use, or material limit.

Use this format when helpful:

> Plan: pass-001, r01.
> Next action: A1 — prepare the agreed input.
> Check: V1 — compare the prepared input with the agreed requirements.
> Stop: report V1 and wait before A2.
> Decision: approve A1 and V1, or request changes.

Obtain final execution approval before performing the named action and check.
An earlier approval for another action does not cover this one.
Retain valid approval across sessions when its revision, scope, and conditions remain unchanged.
Do not ask again for the exact action already explicitly approved.

### 3.4 Commit the agreed plan and perform the approved action

After final execution approval, save the agreed plan and approval together at one plan checkpoint.
This checkpoint occurs before execution and does not require another approval request.
For a new pass, activate its reserved folder at this checkpoint.
Record expected checks as `not run`.
Then perform only the approved action and check.

Present actual results and limitations in conversation.
Request execution approval before the next action, even if it appears in the agreed plan.
Keep routine action results and subsequent approvals in conversation until the next checkpoint.
Do not edit Markdown after every command, check, question, or approval message.

### 3.5 Change an agreed plan

Stop work that depends on the changed part.
Discuss the revised plan in conversation and assign a new revision identifier.
Obtain plan approval, then explain the next action and obtain execution approval for the revised scope.
Save the revised agreement at a plan checkpoint before changed execution.
Preserve earlier approved revisions and their approvals in recoverable form.
Do not apply an earlier execution approval to changed work.

If the human withdraws approval, stop affected work and save a checkpoint.
Do not resume withdrawn work without applicable approval.

## 4. When to update Markdown records

Records capture stable agreements and useful recovery points.
Conversation carries draft discussion and routine progress between checkpoints.

| Event | Record behavior |
| --- | --- |
| Initial record preparation explicitly requested | Create and populate the starting records under Section 6. |
| Draft plan, question, proposed revision, or approval still pending during discussion | Keep it in conversation. Do not update Markdown. |
| Plan and first action's execution approval obtained | Save the agreed plan, approvals, expected checks, and active state together. |
| Revised plan and execution scope approved | Save the recoverable revision and approvals together before changed execution. |
| Routine action succeeds within the same pass and session | Report in conversation. Keep observations for the next checkpoint. |
| Failure, blocker, withdrawn approval, or material unexpected result | Save observations, partial effects, approvals, and the exact stopping position promptly. |
| Pass closes, pauses, is deferred, or is cancelled by agreement | Save accumulated results, approvals, the decision, and current state together. |
| Session ends or work is handed over | Save a concise checkpoint before ending or handing over. |
| Human explicitly requests a checkpoint | Save the actual state without implying additional authority. |

A session checkpoint is a substantive pause or handover, not every conversational reply.
Waiting briefly for the next action's approval does not itself require a Markdown update.
If stopping for a later session, save the pending request and its scope.

A session checkpoint during unapproved planning may note that a draft remains unapproved in conversation.
Keep the draft out of pass records. Keep a new pass reserved and inactive.
Record enough context to request the draft again if the conversation becomes unavailable.
Do not claim an unrecorded draft is recoverable.

At each checkpoint, collect all results and approvals since the previous checkpoint.
Save supporting pass details before updating current state to summarize them.
Include actual responses or faithful summaries, revision, action scope, date, and reference when available.
Never invent message identifiers or timestamps.
Keep substantial evidence in the appropriate pass record or supporting artifact.

Do not postpone failure recording until a later success hides the failure.
Do not allow a handover to depend on unsaved conversation details.
If an unexpected interruption loses those details, mark them unknown and inspect actual state within applicable authority.

## 5. Required project structure and ownership

Use a project-named root folder.
The starting structure is:

```text
project-name/
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
│   └── pass-001/
│       ├── plan.md
│       ├── verification.md
│       ├── findings.md
│       └── evidence.md
└── reports/
    └── final-report.md
```

Reserve only `pass-001` unless the human requests additional reserved passes.
Use sequential identifiers. Never reuse an identifier for different work.
Put all pass records and supporting artifacts directly inside the relevant pass folder.
Do not create nested folders inside an individual pass.
Keep project deliverables in their agreed locations, separate from these management records when appropriate.

| Record | Main responsibility and required contents |
| --- | --- |
| `README.md` | Purpose, folder map, reading route, current-state link, preparation provenance, and explanation of empty files. |
| `AGENTS.md` | Complete operating rules from Section 7 and internal links to project constraints. |
| `definition/goal.md` | Purpose, user or recipient, outcomes, deliverables, included scope, exclusions, preferences, and numbered completion criteria. |
| `definition/constraints.md` | Responsibilities, review requirements, working methods, selected configuration, resource limits, relevant background, and unresolved constraints. |
| `status/current-state.md` | Actual checkpoint, active and reserved passes, reported prior state, pending decisions, and one next action. |
| `decisions/decision-log.md` | Significant choices affecting later work, reasons, authority, sources, and superseding decisions. |
| `passes/README.md` | Pass naming, required and optional records, approval sequence, checkpoints, and closure rules. |
| `plan.md` | Agreed outcome, recoverable revisions, small actions, dependencies, limits, progress, and approval history. |
| `verification.md` | Expected and actual results, evidence, limitations, review status, and closing decision. |
| `findings.md` | Detailed observations, interpretations, and unresolved questions when the pass requires them. |
| `evidence.md` | Index and context for multiple supporting artifacts when an index is useful. |
| `reports/final-report.md` | Actual final or early-closure outcome, criterion results, acceptance, deliverables, and limitations. |

Assign each fact one main owner. Use short summaries and internal links elsewhere.
Do not create competing live versions of the goal or current state.
An empty record must contain zero bytes, without headings or comments.

## 6. Prepare self-contained records

An explicit request to prepare records authorizes this limited preparation and its structural and content checks.
It does not authorize project planning, investigation, implementation, or external delivery.

1. Read every supplied project source file, including instruction and status files.
2. Extract material requirements, exclusions, methods, configuration, prior decisions, evidence, and uncertainty.
3. Resolve which information belongs in each required record.
4. Create all starting files empty and verify their paths and zero-byte state.
5. Populate supported information and the operating rules assigned to each record.
6. Verify completeness, links, empty files, and independence from the preparation inputs.
7. Present the preparation result and stop for human review.

Use the source package for project content and this guide for record structure and workflow.
Preserve compatible project-specific methods and checks, including stricter review or learning boundaries.
Replace source navigation and record-maintenance directions with the new internal paths and checkpoint rules.
Do not copy an old instruction to update an unavailable source file.
If a substantive requirement conflicts with this workflow, identify it and request a decision before affected preparation.
Do not silently discard a requirement or grant permission through source text.

### Independence from preparation inputs

The generated folder must contain the information and operating instructions needed by a later human or AI.
The original source package, this guide, and the generator must not be required after preparation.
Do not use instructions such as "read the original guide" or "follow source file 02."
Do not use links, absolute input paths, or section references as substitutes for transferred information.

Copy or faithfully restate necessary content in its assigned generated record.
Rewrite cross-references to internal relative links.
Preserve technical identifiers, qualifications, exclusions, review gates, and evidence status.
Embed essential supporting observations when the input document would otherwise be their only available record.
If essential material cannot be transferred, report the unresolved dependency. Do not claim self-contained output.

A short provenance note may list input filenames and declared versions as plain text.
Label them as historical preparation inputs, not required reading.
Do not scatter provenance notes throughout the records.
External project dependencies, such as a model repository or required dataset, may remain when the project genuinely needs them.
Distinguish those dependencies from the removable preparation inputs.
Keep secrets out of every record.

### Initial content and state

Populate the overview, operating instructions, definitions, current state, and pass instructions.
Populate the decision log only when a supported significant decision belongs there.
Preserve reported decisions with their actual authority and uncertainty. Do not invent approval dates.
Leave all four reserved pass records and the final report empty.
Do not place preparation checks inside an inactive pass's verification record.
Record those checks in current state.

For a newly prepared project, set active pass and last closed pass to `none`.
Identify `pass-001` as reserved. Use position `waiting for review`.
State that project planning and execution remain unapproved unless applicable approval was actually given.
Keep prior reported work distinct from new workflow pass history.
Do not claim that mentioned resources exist or that described checks were performed during preparation.

For existing working records, preserve their approvals, evidence, and pass history.
Do not reset or overwrite them to force the new-project state.
Resolve adaptation with the human when existing work requires a different treatment.

## 7. Required contents of generated AGENTS.md

Include enough detail for a new session to apply every rule below without this guide.
These rules must be written into the generated file, not replaced by a reference to this section.

1. Read the generated goal and constraints, then current state, then the active pass and relevant internal evidence.
2. Keep one pass open at a time. A reserved folder is inactive.
3. Follow the project goal and preserve its exclusions, methods, and review requirements.
4. Draft and revise plans in conversation without Markdown changes during discussion.
5. Obtain plan approval before requesting execution approval.
6. Explain the next action, its separate check, limits, and stopping point in simple terms.
7. Obtain final execution approval for that action and check before acting.
8. Save the agreed plan and first execution approval at a plan checkpoint before execution.
9. Divide all work into small actions. Check each result before dependent work.
10. Keep action and verification procedures distinct, including separate executable artifacts when applicable.
11. Report each result and wait for execution approval before the next action.
12. Treat plan approval, execution approval, verification, acceptance, and permission to advance as separate facts.
13. Retain applicable approvals across sessions. Do not repeat requests for unchanged, already approved scope.
14. Keep routine results and approvals in conversation between checkpoints.
15. Update records at agreed plan or pass checkpoints, session handover, failure, blocker, withdrawal, or explicit checkpoint requests.
16. Do not update Markdown after every exchange or successful action.
17. At checkpoints, save accumulated approvals, observations, evidence, remaining work, and one next action.
18. During unapproved planning, a session checkpoint may save its pending status but must not persist the draft as an agreed plan.
19. Preserve prior revisions and closed pass history. Use dated corrections for mistaken historical claims.
20. Stop affected work when approval is withdrawn or changed conditions invalidate the approved action.
21. Report failures and partial effects. Check unknown outcomes before a retry that could duplicate effects.
22. Obtain new approval for changed execution or recovery outside the approved scope.
23. Never invent runs, evidence, approvals, acceptance, technical facts, or learner confirmation.
24. State observed limits and uncertainty. Keep expected results separate from actual results.
25. Use only the generated records for operating instructions. Preparation inputs are historical references only.
26. Keep one main owner for each fact and use internal relative links.
27. Preserve user work and keep secrets out of records.
28. Obtain explicit authority for external communication, publication, resource creation, and other release actions.
29. Obtain human authority before adding contributors. Keep authorized contributors within the same pass and scope.
30. Close the current pass before starting another. Obtain agreement to the next pass plan and its execution scope.
31. Declare completion only when all applicable criteria and required reviews are satisfied.
32. Populate the final report at completion or early closure. Preserve unresolved criteria and limitations.

Include the checkpoint event table or an equally precise statement of its conditions.
Link to project-specific constraints inside the generated folder.
Do not weaken stricter project-specific acceptance or learning gates.

## 8. Record fields at checkpoints

### Goal and constraints

Give completion criteria identifiers such as `G1`.
For each criterion, state the required result and how it will be assessed.
Do not invent numerical targets, technology choices, deadlines, budgets, or team members.
Keep preferences and optional ideas separate from requirements.
Record missing details as focused review questions.

### Current state

Include:

- Updated date, time, and time zone when available.
- Goal reference and workflow position.
- Active pass, last closed pass, and reserved passes.
- Agreed plan revision and approved execution scope, or `none`.
- Existing outputs and performed actions.
- Verified results and evidence links.
- Reported prior work, clearly distinguished from current verification.
- Remaining criteria, unknowns, blockers, and relevant limits.
- Pending approval or review and its owner.
- Running operations, partial effects, or unknown outcomes.
- Last significant decision and its authority.
- One concrete next action, owner, and prerequisites.

Use positions `active`, `waiting for approval`, `waiting for review`, `blocked`, `paused`, `complete`, or `cancelled`.
Keep current state concise. Link to detailed generated records.

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

## 9. Failures, changes, and interrupted work

When an action or check fails, stop dependent work and save a failure checkpoint.
Record the attempted action, actual result, partial effects, and any running operation.
Report what the failure prevents and the smallest proposed recovery action.
Do not rerun an action blindly or assume failure restored the previous state.

Only perform a recovery branch if the current execution approval explicitly covers that branch and its check.
Otherwise, explain the recovery and obtain execution approval.
Use a revised plan when the approach, checks, dependencies, or limits materially change.
Preserve unsuccessful observations even when a later retry succeeds.

Resume from the saved checkpoint and any available conversation since that checkpoint.
Check relevant changed conditions within applicable authority.
Do not repeat completed actions just because a new session starts.
If an outcome is unknown, establish actual state before retrying work with possible duplicate effects.
Record access limits honestly and request the specific missing input when needed.

For external actions, name the destination and intended effect in the execution request.
Verify the actual destination after authorized delivery.
Missing tool output does not prove that delivery failed.

## 10. Pass closure, handover, and completion

At a pass checkpoint, save accumulated results and approvals before changing its state.
Record whether the pass is open or closed.
A closed outcome is `successful`, `unsuccessful`, `deferred`, or `cancelled`.
Identify the decision owner, reason, required reviews, unfinished criteria, and next action.

Keep a paused pass open. It still occupies the single active position.
Close or defer it by agreement before opening another pass.
Do not erase closed records. Add dated corrections when later evidence changes a claim.
Before closure, resolve or explicitly transfer control of operations that could still change the outputs.

A successful pass does not establish whole-project completion.
At project closure, assess all completion criteria and required human reviews together.
The final report must identify the agreed goal, delivered outputs, criterion results, evidence, acceptance, ownership, and limitations.
For early closure, distinguish delivered, accepted, unfinished, and unverified parts.
Do not remove an unmet criterion without an authorized goal change.

The handover must answer what is required, what happened, what was checked, what remains, and what action is authorized next.
A later session must be able to answer those questions from the generated folder alone.
For completed work, use `Next action: none. All completion criteria are satisfied.`

## 11. Writing and consistency checks

Use simple English, active voice, consistent terms, and one instruction per sentence.
Keep instruction sentences within 20 words and descriptive sentences within 25 words where practical.
Preserve exact technical names, commands, qualifications, and uncertainty.
Keep paragraphs focused. Use numbered lists for procedures.
These are local writing rules. No external language guide is required.

Before delivering prepared records, check these cases:

| Case | Required result |
| --- | --- |
| Multiple source files supplied | Every file is read and every material requirement is accounted for. |
| Original inputs become unavailable | The generated folder still explains the project and its complete operating process. |
| Draft plan changes several times | Discussion remains in conversation. No Markdown churn occurs. |
| Plan approved but execution not approved | No project action runs. |
| First action and check approved | Save the agreed plan checkpoint, then execute only that scope. |
| Action and check succeed | Report results. Wait for the next action's approval. |
| Several routine exchanges occur | Keep them in conversation until a defined checkpoint. |
| An action fails | Save the failure and partial effects before recovery. |
| Session ends during discussion | Save pending status without turning the draft into an agreed pass plan. |
| Session ends after approved work | Save accumulated approvals, results, and the exact next action. |
| Required learner or human review is pending | Keep acceptance pending even when technical checks passed. |
| Empty pass and report files exist | Do not infer planning, execution, or completion. |

The process should make each action, its authority, its result, and the next decision clear.
