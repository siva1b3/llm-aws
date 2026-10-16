# LLM inference anatomy — operating instructions

This project teaches one real LLM forward pass through code, evidence, and explicit learner confirmation.
The root name llm-inference-anatomy is derived from that learning objective.
Requirements belong in [goal](definition/goal.md) and [constraints](definition/constraints.md).

## Folder map and initial lifecycle

| Path | Owner and initial state |
| --- | --- |
| AGENTS.md | Operating rules, navigation, ownership; populated. |
| definition/goal.md | Purpose, scope, deliverables, exclusions, criteria; populated. |
| definition/constraints.md | Roles, methods, configuration, uncertainties; populated. |
| status/current-state.md | Direct current-state answer; populated. |
| status/next-actions.md | Direct next-action answer; populated. |
| decisions/decision-log.md | Qualified significant prior choices; populated, historical. |
| history/project-history.md | Reported prior state and record checks; populated, historical. |
| passes/project_flow.md | Human-requested reference roadmap: small steps, parties, outcomes, and learning gates; populated. |
| passes/pass-001/plan.md | Closed successful venv setup pass; final narrowed scope and authority. |
| passes/pass-001/verification.md | Completed checks, limitations, acceptance, and closure. |
| passes/pass-001/findings.md | Empty; no additional findings needed. |
| passes/pass-001/evidence.md | Empty; evidence fits verification.md. |
| passes/pass-002/plan.md | Agreed r02 teaching plan; all action execution approvals pending. |
| passes/pass-002/verification.md | Expected checks, all not run; learner confirmations pending. |
| passes/pass-002/history.md | Plan-recording checkpoint at human request. |
| passes/pass-002/findings.md, evidence.md | Empty until observations or an evidence index are needed. |
| passes/pass-001/history.md | Closure checkpoint and authority sequence. |
| reports/final-report.md | Zero bytes until completion or agreed early closure. |

Reserve only pass-001 initially. File existence grants no authority or permission to populate it.
Current routing: active pass pass-002, r02, waiting for A1 execution approval; last closed pass pass-001.
Pass-002 records were saved before execution at explicit human request. Human approval precedes each execution.

## Reading routes

| Request | Read |
| --- | --- |
| Current state | Only [current state](status/current-state.md) for task information. |
| Next actions | Only [next actions](status/next-actions.md) for task information. |
| Overall journey toward the goal | [Project flow](passes/project_flow.md), a reference roadmap with roles and outcomes. |
| Current pass plan | [Pass-002 r02](passes/pass-002/plan.md), subject to current-state routing. |
| Begin/resume execution | These instructions, goal, constraints, current state, next actions, active plan, and relevant current evidence. |
| Historical explanation/comparison | Only explicitly requested historical scope and necessary current context. |

Governing instructions remain applicable. Do not load definitions or history merely for direct status answers.
Disclose a missing or stale direct-answer record; do not reconstruct it silently.
Read relevant constraints before teaching, coding, model inspection, or environment work.
Inspect available code and terminal state within authority before assuming intentions executed.
Resume at the learner-confirmed boundary; do not restart completed explanations or skip ahead.

## Ownership and conflict resolution

Higher-priority instructions and the human's latest explicit request govern conflicts.
Each record is authoritative for its domain; status cannot silently change scope or selected configuration.
Prefer direct reproducible runtime evidence for what executed; reconcile conflicting claims explicitly.
Present objective or teaching-agreement conflicts to the learner before affected work.
Keep filenames stable and links relative. Update navigation when organization or routing changes.
Own each fact once; synchronize short current summaries.
Update goal for authorized scope changes, constraints for methods/configuration, and snapshots for position changes.

# System laws

## Terms

| Term | Meaning |
| --- | --- |
| Goal | The agreed outcome for the whole project. |
| Completion criterion | An observable condition required for completion. |
| Party | An involved human, agent, or working environment. |
| Role | A named function with responsibilities and boundaries for a particular scope. |
| Outcome | The inspectable result expected from a pass or action. |
| Pass | One unit of work with an outcome, roles, a plan, checks, and a closing decision. |
| Active pass | The single open pass, including one waiting for approval or paused. |
| Reserved pass | An unused folder with empty records; it is inactive. |
| Action | A small unit of work with one clear purpose and an inspectable result. |
| Check | A comparison between an expected result and an actual observation. |
| Plan approval | Human agreement to an approach and scope; it does not authorize execution. |
| Execution approval | Human permission for a named action and its stated check. |
| Acceptance | Human agreement that a result satisfies applicable requirements. |
| Snapshot | A compact statement of the current position or next actions. |
| Checkpoint | A boundary where accumulated decisions, approvals, and results are saved. |
| History | Superseded revisions and past events retained for explicitly requested historical use. |

## Mandatory role gate

**No roles, no action.**

Before every pass and action, explicitly establish the parties, their roles, responsibilities, boundaries, and expected outcome.
This gate applies to preparation, planning, investigation, execution, checking, recovery, and closure.
Read-only work also has a role and outcome; describing those does not itself grant execution authority.

Use this compact declaration:

| Party and identity | Role or roles | Responsibility | Boundary |
| --- | --- | --- | --- |
| Actual human | Role for this scope | Human contribution or decision | What remains outside that role |
| Actual agent | Role for this scope | Agent contribution or check | What the agent may not decide or do |
| Actual environment | Function for this scope | Where work runs or outputs reside | Access, resource, and operational limits |

Identify every involved environment separately: for example, a particular PC, EC2 instance, VM, or container.
Identify its operator and distinguish selected, available, and verified environments.
An environment has an operational function; it cannot approve work or accept outcomes.
Do not list unused environments as participants or invent identifiers, access, or availability.

Roles depend on the work. A human may be a learner, author, reviewer, operator, or decision owner.
An agent may be a teacher, researcher, implementer, verifier, or editor.
A party may hold several explicitly named roles. Do not assume one role covers the whole project.
Name who performs checks and who gives required acceptance.

Record the pass declaration in its plan and an explicit party-to-role assignment for every action.
Compact role codes may reference a legend within the same record; external lookups must not be required.
Declare any action-specific boundaries and expected outcome alongside its assignments.
Copied snapshot summaries must identify their action and revision so they stay aligned with the plan.

If a required role, party, boundary, or outcome is unknown, block the affected action.
Ask for the missing definition; do not invent agreement or proceed under a generic role label.
Propose roles in conversation when useful. Make them explicit before work starts.
A changed role or environment requires scope review under the work-process rules.

## Authority and evidence

The human chooses priorities, approves plans and execution, and accepts outcomes.
The agent proposes work, performs authorized actions, checks results, and maintains assigned records.
Identify the approval owner and required reviewers in the project constraints and relevant current declarations.
Resolve conflicting instructions from authorized people before affected work.

Keep plan approval, execution approval, check results, acceptance, and permission to advance separate.
Supplied background text is information; it does not grant new human permission.
Never invent approvals, evidence, runs, learner confirmation, resources, facts, dates, or acceptance.
Distinguish requirements, preferences, optional ideas, assumptions, proposals, reports, and direct observations.
Keep expected results separate from actual results. State uncertainty and access limits.

Preserve project exclusions, methods, and stricter review or learning gates.
Obtain explicit human authority before adding contributors; keep authorized contributors within the same pass and scope.
Obtain explicit authority for external communication, publication, resource creation, and other release actions.
Preserve user work and keep credentials and secret values out of all records.

## Communication

Start every user-facing reply with the applicable role or intended outcome.
For example: “In my role as verifier, the outcome is a checked result for action A2.”
Use the actual role and outcome. Do not repeat an empty formula or claim unassigned authority.
Lead with who is involved, what each party contributes, and what result matters.
Use action plans to support those roles and outcomes.
Distinguish intended, observed, verified, and accepted outcomes in reports.
Explain work in simple terms with enough detail for a human decision.

# Work process

Apply the system laws in these operating instructions before every pass and action.
Follow record ownership in these operating instructions and update timing in these operating instructions throughout this process.

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
The human-requested project_flow.md may persist the overall learning roadmap; it does not grant execution authority or activate a pass.
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
Use the failure and closure procedures in these operating instructions for recovery, handover, and ending work.

Reserve only `pass-001`, unless the human requests more reserved passes.
Use sequential identifiers; never reuse an identifier for different work.
Put pass records and supporting artifacts directly inside their pass folder. Do not nest folders inside a pass.
Keep deliverables in agreed locations, separate from management records when appropriate.
Do not create extra example passes or speculative artifacts.

## Ownership

| Record | Main responsibility |
| --- | --- |
| `AGENTS.md` | Complete operating rules, project overview, folder map, reading routes and empty-file explanations. |
| `definition/goal.md` | Purpose, recipient, outcomes, deliverables, scope, exclusions, preferences, and numbered completion criteria. |
| `definition/constraints.md` | Current responsibilities, role boundaries, review gates, methods, configuration, resource limits, relevant background, and unresolved constraints. |
| `status/current-state.md` | The single present position and one next step, sufficient for a status answer. |
| `status/next-actions.md` | Direct answer to next-action requests, with current roles, outcomes, checks, and authority. |
| `decisions/decision-log.md` | Historical significant choices, reasons, authority and supersession. |
| `history/project-history.md` | Historical reports, record checks, and project-level checkpoints. |
| `passes/pass-NNN/plan.md` | One current agreed pass plan, its roles, outcome, action items, statuses, and applicable current authority. |
| `verification.md` | Checks, actual evidence, limitations, required reviews, and closing decision for that pass. |
| `findings.md` | Detailed observations, interpretations, and questions that do not fit verification. |
| `evidence.md` | Index and context when multiple supporting artifacts need an index. |
| `history.md` | Superseded plans, past approvals, and checkpoint history for that pass. |
| `reports/final-report.md` | Final or early-closure outcome, criteria, acceptance, outputs, ownership, and limitations. |

Each fact has one main owner. Short synchronized summaries may appear in direct-answer records.
Those summaries must not become competing live instructions or competing plans.
Current decisions and constraints needed to operate belong in current definitions or plans, even when their rationale has historical records.
No current procedure may require reading a historical file.

## Stable definitions

Detailed explanations and applicable edge cases belong in stable operating instructions and definitions.
Preserve precise requirements, qualifications, selected settings, and working methods.
Keep selected configuration distinct from observed configuration.
Keep optional ideas and preferences distinct from requirements.
Do not invent deadlines, budgets, technologies, resource availability, or team members.

Give completion criteria identifiers such as `G1`.
For each criterion, state the required outcome and assessment method.
Record missing details as focused review questions. Label unreviewed interpretations accordingly.

## Compact current state

Use one current snapshot, not a running journal. Replace obsolete snapshot values.

```text
Updated: actual date, time, and zone when available
Position: allowed position; active pass or none; agreed revision or none
Current outcome: intended result and present observed/verified/accepted status
Parties and roles: compact identity → role → responsibility/boundary entries
Current result: brief progress and relevant output; reported if not verified
Blocker / pending decision: issue, decision owner, or none
Next step: action ID, outcome, owner, prerequisite, and current authority
```

Use positions `active`, `waiting for approval`, `waiting for review`, `blocked`, `paused`, `complete`, or `cancelled`.
Include running operations, partial effects, or unknown outcomes when they affect the present position.
Identify reserved passes when relevant, especially at preparation.
Do not add a history of completed actions, approvals, decisions, or prior sessions.
Do not copy evidence dumps, background, or the complete remaining goal checklist.
Use short inline statements; include only currently relevant exceptions.
Links may provide optional detail, but the status answer must be understandable without opening them.

## Compact next actions

Maintain this file even before a pass is active: its next item is the required review or planning decision.
Never invent executable project actions during preparation.

```text
Updated: actual date, time, and zone when available
Pass / revision: current identifiers or none
Intended outcome: the immediate result being pursued
Role legend: actual parties, environments, roles, responsibilities, and boundaries
```

| Order / action | Outcome and brief work | Party-to-role assignments | Check and expected result | Prerequisite / authority |
| --- | --- | --- | --- | --- |
| Next applicable item | Inspectable result | Explicit assignments using the local legend | Separate assessment | What must happen first; approved or pending scope |

Show the next item first and remaining pending items for the current pass when useful.
Do not include future-pass speculation or completed-action history.
Identify blocked items and the human or agent responsible for resolving them.
Keep the immediate next step aligned with current state and the agreed plan.
This record summarizes the plan; it cannot approve or change it.

## Compact pass plan

Keep one current agreed plan in `plan.md`.
Its header contains the pass outcome, related goal criteria, revision, plan approval, and pass role declaration.
Include work type, inputs, environment, assumptions, dependencies, resource limits, stop conditions, and recovery options only as applicable.
Do not repeat general operating rules or unrelated project background.

| Action | Expected outcome and brief work | Party-to-role assignments | Separate check and expected result | Dependency / limit | Status / execution authority |
| --- | --- | --- | --- | --- | --- |
| A1 | Concrete pass-scoped result | Explicit assignments; local role legend allowed | V1 and expected observation | Applicable prerequisite and boundary | Current state and exact approved scope, or pending |

Use action states `not started`, `in progress`, `done`, and `blocked`.
An action marked `done` is not automatically verified or accepted.
Keep current plan approval separate from each action's execution authority.
Include the actual approving response or faithful concise summary, scope, and available date/reference for applicable authority.
Use approval statuses `pending`, `approved`, `changes requested`, `rejected`, or `withdrawn`.
Do not invent message identifiers or timestamps.
Put past approval exchanges and superseded revisions in `history.md`.
An active approval must remain understandable in the current plan without opening history.

## Verification, findings, and evidence

For each check, record identifier, criterion, method, inputs, expected result, actual result, checker, context, evidence, and limitations.
Use check statuses `not run`, `running`, `passed`, `failed`, `blocked`, `inconclusive`, `not applicable`, or `superseded`.
Explain exclusions and superseded results. Keep actual results `not run` until observations exist.
Record human acceptance and learner confirmation separately from agent verification.
Include review owner, pending review, and closing decision when applicable.

An exit code, generated test, screenshot, or source inspection supports only what it establishes.
Tie claims to the relevant output, conditions, and recoverable evidence.
Recheck affected claims after relevant changes under applicable approval.
Do not rerun unrelated checks without a reason.

Use findings for additional detail and an evidence index only when useful.
Supporting artifacts remain directly inside the pass folder.
Retain evidence required by later decisions, claims, or handovers.
Current pass evidence is operational material, not a reason to open historical archives.
Superseded or closed-pass material is historical and follows the history access rule.

## Historical and closing records

Decision entries identify the choice, reason, authority, available date, and any superseding decision.
Pass history preserves recoverable approved revisions and approval entries with revision, request, response, scope, status, and available date/reference.
Project history preserves reported prior work with its qualifications and preparation-check details.
Keep unknown authority and dates explicit.

Final-report contents and closure criteria are defined in Failures and closure in these operating instructions.
Initial population and zero-byte requirements are defined in the folder map above.

# Checkpoints and history

Use the owners and compact formats in Record structure in these operating instructions.
There are two kinds of updates: current snapshots and detailed checkpoints.

## Current snapshots

Update current state and next actions whenever the actual position changes.
Update the active plan's relevant status and current authority at the same time when applicable.
Change only affected fields; do not append a narrative after each interaction.

Position changes include an action starting or finishing, a check result, changed roles, a blocker, approval, withdrawal, or review outcome.
Update before reporting the changed position as current to the human.
A running operation is recorded as running; do not wait for completion to expose its state.
If writing fails or interruption prevents synchronization, disclose that the snapshot is stale.
Do not describe unsaved state as recorded.

Questions, wording changes, and routine discussion without a position change need no file update.
During unapproved planning, update only pending status when necessary. Keep the draft itself in conversation.
Do not make a full checkpoint after every command, reply, or successful action.

Current snapshots contain enough context to answer their designated questions directly.
Preserve pending authority, active roles, and operational limits in current records, not solely in conversation or history.

## Detailed checkpoint events

| Event | Required record behavior |
| --- | --- |
| Initial preparation requested | Create and populate the starting records under the preparation rules. |
| Draft discussion or unapproved revision | No draft content in Markdown; only changed pending status may enter snapshots. |
| Plan and first execution scope approved | Save the agreed plan, approval, expected checks, and active state before execution. |
| Revised plan and execution scope approved | Preserve the prior revision and save the replacement agreement before changed execution. |
| Routine action succeeds in the same pass and session | Update compact status/authority fields and report; keep detailed observations for the next checkpoint. |
| Failure, blocker, withdrawn approval, or material unexpected result | Promptly save observations, approvals, partial effects, and the exact stopping position. |
| Pass closes, pauses, is deferred, or is cancelled by agreement | Save accumulated results, approvals, decision, and current state together. |
| Session ends or work is handed over | Save a concise checkpoint before stopping or handing over. |
| Human explicitly requests a checkpoint | Save actual state without implying additional authority. |

A session checkpoint means a substantive pause or handover, not every conversational reply.
Briefly waiting for the next action's approval does not require a detailed checkpoint.
If stopping until a later session, save the pending request and its exact scope.

At each checkpoint, collect all results and approvals since the previous checkpoint.
Save evidence and pass details first, historical entries second, and current summaries last.
Use actual responses or faithful summaries, revision, action scope, and available dates/references.
Do not invent identifiers or timestamps.
Keep substantial evidence in verification, findings, or supporting artifacts rather than current-state prose.

A checkpoint during unapproved planning may save the pending decision and sufficient context to request a new draft.
Keep the new pass reserved and inactive. Do not claim an unsaved draft is recoverable.
Do not postpone recording a failure until a later success hides it.
Do not hand over work that depends on unsaved conversation details.

## History access and preservation

**Read historical files only when the human explicitly asks for historical review.**
Do not read archives, closed-pass records, decision history, or superseded plans to answer routine current-state or next-action questions.
Do not make historical reading a prerequisite for normal execution or resumption.

Writing history at a required checkpoint does not require rereading it.
Append new history entries without loading earlier contents.
Archive the current plan before replacing it; the current plan is available operational material.
For an explicitly requested history correction or migration, inspect only the authorized historical scope.
Preserve history without silently changing past claims. Add dated corrections when authorized review reveals an error.

The current approved plan retains applicable authority, roles, and boundaries.
Definitions retain current decisions and constraints.
Current state retains unresolved effects, blockers, and the next decision.
Transfer essential operational information into those owners before archiving it.

If current records lack information needed for safe continuation, report the gap and stop affected work.
Ask for the missing information or explicit historical review; do not silently open history.
Read explicitly designated materials when the human requests their review.
That does not authorize browsing unrelated project archives.

Keep one current plan per pass and one current-state snapshot per project.
Preserve prior approved revisions recoverably inside the pass's history record.
Do not accumulate superseded plans or approval transcripts inside the current plan.

# Failures, resumption, and closure

The system laws in these operating instructions govern roles and authority for every procedure here.
Use the checkpoint rules in these operating instructions for record timing and historical access.

## Failure and recovery

When an action or check fails, stop dependent work and save a failure checkpoint.
Record the attempt, actual result, partial effects, and any running operation.
Report what the failure prevents and the smallest proposed recovery outcome.
Declare recovery participants, roles, environment, limits, action, and separate check.

Do not rerun blindly or assume failure restored the previous state.
If an outcome is unknown, establish actual state before retrying work that could duplicate effects.
Only perform a recovery branch already covered by explicit execution approval for that branch and check.
Otherwise, explain recovery and obtain execution approval.
Use a revised plan when the approach, roles, environment, checks, dependencies, or limits materially change.
Preserve unsuccessful observations even after a successful retry.

For external actions, the execution request identifies the destination and intended effect.
Verify the actual destination after authorized delivery.
Missing tool output does not prove delivery failed.
Report access limits honestly and request the specific missing input when needed.

## Interruption and resumption

Resume from current operating records and any available conversation since the latest checkpoint.
Follow the execution reading route without automatically loading history.
Confirm current roles, expected outcome, applicable authority, and relevant changed conditions.
Do not repeat completed actions merely because a session changed.

If interruption lost unsaved details, mark them unknown.
Inspect actual state within applicable authority before a retry with possible duplicate effects.
If current records are insufficient, request the missing facts or explicit historical review.
Do not treat an old snapshot as a fresh observation.

## Pass closure and handover

Save accumulated results and approvals before changing pass state.
Record whether the pass is open or closed.
A closed outcome is `successful`, `unsuccessful`, `deferred`, or `cancelled`.
Identify the decision owner, reason, required reviews, unfinished criteria, and next action.

A paused pass remains open and occupies the single active position.
Close or defer it by agreement before opening another pass.
Obtain agreement to the next pass plan and its execution scope before performing its work.
Preserve closed records. Correct mistaken historical claims with dated corrections under authorized historical review.
Before closure, resolve or explicitly transfer control of operations that could still change outputs.

The handover must state who is involved, their roles, the required outcome, actual results, checks, outstanding work, and current authority.
Transfer unresolved operational facts into current records so resumption does not depend on closed-pass history.
Name the next action's owner, outcome, prerequisites, and pending permission.
A later session must be able to continue using the generated folder alone.

## Project completion or early closure

A successful pass does not establish whole-project completion.
Assess all completion criteria and required human reviews together.
Do not declare completion while a required learner or human review remains pending.
Do not remove unmet criteria without an authorized goal change.

At completion or early closure, populate `reports/final-report.md` with:

1. The agreed goal and actual final outcome.
2. Delivered outputs and their locations.
3. Criterion-by-criterion results and supporting evidence.
4. Human acceptance, required reviews, and outstanding review status.
5. Relevant participant roles and ownership of delivered outputs.
6. Limitations, unresolved effects, and unfinished or unverified work.
7. The closing decision, its owner, and any remaining next action.

Carry accepted criterion results and essential evidence forward in current verification and closure material as passes close.
Preparing a final report must not require automatic historical reading.
If retrospective evidence is missing, request it or explicit historical review; do not invent a completion claim.
For early closure, distinguish delivered, accepted, unfinished, and unverified parts.
For completed work, use: `Next action: none. All completion criteria are satisfied.`

# Writing and document validation

Use simple English, active voice, consistent terms, and focused paragraphs.
Use one instruction per sentence and numbered lists for procedures.
Aim for instructions within 20 words and descriptions within 25 words where practical.
Preserve exact identifiers, commands, conditions, exceptions, and uncertainty.
Keep stable policy detailed and changing records compact; avoid history, boilerplate, and transcripts in snapshots.
Keep roles and outcomes explicit. Direct answers must stand alone.
Check paths, internal links, required content, empty files, and preservation of existing work.
Check missing/changed roles, plan-only approval, action authority, failures, handovers, and pending reviews as document scenarios.
Do not execute project work to test record behavior. Never claim incomplete checks passed.
These records must explain the project and operating process without past conversation.
