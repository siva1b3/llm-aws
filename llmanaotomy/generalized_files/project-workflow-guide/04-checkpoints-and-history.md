# Checkpoints and history

Use the owners and compact formats in [Record structure](03-record-structure.md).
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
For explicitly designated source inputs, preparation reads the supplied package as requested.
That does not authorize browsing unrelated project archives.

Keep one current plan per pass and one current-state snapshot per project.
Preserve prior approved revisions recoverably inside the pass's history record.
Do not accumulate superseded plans or approval transcripts inside the current plan.
