# Failures, resumption, and closure

The [system laws](01-system-laws.md) govern roles and authority for every procedure here.
Use the [checkpoint rules](04-checkpoints-and-history.md) for record timing and historical access.

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
