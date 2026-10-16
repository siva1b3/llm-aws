# Project workflow guide — modular edition 1

This seven-file package is one complete workflow guide.
It applies to learning, research, writing, design, analysis, software, and other projects.
It defines roles, outcomes, process, approvals, records, preparation, recovery, and closure.
Earlier guide versions are not required reading.

This package is a specification for preparing and operating project records.
Reading or maintaining this guide does not authorize execution of a source project's work.
The guide package and the project folder generated from it are different artifacts.

## Authoritative parts

| File | Owns |
| --- | --- |
| [System laws](01-system-laws.md) | Terms, roles, outcomes, authority, truthfulness, and communication. |
| [Work process](02-work-process.md) | Action size, checks, planning, approvals, execution, and revisions. |
| [Record structure](03-record-structure.md) | Generated paths, record ownership, fields, and concise formats. |
| [Checkpoints and history](04-checkpoints-and-history.md) | Update timing, current snapshots, historical preservation, and history access. |
| [Failures and closure](05-failures-and-closure.md) | Recovery, interruption, resumption, pass closure, and project completion. |
| [Preparation and validation](06-preparation-and-validation.md) | Preparation boundaries, independence, initial state, writing, and validation. |

Each rule has one authoritative owner. Other parts may reference it or test its application.
Examples illustrate rules; they do not add project requirements.
Validation scenarios check rules; they do not establish alternative policy.
If parts conflict, identify the conflict and resolve it before dependent work.

## Reading routes

For generation, read this entry point and all six parts completely, plus the generator and designated source package.
Do not use this entry point alone as the guide.

For maintenance, read the affected part and its dependencies.
Check changes against the complete package before delivering a revised guide.
Preserve older versions unless the human requests changes to them.

Generated projects must provide these request-specific routes:

| Request | Read |
| --- | --- |
| Current state | Only `status/current-state.md` for the answer. |
| Next action items | Only `status/next-actions.md` for the answer. |
| Current pass plan | The active pass's `plan.md`, identified by the current routing information. |
| Begin or resume execution | Generated operating instructions, goal, constraints, current state, next actions, active plan, and relevant current evidence. |
| Historical explanation or comparison | Only the history explicitly requested, plus any current context needed for that request. |

These routes concern task information; they do not bypass governing instructions already applicable to the agent.
Do not load definitions, pass details, or history just to answer a status or next-action question.
If a direct-answer record is missing or stale, say so. Do not silently reconstruct it from history.
Follow the interruption procedure when actual state cannot be established.

## Preparing generated operating instructions

The generated project's `AGENTS.md` is its operating entry point, overview, and navigation file.
Populate it with the full applicable operating rules from this package, grouped by their authoritative topic.
Include the reading routes, role gate, approval sequence, update events, recovery, closure, and writing requirements.
Include a concise folder map and explanations of initially empty records.
Link project-specific content to the generated goal and constraints.
Do not replace required behavior with references to this preparation guide.
Do not copy these seven preparation files wholesale into a second live rules system.

The generated instruction file may be detailed. Frequently updated records must follow the compact formats in the structure part.
Keep project content in its assigned owner rather than placing arbitrary source text in operating instructions.

## Migration coverage from version 3

This table is a guide-maintenance map, not project history.

| Previous guide section | New authoritative location |
| --- | --- |
| 1. Terms and responsibilities | System laws |
| 2. Small actions and separate checks | Work process |
| 3. Conversation, approval, and execution | Work process |
| 4. When to update Markdown records | Checkpoints and history |
| 5. Required project structure and ownership | Record structure |
| 6. Prepare self-contained records | Preparation and validation |
| 7. Required contents of generated AGENTS.md | This entry point's generation contract and the applicable rules in all six parts |
| 8. Record fields at checkpoints | Record structure |
| 9. Failures, changes, and interrupted work | Failures and closure |
| 10. Pass closure, handover, and completion | Failures and closure |
| 11. Writing and consistency checks | Preparation and validation |

Deliberate changes: explicit roles and outcomes; role/outcome-led replies; direct-answer reading routes; compact live records; separate history.
Current snapshots now update when the actual position changes, while detailed records retain checkpoint timing.
Generated navigation uses `AGENTS.md`; no additional project or passes README is required.
The structure adds a direct next-actions record and dedicated historical owners.
