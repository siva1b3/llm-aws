# Record structure and ownership

Use a filesystem-safe project-named root. The generated starting structure is:

```text
project-name/
├── AGENTS.md
├── definition/
│   ├── goal.md
│   └── constraints.md
├── status/
│   ├── current-state.md
│   └── next-actions.md
├── decisions/
│   └── decision-log.md
├── history/
│   └── project-history.md
├── passes/
│   └── pass-001/
│       ├── plan.md
│       ├── verification.md
│       ├── findings.md
│       ├── evidence.md
│       └── history.md
└── reports/
    └── final-report.md
```

Reserve only `pass-001`, unless the human requests more reserved passes.
Use sequential identifiers; never reuse an identifier for different work.
Put pass records and supporting artifacts directly inside their pass folder. Do not nest folders inside a pass.
Keep deliverables in agreed locations, separate from management records when appropriate.
Do not create extra example passes or speculative artifacts.

## Ownership

| Record | Main responsibility |
| --- | --- |
| `AGENTS.md` | Complete operating rules, project overview, folder map, reading routes, provenance, and empty-file explanations. |
| `definition/goal.md` | Purpose, recipient, outcomes, deliverables, scope, exclusions, preferences, and numbered completion criteria. |
| `definition/constraints.md` | Current responsibilities, role boundaries, review gates, methods, configuration, resource limits, relevant background, and unresolved constraints. |
| `status/current-state.md` | The single present position and one next step, sufficient for a status answer. |
| `status/next-actions.md` | Direct answer to next-action requests, with current roles, outcomes, checks, and authority. |
| `decisions/decision-log.md` | Historical significant choices, reasons, authority, sources, and supersession. |
| `history/project-history.md` | Historical source reports, preparation checks, and project-level checkpoints. |
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
Current result: brief progress and relevant output; source-reported if not verified
Blocker / pending decision: issue, decision owner, or none
Next step: action ID, outcome, owner, prerequisite, and current authority
```

Use positions `active`, `waiting for approval`, `waiting for review`, `blocked`, `paused`, `complete`, or `cancelled`.
Include running operations, partial effects, or unknown outcomes when they affect the present position.
Identify reserved passes when relevant, especially at preparation.
Do not add a history of completed actions, approvals, decisions, or prior sessions.
Do not copy evidence dumps, source background, or the complete remaining goal checklist.
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

Decision entries identify the choice, reason, authority, source, available date, and any superseding decision.
Pass history preserves recoverable approved revisions and approval entries with revision, request, response, scope, status, and available date/reference.
Project history preserves reported prior work with its qualifications and preparation-check details.
Keep unknown authority and dates explicit.

Final-report contents and closure criteria are defined in [Failures and closure](05-failures-and-closure.md).
Initial population and zero-byte requirements are defined in [Preparation and validation](06-preparation-and-validation.md).
