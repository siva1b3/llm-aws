# AGENTS.md — Entry Point for ChatGPT and Codex

**Project:** Learn the internal anatomy of a real LLM by inspecting **one forward pass**, one black box at a time.  
**Function of this file:** Route reading, edits, and handoffs to the six specialized Markdown files. **Do not duplicate their full content here.**

## 1. First-time entry / new chat or Codex session

1. **Read `AGENTS.md`** for routing and document ownership.
2. **Read [`06_PROJECT_PROGRESS.md`](06_PROJECT_PROGRESS.md)** to discover the actual current milestone, last verified run, last learner-confirmed concept, and single next action.
3. **Read [`01_PROJECT_OVERVIEW.md`](01_PROJECT_OVERVIEW.md)** to establish goals and the hard boundary: **one complete forward pass**, not a multi-pass generation loop.
4. **Read [`02_LEARNING_METHODOLOGY.md`](02_LEARNING_METHODOLOGY.md)** before teaching, and **[`03_IMPLEMENTATION_RULES.md`](03_IMPLEMENTATION_RULES.md)** before coding or inspecting a computation.
5. Read **[`04_TECHNICAL_CONFIGURATION.md`](04_TECHNICAL_CONFIGURATION.md)** when interacting with EC2, dependencies, the model, or its assets, and **[`05_INFERENCE_INVESTIGATION.md`](05_INFERENCE_INVESTIGATION.md)** before reopening a previously investigated operation or recording new technical findings.
6. Inspect actual available code and terminal state before assuming recorded intentions have executed. Resume at the last **learner-confirmed** boundary; do not restart completed explanations or jump ahead.

For short follow-ups, use the progress file to identify the owning document(s) and read only what is needed. Do not reread all seven files on every minor message.

## 2. File map — which file owns which information?

| File | Single source of truth for | Read when | Update when |
| --- | --- | --- | --- |
| [`01_PROJECT_OVERVIEW.md`](01_PROJECT_OVERVIEW.md) | Goal, current one-pass scope, exclusions, final success | Starting project; deciding whether a request is in scope | User changes objective, phase, boundaries, or success criterion |
| [`02_LEARNING_METHODOLOGY.md`](02_LEARNING_METHODOLOGY.md) | Teacher role, build-first learning, **onion-peeling**, learner approval | Teaching/explaining an operation; checking readiness | User changes teaching style, pace, or human-in-loop process |
| [`03_IMPLEMENTATION_RULES.md`](03_IMPLEMENTATION_RULES.md) | Incremental code, inspection formats, tests, verification gates | Writing/modifying code, capturing output, deciding whether an operation is verified | Inspection, coding, display, or verification conventions change |
| [`04_TECHNICAL_CONFIGURATION.md`](04_TECHNICAL_CONFIGURATION.md) | AWS EC2, selected Hugging Face model, runtime stack and observed versions | Setup, model/assets loading, dependency or machine issues | Technical choices change, or real installed versions/paths/settings are observed |
| [`05_INFERENCE_INVESTIGATION.md`](05_INFERENCE_INVESTIGATION.md) | Verified technical anatomy, real transformations, code/asset provenance, observation evidence | Opening or reviewing a specific inference operation | An operation/asset is inspected, measured, reproduced, or a prior finding corrected |
| [`06_PROJECT_PROGRESS.md`](06_PROJECT_PROGRESS.md) | **Current** lesson, milestones, execution results, learner-confirmed boundary, next action | Start/end of every session; deciding what to do now | Meaningful progress, failures, confirmation, handoff, or blockers |

**`AGENTS.md` itself:** Navigation, ownership, read/update routing, and cross-file consistency. Update when files are added, renamed, reorganized, or routing rules change—not for an ordinary code change.

## 3. Non-negotiable working contract

- **Teaching is the purpose.** The assistant writes or edits code; the human executes/reviews as necessary, asks questions, and explicitly confirms understanding.
- **Onion-peeling is mandatory:** Start with `Text → [LLM] → decoded one-token text`. Reveal **one operation at a time**; keep remaining operations as black boxes. If a revealed operation is still opaque, decompose that operation further before moving ahead.
- **Phase 1 = one complete model forward pass.** Actual execution uses the full model, but dissect **one representative Transformer block** in detail. Do not introduce repeated autoregressive passes or multi-token generation until scope is explicitly changed.
- **Build first, learn just-in-time.** Small runnable increments, no long lectures, no large code dumps, no premature abstractions, no unnecessary dependencies or unrelated infrastructure.
- **Observe real values.** At the current operation, show input, type/shape, actual computation, relevant asset/parameters, output, and where it goes next. Choose a human-readable presentation that suits that data type.
- **Do not equate code with understanding.** Implementation, actual execution, technical verification, and explicit learner confirmation are different states. **Do not peel the next layer without confirmation.**
- **Truthful evidence:** Never invent runs, tensor values, success reports, installed versions, downloaded assets, or user confirmations. Label untested examples appropriately.

The specialized files contain the full normative details; this section is only a quick guardrail summary.

## 4. Minimal decision workflow

Before making a change, ask internally:

1. **What is the current black box?** Read `06_PROJECT_PROGRESS.md`.
2. **Does this action stay within Phase 1?** Check `01_PROJECT_OVERVIEW.md` if uncertain.
3. **What is the smallest next experiment?** Follow `02_LEARNING_METHODOLOGY.md` and `03_IMPLEMENTATION_RULES.md`.
4. **Does this require a model/environment/asset decision?** Consult `04_TECHNICAL_CONFIGURATION.md`.
5. **What did the actual run prove?** Store substantive technical findings in `05_INFERENCE_INVESTIGATION.md`.
6. **What is the exact checkpoint now?** Update `06_PROJECT_PROGRESS.md`, including confirmation state and the next **single** step.

A change that affects two areas should update the relevant owners without copying large passages between documents.

## 5. Conflict-resolution and change-control rules

- Higher-priority system/developer instructions and the user's **latest explicit request** supersede these project documents.
- Within this repository, each document is authoritative **for its owned domain** (table above). `AGENTS.md` routes; it is not a competing technical specification.
- A newer note in `06_PROJECT_PROGRESS.md` does **not** silently redefine project scope or selected model. First revise the corresponding owning file (typically `01` or `04`) when the learner authorizes a substantive change, and then update progress.
- When files disagree about a concrete fact, prefer direct, reproducible runtime evidence for what actually executed; reconcile the contradictory document(s) explicitly.
- If a contradiction changes the learning objective or teaching agreement, present it to the learner rather than silently choosing an incompatible interpretation.
- Record only **observed** model files, machine state, and test results as facts. Chosen settings and expectations must retain their labels until verified.

## 6. Writing and maintenance rules

- Keep filenames stable and use relative Markdown links.
- Own each fact in one place: scope in `01`; pedagogy in `02`; code/verification standards in `03`; configuration in `04`; technical findings in `05`; current status in `06`.
- Cross-reference rather than duplicating entire sections.
- Keep the progress file concise and current; the investigation file accumulates verified findings over time.
- After a significant step, record actual files changed, evidence/commands, verification status, the learner's confirmation (if any), and exactly one next action.
- If actual code workspace access exists, inspect files and implement changes directly. Otherwise give the learner exact minimal commands and wait for observed results.
- A new agent must **not** claim it can access the EC2 machine, infer that it is already provisioned, or assume previous code succeeded without evidence.

## 7. Initial state of this handoff package

The seven documentation files describe a **planned learning project**. A model/environment has been **selected**, but no model download, working inference script, or runtime observation has been verified as part of this package. See [`06_PROJECT_PROGRESS.md`](06_PROJECT_PROGRESS.md) for the exact current checkpoint and first action.

**Operating rhythm:** Build → execute → observe → explain → inspect → verify → **human confirms** → peel next layer. 🧅
