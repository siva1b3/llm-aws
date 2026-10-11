# 06 — Project Progress and Cross-Session Handoff

**Document role:** Authoritative **current state**, learning boundary, outstanding questions, and exact next action  
**Update when:** Meaningful work occurs, a run succeeds/fails, a learning milestone is confirmed, or a session ends.

> **Current status is an initial handoff, not a claim of implementation.** The project specification and decisions exist; the EC2 environment, scripts, runtime observations, and learner confirmations for the implementation have not been verified in this conversation.

## 1. Snapshot

| Field | Current state |
| --- | --- |
| Project | LLM inference anatomy through real code and onion-peeling |
| Phase | Phase 1: exactly one forward pass |
| Learner goal | Understand every major transformation from text input to a single next-token prediction and decoded token |
| Model selected | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| Instance selected | AWS EC2 `t3.large`, Mumbai, Ubuntu 24.04, 30 GiB gp3, CPU-only |
| Implementation milestone | **Not started / no verified baseline** |
| Learning milestone | Initial conceptual contract agreed; no runtime black box yet learner-confirmed |
| Active black box | The entire single-forward-pass LLM interface |
| Runtime execution | **Not performed/verified here** |
| Model download / assets | **Not performed/verified here** |
| Blocking issues | None known; environment existence needs confirmation before running |

## 2. Status meanings — do not conflate

- **Planned:** Specified or decided, but not implemented.
- **Implemented:** Code or configuration exists, but execution has not necessarily succeeded.
- **Executed:** Command actually ran with recorded output.
- **Technically verified:** Observations/tests support the claimed operation.
- **Learner-confirmed:** Human explicitly states they understand the specific operation after evidence/explanation.

An operation becomes an **understood, closed black box** only at the last status. Do not infer learner confirmation from silence, an earlier general agreement, or successful execution.

## 3. Current onion / learning boundary

```text
Input text → [ CURRENT: entire LLM / one forward pass ] → next-token scores
                                                        ↓
                                              one token → decoded text
```

**Previously verified components:** None yet in this implementation.  
**Currently being investigated:** Establishing that the single-pass black box really runs with the chosen model.  
**Hidden remainder:** Every internal operation, to be exposed **one by one** after the baseline is verified and the learner confirms.

## 4. Milestone register

| Milestone | Code status | Runtime verification | Learner confirmation | Notes |
| --- | --- | --- | --- | --- |
| Seven-document specification and navigation | Created in this handoff package | All seven Markdown files present; local Markdown links checked | Not required for individual inference concept | Project instructions only |
| EC2 availability and runtime versions | Not started | Not run | Pending | Technical choice ≠ provisioned resource |
| Pretrained model + asset download | Not started | Not run | Pending | Confirm actual file inventory |
| One-call forward-pass baseline | Not started | Not run | Pending | Select and decode one token only |
| First internal box inspection | Not started | Not run | Pending | Do not open before baseline gate |

## 5. Files in this documentation package

- [`AGENTS.md`](AGENTS.md) — document routing and maintenance instructions.
- [`01_PROJECT_OVERVIEW.md`](01_PROJECT_OVERVIEW.md) — objective, one-pass scope, final success.
- [`02_LEARNING_METHODOLOGY.md`](02_LEARNING_METHODOLOGY.md) — teacher, onion-peeling, human approval.
- [`03_IMPLEMENTATION_RULES.md`](03_IMPLEMENTATION_RULES.md) — small code changes, displays, and verification gate.
- [`04_TECHNICAL_CONFIGURATION.md`](04_TECHNICAL_CONFIGURATION.md) — selected AWS/model/software and observed environment facts.
- [`05_INFERENCE_INVESTIGATION.md`](05_INFERENCE_INVESTIGATION.md) — technical evidence and opened operations.
- `06_PROJECT_PROGRESS.md` — this current checkpoint.

**Application source files:** None created as part of this specification package. If the learner supplies an existing codebase, inspect it first; do not assume an empty repository.

## 6. Exact next small action

**First:** Establish the actual working environment context (whether the chosen EC2 instance is provisioned/reachable, and if files already exist). If direct workspace or shell access is available, inspect rather than ask the learner to repeat information already available. If it is unavailable, request the single minimum missing fact/action needed to proceed.

**Then:** Create or run a minimal Python script that executes **exactly one** model forward pass for the fixed input, reads next-token logits, selects a single argmax ID, decodes that ID, and prints bounded, human-readable results. The assistant writes the code; the learner executes if necessary. Do not use a multi-token generation loop.

**Stop after:** The actual baseline output is examined and explained. Request the learner's confirmation before opening the tokenizer or the next internal box.

## 7. Run and evidence record — empty until tested

| Field | Result |
| --- | --- |
| Last command executed | None recorded |
| Last actual terminal output | None recorded |
| Last run success/failure | Not run |
| Evidence of next-token prediction | None recorded |
| Last verified model revision | Not recorded |
| Last test status | Not run |
| Current unresolved technical question | Whether the real execution environment is already set up |
| Current unresolved learning question | None yet; baseline lesson has not begun |

Do not fill gaps with fabricated commands or output. Add evidence and update only the affected fields after an actual run.

## 8. End-of-session handoff checklist

At each meaningful stopping point, update the following succinctly:

1. **Last completed technical step:** What code changed; what ran; what was verified, with evidence.
2. **Last learner-confirmed concept:** Quote or paraphrase the exact confirmation; do not infer it.
3. **Current onion position:** A three-part diagram (understood → current → hidden remainder).
4. **Files changed:** Exact paths; include new inspection scripts and assets as applicable.
5. **Reproduction:** Smallest command(s) to rerun the last observed result.
6. **Outstanding issues:** A direct, finite set; indicate blockers versus optional curiosities.
7. **Next single step:** Only one concrete action; do not bundle future operations.

Technical findings belong in `05_INFERENCE_INVESTIGATION.md`; the *latest state* belongs here. If the learner changes the objective, update `01_PROJECT_OVERVIEW.md` instead of burying a new scope inside this progress note.

## 9. For a new ChatGPT or Codex session

Read [`AGENTS.md`](AGENTS.md) first, this file second, and then the specialized document(s) identified by the router. Honor the **last verified, learner-confirmed boundary**. Do not restart completed work, claim unexecuted code has passed, select a different model, or open multiple black boxes at once.
