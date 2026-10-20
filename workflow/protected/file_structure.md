# File Structure

All paths are relative to the project root. The archive's top-level folder is `agent_workflow/`; it may be renamed when extracted.

## Required Core Files

| Path | Purpose | Editing permission |
| --- | --- | --- |
| `AGENTS.md` | Agent entry point and reading order | Explicit human authorization |
| `protected/laws.md` | Self-contained laws | Explicit human authorization |
| `protected/goal.md` | Goal, parties, resources, and overall criteria | Explicit human authorization |
| `protected/workflow.md` | Execution and recovery procedure | Explicit human authorization |
| `protected/file_structure.md` | Locations and permissions | Explicit human authorization |
| `protected/contracts/pass_input.schema.json` | JSON Schema for every pass input | Explicit human authorization |
| `protected/contracts/pass_output.schema.json` | JSON Schema for every pass output | Explicit human authorization |
| `current_state.md` | Short factual progress paragraph | Update during authorized planning and execution |
| `action_plan.json` | Actions, revisions, authorization references, dependencies, and status | Update within authorized scope |

Protection covers all files below `protected/`, including JSON files and subdirectories.

## Execution Records

| Path | Purpose | Creation and editing |
| --- | --- | --- |
| `passes/` | Real execution history | Initially empty |
| `passes/pass_001/input.json` | Exact input snapshot for an attempt | Create when ready; freeze on start |
| `passes/pass_001/output.json` | Running record, then final outcome | Update while running; preserve once finalized |
| `passes/pass_001/evidence/` | Supporting logs and artifacts | Create only when evidence files are needed |
| `passes/pass_001/corrections/correction_001.json` | Explicit correction to a finalized record | Create only when needed; never replace the original |

Allocate monotonically increasing pass IDs, padded to at least three digits. Never reuse an existing ID. Each retry uses a new folder. Evidence paths inside an output are relative to that pass folder. All other record paths are relative to the project root unless explicitly stated.

Application code and its data live outside `protected/`, `passes/`, and `examples/`. Choose and record their exact locations in the action plan and pass input before implementation. This package contains workflow files, not an implemented task tracker.

## Examples

`examples/four_pass_walkthrough/README.md` explains four synthetic passes. Each example pass contains `input.json`, `output.json`, and `evidence/checks.txt`.

Examples use `record_kind: "example"`. Their timestamps, authorization, commands, and outputs are illustrative, not observations. They never update the real action plan or current state. Their pass IDs are local to the example and do not reserve real pass IDs.

## Missing Files

Create needed directories without approval and never overwrite existing files. Missing governing content is a blocker for execution that depends on it; do not create empty placeholders and treat them as valid instructions. A missing plan or state must be reconstructed from verified records or supplied requirements, with unknowns explicit. Do not populate execution folders until a real pass starts.

## Sources of Truth

- Protected instructions define the governing requirements.
- `action_plan.json` defines the current intended work and preserves definition revisions.
- Each pass input records the exact execution contract for that attempt.
- Each pass output and its evidence record what was performed and verified.
- `current_state.md` is a summary, not independent proof of execution.

On disagreement, inspect relevant evidence and actual state. Correct derived summaries; preserve finalized history. No document can replace a current environment check when the real system may have changed.
