# File Structure

All paths are relative to the project root.

## Folders and Files

```text
project/
├── AGENTS.md
├── protected/
│   ├── laws.md
│   ├── goal.md
│   ├── workflow.md
│   └── file_structure.md
├── current_state.md
└── action_plan.json
```

## Required Files and Permissions

| File | Purpose | AI editing permission |
| --- | --- | --- |
| `AGENTS.md` | Entry point and reading order. | Only with explicit human authorization. |
| `protected/laws.md` | Self-contained governing laws. | Never edit an existing file. |
| `protected/goal.md` | Fixed goal and parties involved. | Never edit an existing file. |
| `protected/workflow.md` | Order of work and approval process. | Never edit an existing file. |
| `protected/file_structure.md` | File locations, permissions, and placeholders. | Never edit an existing file. |
| `current_state.md` | Short factual paragraph about the present system state. | Read and update without separate approval. |
| `action_plan.json` | Next actions and their individual party definitions. | Read and update without separate approval. |

The protected folder is `protected/`, including its subfolders. Existing Markdown files there must not be edited by the AI. The human maintains their contents.

## Missing Files and Folders

- All seven files above are required.
- Create missing parent folders without approval.
- Create missing Markdown files as empty placeholders.
- Create a missing `action_plan.json` with `{"actions":[]}`.
- Never overwrite an existing file when creating placeholders.
- An empty placeholder does not replace missing instructions. Ask the human for missing governing content before executing project actions.

Code and other non-Markdown files may be created or modified as part of approved actions.
