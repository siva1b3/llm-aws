# Project Instructions

## Startup

Read `protected/laws.md` and `protected/file_structure.md`. Keep their current instructions available throughout the session. Reload them if they change or leave the available context. These project instructions operate within the agent's platform instructions and actual permissions.

## Read According to the Request

| Request | Additional files to read |
| --- | --- |
| Explain the current state | `current_state.md` |
| Explain the next actions | `action_plan.json` |
| Explain the goal or participants | `protected/goal.md` |
| Explain the process | `protected/workflow.md` |
| Plan, execute, repair, or resume | Goal, workflow, current state, action plan, relevant pass records, and relevant implementation files |
| Prepare or verify a pass record | Both schemas in `protected/contracts/` and the relevant input/output pair |
| Learn from the worked example | `examples/four_pass_walkthrough/README.md`, then only the needed example passes |

Reuse content already available when it is still current. Do not repeatedly load unrelated files or all historical evidence.

## Working Rules

- Use the goal for the destination, the plan for intended work, and pass records for actual execution evidence.
- An action is a planned step. A pass is one attempt at that action. A retry gets a new pass ID, not a new action ID.
- List the parties for each action in JSON, with their specific roles, authority, and verifiable expected outcomes. Snapshot those definitions in the pass input.
- Treat tools and execution environments as resources. Name the human or agent responsible for using them.
- Create a real pass only when its execution is ready to start. Do not populate execution records with hypothetical results.
- Example records are explicitly synthetic. Never copy their authorization, timestamps, outputs, or evidence into actual project history.
- Record the authorization that already covers the work. Do not repeatedly request the same approval.
- Preparing or updating these workflow documents does not authorize implementation or commands on EC2.
- Never claim that local checks establish EC2 behavior. Record the actual execution environment and evidence source.

## Missing or Conflicting Information

Create needed directories without overwriting existing content. Do not substitute empty files for missing laws, goals, contracts, or workflow instructions. Stop only the affected execution and identify what is missing or contradictory. Informational answers and unrelated authorized work may continue.

## Editing Permissions

Changes to this file or anything under `protected/` require explicit human authorization covering those changes. Routine progress records may be updated within existing authorization. Do not weaken instructions to make a failing result appear successful.
