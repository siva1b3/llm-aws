# Goal

## Destination

Build a small Python command-line task tracker on the EC2 server. A user must be able to add a task, list tasks, and mark a task complete. Tasks must remain available after the program exits and restarts.

## Required Behavior

- Use Python 3 and its standard library only; no third-party application packages.
- Provide `add "task title"`, `list`, and `complete TASK_ID` commands.
- Store a unique positive integer ID, title, and completion status for each task in a local JSON file.
- Keep IDs unique across separate invocations and restarts.
- List task IDs, titles, and completion status.
- Reject blank titles, invalid IDs, and unknown task IDs with clear messages.
- Treat a missing data file as an empty task collection.
- Report malformed data and filesystem errors without silently resetting or overwriting existing data.
- Keep the implementation small and readable, without unnecessary frameworks or design patterns.

## Overall Acceptance Criteria

| ID | Required result | Verification |
| --- | --- | --- |
| GOAL-01 | Runs on EC2 with Python 3 and the standard library only | Record the EC2 environment and execute the final application; inspect imports |
| GOAL-02 | Adds and lists tasks with unique positive IDs, titles, and status | Add two tasks, list them, and inspect all fields |
| GOAL-03 | Tasks persist and IDs remain unique across invocations | Read existing tasks in a fresh invocation, then add another task |
| GOAL-04 | Completes the selected task and preserves unrelated tasks | Complete each of two tasks separately; check stored state after each; repeat completion of an already completed task |
| GOAL-05 | Invalid commands, blank titles, invalid IDs, and unknown IDs fail clearly | Exercise each error case and record messages and exit statuses |
| GOAL-06 | Missing data is handled as an empty collection | List with an absent test data file |
| GOAL-07 | Malformed JSON, invalid task records, and filesystem errors are reported without damaging existing data | Use isolated fixtures; compare stored bytes before and after each failed operation, including a controlled save failure |
| GOAL-08 | Usage and data location are documented for the final version | Follow the documented commands on EC2 and record the exact data path |

Record evidence in real pass outputs and their evidence files. All criteria must be verified for the delivered version; use isolated test data for destructive or failure scenarios.

## Parties Involved

| Stable ID | Party |
| --- | --- |
| ai_assistant | The AI assistant working on this project |
| human_owner | The human user who owns the project |

This section identifies the parties. Each action defines their specific responsibilities, authority, and expected outcomes. Add other agents only when needed and authorized; record their stable IDs and ownership before they participate.

## Resources

| ID | Resource | Known state |
| --- | --- | --- |
| ec2_server | Target EC2 execution environment | Connection and Python availability were previously human-reported; current access remains unverified |
| project_directory | Application and workflow directory on EC2 | Exact path must be established during step_01 |
| task_data | Local JSON task storage | Location will be documented during implementation |

## Scope

Keep five planned actions: environment verification; command-line interface; storage/add/list; completion; final verification and usage documentation. Retries create additional passes, not additional planned actions. A change to this five-action structure requires human authorization because it changes the agreed scope.

No web interface, API, database server, authentication, deployment pipeline, or multi-user concurrency is required.

The goal stays fixed unless the human explicitly changes it. Authorization to rewrite the workflow files is recorded separately from authorization to execute the application plan.
