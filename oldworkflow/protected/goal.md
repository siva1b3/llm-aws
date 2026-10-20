# Goal

## Destination

Build a small Python command-line task tracker on the EC2 server. A user must be able to add a task, list tasks, and mark a task complete. Tasks must remain available after the program exits and restarts.

## Required Behavior

- Run with Python 3 and its standard library; no third-party packages.
- Provide `add "task title"`, `list`, and `complete TASK_ID` commands.
- Store each task's unique integer ID, title, and completion status in a local JSON file.
- Assign IDs that remain unique across program restarts.
- Show task IDs, titles, and completion status when listing tasks.
- Reject blank titles, invalid IDs, and unknown task IDs with clear messages.
- Treat a missing data file as an empty task collection.
- Report malformed data and filesystem errors without silently resetting or overwriting existing data.
- Keep the implementation small and readable; introduce no unnecessary frameworks or design patterns.

## Completion Criteria

On EC2, verify that two tasks can be added, listed, retained across separate invocations, and individually marked complete. Verify the error cases above and confirm that a failed operation leaves existing task data unchanged. Evidence must be recorded in the relevant step histories before the goal is marked achieved.

## Parties Involved

- AI assistant: ChatGPT.
- Human: The user.
- Execution environment: EC2 server.

## Scope

Deliver through five planned steps recorded in `action_plan.json`. The workflow governs execution and retries; step history JSON files retain the attempts. Retry attempts do not count as additional planned steps.

No web interface, API, database server, authentication, deployment pipeline, or multi-user concurrency is required.

The goal remains fixed during execution unless the human explicitly changes it. This document defines the proposed new goal; execution approval is recorded separately.

