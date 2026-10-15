# Core engineering guides

These guides provide the shared foundation for software and AI projects.
The core workflow specializes the generalized version 3 workflow and includes its complete operating rules.
Read the foundation, then select technical guidance from the [domain index](../domains/README.md).

## Guides and ownership

| Guide | Responsibility | Main contents |
| --- | --- | --- |
| [Software and AI development workflow](software-ai-development-workflow.md) | Records and operating process. | Full-source preparation, independence, small actions, approvals, checkpoints, pass lifecycle, and completion. |
| [Architecture and design](architecture-and-design-guide.md) | Design content and review. | HLD, LLD, interfaces, data ownership, failure behavior, tradeoffs, and design evolution. |
| [Engineering quality](engineering-quality-guide.md) | Engineering checks and review. | Maintainability, correctness, security, tests, model evaluation, GPU checks, and release readiness. |

## Applying the foundation

1. Read the workflow to establish record ownership, required authority, and the current position.
2. Use architecture guidance for design detail justified by the actual change.
3. Use quality guidance to select separate checks that support the intended claims.
4. Add relevant domain content without creating another workflow or record structure.

The operating sequence is **Discuss the plan → approve the plan → explain the next action → approve execution → act → check → review**.
Keep drafts in conversation. Save agreed plans after final execution approval and before execution.
Obtain execution approval for each next action and its separate check.
Save accumulated results and approvals at the workflow's defined checkpoints.

Define, Understand, Plan, Work, Verify, and Decide remain engineering responsibilities within this operating process.
A local correction may need a short design explanation and focused checks.
Cross-service changes, model work, or GPU workloads may need deeper design and verification.
Examples do not mandate technologies, numerical targets, or extra documents.

## Preparing records

Use Sections 5–8 of the [core workflow](software-ai-development-workflow.md) for structure, preparation, operating instructions, and fields.
Read every supplied source and derive the starting paths and content conditions from the actual workflow.
Create and verify empty records before filling supported content.
Keep generated records in the intended project destination, outside this reusable library.
Keep reserved pass records and the final report empty until their content conditions are met.

Transfer all necessary operating rules and applicable engineering guidance into generated owners.
Generated projects must work without this library, the generator, the original source package, or previous conversation.
Record preparation checks in current state, verify independence, and stop for human review.

This index summarizes ownership; the workflow defines policy.
See the [library overview](../README.md) for navigation and the [agent instructions](../AGENTS.md) for maintenance guidance.
