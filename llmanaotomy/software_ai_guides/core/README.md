# Core engineering guides

These three guides provide the shared foundation for software and AI projects using this library.
Read them together, then select technical guidance from the [domain guide index](../domains/README.md).

## Guides and ownership

| Guide | Responsibility | Main contents |
| --- | --- | --- |
| [Software and AI development workflow](software-ai-development-workflow.md) | Process and project records. | Goal criteria, one active pass, planning and execution approvals, evidence, current state, closure, and handover. |
| [Architecture and design](architecture-and-design-guide.md) | Design content and review expectations. | High-level design (HLD), low-level design (LLD), interfaces, data ownership, failure behavior, tradeoffs, and design evolution. |
| [Engineering quality](engineering-quality-guide.md) | Engineering checks and review expectations. | Maintainability, correctness, security, testing, dependencies, model evaluation, GPU checks, and release readiness. |

## How to apply the foundation

1. Start with the workflow to understand the governing records and authorized stage of work.
2. Use architecture guidance to define the design detail justified by the change.
3. Use quality guidance to choose checks that support the intended claims.
4. Add relevant domain guidance without creating a competing workflow or record structure.

Design and quality apply throughout **Define → Understand → Plan → Work → Verify → Decide**.
Keep effort proportionate: a local correction may need a short design note and focused checks.
Cross-service changes, model work, or GPU workloads may need deeper design and verification.
Examples and checklists do not automatically become project requirements.

## Preparing project records

Use section 4 of the [core workflow](software-ai-development-workflow.md) directly when record preparation is requested.
Derive the starting structure and file responsibilities from the workflow's current contents.
Keep generated records in the intended project destination, separate from this reusable guide library.

This folder's README is an index, not a replacement for the guides or an additional approval policy.
See the [library overview](../README.md) for reading order and portability and the [agent instructions](../AGENTS.md) for maintenance guidance.
