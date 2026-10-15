# Domain engineering guides

These guides add technical detail to the [shared foundation](../core/README.md).
Select them from actual project requirements; reading one does not require every related guide.
They do not grant execution authority or define another project record structure.
All eight domain guides apply the version 3 [core workflow](../core/software-ai-development-workflow.md).

## Apply the version 3 process

Discuss and revise plans in conversation; keep unapproved drafts out of Markdown.
Obtain plan approval, explain the next action and separate check, then obtain final execution approval.
Save the agreed plan checkpoint before execution and obtain execution approval before each next action.
Keep routine results and subsequent approvals in conversation until the core's defined checkpoints.
Save failures, blockers, partial effects, and substantive handovers promptly under those rules.
Preserve valid approvals and stricter project-specific learning or review gates.

Domain instructions to record observations follow core checkpoint timing for management records.
Edit deliverables within their approved action scope.
Runtime logs, business state, training checkpoints, and backfill progress follow their technical requirements independently of Markdown checkpoint timing.
The core remains the authority for the complete operating rules and precise checkpoint conditions.

## Prepare self-contained records

Read every designated project source and each selected guide before transferring applicable requirements.
Preserve selected settings, working methods, review obligations, reported prior state, and uncertainty in their generated owners.
Keep requirements distinct from examples and reported outcomes distinct from newly verified results.
Do not perform project investigation, design, implementation, experiments, or deployment during preparation.

Derive paths and content permissions from the actual core workflow.
Create and verify empty records before populating supported contents.
Keep reserved pass records at zero bytes and the final report empty until the core permits their contents.
Record preparation checks in current state and stop for human review.

Transfer necessary guidance into generated definitions and operating instructions using internal relative links.
Generated projects must work without this library, the original source package, the generator, or previous conversation.
A list of selected guides cannot replace their necessary contents.
Keep preparation provenance in one historical note and verify independence before claiming self-contained output.

## Choose the applicable guides

| Guide | Use when | Main concerns |
| --- | --- | --- |
| [Frontend development](frontend-development-guide.md) | Building websites or browser interfaces. | User journeys, state, accessibility, rendering, browser security, performance, and compatibility. |
| [Backend and API development](backend-api-development-guide.md) | Building services, APIs, business logic, or background jobs. | Contracts, validation, authorization, transactions, asynchronous work, resilience, and observability. |
| [Database engineering](database-engineering-guide.md) | Managing persistent application data or storage changes. | Data models, integrity, queries, concurrency, migrations, retention, backups, and restoration. |
| [Distributed systems and microservices](distributed-systems-microservices-guide.md) | Crossing process, service, machine, or independent failure boundaries. | Communication, consistency, retries, idempotency, coordination, partial failure, and compatibility. |
| [ML and deep learning](ml-deep-learning-guide.md) | Training, fine-tuning, evaluating, or managing models. | Data lineage and splits, baselines, experiment limits, reproducibility, checkpoints, evaluation, and monitoring. |
| [LLM application and model engineering](llm-engineering-guide.md) | Building language-model applications, retrieval, prompts, tools, or model adaptation. | Context, grounding, output validation, action authorization, evaluation, fallback, latency, and cost. |
| [Infrastructure, deployment, and operations](infrastructure-deployment-operations-guide.md) | Building, hosting, releasing, or operating systems. | Environments, configuration, artifacts, containers, orchestration, observability, rollout, recovery, and ownership. |
| [GPU infrastructure and performance](gpu-infrastructure-performance-guide.md) | Executing training, inference, or other computations on accelerators. | Compatibility, memory, device allocation, representative benchmarks, optimization, multi-GPU work, and isolation. |

## Combining guides

A browser application with persistent data may need frontend, backend, database, and infrastructure guidance.
A hosted LLM application may need LLM and backend guidance without training or GPU guidance.
Self-hosted model training may need ML, GPU, and infrastructure guidance, plus distributed-systems guidance when workers coordinate.
These are selection examples, not prescribed architectures.

Keep each concern with its owner.
For example, database guidance owns data restoration expectations; infrastructure guidance covers operational recovery execution.
Model guides assess model quality; GPU throughput alone cannot establish that quality.

Record applicability and its reasons in the project's existing records.
Use `applicable`, `not applicable`, or `pending clarification` where the distinction matters.
Revisit selection when project scope changes.
Do not invent technologies, numerical targets, budgets, or acceptance thresholds from guide examples.

## Shared references

- [Core workflow](../core/software-ai-development-workflow.md): approvals, records, lifecycle, and completion.
- [Architecture and design](../core/architecture-and-design-guide.md): HLD, LLD, and design review.
- [Engineering quality](../core/engineering-quality-guide.md): shared checks and evidence expectations.
- [Library overview](../README.md): reading route and portability.
