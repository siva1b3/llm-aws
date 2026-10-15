# Domain engineering guides

These guides add technical detail to the [shared foundation](../core/README.md).
Select them from actual project requirements; reading one does not require every related guide.
They do not grant execution authority or define another project record structure.

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
- [Project records generator](../software-ai-project-records-generator.md): preparation from a supplied goal.
- [Library overview](../README.md): reading route and navigation limitations.
