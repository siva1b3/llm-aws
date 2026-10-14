# Architecture and design guide

## 1. Purpose and boundaries

Use this guide for high-level design (HLD) and low-level design (LLD) in software and AI projects.
Follow the [core workflow](software-ai-development-workflow.md) for approvals, records, and pass lifecycle.
Use the [quality guide](engineering-quality-guide.md) to translate design claims into checks.

This guide owns design content and review expectations. It does not authorize work or define another project record tree.
Apply relevant sections throughout development, deployment, and maintenance.
Do not produce large design documents solely to satisfy a template.

## 2. HLD and LLD

| Design level | Main question | Typical contents |
| --- | --- | --- |
| HLD | How does the system satisfy its goals as a whole? | Boundaries, major components, data flow, trust boundaries, deployment topology, quality requirements, and tradeoffs. |
| LLD | How will the affected components behave and interact? | Interfaces, data structures, schemas, algorithms, state transitions, errors, concurrency, and testable contracts. |

HLD and LLD are connected views, not mandatory separate files for every change.
A small change can use a design section in its pass plan.
Substantial designs should use agreed repository documentation locations, linked from the plan.
Preserve decisions that affect later work in the core workflow's decision log.

Distinguish existing design, proposed design, and implemented design.
Identify the version or state that each diagram describes.

## 3. Determine design depth

| Change | Proportionate design expectation |
| --- | --- |
| Local correction with unchanged interfaces | Explain the cause, behavior change, affected dependencies, and regression check. |
| New component or feature | Describe affected HLD boundaries and the component's LLD. |
| Public API or schema change | Define compatibility, consumers, migration, and failure behavior. |
| Cross-service change | Define communication, consistency, retries, ownership, rollout order, and partial failure. |
| ML or LLM change | Define data and model flow, evaluation, resource needs, and behavioral limits. |
| GPU or deployment topology change | Define hardware and software constraints, capacity, scheduling, failure handling, and recovery. |

Record why existing design remains sufficient when no larger design update is needed.
Revisit design when implementation reveals a material missing assumption or incompatible contract.

## 4. Inputs before design

Read the goal, constraints, existing design, affected code, and relevant evidence within approved scope.
Identify users, operators, consumers, and external dependencies.
Separate requirements, preferences, assumptions, and open decisions.

Capture applicable quality requirements with a verification method:

- Latency and throughput under stated workload conditions.
- Availability, recovery needs, and acceptable data loss.
- Security, privacy, accessibility, and compatibility.
- Data volume, retention, consistency, and integrity.
- Compute, GPU memory, storage, network, and cost limits.
- Model quality and acceptable failure behavior for AI systems.

Do not invent numerical targets. Record missing targets for human review.
Use discovery when evidence is insufficient to choose a defensible design.

## 5. HLD contents

An HLD should answer the applicable questions below.

| Area | Questions |
| --- | --- |
| Context | Who uses the system? What lies inside and outside its responsibility? |
| Components | What are the major responsibilities and their owners? |
| Interfaces | Which boundaries use APIs, events, files, queues, or shared data? |
| Data flow | Where does data originate, move, change, persist, and expire? |
| Trust | Where are authentication, authorization, isolation, and sensitive-data boundaries? |
| Deployment | Where do components run? Which environments, networks, and failure domains apply? |
| Dependencies | Which outside services, packages, models, datasets, or platforms are required? |
| Reliability | What happens when a dependency is slow, unavailable, or partially successful? |
| Scale | What limits throughput, storage, memory, concurrency, or cost? |
| Operations | How will the system be observed, released, maintained, and recovered? |
| Evolution | How can interfaces, schemas, and components change compatibly? |
| Alternatives | What options were considered, and why does the proposed choice fit? |

Use diagrams when they clarify boundaries or flow.
Label components, direction, protocols, trust boundaries, and persistence where relevant.
Explain important behavior in text. A diagram alone is not a complete design.

Prefer the least complex architecture that satisfies the actual requirements.
Do not assume microservices are superior to a modular monolith.
Account for operational cost and team capability when introducing distribution.

## 6. LLD contents

An LLD should define the behavior needed for implementation and verification.

### Components and interfaces

- Define each module's responsibility, inputs, outputs, dependencies, and side effects.
- Specify API methods, request and response schemas, validation, authorization, and error semantics.
- Define pagination, ordering, limits, compatibility, and versioning when applicable.
- Specify event schemas, producer and consumer ownership, delivery assumptions, and schema evolution.
- Define timeouts, cancellation, retries, and idempotency where operations can repeat.
- Identify extension points only where a demonstrated need justifies them.

### State, data, and algorithms

- Define entities, relationships, constraints, indexes, and transaction boundaries.
- State invariants that must remain true across updates and failures.
- Describe state transitions, invalid transitions, and recovery from incomplete transitions.
- Identify concurrency hazards, locking or coordination, and ownership of mutable state.
- Explain algorithm choices when complexity, accuracy, or resource usage affects requirements.
- Define migration, backfill, deletion, retention, and compatibility behavior where applicable.

### Errors and resource lifecycle

- Distinguish invalid input, denied access, missing resources, conflicts, and dependency failures.
- Define user-visible behavior and diagnostic information for each material failure.
- Specify resource acquisition, limits, cleanup, and cancellation.
- Define backpressure and overload behavior instead of assuming unlimited queues or memory.
- Identify which operations can safely retry and which need reconciliation.

### Testability

Map important contracts and failure paths to proposed checks.
Identify controllable boundaries for dependencies, time, randomness, and external state.
Avoid test-only abstractions that complicate production behavior without a clear benefit.

## 7. Domain-specific design attention

Use only the rows applicable to the project.

| Domain | Design attention |
| --- | --- |
| Frontend | User journeys, loading and empty states, accessibility, state ownership, validation, browser behavior, and client/server boundaries. |
| Backend | Domain rules, API contracts, authorization, asynchronous work, transactions, resource limits, and dependency behavior. |
| Database | Integrity, query patterns, indexing, concurrency, ownership, migration, retention, and restoration. |
| Distributed systems | Consistency expectations, delivery semantics, duplicate messages, ordering, retries, timeouts, partitions, and reconciliation. |
| Containers and orchestration | Runtime identity, configuration, persistence, startup, readiness, shutdown, resource requests, and scheduling constraints. |
| Cloud and on-premises | Network paths, identity, storage, capacity, failure domains, provisioning, maintenance, and operational ownership. |

Do not claim end-to-end exactly-once behavior without defining its scope and supporting mechanism.
Make retry limits and idempotency explicit for operations with external effects.
Account for version skew during rolling releases and asynchronous processing.

## 8. ML, DL, and LLM design

Machine learning (ML) and deep learning (DL) introduce data-dependent and often variable behavior.
Large language model (LLM) applications also require explicit boundaries around generated outputs and tool access.

At HLD level, identify:

- Whether the system trains, fine-tunes, retrieves information, performs inference, or combines these activities.
- Data sources, permission to use data, lineage, preprocessing, and storage boundaries.
- Training, validation, and test separation where applicable.
- Model selection constraints, serving topology, evaluation ownership, and release criteria.
- External model services and what data crosses their boundaries.
- Human review, fallback, and escalation when incorrect output has significant consequences.

At LLD level, define applicable details:

- Input schemas, preprocessing, tokenization, feature transformations, and output interpretation.
- Model, tokenizer, prompt, retrieval index, dataset, and configuration version references.
- Batching, streaming, cancellation, timeouts, caching, and concurrency behavior.
- Evaluation datasets, baseline comparisons, metrics, slices, and acceptable variation.
- Handling of unsupported answers, malformed outputs, missing context, and unavailable models.
- Validation and authorization before generated content causes a system action.
- Prompt-injection and untrusted-content boundaries for retrieval and tool use.
- Checkpointing, experiment tracking, artifact storage, and resumability for training.

Do not treat plausible output as verified correctness.
Do not rely on a prompt as the only authorization or data-isolation boundary.
Keep evaluation data protected from training leakage where the evaluation requires independence.

## 9. GPU design considerations

Identify the hardware and software compatibility assumptions that the workload depends on.
Verify version-sensitive compatibility against applicable vendor documentation during approved technical work.
Do not embed a universal driver, runtime, framework, or GPU version matrix in this guide.

Design for applicable concerns:

- Device capability and available accelerator memory.
- Memory used by parameters, activations, optimizer state, batches, and inference caches.
- Numerical precision and its effect on quality, stability, and throughput.
- Host memory, storage throughput, data loading, and host-to-device transfer.
- Device placement, scheduling, isolation, and competing workloads.
- Multi-GPU communication, synchronization, scaling limits, and network topology.
- Out-of-memory handling, checkpoint integrity, worker failure, and restart behavior.
- Representative warm-up, batch sizes, sequence lengths, and concurrency for benchmarks.
- Utilization, memory, latency, throughput, failure rates, and compute cost observability.

Separate measured capacity from estimates. State workload assumptions behind each estimate.
Changing precision, batching, quantization, or parallelism can require quality reevaluation as well as performance checks.

## 10. Design decisions and review

For significant choices, record the problem, options, chosen approach, evidence, tradeoffs, and consequences.
Include authority, date, affected components, and conditions that could justify reconsideration.
Link the decision from the design and relevant pass.
Preserve superseded decisions and explain replacements.

Before design-dependent execution, assess:

1. Does the design satisfy the selected goal criteria?
2. Are component responsibilities, interfaces, and data ownership clear?
3. Are important failure, security, concurrency, and resource cases addressed?
4. Can the proposed behavior be tested and operated?
5. Are compatibility, migration, rollout, and recovery needs addressed?
6. Are assumptions and unresolved decisions visible?
7. Is complexity justified by the requirements?
8. Have required reviewers completed their review?

Mark each applicable item satisfied, unresolved, or not applicable with a reason.
Do not label a proposed design as reviewed when review remains pending.
The core workflow determines whether unresolved items block execution or permit a bounded discovery branch.

## 11. Keep design current

Check design impact when requirements, code, data, interfaces, models, or infrastructure change.
Update affected design descriptions before later work relies on them.
Obtain revised execution approval when design changes alter approved actions or checks.
Do not rewrite design history to hide a failed assumption.

At handover, link the implemented architecture, important decisions, interface contracts, and operational limitations.
Record known differences between the approved design and the delivered system.
Resolve or explicitly accept those differences through the core workflow.

## 12. Compact design outline

Use this outline when useful. Omit irrelevant sections with a reason where the omission affects review.
Do not populate reserved pass records with this outline before planning approval.

```markdown
# Design: <change or system>

- State: proposed / reviewed / implemented / superseded
- Related goal criteria and pass:
- Scope and exclusions:
- Existing behavior and affected components:
- Requirements and quality targets:
- HLD: boundaries, components, flows, deployment, and trust
- LLD: interfaces, schemas, state, algorithms, and errors
- Security, failure, concurrency, and resource behavior:
- AI and GPU considerations, if applicable:
- Compatibility, migration, rollout, and recovery:
- Alternatives and decision references:
- Verification approach:
- Assumptions, unresolved questions, and required reviewers:
- Implemented version and remaining differences:
```
