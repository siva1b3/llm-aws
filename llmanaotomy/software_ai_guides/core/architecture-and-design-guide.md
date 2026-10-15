# Architecture and design guide

## 1. Purpose and boundaries

Use this guide for high-level design (HLD) and low-level design (LLD) in software and AI projects.
Follow the [core workflow](software-ai-development-workflow.md) for approvals, records, and pass lifecycle.
Use the [quality guide](engineering-quality-guide.md) to translate design claims into checks.

This guide owns design content and review expectations. It does not authorize work or define another project record tree.
Apply relevant sections throughout development, deployment, and maintenance.
Do not produce large design documents solely to satisfy a template.

### Apply version 3 boundaries

During preparation, transfer applicable design obligations and supplied design facts into the generated definitions.
Do not draft new HLDs, LLDs, or design passes during record preparation.
Generated projects must retain necessary design guidance without requiring this library later.

Discuss proposed plan revisions and proportionate design explanations in conversation until the core's plan checkpoint.
Producing a design artifact or investigating an uncertain design is an action with its own approval and separate check.
Identify the design output, review method, expected result, limits, and stopping point before execution.
Keep design creation and verification distinct, including their executable artifacts when applicable.
Report the result and obtain execution approval before dependent implementation or the next design action.
Required human design review remains pending until the reviewer actually completes it.

Save management records at the core's checkpoints; keep routine observations in conversation between checkpoints.
Update design deliverables within their approved action scope and identify the reviewed version.

## 2. HLD and LLD

| Design level | Main question | Typical contents |
| --- | --- | --- |
| HLD | How does the system satisfy its goals as a whole? | Boundaries, major components, data flow, trust boundaries, deployment topology, quality requirements, and tradeoffs. |
| LLD | How will the affected components behave and interact? | Interfaces, data structures, schemas, algorithms, state transitions, errors, concurrency, and testable contracts. |

HLD and LLD are connected views, not mandatory separate files for every change.
A small change can use a design section saved with its agreed plan at the plan checkpoint.
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

### Turn quality goals into design scenarios

For each material quality goal, specify the stimulus, operating condition, affected component, expected response, and measurement.
Use agreed targets or mark them pending. Do not replace unknown targets with convenient example values.

| Scenario | Design questions |
| --- | --- |
| Traffic burst | Which work is admitted, queued, delayed, or rejected? How is user impact measured? |
| Dependency outage | Which journeys remain usable, and which fail with an explicit outcome? |
| Concurrent updates | Which invariant must hold, and how are conflicts observed and resolved? |
| Access revocation | When do sessions, caches, indexes, and ongoing work stop exposing protected data? |
| Model uncertainty | When does the system abstain, request review, or use a bounded fallback? |
| Recovery | Which data position and service functions must be restored, and how are they checked? |

Link scenarios to goal criteria and proposed verification.
Use them to compare alternatives and expose conflicts between latency, consistency, cost, and availability.

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

### Evaluate boundaries and alternatives

Assign a clear owner to each business rule, authoritative data set, and externally visible contract.
Define allowed dependency directions and identify cycles or shared state that make components difficult to change independently.
Keep volatile provider or infrastructure details behind useful boundaries when replacement is a realistic need.
Avoid adding interfaces solely to create hypothetical flexibility.

For significant choices, compare the simplest viable option with credible alternatives using the same requirements.
Include delivery effort, operating effort, failure modes, portability, recurring cost, and exit cost where material.
For external products or services, consider support, data export, version changes, and service withdrawal.
Separate measured evidence from estimates and unresolved assumptions.
Identify expensive-to-reverse decisions and validate their critical assumptions before committing dependent work.

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

### Cache and configuration contracts

For each material cache, define the authoritative source, key, scope, lifetime, and invalidation mechanism.
Include user or tenant identity in cache isolation when protected results depend on that identity.
Specify acceptable staleness, concurrent refresh behavior, and behavior when the cache is unavailable.
Account for synchronized expiration or refresh causing sudden load on the authoritative store.
Do not assume a cache is harmless because its contents can be rebuilt.

Define configuration precedence, defaults, validation, and whether changes require restart or support live reload.
Specify behavior for missing, malformed, or incompatible configuration.
For feature flags, identify the owner, safe default, supported combinations, and removal condition.
Design disable behavior explicitly when a flag controls writes, migrations, background jobs, or model actions.

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

### Failure containment and recovery

Trace a critical journey across its dependencies, including identity, name resolution, storage, and external providers where relevant.
Identify shared dependencies that can fail several supposedly redundant components together.
For each material failure, describe detection, user-visible behavior, containment, recovery, and residual effects.
Specify behavior when recovery itself fails or a fallback becomes overloaded.

Allocate end-to-end deadlines across dependency calls, queueing, retries, and cleanup.
Bound concurrent work and admission before resources are exhausted.
Define the scope of a failure: one request, user, tenant, worker, node, or larger deployment.
Use bulkheads or other isolation mechanisms only when they provide a required containment boundary.
Specify whether degraded operation preserves correctness, becomes read-only, or rejects work.
Do not silently weaken authorization or data-integrity guarantees during degradation.

### Capacity and service measurement

Estimate request rates, concurrency, payload sizes, data growth, and retention from stated workload assumptions.
Include peak demand, background work, deployment overlap, and loss of capacity during failure where relevant.
Account for quotas, connection pools, storage throughput, and other limits outside the main compute tier.
Identify the first expected bottleneck and the observation that would justify scaling or redesign.

For operated services, define a service level indicator (SLI) for a meaningful user outcome.
Define its service level objective (SLO), measurement window, included population, and exclusions when agreed.
Use distributions or relevant percentiles when averages would hide poor experiences.
Specify how operators detect failure and distinguish application, dependency, and capacity problems.
See [Google SRE: Service Level Objectives](https://sre.google/sre-book/service-level-objectives/) for measurement terminology and tradeoffs.

### Threat and abuse modeling

Identify protected assets, actors, entry points, trust boundaries, and privileged operations.
Consider misuse of legitimate features as well as malformed requests.
Trace how an attacker or mistaken user could cross a boundary or exceed intended authority.
For material threats, link the mitigation, verifying check, and residual risk to the design.
Revisit this analysis when data exposure, privilege, integrations, or execution capabilities change.
Use [OWASP threat modeling](https://community.owasp.org/Threat_Modeling) for a structured threat-modeling approach.

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
Stop dependent work when a design change invalidates the agreed plan.
Discuss a revised plan, obtain plan approval, and obtain execution approval for its next action and separate check.
Save the approved revision at a plan checkpoint before changed execution.
Update affected design deliverables within that scope before later work relies on them.
Do not rewrite design history to hide a failed assumption.

At handover, link the implemented architecture, important decisions, interface contracts, and operational limitations.
Record known differences between the approved design and the delivered system.
Resolve or explicitly accept those differences through the core workflow.

### Evolution and retirement

Identify consumers and retained artifacts affected by contract, schema, model, or configuration changes.
Define supported version overlap, deprecation conditions, and evidence needed before removing an old path.
Check that queued work, exported data, and saved client state remain interpretable where required.
Plan replacement or export paths for dependencies whose removal would otherwise trap required data or behavior.
For retirement, identify remaining consumers, retained data, credentials, infrastructure, and operating obligations.
Treat retention and deletion requirements as inputs to retirement design, not as assumptions about immediate deletion.

## 12. Compact design outline

Use this outline when useful. Omit irrelevant sections with a reason where the omission affects review.
Keep reserved pass records empty until plan approval and final execution approval permit the core's plan checkpoint.
Keep unapproved outlines in conversation; creating a separate design deliverable requires its named action's execution approval.

```markdown
# Design: <change or system>

- State: proposed / reviewed / implemented / superseded
- Related goal criteria and pass:
- Scope and exclusions:
- Existing behavior and affected components:
- Requirements and quality targets:
- Quality scenarios, workload assumptions, and capacity limits:
- HLD: boundaries, components, flows, deployment, and trust
- LLD: interfaces, schemas, state, algorithms, and errors
- Security, failure, concurrency, and resource behavior:
- Threats, mitigations, failure containment, and residual risks:
- AI and GPU considerations, if applicable:
- Compatibility, migration, rollout, and recovery:
- Observability, operating ownership, deprecation, and retirement:
- Alternatives and decision references:
- Verification approach:
- Assumptions, unresolved questions, and required reviewers:
- Implemented version and remaining differences:
```
