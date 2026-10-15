# Distributed systems and microservices guide

## 1. Purpose and authority

Use this guide when work crosses independent processes, machines, services, or failure boundaries.
It owns communication semantics, coordination, consistency, and partial-failure design.
Follow the [core workflow](software-ai-development-workflow.md), [architecture guide](architecture-and-design-guide.md), and [quality guide](engineering-quality-guide.md).
Distribution is an architectural choice, not a default requirement.

## 2. Related guides

- [Backend and APIs](backend-api-development-guide.md): service contracts and domain operations.
- [Database](database-engineering-guide.md): local integrity, migrations, storage, and recovery.
- [Infrastructure](infrastructure-deployment-operations-guide.md): networking, deployment, scheduling, and operation.
- [ML and deep learning](ml-deep-learning-guide.md): distributed training and experiment requirements.
- [GPU infrastructure](gpu-infrastructure-performance-guide.md): device topology and accelerator communication.

A modular monolith may satisfy the project with less complexity.
Record the benefit and operating cost before introducing independent services.

## 3. HLD and ownership

Describe service boundaries, responsibilities, owners, and authoritative data.
Map synchronous calls, asynchronous messages, shared dependencies, and failure domains.
Identify which user operations cross multiple boundaries.
Specify consistency, availability, latency, and recovery expectations for those operations.
Avoid vague system-wide guarantees when different operations have different semantics.

Identify deployment and organizational dependencies created by each service boundary.
Do not split services solely by database tables or technical layers without a domain reason.

## 4. Contracts and communication

Define request, response, and event schemas with explicit compatibility expectations.
Identify producers, consumers, schema owners, and supported version overlap.
State size limits, deadlines, cancellation, and error behavior.
Define message delivery assumptions and the consequences of duplicates or missing messages.

For each important event, specify:

- Event identity, type, schema version, and business meaning.
- Ordering scope and whether consumers require that order.
- Persistence and acknowledgment behavior.
- Retry, dead-letter, replay, and reconciliation procedures.
- Retention and access controls.

Do not describe a transport acknowledgment as proof that the business operation completed.
Do not claim end-to-end exactly-once behavior without defining and validating its complete scope.

## 5. Partial failure and retries

A timeout can leave an operation's result unknown.
Determine actual state or use an idempotent protocol before retrying an operation with external effects.
Define idempotency-key scope, storage, expiration, and response behavior where used.
Handle duplicate requests with differing payloads explicitly.

Bound retries and use appropriate delay and jitter where repeated attempts can amplify load.
Define retryable and non-retryable errors.
Coordinate retries across layers to avoid multiplying attempts unexpectedly.
Define overload rejection, queue bounds, and backpressure.
Use isolation or circuit-breaking mechanisms only where their behavior and recovery are understood.

## 6. Consistency and coordinated changes

Identify invariants that cannot tolerate temporary inconsistency.
Define observable behavior while replicated or derived data catches up.
State conflict detection, resolution, and reconciliation ownership.
Avoid relying on synchronized wall clocks for correctness unless the necessary assumptions are established.

When a workflow spans independent transactions, describe its intermediate states and recovery.
Use appropriate patterns only after assessing their guarantees and failure modes.
An outbox or saga is not a substitute for defining the operation's semantics.
Compensation can be a new business action rather than a perfect reversal.
Record irreversible effects and decisions needed after partial success.

## 7. Concurrency and coordination

Specify ownership of work, leases, leader changes, and stale workers where relevant.
Define what prevents a worker from applying changes after losing authority.
Account for process pauses, network partitions, and delayed messages.
Do not assume an expiring distributed lock alone guarantees all application invariants.
Use documented mechanisms appropriate to the selected system and validate critical assumptions.

For distributed training, distinguish communication failure from model-quality failure.
Coordinate checkpoint and worker-restart semantics with the ML and GPU guides.

## 8. Observability and operations

Correlate requests and events across relevant services.
Measure end-to-end outcomes as well as individual service health.
Track queue age, retries, dead letters, dependency latency, and reconciliation failures where applicable.
Avoid high-cardinality or sensitive telemetry without an assessed need.

Define operational ownership for stuck workflows and inconsistent data.
Document safe replay and repair procedures.
Do not replay messages or jobs without checking duplicate effects and applicable authority.

## 9. Verification

Select representative failure scenarios:

| Scenario | Expected behavior to define |
| --- | --- |
| Duplicate request or message | Safe repeated processing or explicit rejection. |
| Out-of-order delivery | Correct ordering enforcement or documented tolerance. |
| Dependency timeout | Bounded waiting and known treatment of unknown outcomes. |
| Worker restart | Recoverable work and preserved invariants. |
| Network partition | Defined availability and consistency behavior. |
| Sustained overload | Bounded resources and controlled degradation. |
| Mixed service versions | Compatible interactions during rollout. |
| Partial workflow success | Observable state and an actionable recovery path. |

Run fault injection only in authorized environments with bounded effects.
Record the topology, versions, workload, injected fault, observations, and limitations.
Unit tests alone do not establish distributed failure behavior.

## 10. Release and evolution

Use the infrastructure guide for rollout and destination verification.
Coordinate API, event, and schema changes with dependent consumers.
Specify deployment order where compatibility requires it.
Preserve replay compatibility for retained messages when required.
Verify recovery before removing old consumers or data representations.
Update diagrams, contracts, decision records, and operational procedures as the system evolves.
Close work using the core workflow and retain unresolved guarantees explicitly.
