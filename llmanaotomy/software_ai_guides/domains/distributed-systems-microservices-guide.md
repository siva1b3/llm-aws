# Distributed systems and microservices guide

## 1. Purpose and authority

Use this guide when work crosses independent processes, machines, services, or failure boundaries.
It owns communication semantics, coordination, consistency, and partial-failure design.
Follow the [core workflow](../core/software-ai-development-workflow.md), [architecture guide](../core/architecture-and-design-guide.md), and [quality guide](../core/engineering-quality-guide.md).
Distribution is an architectural choice, not a default requirement.

### Apply the version 3 workflow

Use the core's conversation, approval, action, check, and checkpoint sequence for engineering work.
Draft and revise plans in conversation without changing Markdown during discussion.
Obtain plan approval, then explain the next action, its separate check, limits, and stopping point.
Obtain final execution approval for that action and check before acting.
Save the agreed plan and first execution approval at the core's plan checkpoint before execution.
Keep reserved pass records empty and inactive until that checkpoint.

Divide investigation, contract changes, migrations, deployments, and recovery into small actions with inspectable results.
Keep action and verification procedures distinct, including separate executable artifacts when applicable.
Report each result and obtain execution approval before dependent work or the next action.
Do not create an unattended procedure that crosses deployment, migration, fault-injection, or human review boundaries.
Retain valid approvals for unchanged scope and preserve stricter project-specific learning or review gates.

Save management records at the core's defined checkpoints, including failures, blockers, withdrawal, and substantive session handovers.
Keep routine results and subsequent approvals in conversation between checkpoints.
Preserve partial effects, running operations, unknown outcomes, accumulated approvals, and the exact next action at checkpoints.
Explain recovery and obtain new execution approval unless the approved scope explicitly covers that recovery branch and check.

These rules govern project engineering actions; application retries and coordination follow the agreed system design.
Durable business state, message processing, and operational telemetry are independent of project Markdown checkpoint timing.

### Transfer guidance during record preparation

Read all designated sources, including topology, contract, configuration, instruction, incident, and status records.
Transfer applicable requirements into generated definitions and operating instructions under the core's ownership rules.
Preserve supplied service boundaries, data owners, invariants, delivery semantics, dependencies, recovery needs, and review obligations.
Keep selected topology distinct from observed topology, and reported incidents or deployments distinct from newly verified facts.
Record unresolved guarantees or ownership as focused questions; do not invent services, platforms, targets, or owners.

Preparation does not authorize service probing, new designs or migration plans, fault injection, replay, provisioning, or deployment.
Keep preparation checks in current state, reserved pass records at zero bytes, and the final report empty until closure.
Transfer necessary guidance into internal owners so future operation requires neither this guide nor other preparation inputs.
Use internal relative links; keep preparation provenance in the overview's historical note.

## 2. Related guides

- [Backend and APIs](backend-api-development-guide.md): service contracts and domain operations.
- [Database](database-engineering-guide.md): local integrity, migrations, storage, and recovery.
- [Infrastructure](infrastructure-deployment-operations-guide.md): networking, deployment, scheduling, and operation.
- [ML and deep learning](ml-deep-learning-guide.md): distributed training and experiment requirements.
- [GPU infrastructure](gpu-infrastructure-performance-guide.md): device topology and accelerator communication.

A modular monolith may satisfy the project with less complexity.
Record the benefit and operating cost before introducing independent services.
Keep proposed tradeoffs in conversation until the relevant agreement is saved at a core checkpoint.

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
Name the affected environment, records or messages, expected effect, separate check, limits, and stopping point before requesting execution approval.
Stop dependent engineering work and save a failure checkpoint when an action or check fails.
Do not infer that a failed command restored the previous distributed state.

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
Explain the fault action, separate observation method, expected result, recovery scope, and stopping point before execution approval.
Keep fault procedures and verification procedures separate when executable artifacts are used.
Record expected checks as `not run` at the agreed plan checkpoint.
Save topology, versions, workload, injected fault, observations, and limitations at the applicable checkpoint.
Save failures and partial effects promptly; do not wait for a later successful retry.
Unit tests alone do not establish distributed failure behavior.
Technical checks do not satisfy pending human acceptance or required learner confirmation.

## 10. Release and evolution

Use the infrastructure guide for rollout and destination verification.
Coordinate API, event, and schema changes with dependent consumers.
Specify deployment order where compatibility requires it.
Propose each rollout or migration action with its own separate check and execution scope.
Check its result and obtain approval before the next action, even when the overall plan is approved.
Preserve replay compatibility for retained messages when required.
Verify recovery before removing old consumers or data representations.
Update diagrams, contracts, and operational deliverables within their approved action scope.
Save significant decisions, accumulated approvals, results, and remaining work at core checkpoints.
Resolve or explicitly transfer control of running deployments, migrations, replays, and repair jobs before pass closure.
Close work using the core workflow and retain unresolved guarantees explicitly.
