# Backend and API development guide

## 1. Purpose and authority

Use this guide for services, application servers, APIs, background jobs, and business logic.
It owns backend contracts, request handling, domain behavior, and service integration.
Follow the [core workflow](../core/software-ai-development-workflow.md) for approvals, records, and lifecycle.
Use [architecture and design](../core/architecture-and-design-guide.md) and [engineering quality](../core/engineering-quality-guide.md) as the shared foundation.
This guide does not authorize execution or require a particular language, framework, or deployment model.

### Apply the version 3 workflow

Use the core's conversation, approval, action, check, and checkpoint sequence for engineering work.
Draft and revise plans in conversation without changing Markdown during discussion.
Obtain plan approval, then explain the next action, its separate check, limits, and stopping point.
Obtain final execution approval for that action and check before acting.
Save the agreed plan and first execution approval at the core's plan checkpoint before execution.
Keep reserved pass records empty and inactive until that checkpoint.

Divide service investigation, contract changes, implementation, integration checks, and release into small actions with inspectable results.
Keep action and verification procedures distinct, including separate executable artifacts when applicable.
Report each result and obtain execution approval before dependent work or the next action.
Do not create an unattended procedure that crosses action approval or required human review boundaries.
Retain valid approvals for unchanged scope and preserve stricter project-specific learning or review gates.

Save management records at the core's defined checkpoints, including failures, blockers, withdrawal, and substantive session handovers.
Keep routine results and subsequent approvals in conversation between checkpoints.
Preserve partial effects, running operations, unknown outcomes, accumulated approvals, and one next action at checkpoints.
Check unknown outcomes within applicable authority before retries that could duplicate effects.
Obtain new approval for recovery unless the approved scope explicitly covers that recovery branch and its check.

Request handling, background job state, retries, and audit events follow product requirements independently of project Markdown checkpoint timing.

### Transfer guidance during record preparation

Read all designated sources, including API contracts, configuration notes, integration reports, instruction files, and status records.
Transfer applicable requirements into generated definitions and operating instructions under the core's ownership rules.
Preserve supplied consumers, operation contracts, authorization boundaries, data ownership, compatibility needs, resource limits, and review obligations.
Keep selected configuration distinct from observed configuration, and source-reported results distinct from newly verified facts.
Record unresolved choices as focused questions; do not invent frameworks, endpoints, service targets, environments, or owners.

Preparation does not authorize service probing, new designs or test plans, API calls, background jobs, resource creation, or deployment.
Keep preparation checks in current state, reserved pass records at zero bytes, and the final report empty until closure.
Transfer necessary guidance into internal owners so future operation requires neither this guide nor other preparation inputs.
Use internal relative links; keep preparation provenance in the overview's historical note.

## 2. Related guides

| Guide | Use when |
| --- | --- |
| [Frontend](frontend-development-guide.md) | Browser clients consume the API. |
| [Database](database-engineering-guide.md) | The service persists or queries data. |
| [Distributed systems](distributed-systems-microservices-guide.md) | Work crosses services, queues, or independent failure boundaries. |
| [LLM engineering](llm-engineering-guide.md) | The service calls models, retrieves context, or exposes AI tools. |
| [Infrastructure](infrastructure-deployment-operations-guide.md) | The service is deployed and operated. |

Database integrity and migration rules belong in the database guide.
Distributed delivery and coordination details belong in the distributed systems guide.

## 3. Requirements and design

Identify consumers, business operations, data sensitivity, and expected workloads.
Record compatibility requirements, service limits, and applicable response targets.
Do not invent latency, scale, availability, or retention requirements.

At HLD level, describe domain boundaries, service responsibilities, dependencies, and deployment relationships.
At LLD level, specify operation contracts, validation, authorization, errors, transactions, and side effects.
Keep transport details separate from domain rules when that separation improves clarity and testability.
Avoid excessive abstraction or service splitting without a demonstrated need.

## 4. API contracts

For each material operation, define:

- Purpose, caller, protocol, endpoint or operation name, and version policy.
- Input schema, required fields, limits, defaults, and validation behavior.
- Output schema, status or error semantics, and sensitive-field handling.
- Authentication and resource-level authorization requirements.
- Side effects, transactional boundaries, and duplicate-request behavior.
- Pagination, ordering, filtering, and consistency expectations where applicable.
- Timeout, cancellation, and asynchronous completion behavior.
- Compatibility expectations for existing consumers.

Use a machine-readable contract when it provides practical value to the project.
Keep one authoritative contract and check generated or copied representations against it.
Do not expose internal exceptions as the public error model.
Distinguish a rejected operation from an operation whose final outcome is unknown.

## 5. Domain logic and data access

Enforce business invariants at the authoritative service or data boundary.
Do not rely on client validation or a hidden UI control for protection.
Define ownership of each mutation and its transaction scope.
Use parameterized data access and bounded queries appropriate to the selected storage technology.
Avoid unbounded collection reads or accidental repeated queries under normal workloads.

Coordinate schema changes, indexes, migrations, and restoration with the database guide.
Document behavior when concurrent operations update the same resource.
Specify conflict detection or resolution instead of silently losing updates.
Identify external side effects that cannot participate in the same transaction.

## 6. Authentication, authorization, and isolation

Identify the trusted identity source and credential lifecycle.
Authorize each protected operation against its target resource and relevant tenant.
Do not trust caller-supplied ownership fields without validation.
Define administrative privileges and service-to-service permissions explicitly.
Avoid leaking sensitive resource existence through error details where that matters.
Record audit events for operations that require traceability.
Apply the shared security guidance to sessions, secrets, outbound requests, uploads, and dependency use.

## 7. Background work and resilience

Define job submission, ownership, status, cancellation, and completion reporting.
Use durable work tracking when losing an accepted job would violate requirements.
Bound concurrency, queue size, timeouts, and retries according to the workload.
Retry only where the failure and operation semantics make retry appropriate.
Account for duplicate execution and worker restarts.
Use the distributed systems guide for delivery guarantees, idempotency, and reconciliation.

Propagate request deadlines when relevant.
Distinguish cancellation of a caller's wait from cancellation of the underlying operation.
Ensure resource cleanup after errors and cancellation.

## 8. Observability and performance

Define useful logs, metrics, and traces around important operations and dependencies.
Use correlation identifiers without logging secrets or unnecessary personal data.
Track error rate, latency, throughput, saturation, and queue delay where applicable.
Separate readiness to accept work from mere process existence.

Measure representative workloads before optimizing.
Record cache keys, validity, invalidation, and failure behavior when caching is used.
Protect against overload with appropriate bounds and rejection behavior.
Do not promise unlimited capacity or use retries to conceal persistent overload.

## 9. Verification and release

Check domain rules, boundary validation, resource authorization, and meaningful failure cases.
Test API compatibility and real integrations where mocks would hide relevant behavior.
Include concurrency, duplicate submission, cancellation, and partial-failure checks when they affect correctness.
Check deployment configuration and secret handling through the infrastructure guide.
Verify database compatibility during mixed-version releases.
Name the target environment, affected operations, test data, expected effects, limits, and stopping point before requesting execution approval.
Keep implementation and verification procedures separate, including their executable artifacts when applicable.
Keep expected and actual results distinct; actual results remain `not run` until observations exist.

Save the tested service version, dependency conditions, inputs, results, and limitations at the applicable core checkpoint.
After an approved release, check actual critical operations at the destination.
Report the action and check results, then obtain execution approval before the next action.
Update consumer contracts, design notes, tests, and operating instructions within their approved action scope.
Technical checks do not satisfy pending human acceptance or required learner confirmation.
Resolve or explicitly transfer control of running jobs and release operations before pass closure.
Keep unresolved defects visible and close the pass through the core workflow.
