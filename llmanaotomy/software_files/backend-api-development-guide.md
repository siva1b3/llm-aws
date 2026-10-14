# Backend and API development guide

## 1. Purpose and authority

Use this guide for services, application servers, APIs, background jobs, and business logic.
It owns backend contracts, request handling, domain behavior, and service integration.
Follow the [core workflow](software-ai-development-workflow.md) for approvals, records, and lifecycle.
Use [architecture and design](architecture-and-design-guide.md) and [engineering quality](engineering-quality-guide.md) as the shared foundation.
This guide does not authorize execution or require a particular language, framework, or deployment model.

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

Record the tested service version, dependency conditions, inputs, results, and limitations.
After an approved release, check actual critical operations at the destination.
Update consumer contracts, design notes, tests, and operating instructions when behavior changes.
Keep unresolved defects visible and close the pass through the core workflow.
