# Engineering quality guide

## 1. Purpose and authority

Use this guide to make software and AI systems easier to understand, verify, secure, operate, and change.
The aim is to reduce preventable defects and detect problems early. No checklist guarantees bug-free software.

Follow the [core workflow](software-ai-development-workflow.md) for approvals, records, evidence, and completion.
Use the [architecture guide](architecture-and-design-guide.md) for HLD, LLD, and technical decisions.
This guide owns engineering checks and review expectations. It does not authorize new work.

Apply relevant practices at every stage, proportionate to the change's consequences.
Avoid requiring every tool, test type, or document for every project.
Convert "best practices" into explicit checks with an owner, expectation, and evidence.
Do not claim compliance with a named standard without the required assessment.

## 2. Quality throughout the workflow

| Stage | Quality responsibility |
| --- | --- |
| Define | Agree on observable behavior and applicable quality requirements. |
| Understand | Identify existing failures, vulnerable boundaries, technical constraints, and uncertainty. |
| Plan | Include relevant tests, reviews, failure cases, and release checks. |
| Work | Maintain readable code, safe boundaries, tests, and affected documentation. |
| Verify | Run approved checks against identified outputs and report limitations. |
| Decide | Assess evidence and unresolved defects before acceptance or release. |

Plan expectations before implementation where possible.
Do not design tests solely to match the implementation after seeing its behavior.
Use existing repository tools and conventions unless an approved change has a clear reason.

## 3. Clean code and maintainability

Review these properties in changed code:

- Names communicate domain meaning and distinguish concepts consistently.
- Functions and modules have coherent responsibilities.
- Control flow makes normal behavior and exceptional behavior understandable.
- Dependencies and side effects are explicit at meaningful boundaries.
- State ownership and mutation are clear.
- Interfaces expose only what their consumers need.
- Error handling preserves useful context without leaking sensitive information.
- Configuration is validated and separated from secrets.
- Comments explain constraints or reasons that code alone cannot express.
- Dead code and obsolete comments are removed when within scope.
- Repeated logic is consolidated when it represents the same rule.
- New abstractions solve demonstrated needs rather than speculative future requirements.

Use principles such as separation of concerns and dependency inversion where they improve the design.
Do not apply design patterns mechanically or enforce arbitrary function-length limits.
Prefer clear, direct code over cleverness.
Avoid broad rewrites mixed with a focused behavior change unless the approved outcome requires them.

Preserve public contracts during refactoring unless the change explicitly includes compatibility work.
Review complexity in terms of behavior, dependencies, and change risk, not line count alone.

## 4. Correctness at boundaries

Validate untrusted inputs at system boundaries.
Define missing, empty, invalid, oversized, duplicate, and conflicting input behavior.
Do not assume frontend validation replaces server-side validation.
Enforce authorization on the actual protected operation and resource.

For affected behavior, consider:

- Numerical boundaries, precision, overflow, and unit conversion.
- Time zones, date boundaries, clock assumptions, and expiration.
- Text encoding, normalization, and locale where relevant.
- Ordering, duplicate requests, concurrency, and repeated events.
- Partial writes, transaction boundaries, and cleanup after failures.
- Timeout, cancellation, retry, and resource exhaustion.

Protect important data invariants at the authoritative layer.
Avoid silent error suppression and success responses after incomplete critical operations.

## 5. Test selection

Select tests according to the claim and risk. Record justified exclusions where they affect confidence.

| Check type | What it can establish |
| --- | --- |
| Formatting, linting, and static analysis | Conformance to configured rules and detectable source-level issues. |
| Type checking and compilation | Compatibility with configured language and build constraints. |
| Unit tests | Behavior of a bounded unit under controlled conditions. |
| Integration tests | Behavior across real component or dependency boundaries. |
| Contract tests | Agreement between producers and consumers of an interface. |
| End-to-end tests | Selected user journeys through the integrated system. |
| Regression tests | Protection against a specified previously observed failure. |
| Security checks | Particular vulnerability classes or security requirements within their scope. |
| Performance tests | Latency, throughput, memory, or cost under recorded workload conditions. |
| Resilience checks | Behavior under specified failures, restarts, or dependency degradation. |
| Accessibility checks | Selected accessibility requirements, with human checks where automation is insufficient. |
| Migration and restoration checks | Data transition and recovery behavior under stated conditions. |
| Model evaluations | Measured AI behavior on specified data, metrics, and execution conditions. |

Do not treat one check type as a substitute for all others.
Use a focused regression test for a bug when practical and meaningful.
Avoid tests that merely reproduce implementation logic or assert irrelevant internal details.
Use realistic dependency behavior where mocks would hide the failure being investigated.

Keep tests isolated and repeatable where possible.
Control clocks, random seeds, inputs, and external dependencies where those controls support the intended claim.
Investigate flaky tests. Do not hide them with unlimited retries or delete their assertions to obtain a pass.
Record justified quarantines with an owner, consequence, and follow-up condition.

Run affected checks first. Run broader checks when integration risk or repository requirements justify them.
Do not repeat unchanged checks without a reason.

## 6. Security and privacy

Identify trust boundaries, sensitive data, and actors before selecting security checks.
Apply the following where relevant:

- Least-privilege access for users, services, containers, and deployment identities.
- Authentication and resource-level authorization, including tenant isolation.
- Input handling that prevents injection into queries, commands, templates, and interpreters.
- Output handling appropriate to the browser, file, or other destination.
- Secure secret storage, delivery, rotation, and redaction.
- Session, cookie, cross-origin, and request-forgery protections for the actual web architecture.
- Restrictions on outbound requests and untrusted file access.
- Safe upload handling and bounded processing of untrusted content.
- Encryption and certificate handling appropriate to project requirements.
- Data minimization, retention, deletion, and access logging where applicable.

Keep secrets out of source, examples, logs, screenshots, test fixtures, and model prompts.
Use approved handling when tests require sensitive or production-like data.
Security testing against external or production systems requires applicable authority and scope.

Check current advisories and vendor guidance during approved version-specific security work.
Do not preserve a static claim that a dependency or configuration is permanently secure.
Track unresolved material findings and their disposition. Do not mark a scan pass as proof of complete security.

## 7. Dependencies and builds

Use the ecosystem's agreed dependency declaration and reproducibility mechanism.
Record relevant runtime, build-tool, framework, and platform versions.
Review new dependencies for necessity, compatibility, maintenance, license constraints, and security implications.
Avoid adding a large dependency for a small need without considering the tradeoff.

Build and test the artifact intended for delivery.
Identify the artifact and source revision so results can be traced to the release.
Keep environment-specific configuration outside immutable build artifacts where the architecture requires it.
Check that generated artifacts do not contain unintended secrets or local-only assumptions.

For GPU workloads, check the relevant driver, runtime, framework, extension, and hardware combination.
Use applicable vendor sources for current compatibility claims.
Record the actual tested combination instead of assuming compatibility from version names.

## 8. Domain quality checks

| Domain | Checks to consider |
| --- | --- |
| Frontend | Critical user journeys, loading and error states, keyboard operation, responsive behavior, browser compatibility, and unsafe rendering. |
| Backend and APIs | Validation, authorization, contracts, errors, pagination, concurrency, idempotency, timeouts, and transaction behavior. |
| Databases | Constraints, representative query performance, migrations, concurrent updates, backup restoration, and retention behavior. |
| Distributed systems | Duplicate and out-of-order messages, worker restart, dependency failure, backpressure, consistency, and reconciliation. |
| Containers | Reproducible build, startup, runtime identity, configuration, persistence, signal handling, and shutdown. |
| Kubernetes | Scheduling, resources, configuration, readiness, startup, shutdown, persistence, access boundaries, and rollout behavior. |
| Cloud and on-premises | Network and identity paths, target capacity, storage, monitoring, deployment, recovery, and operating ownership. |

These are selection prompts, not automatic requirements to introduce those technologies.
Define expected results for the checks selected by the project.

## 9. ML, DL, and LLM quality

Use checks that distinguish software correctness from model quality.
A working inference endpoint can still produce unacceptable model results.
A strong offline score can coexist with unacceptable latency or operational behavior.

### Data and experiments

- Validate schemas, missing values, labels, transformations, and material data distributions.
- Track dataset origin, permitted use, lineage, and version where applicable.
- Prevent unintended leakage between training and independent evaluation data.
- Record experiment configuration, code, model, data, preprocessing, and relevant environment versions.
- Compare against an agreed baseline and meaningful metrics.
- Evaluate important data slices and failure cases, not only an overall average.
- Report variability and sampling limits. Do not select only favorable runs.

### LLM applications

- Evaluate representative tasks with explicit success criteria.
- Check grounding and source support when factual attribution is required.
- Check structured output validity and downstream contract handling.
- Test missing context, ambiguous requests, malformed input, and unavailable dependencies.
- Test untrusted retrieved content and prompt-injection attempts within approved scope.
- Validate and authorize tool actions outside the model's generated text.
- Test sensitive-data leakage and isolation where relevant.
- Assess latency, token usage, cost, and fallback behavior under representative workloads.

Prompts, retrieval settings, model versions, and tool contracts are behavior-affecting changes.
Reevaluate affected claims when any of them changes.
Automated model-based judging needs its own validation and stated limitations when used as evidence.
Use human review for cases where the agreed evaluation requires it.

## 10. GPU quality and performance

Verify that the workload uses the intended device and execution path.
Do not infer GPU execution merely from installed accelerator packages.
Record hardware, software, precision, workload shape, batch size, and concurrency for relevant checks.

Consider these checks when applicable:

- Correctness and numerical stability at the chosen precision.
- Peak and sustained device memory usage under representative inputs.
- Out-of-memory behavior and cleanup after failed work.
- Warm-up and steady-state performance measured separately.
- Synchronization and timing methods appropriate to asynchronous accelerator execution.
- Data-loading, CPU, storage, and network bottlenecks.
- Multi-device communication, scaling efficiency, and worker failure behavior.
- Checkpoint creation, restoration, and integrity.
- Isolation and contention under shared GPU use.
- Cancellation, process exit, and resource release.

Evaluate quality alongside speed when changing quantization, precision, batching, or parallelism.
Do not compare benchmarks with materially different workloads without explaining the difference.
Avoid universal utilization targets. Derive acceptance from the project's workload and constraints.

## 11. Code review

Review the change against its goal and design, not only its syntax.
The reviewer should be able to identify:

1. The intended behavior and scope.
2. The relevant design and compatibility decisions.
3. Important correctness, security, concurrency, and resource risks.
4. Tests performed, their results, and known gaps.
5. Migration, release, and recovery implications.
6. Documentation and operating changes.
7. Remaining issues and required decisions.

Prioritize actionable defects and material maintainability issues.
Separate correctness findings from optional style suggestions.
Give findings a location, triggering condition, consequence, and recommended direction.
Review AI-generated code and tests with the same standards as other contributions.
An AI's confidence is not evidence that its code works.

## 12. Release readiness

Before a release within approved scope, confirm applicable items:

- The reviewed artifact is the artifact being released.
- Required build, test, security, and model-evaluation checks passed for identified versions.
- Compatibility and migration order are defined and checked.
- Configuration and secrets are available through the agreed mechanism.
- Health checks assess readiness for relevant traffic or work.
- Logging, metrics, alerts, and operational ownership cover material failure modes.
- Rollout limits, stop conditions, and recovery actions are explicit.
- Backup and restoration evidence exists where required by the data risk.
- Known defects and exclusions have a recorded disposition.
- Required human review and release authority are present.

After release, verify the actual destination, version, and relevant user behavior.
Do not call a deployment successful solely because the deployment command exited successfully.
Keep checks safe for the target environment and within approved authority.

## 13. Defects, exceptions, and technical debt

Record a defect's observed behavior, expected behavior, reproduction conditions, effect, and evidence.
Assess severity using user impact, data impact, security exposure, and operational consequences.
Preserve failed results when a correction passes later checks.

Use existing core records for issues and follow-up proposals.
Do not create a parallel permanent tracking system without an agreed need.
For a deferred defect or quality exception, record:

- The affected criterion or practice.
- The reason and consequences of deferral.
- The approving owner and accepted scope.
- Any compensating measure.
- The follow-up trigger or agreed date, when applicable.

Deferral does not turn a failed criterion into a passed criterion.
The human must explicitly change the goal if that criterion will no longer be required.

## 14. Evidence and completion

Use the core verification fields for all recorded checks.
Keep expected results separate from observations.
Identify the tested version, conditions, evidence, and limitations.
Retain enough information to explain significant claims and decisions.

Before pass closure, confirm that the outcome meets its agreed criteria and required reviews.
Before project completion, assess the integrated result and any required delivery or operating checks.
Do not declare completion while required tests, model evaluations, restoration checks, or human reviews remain pending.

Keep checks proportionate. Add a check because it supports a meaningful claim or addresses a material risk.
Update this guidance when experience reveals a recurring failure or an obsolete practice.
Preserve the separation between reusable guidance and a project's specific acceptance requirements.
