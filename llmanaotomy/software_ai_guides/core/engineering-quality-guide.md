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

### Apply version 3 boundaries

During preparation, transfer applicable quality requirements and review obligations into generated definitions and operating instructions.
Preserve supplied evaluation methods, selected configuration, and evidence status without creating new test plans or running technical checks.
Generated projects must retain necessary quality guidance without requiring this library later.

Propose each action with a separate check, expected result, limits, and stopping point.
Obtain plan approval before the next action's final execution approval under the core workflow.
Save agreed expectations as `not run` at the plan checkpoint before execution.
Keep the action procedure and checking procedure distinct; use separate executable artifacts when applicable.
Do not create an unattended sequence that crosses an action approval or human review boundary.
After the action and check, report actual results and obtain execution approval before the next action.

Keep routine results in conversation until a defined checkpoint.
Save failures, blockers, material unexpected results, and partial effects promptly under the core's failure rules.
Save accumulated quality evidence and approvals at plan, pass, session, and other required checkpoints.
Human acceptance and required learner confirmation remain separate from passing technical checks.

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
These stages describe engineering responsibilities within the core's conversation, approval, action, check, and checkpoint sequence.
Draft expectations in conversation; persist agreed plans only when the core permits the plan checkpoint.
Do not design tests solely to match the implementation after seeing its behavior.
Use existing repository tools and conventions unless an approved change has a clear reason.

### Choose checks by consequence

Use the core workflow's criterion mapping and risk assessment to select checks.
Prioritize data integrity, access boundaries, irreversible effects, critical user journeys, and difficult-to-detect failures.
Include negative cases and recovery behavior, not only successful requests.
Identify who evaluates subjective or specialist criteria before relying on those checks for acceptance.
For each excluded material check, explain the confidence gap and any compensating evidence.
Keep an unperformed required check pending rather than treating a waiver as a passing result.

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

Check that success is reported only after the contract's promised work has reached its required completion state.
For asynchronous work, distinguish accepted, running, completed, failed, and unknown outcomes where applicable.
Verify cleanup of files, handles, connections, locks, subscriptions, and temporary state after failures.
Check that permission changes affect caches and ongoing work according to the agreed access contract.
Exercise configuration validation, safe defaults, and relevant feature-flag combinations when they change behavior.

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

### Test design and test data

Derive expected results from requirements, contracts, known examples, or an independent reference where available.
Check that a regression test exposes the original defect when practical.
Use boundary values, equivalence classes, and state-transition cases to cover meaningful behaviors without redundant examples.
Use property-based testing for invariants across varied inputs when fixed examples leave important gaps.
Use fuzzing for parsers or other exposed input handlers when malformed input is a material risk.
Use mutation testing selectively when uncertainty concerns whether assertions detect meaningful changes.
No single technique is mandatory for every change.

Version or reconstruct important fixtures and evaluation inputs.
Prevent tests from depending on execution order or shared mutable state unless that interaction is the behavior under test.
Test realistic data shapes, sizes, encodings, and permission scopes.
Keep test credentials and test tenants separate from production access.
Ensure cleanup cannot delete unrelated data, including after partial test failure.

### Continuous integration and environment confidence

Keep the project's agreed fast checks runnable locally and in continuous integration (CI).
Record the commands, required services, configuration, and supported environment needed to reproduce failures.
Select broader suites according to affected boundaries and delivery risk.
Inspect test discovery and skipped-test counts when a surprisingly fast or empty run appears successful.
Distinguish command failure, test failure, timeout, unavailable infrastructure, and missing results.

Check the resulting combined change after merges or conflict resolution when earlier results no longer identify the delivered work.
Do not reuse stale results solely because the branch name is unchanged.
Compare important local, CI, staging, and production differences before generalizing a result.
Include operating system, runtime, database, feature flags, permissions, and architecture where they affect behavior.
Use clean-environment setup checks when hidden local state could explain success.
Treat generated code and configuration as deliverables requiring appropriate checks.

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

Translate the architecture's material threats into negative tests and review questions.
Test unauthorized roles, cross-tenant identifiers, expired or revoked access, and privilege changes where applicable.
Check that rejected requests cannot trigger protected side effects before the rejection occurs.
Verify that logs, exports, caches, and error responses respect the same data boundaries as normal responses.
Provide a responsible owner and remediation path for vulnerabilities discovered after release.

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

### Software supply chain

Verify package identity and origin before adopting an unfamiliar dependency or generated installation command.
Review relevant transitive dependencies and installation scripts, not only direct package declarations.
Use the project's supported pinning, lockfiles, integrity checks, and artifact verification mechanisms.
Restrict build identities and protect release credentials from untrusted code execution.
Track required dependency inventories or software bills of materials when the delivery context requires them.
Define ownership for updates, security advisories, unsupported components, and license obligations.
Confirm that a dependency update preserves required runtime and application behavior.

Use [NIST's Secure Software Development Framework](https://csrc.nist.gov/Projects/ssdf) as supporting lifecycle guidance when applicable.
Identify the adopted publication version before mapping project practices to its requirements.
Referencing the framework does not establish compliance.

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

Version evaluation sets and scoring rules alongside the behavior being assessed.
Protect held-out cases from prompt tuning, example selection, and repeated candidate selection when independence is required.
Compare candidates on equivalent tasks and record meaningful regressions even when the overall score improves.
Evaluate abstention, escalation, and fallback behavior as well as successful answers.
Test tool-loop limits, repeated actions, interrupted streams, and failures after partial action completion where applicable.
Include relevant language, user, and task slices when average results could hide important failures.
Check judge disagreement against reviewed examples before treating a model judge as reliable evidence.

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

Verify that referenced packages, APIs, configuration options, and commands actually exist in the selected versions.
Review generated tests for shared assumptions with the generated implementation.
Propose corrections for material review comments and identify affected checks.
Perform corrections and reruns only within applicable execution approval; obtain revised authority when their scope changes.
Record reviewer identity and scope. Distinguish author self-review from an independent review when independence is required.

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

### Performance, resilience, and operational checks

Choose the workload and expected result before interpreting performance measurements.
Record request mix, arrival rate, concurrency, input size, data volume, cache state, and test duration where relevant.
Report latency distributions, throughput, errors, resource saturation, and cost together when they affect acceptance.
Check that the load generator or test environment is not the limiting factor.

| Check | Question it addresses |
| --- | --- |
| Expected-load test | Does the system meet agreed behavior at representative demand? |
| Burst or overload test | Does admission control keep resource use and degradation within agreed bounds? |
| Sustained-load test | Do memory, connections, queues, or storage grow unexpectedly over time? |
| Dependency-failure test | Are deadlines, retry limits, fallbacks, and recovery effective? |
| Recovery rehearsal | Can operators restore the required service and data state using available instructions and access? |

Select only relevant checks and bound their environmental effects.
Check recovery after load or faults are removed, not only behavior during the disturbance.
Verify alarms with representative signals and confirm that they reach the responsible destination within authorized scope.
Check that logs and traces explain failures without exposing secrets or overwhelming storage.
Exercise kill switches or feature-disable paths when the release relies on them for containment.
Verify retained rollback artifacts and their compatibility with the current data state.

### Documentation and usability checks

Verify setup, configuration, migration, and recovery instructions against the identified version when they support a delivery claim.
Check example commands, links, prerequisites, and expected outputs.
Review important user journeys for understandable errors, recoverability, and clear completion feedback.
Include keyboard, assistive-technology, localization, or time-zone checks according to the actual supported audience.
Ensure operators can identify the running version and locate its runbook and known limitations.

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

Before marking a pass successful, confirm that the outcome meets its agreed criteria and required reviews.
For unsuccessful, deferred, or cancelled passes, record unmet criteria, unresolved reviews, and the closing decision under the core workflow.
Before project completion, assess the integrated result and any required delivery or operating checks.
Do not declare completion while required tests, model evaluations, restoration checks, or human reviews remain pending.

Keep checks proportionate. Add a check because it supports a meaningful claim or addresses a material risk.
Update this guidance when experience reveals a recurring failure or an obsolete practice.
Preserve the separation between reusable guidance and a project's specific acceptance requirements.
