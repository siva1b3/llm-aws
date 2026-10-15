# Infrastructure, deployment, and operations guide

## 1. Purpose and authority

Use this guide for build and delivery systems, Docker, Kubernetes, cloud, and on-premises operation.
It owns environment provisioning, artifact delivery, runtime configuration, observability, and operational recovery.
Follow the [core workflow](../core/software-ai-development-workflow.md), [architecture guide](../core/architecture-and-design-guide.md), and [quality guide](../core/engineering-quality-guide.md).
It does not authorize deployment, paid resources, external communication, or destructive recovery.

## 2. Related guides

- [Frontend](frontend-development-guide.md) and [backend](backend-api-development-guide.md): application behavior and runtime contracts.
- [Database](database-engineering-guide.md): migration, backup, data integrity, and restoration requirements.
- [Distributed systems](distributed-systems-microservices-guide.md): service interactions and partial failure.
- [ML](ml-deep-learning-guide.md) and [LLM](llm-engineering-guide.md): model artifacts, evaluations, and serving behavior.
- [GPU](gpu-infrastructure-performance-guide.md): device compatibility, allocation, memory, and performance.

Use only applicable platforms. Docker and Kubernetes are not mandatory deployment layers.
Cloud and on-premises environments require explicit ownership and recovery arrangements.

## 3. Requirements and HLD

Identify environments, accounts or sites, regions where relevant, and responsible operators.
Record availability, capacity, security, recovery, budget, and maintenance constraints.
Identify network paths, trust boundaries, identity sources, and external dependencies.
Describe compute, storage, data services, and failure domains.
Record who can provision, release, administer, and recover each environment.

For on-premises systems, identify hardware ownership, network access, storage, power-related dependencies, and maintenance responsibilities where relevant.
For cloud systems, identify account boundaries, resource ownership, quotas, and spending controls.
Do not infer that an environment is safe to modify from its name alone.

## 4. Configuration and infrastructure definitions

Keep environment definitions reviewable and recoverable using the project's selected tools.
Separate secrets from ordinary configuration.
Validate required configuration at an appropriate startup or deployment boundary.
Record actual target identity before state-changing operations.

Review infrastructure changes for replacement, deletion, permission expansion, and cost effects.
Inspect drift where it affects the proposed action.
Protect state files and other privileged control artifacts.
Do not adopt unmanaged resources or overwrite manual changes without an agreed decision.

Use current vendor documentation for version-sensitive configuration and upgrade procedures.
Do not embed permanent assumptions about changing platform defaults in project plans.

## 5. Builds and delivery pipelines

Identify source revision, dependencies, build environment, and resulting artifact.
Use the agreed reproducibility and provenance mechanisms.
Prevent secrets and unnecessary development content from entering published artifacts.
Run applicable approved checks before promotion.

Define who can trigger builds and deployments and which identities they use.
Separate untrusted contribution execution from privileged deployment credentials.
Use the same verified artifact through promotion where the architecture permits it.
Record configuration differences separately from artifact identity.
Do not assume a successful build authorizes publishing or production deployment.

## 6. Docker and container operation

Define base image selection, dependency installation, build inputs, and runtime entry point.
Use appropriate minimal privileges and filesystem permissions.
Specify configuration, secret injection, networking, and persistent storage.
Do not rely on ephemeral container storage for required durable data.

Define startup, health behavior, signals, graceful shutdown, and restart expectations.
Bound memory, CPU, and other resources where needed.
Check that builds and images support the intended platform architecture.
Identify images by a recoverable version or digest for release evidence.
Use the quality guide for vulnerability checks and dependency review.

## 7. Kubernetes operation

Use this section only when Kubernetes is selected.
Identify cluster, namespace, service accounts, workload types, and access boundaries.
Define networking, discovery, configuration, secrets, and persistent volumes.
Specify resource requests and limits from workload evidence and operational requirements.

Distinguish startup, readiness, and liveness behavior.
Avoid probes that cause destructive restart loops during ordinary dependency failures.
Define scheduling, placement, disruption, and graceful termination requirements.
Account for rollout overlap, storage attachment, and mixed application versions.
Verify that policies and device integrations are actually enforced in the target environment.
Object creation alone does not establish application readiness.

## 8. Release planning

Before execution approval, identify:

1. Artifact and target environment.
2. Configuration, secret, identity, and dependency prerequisites.
3. Database and interface compatibility requirements.
4. Rollout strategy and expected service effect.
5. Health and user-behavior checks.
6. Monitoring period or evidence needed for the release decision.
7. Stop conditions and responsible decision owner.
8. Rollback, forward repair, or restoration actions and limitations.

Do not invent a universal rollout strategy.
Use staged exposure where the risk and platform justify it.
Coordinate data changes with the database guide.
Application rollback may not reverse data migrations or external side effects.
Check the actual destination before repeating a release with an unknown outcome.

## 9. Observability and service ownership

Define useful logs, metrics, traces, and health signals for agreed service outcomes.
Set alerts around actionable conditions with a responsible owner.
Avoid claiming reliability from dashboards that do not measure user-visible behavior.
Redact secrets and limit sensitive telemetry retention.

Identify operating procedures for startup, shutdown, common failures, capacity problems, and escalation.
Provide access references without copying credentials into documentation.
Record maintenance and upgrade responsibilities.
Use the GPU guide for accelerator-specific telemetry and the model guides for quality monitoring.

## 10. Recovery and incidents

Define recovery requirements and verify relevant procedures in an authorized environment.
Use the database guide for data restoration claims.
Account for configuration, credentials, artifact availability, and external dependencies during recovery.
Record partial effects and running operations before retrying.

During incidents, operate within existing response authority and stop conditions.
Do not treat urgency as unlimited permission to delete, restore, publish, or spend.
Preserve evidence needed to explain the incident without exposing private information.
After recovery, distinguish restored service from unresolved root causes.
Record corrective work through the core workflow.

## 11. Verification and handover

Verify actual artifact identity, network paths, access controls, health, and important user operations after deployment.
Check capacity and failure recovery where required by the goal.
Record environment, versions, inputs, results, and limitations.
Do not claim production behavior from local-only checks.

At handover, identify operators, deployed versions, configuration sources, runbooks, monitoring, and recovery evidence.
Document known limitations and unfinished work.
Close the release pass only when required checks and reviews are complete or its unresolved outcome is explicitly recorded.
