# Database engineering guide

## 1. Purpose and authority

Use this guide for persistent application data, relational databases, document stores, and other selected storage systems.
It owns data modeling, integrity, query behavior, migrations, retention, backup, and restoration expectations.
Follow the [core workflow](software-ai-development-workflow.md), [architecture guide](architecture-and-design-guide.md), and [quality guide](engineering-quality-guide.md).
This guide does not select a database or authorize access to production data.

## 2. Related guides

- [Backend and APIs](backend-api-development-guide.md): application operations and data access contracts.
- [Distributed systems](distributed-systems-microservices-guide.md): replication, cross-service consistency, and event-based data movement.
- [ML and deep learning](ml-deep-learning-guide.md): dataset lineage and evaluation data boundaries.
- [LLM engineering](llm-engineering-guide.md): retrieval indexes and source-level access rules.
- [Infrastructure](infrastructure-deployment-operations-guide.md): hosting, identity, storage, monitoring, and recovery execution.

Use relevant sections only. A cache or derived index may have different durability needs from authoritative data.
Record which store is authoritative and which stores can be rebuilt.

## 3. Requirements and HLD

Identify data owners, consumers, sensitivity, expected volume, and query patterns.
Record integrity, availability, consistency, retention, and recovery requirements.
Define acceptable data loss and recovery time when they matter. Do not invent targets.
Identify geographic, legal, or organizational restrictions supplied by the project.

Explain storage choices against the workload and operating constraints.
Describe data flow, service ownership, replication, derived stores, and trust boundaries.
Do not select a storage engine solely because it is popular.
Keep business data ownership clear when several services use related information.

## 4. Data model and LLD

Define entities or documents, identifiers, relationships, types, and required fields.
Record uniqueness, referential, domain, and lifecycle invariants.
Specify null and missing-value meanings.
Choose units, timestamps, precision, and identifier formats deliberately.
Enforce constraints at the authoritative layer where the selected store supports them.

Describe indexes in relation to actual access patterns.
Explain denormalization, duplication, or partitioning when used.
Define how duplicated data is updated and reconciled.
Keep schema descriptions synchronized with executable migrations and actual behavior.

## 5. Transactions and concurrency

Identify operations that must succeed or fail together.
Define isolation expectations and behavior under concurrent updates.
Account for lost updates, duplicate inserts, write conflicts, and retryable failures where relevant.
Specify lock scope and ordering when locks are needed.
Bound transactions and avoid holding locks across slow external calls without a justified design.

Do not assume a local database transaction includes messages or external API effects.
Use the distributed systems guide when coordination spans independent systems.
Test invariants under the concurrency patterns that matter to the application.

## 6. Queries and capacity

Use representative data distributions and query parameters for performance checks.
Inspect execution behavior where needed to explain slow or expensive access.
Bound result sets, connection use, and expensive operations.
Consider pagination stability when data changes between requests.
Assess index benefits against write cost and storage overhead.

Record growth assumptions and capacity limits.
Do not infer production performance from tiny fixtures alone.
Measure before adding caches, replicas, or partitioning that increases operating complexity.

## 7. Migrations and backfills

Treat schema and data changes as versioned delivery work.
For a material migration, identify:

1. Starting schema and supported application versions.
2. Intended schema and data transformation.
3. Compatibility during deployment and any mixed-version period.
4. Locking, execution time, storage, and operational effects.
5. Backup or other recovery prerequisites.
6. Interruption, restart, and duplicate-execution behavior.
7. Validation of row counts, constraints, and business meaning where applicable.
8. Rollback, forward repair, or restoration procedure and its limits.

Use staged changes when incompatible consumers cannot change simultaneously.
Separate expanding a schema from removing old structures when compatibility requires it.
Bound and checkpoint large backfills where needed.
Do not claim rollback is safe when a transformation loses information.
Obtain the applicable execution authority before destructive or production-changing actions.

## 8. Backup and restoration

Define what is backed up, how often, where it is stored, and who can restore it.
Include encryption, access, retention, and dependency requirements.
Identify whether the required recovery needs a consistent snapshot, transaction history, or other mechanisms.

Verify restoration in an authorized environment when recovery claims require it.
Check restored application behavior and data integrity, not only successful file extraction.
Record achieved recovery time and data position against the agreed targets.
A successful backup job does not establish successful restoration.
Do not treat replication alone as protection from accidental deletion or corruption.

## 9. Security and data lifecycle

Use least-privilege identities and appropriate separation of administrative access.
Protect connections and stored sensitive data according to project requirements.
Keep credentials and private production records out of source, fixtures, and evidence.
Use approved masking or synthetic data for development where needed.

Define retention and deletion across authoritative stores, replicas, caches, indexes, and backups as applicable.
State limitations where deletion cannot take effect immediately.
Log sensitive operations without unnecessarily reproducing the sensitive data.

## 10. Verification and maintenance

Check schema constraints, query correctness, transactions, and meaningful failure cases.
Test representative migrations and recovery paths before approved production use when risk requires it.
Monitor capacity, query latency, lock contention, replication lag, and backup failures where relevant.
Record exact database versions and relevant configuration for compatibility-sensitive checks.
Consult current vendor documentation for engine-specific guarantees and migration behavior.

Preserve migration history and evidence of significant changes.
Update application contracts and operating instructions when data behavior changes.
Report unresolved integrity or recovery gaps before acceptance through the core workflow.
