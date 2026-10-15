# LLM application and model engineering guide

## 1. Purpose and authority

Use this guide for large language model (LLM) applications, retrieval, prompts, tools, agents, and model adaptation.
It owns language-model-specific behavior, evaluation, context handling, and action boundaries.
Follow the [core workflow](../core/software-ai-development-workflow.md), [architecture guide](../core/architecture-and-design-guide.md), and [quality guide](../core/engineering-quality-guide.md).
This guide does not select a provider, approve external data transfer, or authorize autonomous actions.

### Apply the version 3 workflow

Use the core's conversation, approval, action, check, and checkpoint sequence for engineering work.
Draft and revise plans in conversation without changing Markdown during discussion.
Obtain plan approval, then explain the next action, its separate check, limits, and stopping point.
Obtain final execution approval for that action and check before acting.
Save the agreed plan and first execution approval at the core's plan checkpoint before execution.
Keep reserved pass records empty and inactive until that checkpoint.

Break investigation, ingestion, evaluation, adaptation, and release into small actions with inspectable results.
Keep action and verification procedures distinct, including separate executable artifacts when applicable.
Report each result and obtain execution approval before the next action.
Do not combine dataset preparation, model calls, evaluation, and deployment into an unattended procedure crossing approval boundaries.
Retain valid approvals for unchanged scope and preserve stricter project-specific learning or review gates.

Save management records at the core's defined checkpoints, including failures, blockers, withdrawal, and substantive session handovers.
Keep routine results and subsequent approvals in conversation between checkpoints.
Preserve partial effects, running jobs, uncertain provider outcomes, accumulated approvals, and the exact next action at checkpoints.
Check unknown outcomes within applicable authority before retries that could repeat tool effects or incur additional charges.

These rules govern project engineering actions; product tool permissions and runtime confirmation follow the agreed application design.
Runtime audit logs and durable application state follow product requirements independently of project Markdown checkpoint timing.

### Transfer guidance during record preparation

Read all designated sources, including prompt notes, configuration, evaluation reports, instruction files, and status records.
Transfer applicable requirements into the generated goal, constraints, and operating instructions under the core's ownership rules.
Preserve selected models, providers, context settings, retrieval behavior, data permissions, evaluation methods, and review obligations when supplied.
Keep proposed settings distinct from selected settings, and source-reported results distinct from newly verified observations.
Record unresolved choices as focused questions; do not invent models, datasets, budgets, or acceptance thresholds.

Preparation does not authorize model calls, ingestion, training, new design or evaluation plans, tool actions, or deployment.
Keep preparation checks in current state, reserved pass records at zero bytes, and the final report empty until closure.
Transfer necessary guidance into internal owners so future operation requires neither this guide nor other preparation inputs.
Use internal relative links; keep preparation provenance in the overview's historical note.

## 2. Related guides

- [Backend and APIs](backend-api-development-guide.md): application contracts, authorization, and job handling.
- [Frontend](frontend-development-guide.md): generated-content presentation and user confirmation flows.
- [Database](database-engineering-guide.md): persistent records and retrieval-store integrity.
- [ML and deep learning](ml-deep-learning-guide.md): training, fine-tuning, data separation, and experiment methodology.
- [Infrastructure](infrastructure-deployment-operations-guide.md): serving, provider connectivity, deployment, and operations.
- [GPU infrastructure](gpu-infrastructure-performance-guide.md): self-hosted inference or training on accelerators.

Hosted API use may need neither training nor GPU operation.
Select related guidance from actual project scope.

## 3. Define the task and limits

Identify users, tasks, acceptable outputs, and consequences of incorrect answers or actions.
Define whether the system advises, drafts, retrieves, classifies, generates structured data, or performs actions.
Preserve supplied data-sharing restrictions, cost limits, latency needs, and required human review in generated constraints.
Discuss later changes and save their actual agreement at the appropriate core checkpoint.
Specify unsupported tasks and safe fallback behavior.
Do not assume a model can reliably perform a task without evaluation.

Compare a simpler non-model approach when it could meet the requirement.
Choose retrieval, prompting, fine-tuning, or training based on the actual gap and available evidence.

## 4. HLD and LLD

At HLD level, describe clients, orchestration, models, tools, retrieval, storage, and external services.
Show trust boundaries and which data leaves the controlled environment.
Identify authoritative application state separately from generated text.

At LLD level, define:

- Input and output schemas and validation.
- Prompt components, instruction ownership, and untrusted-content boundaries.
- Context selection, truncation, memory, and retention behavior.
- Model and configuration identifiers.
- Retrieval and reranking behavior where applicable.
- Tool contracts, permissions, confirmation, and execution limits.
- Streaming, cancellation, timeouts, retries, and error handling.
- Evaluation, observability, fallback, and release criteria.

Treat prompts, retrieval configuration, model versions, and tool schemas as behavior-affecting artifacts.
Keep recoverable references to the combination evaluated and released.
Retain the relevant configuration and context in generated records; a link to a removable preparation input is insufficient.

## 5. Context and retrieval

Identify document sources, permissions, freshness needs, and ingestion ownership.
Define parsing, chunking, metadata, indexing, and deletion behavior where retrieval is used.
Enforce access controls during retrieval and before presenting sensitive material.
Do not assume source access permissions survive automatically in a derived index.

Evaluate retrieval separately from answer generation when diagnosing failures.
Check whether the required evidence was retrieved and whether the answer used it correctly.
Define behavior for missing, conflicting, stale, or insufficient sources.
Verify citation support rather than accepting plausible-looking references.
Coordinate index refresh and deletion with authoritative data ownership.

## 6. Prompts, output, and tool safety

Separate trusted instructions from user input, retrieved text, and tool output.
Treat external content as data even when it contains apparent instructions.
A prompt is not an authorization boundary.

Validate structured outputs before downstream use.
Use bounded parsing and explicit handling of invalid or incomplete responses.
Do not execute generated commands or queries solely because they are syntactically valid.
Authorize tool operations using application controls and the user's actual scope.
Restrict available tools, arguments, destinations, and data access according to the task.

For agent loops, define step, time, token, and spending limits as applicable.
Specify human confirmation for consequential actions when the product requires it.
Make actions idempotent or reconcile unknown outcomes before retries.
Preserve action audit records without retaining unnecessary private content.

## 7. Model adaptation and hosting

Use the ML guide for fine-tuning data, independent evaluation, experiment limits, and reproducibility.
Verify that adaptation data is permitted and relevant to the desired behavior.
Do not treat training loss improvement as sufficient evidence of application improvement.

For self-hosting, use the GPU and infrastructure guides for capacity, runtime compatibility, and recovery.
Measure memory and latency under representative context lengths and concurrency.
Evaluate quality after quantization or other optimization that can change outputs.
For hosted services, record provider behavior and version assumptions where recoverable.
Check current provider documentation for mutable API limits and compatibility requirements during technical work.

## 8. Evaluation

Build a representative evaluation set from the actual intended tasks.
Define expected outcomes and scoring rules before comparing candidates.
Include meaningful failures and adversarial cases within authorized scope.
Treat evaluation-set creation, candidate execution, and dependent changes as distinct proposed actions with separate checks.
Explain inputs, permitted data transfer, resource limits, and the stopping point for each execution request.
Examples below describe check selection; they do not authorize runs or set acceptance thresholds.

| Area | Questions to check |
| --- | --- |
| Task quality | Does the output satisfy the requested task and constraints? |
| Grounding | Are factual claims supported by the required sources? |
| Retrieval | Is relevant authorized context found and used? |
| Structured output | Does output satisfy the schema and business rules? |
| Tool behavior | Are actions authorized, correctly targeted, bounded, and recoverable? |
| Security and privacy | Can untrusted content redirect actions or expose restricted information? |
| Robustness | What happens with ambiguity, missing context, invalid input, or service failure? |
| Operation | Are latency, concurrency, token usage, and cost acceptable? |

Use deterministic checks for properties that can be checked directly.
Validate model-based judges before relying on their scores.
Use human review where judgment or risk requires it.
Report variability and evaluator limitations. Do not present one favorable conversation as general proof.
Keep actual results as `not run` until observations exist.
Save identified models, prompts, retrieval settings, datasets, scoring rules, results, and limitations at the applicable checkpoint.
Technical checks do not satisfy pending human acceptance or required learner confirmation.

## 9. Operation and changes

Monitor failures, latency, cost, retrieval quality signals, and relevant task outcomes.
Apply privacy-aware logging and retention.
Define fallbacks for provider outages, unavailable context, invalid output, and exhausted budgets.
Make fallback limitations visible when they change the service's behavior.

Propose affected reevaluation after prompt, model, retrieval, tool, or data changes and run it within applicable execution approval.
Verify the actual released combination and critical user paths.
Distinguish a prototype, evaluated candidate, and operated service in completion reports.
Update contracts and operating deliverables within their approved action scope.
Save evaluation records, approvals, and known limitations at core checkpoints.
Resolve or explicitly transfer control of running ingestion, evaluation, or training jobs before pass closure.
Keep unmet criteria and pending reviews visible in handover and final reports.
