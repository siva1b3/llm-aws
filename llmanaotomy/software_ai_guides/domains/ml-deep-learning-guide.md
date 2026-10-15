# ML and deep learning guide

## 1. Purpose and authority

Use this guide for machine learning (ML), deep learning (DL), training, fine-tuning, evaluation, and model lifecycle work.
It owns data and experiment methodology, model-quality assessment, and training reproducibility requirements.
Follow the [core workflow](../core/software-ai-development-workflow.md), [architecture guide](../core/architecture-and-design-guide.md), and [quality guide](../core/engineering-quality-guide.md).
It does not authorize data acquisition, compute spending, or model deployment.

### Apply the version 3 workflow

Use the core's conversation, approval, action, check, and checkpoint sequence for engineering work.
Draft and revise plans in conversation without changing Markdown during discussion.
Obtain plan approval, then explain the next action, its separate check, limits, and stopping point.
Obtain final execution approval for that action and check before acting.
Save the agreed plan and first execution approval at the core's plan checkpoint before execution.
Keep reserved pass records empty and inactive until that checkpoint.

Divide data preparation, baseline assessment, training, evaluation, and model delivery into small actions with inspectable results.
Keep action and verification procedures distinct, including separate executable artifacts when applicable.
Report each result and obtain execution approval before dependent work or the next action.
Do not create an unattended procedure that crosses action approval or required human review boundaries.
Retain valid approvals for unchanged scope and preserve stricter project-specific learning or review gates.

Save management records at the core's defined checkpoints, including failures, blockers, withdrawal, and substantive session handovers.
Keep routine results and subsequent approvals in conversation between checkpoints.
Preserve partial effects, running operations, unknown outcomes, accumulated approvals, and one next action at checkpoints.
Check unknown outcomes within applicable authority before retries that could duplicate effects.
Obtain new approval for recovery unless the approved scope explicitly covers that recovery branch and its check.

Training checkpoints and experiment telemetry follow the experiment design independently of project Markdown checkpoint timing.

### Transfer guidance during record preparation

Read all designated sources, including dataset notes, experiment configurations, training reports, instruction files, and status records.
Transfer applicable requirements into generated definitions and operating instructions under the core's ownership rules.
Preserve supplied data permissions, splits, transformations, model settings, baselines, evaluation methods, compute limits, and review obligations.
Keep selected configuration distinct from observed configuration, and source-reported results distinct from newly verified facts.
Record unresolved choices as focused questions; do not invent datasets, models, metrics, thresholds, budgets, or reviewers.

Preparation does not authorize data acquisition, dataset inspection beyond supplied material, new experiment plans, training, evaluation runs, or deployment.
Keep preparation checks in current state, reserved pass records at zero bytes, and the final report empty until closure.
Transfer necessary guidance into internal owners so future operation requires neither this guide nor other preparation inputs.
Use internal relative links; keep preparation provenance in the overview's historical note.

## 2. Related guides

- [Database](database-engineering-guide.md): persistent data integrity, access, and retention.
- [LLM engineering](llm-engineering-guide.md): language-model applications, retrieval, prompts, and tool behavior.
- [GPU infrastructure](gpu-infrastructure-performance-guide.md): accelerator compatibility, capacity, and measurements.
- [Distributed systems](distributed-systems-microservices-guide.md): coordination and partial-failure semantics.
- [Infrastructure](infrastructure-deployment-operations-guide.md): workload execution, serving, release, and operations.

Use GPU guidance only when accelerators are relevant.
Using a hosted model API does not automatically require a training pipeline.

## 3. Define the learning problem

Identify the user decision or task the model supports.
Define inputs, outputs, operating conditions, and consequences of errors.
Establish a baseline and measurable acceptance criteria tied to the actual use case.
Separate model-quality targets from software correctness and operational targets.
Do not invent accuracy, fairness, latency, or spending thresholds.

Record whether the work explores feasibility, compares models, trains a candidate, or prepares a release.
A useful experiment may establish that an approach does not meet the goal.
Do not promise that training will produce a successful model.

## 4. Data design and quality

Identify sources, permitted use, ownership, sensitivity, and collection limitations.
Record dataset versions and transformations needed to reconstruct the experiment.
Inspect schema, missing values, duplicates, label quality, and relevant distributions.
Define preprocessing and feature computation consistently between training and inference.

Choose splits appropriate to the data and deployment question.
Account for entities, time, groups, and duplicates where they can leak information across splits.
Use validation data for tuning. Reserve held-out test data for independent evaluation.
Do not repeatedly optimize against a held-out test set while continuing to call it independent.
Record known coverage gaps and label uncertainty.

## 5. HLD and LLD

At HLD level, map ingestion, preparation, training, evaluation, artifact storage, serving, and feedback.
Identify data and model ownership, trust boundaries, and external dependencies.
Explain where batch and online processing differ.

At LLD level, define:

- Input schemas, features, transforms, labels, and output interpretation.
- Model configuration, loss, optimization, and relevant hyperparameters.
- Checkpoint contents and resume semantics.
- Dataset, code, environment, and artifact references.
- Resource limits, stopping conditions, and failed-run handling.
- Evaluation metrics, baseline, slices, and comparison procedure.

Use established implementations where they meet requirements.
Do not add model complexity without assessing measurable benefit and operating cost.

## 6. Experiment control

Define the hypothesis or comparison before launching a material experiment.
Record the approved search space, run limit, compute budget, and stopping conditions.
Discuss these limits in conversation and save the agreed scope at the core's plan checkpoint before execution.
An approved search space does not authorize every later experiment action.
Name the run or bounded experiment action and its separate check when requesting execution approval.
Track successful, failed, and interrupted runs when they affect conclusions.
Separate hyperparameter selection from final performance estimation.

Record random seeds and deterministic settings where useful.
State nondeterminism and environment limitations honestly.
A seed alone does not guarantee identical results across hardware and software conditions.
Use repeated runs or uncertainty estimates when the conclusion requires them.

Preserve run identifiers, logs, metrics, and recoverable artifacts needed for decisions.
Do not retain unnecessary private samples in experiment logs.

## 7. Training and recovery

Validate the data and a bounded execution path before expensive approved runs where practical.
Monitor loss behavior, invalid numerical values, memory, throughput, and relevant resource use.
Define early stopping or abort conditions when they affect cost or validity.
Check checkpoint integrity and restoration when resumability is required.
Training checkpoints preserve model and optimizer state; core workflow checkpoints preserve project agreements, observations, and next actions.
Do not delay required training-state persistence until a project Markdown checkpoint.

For distributed training, record worker count, data partitioning, synchronization, and failure behavior.
Coordinate GPU memory, precision, communication, and resource allocation with the GPU guide.
Do not resume a run under changed data or configuration while presenting it as an unchanged experiment.
Record whether optimizer, scheduler, and random-state restoration are needed for the intended resume claim.

## 8. Evaluation

Choose metrics that match error costs and deployment behavior.
Inspect relevant subgroups, rare cases, and out-of-distribution conditions where required.
Compare against the baseline under comparable conditions.
Report variation, sample sizes, uncertainty, and material limitations.
Include calibration or threshold selection when the application relies on probabilities or operating thresholds.

Inspect representative failures rather than relying only on an aggregate score.
Keep human review separate from measured performance.
Do not call a benchmark score proof of fitness for every deployment population.
Use the LLM guide for language-model-specific evaluation and generated-output risks.
Define expected checks before execution and keep actual results as `not run` until observations exist.
Save dataset, code, model, configuration, run identifiers, results, and limitations at the applicable core checkpoint.
Technical results do not satisfy pending human acceptance or required learner confirmation.

## 9. Model delivery and monitoring

Package the model with required preprocessing, metadata, configuration, and compatibility information.
Identify code, dataset, model, and environment versions supporting the release.
Check inference correctness, latency, throughput, and resource use under representative conditions.
Use the core workflow for release authority and the infrastructure guide for rollout, recovery, and destination checks.

Define monitoring for input changes, quality signals, failures, and operational health where feasible.
Distinguish data drift from demonstrated quality degradation.
Define review and retraining triggers. Monitoring does not automatically authorize retraining or redeployment.
Preserve rollback or fallback artifacts according to project requirements.

## 10. Completion evidence

Report the task, data scope, baseline, experiment references, evaluation results, and limitations.
Identify accepted and unmet criteria separately.
State whether the output is an experiment, candidate model, validated artifact, or deployed system.
Do not present a saved checkpoint as a verified production model.
Update design, data, and operating deliverables within their approved action scope.
Save accumulated approvals, experiment results, and limitations at core checkpoints.
Resolve or explicitly transfer control of running training and evaluation jobs before pass closure.
