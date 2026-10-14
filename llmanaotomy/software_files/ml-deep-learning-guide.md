# ML and deep learning guide

## 1. Purpose and authority

Use this guide for machine learning (ML), deep learning (DL), training, fine-tuning, evaluation, and model lifecycle work.
It owns data and experiment methodology, model-quality assessment, and training reproducibility requirements.
Follow the [core workflow](software-ai-development-workflow.md), [architecture guide](architecture-and-design-guide.md), and [quality guide](engineering-quality-guide.md).
It does not authorize data acquisition, compute spending, or model deployment.

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

## 9. Model delivery and monitoring

Package the model with required preprocessing, metadata, configuration, and compatibility information.
Identify code, dataset, model, and environment versions supporting the release.
Check inference correctness, latency, throughput, and resource use under representative conditions.
Use the infrastructure guide for release authority, rollout, recovery, and destination checks.

Define monitoring for input changes, quality signals, failures, and operational health where feasible.
Distinguish data drift from demonstrated quality degradation.
Define review and retraining triggers. Monitoring does not automatically authorize retraining or redeployment.
Preserve rollback or fallback artifacts according to project requirements.

## 10. Completion evidence

Report the task, data scope, baseline, experiment references, evaluation results, and limitations.
Identify accepted and unmet criteria separately.
State whether the output is an experiment, candidate model, validated artifact, or deployed system.
Do not present a saved checkpoint as a verified production model.
Maintain design, data, and operating records through the core workflow.
