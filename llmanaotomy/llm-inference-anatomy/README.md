# LLM inference anatomy

Learn one complete pretrained LLM forward pass through real code, observations, and progressive explanation.
The name llm-inference-anatomy is derived from the supplied goal.
This is a learning project, not a production inference service.

## Reading route

Read [goal](definition/goal.md), [constraints](definition/constraints.md), [current state](status/current-state.md), and [operating instructions](AGENTS.md).
Then read the [planned roadmap](passes/planned_roadmap.md) for the full journey from EC2 setup to the learning goal, followed by the active pass and relevant internal evidence.
Work through the roadmap one step at a time. Each pass plan covers only its immediate work; execution and learner confirmation retain their separate approval gates.
Use current state for the latest position and the roadmap for the planned workflow.

## Folder map

| Path | Owner |
| --- | --- |
| [AGENTS.md](AGENTS.md) | Operating rules, approvals, checkpoints, completion. |
| [definition/goal.md](definition/goal.md) | Outcomes, scope, exclusions, deliverables, criteria. |
| [definition/constraints.md](definition/constraints.md) | Teaching, inspection, configuration, engineering requirements, unresolved constraints. |
| [status/current-state.md](status/current-state.md) | Current position, learning boundary, and next action. |
| [decisions/decision-log.md](decisions/decision-log.md) | Significant choices. |
| [passes/README.md](passes/README.md) | Pass fields and lifecycle. |
| [passes/planned_roadmap.md](passes/planned_roadmap.md) | Overall workflow. |
| [passes/pass-001/plan.md](passes/pass-001/plan.md) | Immediate Python environment plan. |
| [passes/pass-001/verification.md](passes/pass-001/verification.md) | Reserved checks. |
| [passes/pass-001/findings.md](passes/pass-001/findings.md) | Reserved technical findings. |
| [passes/pass-001/evidence.md](passes/pass-001/evidence.md) | Reserved evidence index. |
| [reports/final-report.md](reports/final-report.md) | Final or authorized early-closure report. |

Pass-001 verification, findings, evidence, and the final report remain empty until needed.
Planning and execution retain separate approvals.

## Historical preparation provenance

Historical inputs, unnecessary for future operation: sources/AGENTS.md; 01_PROJECT_OVERVIEW.md; 02_LEARNING_METHODOLOGY.md; 03_IMPLEMENTATION_RULES.md; 04_TECHNICAL_CONFIGURATION.md; 05_INFERENCE_INVESTIGATION.md; 06_PROJECT_PROGRESS.md.
Procedure: software-ai-project-records-generator.md, declared version 3.
Workflow: software-ai-development-workflow.md, declared version 3.
Supporting inputs: architecture-and-design-guide.md; engineering-quality-guide.md; llm-engineering-guide.md; ml-deep-learning-guide.md; infrastructure-deployment-operations-guide.md.
Applicable content is integrated into this folder. Original inputs were preserved.
The selected model repository remains a genuine external project dependency.
