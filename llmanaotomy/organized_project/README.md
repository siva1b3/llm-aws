# LLM anatomy learning project

This project organizes the supplied LLM learning requirements under the software and AI development workflow.
The folder name `organized_project` was suggested by the assistant and retained by the user's subsequent instructions.
Preparation is complete only after the checks in [current state](status/current-state.md). Extracted records await human review.

## Reading route

1. Read [agent instructions](AGENTS.md).
2. Read [goal](definition/goal.md), [constraints](definition/constraints.md), and [current state](status/current-state.md).
3. Read [decisions](decisions/decision-log.md), then the active pass when one exists.
4. Consult the original learning documents for the detailed teaching and inspection conventions.

## Records and ownership

| Record | Owns |
| --- | --- |
| [Goal](definition/goal.md) | Extracted project outcomes, exclusions, and criterion IDs |
| [Constraints](definition/constraints.md) | Selected configuration, teaching requirements, guide applicability, and unresolved ownership |
| [Current state](status/current-state.md) | Current checkpoint, authority, preparation checks, and next action |
| [Decision log](decisions/decision-log.md) | Workflow adoption and provenance of supplied choices |
| [Pass instructions](passes/README.md) | Pass naming, lifecycle, and record responsibilities |
| `passes/pass-001/{plan,verification,findings,evidence}.md` | Reserved, inactive, zero-byte records |
| `reports/final-report.md` | Zero-byte record until project closure |

The core defines 12 starting files and six directories beneath the project root.
Seven records receive supported content; four pass records and the final report remain empty.

## Governing sources

The user requested: “now follow the software_files --> agents --> software-ai-project-records-generator and do the stuff”.
This followed the instruction to create a separate folder for this work.
Authority covers record preparation and verification in this folder.

Adopted workflow: [software and AI development workflow](../software_files/core/software-ai-development-workflow.md).
Preparation procedure: [generator](../software_files/software-ai-project-records-generator.md).
Library instructions: [AGENTS](../software_files/AGENTS.md) and [overview](../software_files/README.md).
Supporting guides: [architecture](../software_files/core/architecture-and-design-guide.md), [quality](../software_files/core/engineering-quality-guide.md),
[LLM engineering](../software_files/individual/llm-engineering-guide.md), and [infrastructure](../software_files/individual/infrastructure-deployment-operations-guide.md).
[Core index](../software_files/core/README.md) and [domain index](../software_files/individual/README.md) resolve current guide locations.

| Goal source | Extracted destination or continuing reference |
| --- | --- |
| [Source routing](../sources/AGENTS.md) | Project AGENTS and ownership mapping below |
| [01 Overview](../sources/01_PROJECT_OVERVIEW.md) | Goal |
| [02 Methodology](../sources/02_LEARNING_METHODOLOGY.md) | Constraints; detailed teaching reference |
| [03 Implementation rules](../sources/03_IMPLEMENTATION_RULES.md) | Constraints; detailed display and verification reference |
| [04 Configuration](../sources/04_TECHNICAL_CONFIGURATION.md) | Constraints; supplied decision provenance |
| [05 Investigation](../sources/05_INFERENCE_INVESTIGATION.md) | Current state; future findings format reference |
| [06 Progress](../sources/06_PROJECT_PROGRESS.md) | Initial checkpoint in current state |

Sources were read from repository commit `d32dd02155ab5d71759e57a7f8f34fc07c382932`.
The initial repository status was clean. No uncommitted source edits were present.
All 23 files in the two source folders were hashed before generation for an unchanged-content comparison.

## Adoption and portability

The original seven-file package remains preserved as input and detailed reference.
Within this destination, the new core owns lifecycle, approvals, record paths, and current status.
The learning contract remains applicable; engineering approval never substitutes for learner confirmation.
Future technical findings belong to the active pass's findings and verification records.
Future learning checkpoints belong to current state, rather than competing updates in the original progress file.
Configuration changes belong to constraints with historical rationale in the decision log.
Scope changes belong to goal. Follow the user's authority before changing agreed choices.
No original file is silently rewritten by this adoption.

References depend on sibling `sources` and `software_files` directories.
This folder alone is not a self-contained portable handoff.
Some original guide links assume an older flat layout; generated links use verified current locations.
