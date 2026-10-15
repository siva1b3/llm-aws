# Software and AI engineering guide library

This folder contains reusable guidance for planning, designing, building, verifying, and operating software and AI projects.
It contains Markdown guides, not an application or an executable generator.

## Folder map

| Location | Contents and purpose |
| --- | --- |
| [core/](core/README.md) | Shared workflow, architecture, and engineering quality guidance. |
| [individual/](individual/README.md) | Eight domain guides selected according to project scope. |
| [Project records generator](software-ai-project-records-generator.md) | Instructions for preparing project records from a human goal and selected guides. |
| [AGENTS.md](AGENTS.md) | Instructions for agents maintaining or using this library. |

## Reading and use

1. Read the human's actual goal, constraints, and existing project instructions.
2. Read the [software and AI workflow](core/software-ai-development-workflow.md) for record structure, approvals, lifecycle, and completion rules.
3. Read the [architecture guide](core/architecture-and-design-guide.md) and [quality guide](core/engineering-quality-guide.md).
4. Select relevant [domain guides](individual/README.md) and applicable sections within them.
5. Use the [generator](software-ai-project-records-generator.md) when asked to prepare project records under this workflow.

The workflow is **Define → Understand → Plan → Work → Verify → Decide**.
The shared foundation applies across domains; technical guidance depends on the actual work.
Guide availability does not require a particular framework, database, cloud, model, GPU, or deployment platform.

## Responsibilities and boundaries

The human goal defines desired outcomes and constraints.
The core workflow owns project records, approvals, lifecycle, and completion.
Architecture guidance owns design content; quality guidance owns engineering checks and review expectations.
Domain guides supply specialized technical requirements.
The generator owns record preparation, not architecture decisions or implementation.

Adopt one core workflow explicitly when applying this library to a project.
This workflow does not automatically replace the older workflow referenced in the source guides.
Keep project-specific decisions, requirements, evidence, and approvals in that project's records.

Record preparation creates the core-defined starting structure and fills only supported information.
Reserved pass records and the final report initially remain empty.
Preparation does not authorize investigation, design, planning, implementation, experiments, or deployment.
Consult the generator and core workflow for the complete procedure.

## Navigation note

Some existing guide links still assume all guides share one directory.
Use the indexes above to locate files in the current layout: shared guides in `core/`, domain guides in `individual/`.
The generator remains directly in `software_files/`.
Links in these new indexes use the current layout.
