# Software and AI engineering guide library

This folder contains reusable guidance for planning, designing, building, verifying, and operating software and AI projects.
This is a self-contained Markdown guide library; no separate project-records generator is required.

## Folder map

| Location | Contents and purpose |
| --- | --- |
| [core/](core/README.md) | Shared workflow, architecture, and engineering quality guidance. |
| [domains/](domains/README.md) | Eight domain guides selected according to project scope. |
| [AGENTS.md](AGENTS.md) | Instructions for agents maintaining or using this library. |

## Reading and use

1. Read the human's actual goal, constraints, and existing project instructions.
2. Read the [software and AI workflow](core/software-ai-development-workflow.md) for record structure, approvals, lifecycle, and completion rules.
3. Read the [architecture guide](core/architecture-and-design-guide.md) and [quality guide](core/engineering-quality-guide.md).
4. Select relevant [domain guides](domains/README.md) and applicable sections within them.
5. Prepare project records directly from the core workflow when requested.

The workflow is **Define → Understand → Plan → Work → Verify → Decide**.
The shared foundation applies across domains; technical guidance depends on the actual work.
Guide availability does not require a particular framework, database, cloud, model, GPU, or deployment platform.

## Responsibilities and boundaries

The human goal defines desired outcomes and constraints.
The core workflow owns project records, approvals, lifecycle, and completion.
Architecture guidance owns design content; quality guidance owns engineering checks and review expectations.
Domain guides supply specialized technical requirements.

Adopt one core workflow explicitly when applying this library to a project.
This workflow does not automatically replace the older workflow referenced in the source guides.
Keep project-specific decisions, requirements, evidence, and approvals in that project's records.

Record preparation creates the core-defined starting structure and fills only supported information.
Reserved pass records and the final report initially remain empty.
Preparation does not authorize investigation, design, planning, implementation, experiments, or deployment.
Consult the core workflow for preparation rules, record responsibilities, and required operating instructions.

## Portability

Keep this folder's core and domain guides together when copying the library.
All local guide links resolve within this folder.
External web references provide supporting material; they do not replace the library's workflow.
