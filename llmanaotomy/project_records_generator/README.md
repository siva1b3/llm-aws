# Software and AI project records generator

This folder contains a Markdown procedure for preparing project records from a supplied human goal.
It is guidance for an assistant, not an executable program.

## Contents

| File | Purpose |
| --- | --- |
| [Generator](software-ai-project-records-generator.md) | Source selection, extraction, record preparation, validation, and delivery. |
| [AGENTS.md](AGENTS.md) | Instructions for maintaining and applying this procedure. |

## Inputs and use

1. Read the human's goal, destination, constraints, and existing project instructions.
2. Read the [agent instructions](AGENTS.md) and [generator](software-ai-project-records-generator.md).
3. Read the [core workflow](../software_ai_guides/core/software-ai-development-workflow.md), which owns record structure, approvals, and lifecycle.
4. Read the [architecture](../software_ai_guides/core/architecture-and-design-guide.md) and [quality](../software_ai_guides/core/engineering-quality-guide.md) guides.
5. Select applicable [domain guidance](../software_ai_guides/domains/README.md) from actual project scope.
6. Prepare records in the intended project destination, verify them, and stop for human review.

The generator depends on the sibling [software_ai_guides/](../software_ai_guides/README.md) library.
The library works independently and does not require this folder.
When relocating the generator, retain that sibling layout or update its source links.
Derive the starting structure from the actual core workflow; do not maintain a duplicate tree here.

Preparation does not authorize investigation, design, pass planning, implementation, experiments, or deployment.
Preserve existing destination contents and keep reserved pass records and the final report initially at zero bytes.
Keep generated projects outside both reusable source folders.
