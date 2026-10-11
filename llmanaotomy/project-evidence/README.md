# Project evidence

This folder contains accumulated observations, experiment methods, results, explanations, and limitations.

Read the relevant indexed record first. Read the [investigation recording instructions](../project-rules/project-guide.md#investigation-records) before a model investigation.
The [project guide](../project-rules/project-guide.md) defines scope, progressive inspection, displays, and verification gates.

Active blockers and next actions belong in [project status](../project-status/README.md). Governing decisions belong in [project rules](../project-rules/README.md).

## Records

| Record | Evidence purpose |
| --- | --- |
| [Source comparison and context corrections](context-reorganization.md) | Coverage of the six supplied sources, corrections, checks, and limitations |

No completed inference experiment is implied by this index.

## Recording requirements

Create records only after actual work produces observations. Do not create empty records for future model operations.

A coherent record contains:

1. The question addressed.
2. Inputs, conditions, versions, and relevant asset locations.
3. Method and exact commands or scripts.
4. Observations and bounded output.
5. Interpretation and supporting comparisons.
6. Limitations and unresolved questions.

Label hypotheses, expectations, and illustrative examples. An unanswered question is not an observed finding.
Keep experiment questions beside their evidence when readers need that context. Summarize blocking questions in current progress.

Record actual environment details, model revisions, cache paths, dependency versions, dtype/device, and baseline reproduction instructions with their evidence.
Record the exact raw input and any chat template or special-token preparation.
Record actual cache settings and inference mode when they affect interpretation.
Retain useful records after active work changes. Explain corrections and mark earlier conclusions as superseded when applicable.
Add explanations to raw logs so readers can understand their significance.

See the project guide for [component fields](../project-rules/project-guide.md#component-record-format), [asset and comparison fields](../project-rules/project-guide.md#assets-observations-and-comparisons), and the [first baseline record](../project-rules/project-guide.md#first-baseline-record).

