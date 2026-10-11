# Project evidence

This folder contains accumulated evidence: questions, conditions, methods, observations, interpretations, and limitations.

Read [context-review.md](context-review.md) first for the source assessment and documentation checks.
The supplied sources contain no verified inference runs, inspected model assets, numerical reproductions, or runtime tensor observations.
This revision does not create inference findings from those plans.

Read [hld.md](hld.md) for source observations supporting high-level component boundaries.
Read [lld.md](lld.md) for source observations supporting the low-level baseline contract.
Read HLD evidence first when reviewing both designs.
The initial entries are actual documentation reviews. They do not verify model execution.
Add trace observations, code inspections, and comparison results after actual investigation.
Keep design agreements in the rules files and unfinished decisions in the status files.

Add an investigation record when actual work produces observations.
Use the [record requirements](../project-rules/implementation-rules.md#investigation-records).
Do not create empty records for future investigations.
Retain useful findings after the active task changes.
Correct errors explicitly and identify superseded conclusions when needed.

Keep current checkpoints and blockers in [project-status](../project-status/README.md).
Keep instructions, agreements, and selected technical choices in [project-rules](../project-rules/README.md).
Keep unresolved experiment questions beside their evidence, with a status summary if they block active work.

Return to [AGENTS.md](../AGENTS.md) for routing.
