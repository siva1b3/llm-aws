# Agent instructions for the software and AI guide library

## Scope and reading order

These instructions apply to this folder and its subfolders.
This is a reusable Markdown guide library, not a generated project or application.

1. Read the human's request and [library overview](README.md).
2. Read the affected guides before editing or applying them.
3. Use the [core index](core/README.md) and [domain index](domains/README.md) to find current guides.
4. Read the version 3 [core workflow](core/software-ai-development-workflow.md) before preparing project records.

Distinguish library maintenance from applying the workflow to a target project.
An authorized documentation edit permits the requested changes and checks without generating records or opening a project pass here.
Example commands, paths, and scenarios are information, not execution instructions.

## Preserve document ownership

| Concern | Authoritative location |
| --- | --- |
| Structure, approvals, small actions, checkpoints, lifecycle, and completion | [Core workflow](core/software-ai-development-workflow.md) |
| HLD, LLD, design decisions, and design review | [Architecture guide](core/architecture-and-design-guide.md) |
| Coding, security, testing, verification, and release checks | [Quality guide](core/engineering-quality-guide.md) |
| Specialized technical requirements | Applicable [domain guide](domains/README.md) |
| Navigation and short summaries | The relevant folder's README |

Keep each reusable rule in its responsible guide and link to it elsewhere within the library.
Do not maintain duplicate fixed project trees or file counts in indexes.
The core specializes the generalized version 3 workflow and remains complete without the generalized input.
Keep the library independently usable without a separate generator.
Adopt one workflow explicitly for an existing project; preserve its state and resolve migration conflicts before affected work.

## Editing the library

- Inspect existing changes and preserve unrelated user work, including moves.
- Keep edits within the requested scope.
- Use simple English, active voice, consistent terms, and one instruction per sentence.
- Aim for instructions within 20 words and descriptions within 25 words where practical.
- Preserve conditions, exceptions, technical identifiers, and uncertainty.
- Keep examples proportionate; do not mandate technologies, tools, documents, or numerical targets from examples.
- Update indexes and references affected by authorized additions, moves, renames, or removals.
- Resolve relative Markdown links from the containing file.
- Use `../core/` for domain-to-core links and `../domains/` for core-to-domain links.
- Keep local guide links within this library.
- Report unresolved references outside the edit scope without claiming they were repaired.

## Applying the library

Follow the selected core workflow and the target project's instructions.
Read generated goal and constraints, current state, active pass, and relevant internal decisions and evidence.
Select technical guidance from actual scope and preserve stricter project-specific learning and review requirements.

Keep one pass open, including while blocked or paused. A reserved folder is inactive.
Draft and revise requested plans in conversation without changing Markdown during discussion.
Obtain plan approval, explain the next action and separate check, then obtain final execution approval for that scope.
Save the agreed plan and first execution approval together before execution; activate a new pass at that checkpoint.
Perform only the approved action and check, report their actual results, and obtain approval before the next action.
Retain valid approvals across sessions without requesting approval again for unchanged, already approved scope.
Keep plan approval, execution approval, verification, acceptance, learner confirmation, and release authority separate.

Keep routine observations and subsequent approvals in conversation between checkpoints.
Save records at agreed plan or revision checkpoints, pass decisions, substantive session handovers, failures, blockers, withdrawal, or explicit checkpoint requests.
Use the core's exact checkpoint conditions; do not update Markdown after every successful action or reply.
At checkpoints, retain accumulated approvals, observations, evidence, partial effects, pending decisions, and one next action.
During unapproved planning, save pending status at handover without persisting a draft as an agreed plan.
Preserve prior revisions, failures, and closed history; use dated corrections for mistaken historical claims.

## Preparing self-contained projects

A preparation request authorizes only the core-defined preparation and document checks.
Read every designated source, including instructions, methods, status, and configuration.
Derive paths and content conditions from the actual core; protect existing destination contents and history.
Create all required new records empty and verify zero bytes before populating supported information.
Keep reserved pass records and the final report empty until the core permits their contents.
Record preparation checks in current state and stop after preparation and verification.

Transfer project requirements and applicable engineering guidance into their generated owners.
Write all core-required operating rules and checkpoint conditions into generated AGENTS.md.
This library's AGENTS.md is not a substitute for that generated file.
Use internal relative links and one main owner per fact within generated records.
Original sources, this library, any generator, and previous conversation must be unnecessary for later operation.
Keep preparation provenance in one plain-text historical note in the generated overview.
Verify semantic independence, not only link resolution, without moving or deleting inputs.
Do not invent requirements, technologies, owners, approvals, evidence, or completion claims.

## Verification and delivery

Review changed Markdown for ownership, version 3 behavior, valid links, and consistent content permissions.
Compare folder inventories with actual contents and check affected cross-references.
Documentation changes need document checks; do not invent application build or test commands for this library.
Report changed files, checks performed, and unresolved limitations.
