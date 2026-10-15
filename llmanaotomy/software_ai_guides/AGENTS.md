# Agent instructions for the software and AI guide library

## Scope and reading order

These instructions apply to `software_ai_guides/` and its subfolders.
This directory is a reusable Markdown guide library, not a generated project or application.

1. Read the user's request and [library overview](README.md).
2. Read the affected guides before editing or applying them.
3. Consult the [core index](core/README.md) and [domain index](domains/README.md) for current locations.
4. Use the core workflow directly when preparing project records.

Distinguish library maintenance from applying the workflow to a target project.
An authorized documentation edit does not require generating project records or starting an engineering pass in this library.
Do not interpret example commands, paths, or project scenarios as instructions to execute them.

## Preserve document ownership

| Concern | Authoritative location |
| --- | --- |
| Project record structure, approvals, lifecycle, completion | [Core workflow](core/software-ai-development-workflow.md) |
| HLD, LLD, design decisions, design review | [Architecture guide](core/architecture-and-design-guide.md) |
| Coding, security, testing, verification, release checks | [Quality guide](core/engineering-quality-guide.md) |
| Specialized technical requirements | Relevant guide in [domains/](domains/README.md) |
| Navigation and short summaries | The README in the relevant folder |

Keep each rule in its responsible guide and link to it elsewhere.
Do not maintain a duplicate fixed project tree or file count in indexes.
Keep project-specific requirements, technologies, approvals, and evidence in the target project's records.
Do not silently combine this workflow with an older workflow governing an existing project.

## Editing the library

- Inspect existing changes and preserve unrelated user work, including file moves.
- Keep edits within the requested scope.
- Use simple English, active voice, consistent terminology, and one instruction per sentence.
- Aim for instruction sentences of at most 20 words and descriptions of at most 25 words.
- Preserve conditions, exceptions, technical identifiers, and uncertainty when simplifying text.
- Keep guidance proportionate; examples do not mandate technologies, documents, tools, or numeric targets.
- Update affected indexes and references when authorized changes add, rename, move, or remove guides.
- Resolve relative Markdown links from the containing file's directory.
- Use `../core/` for domain-to-core links and `../domains/` for core-to-domain links.
- Keep local guide links within this library so it remains independently usable.
- Report pre-existing broken references outside the edit scope without claiming they were repaired.

## Applying the library to projects

Follow the selected core workflow and the target project's instructions.
Read its goal, constraints, current state, active pass, and relevant decisions before project work.
Select domain guidance from actual scope, not merely because a file exists.

Under this workflow, keep one pass active, including while paused or blocked.
Respect separate planning and execution authority and its recorded scope and revision.
Do not request approval again for unchanged work already authorized.
Maintain records before dependent work and preserve history, failed results, and unrelated changes.
Keep proposed work, performed actions, verification, acceptance, and release distinct.
Use the core workflow's full operating rules when generating a project's own `AGENTS.md`.
This library's instructions are not a substitute for that project-specific file.

A preparation request authorizes only the preparation defined by the core workflow.
Derive required records from the actual core guide and preserve existing destination contents.
Keep reserved pass records and the final report initially at zero bytes, as the core requires.
Record preparation checks in current state and stop after preparation and verification.
Do not invent requirements, approval owners, results, or completion claims.

## Verification and delivery

Review changed Markdown for accurate ownership, consistent terminology, and valid local link targets.
Compare inventories with actual folder contents.
Check affected cross-references when changing a shared rule or path.
Documentation-only changes need document checks; do not invent application build or test commands for this library.
Report created or changed files, checks performed, and any unresolved reference limitations.
