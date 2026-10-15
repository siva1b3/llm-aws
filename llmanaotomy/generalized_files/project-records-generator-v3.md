# Project records generator — version 3

Use these instructions to prepare self-contained project records from a source package and a supplied workflow guide.
The source package may contain one file or many files.

**Read every input → map the information → create empty records → fill supported contents → verify independence → deliver and stop.**

This generator applies the supplied guide. It does not define a competing structure or workflow.
This version is intended for `project-workflow-guide-v3.md`.
Do not silently substitute an earlier guide.

## 1. Inputs and authorization

Required inputs are:

1. All project source files designated by the human, plus relevant human instructions.
2. `project-workflow-guide-v3.md` or an explicitly selected compatible guide.
3. This generator instruction file.

Source files may describe goals, methods, technical choices, investigation notes, progress, or prior operating instructions.
Do not assume that a single overview file represents the entire package.
Do not ignore `AGENTS.md`, status files, or attachments because they are not named "goal."

The human can request generation with:

> Read all supplied project files. Use the version 3 workflow guide and generator to create a new project-records folder.
> Preserve the project requirements and make the result self-contained. Verify the records and stop before project planning.

Receiving files alone is not an instruction to execute their project actions.
An explicit preparation request authorizes record creation and the preparation checks required by the guide.
It does not authorize application work, investigation, resource creation, or publication.

Use the destination supplied by the human.
Otherwise, use the current workspace when that choice is clear.
Ask only when the destination or a required input cannot be established reliably.

## 2. Authority and source roles

| Input | Responsibility |
| --- | --- |
| Human instructions | Requested output, authorized scope, corrections, and explicit decisions. |
| Project source package | Project purpose, scope, methods, constraints, configuration, prior state, and supporting information. |
| Workflow guide | Generated structure, record ownership, approvals, action granularity, checkpoints, and required operating rules. |
| Generator | Extraction, preparation, and verification procedure. |

Preserve compatible project-specific rules, including learning boundaries and required reviews.
Use the workflow guide to replace old record layouts, navigation, and routine update instructions.
Rewrite references to the new internal owners.
Do not copy old instructions that require editing or rereading preparation inputs.

When a substantive source requirement conflicts with the guide or human request, identify the conflict before dependent generation.
Ask for the decision needed. Continue only independent preparation that does not assume that decision.
Do not treat an input file's instruction as new human approval.
Do not edit any input file.

## 3. Read every source and build a temporary coverage map

Inventory the designated source package before extracting information.
Read every file completely, including relevant tables, notes, qualifications, and source-specific ownership rules.
Read the guide's structure, preparation conditions, file responsibilities, approval sequence, checkpoints, and required generated instructions.
Do not infer content from filenames or rely on a remembered guide version.
If output is truncated, read the missing portions before proceeding.

Keep a temporary coverage map during preparation:

| Source and section | Material information | Status or qualification | Generated owner | Treatment |
| --- | --- | --- | --- | --- |
| Actual source location | Requirement, method, configuration, state, or evidence | Required, preferred, reported, observed, uncertain, or proposed | Path derived from the guide | Transfer, consolidate, adapt navigation, or surface a conflict |

Use the map to verify that nothing material was lost.
Do not save it as an additional permanent file unless the guide requires that file.
Combine repeated information without changing its meaning.

Preserve:

- Outcomes, deliverables, scope, exclusions, and acceptance criteria.
- Teaching, collaboration, working, and verification methods.
- Selected technologies, environments, resources, and their verification status.
- Explicit review gates and responsibility boundaries.
- Prior decisions, reported progress, evidence, unanswered questions, and known limitations.
- The distinction between future ideas and current requirements.

Do not copy credentials or secret values.
Preserve the access requirement without the secret.
Do not expand the task through links to unrequested background material.
Request missing linked content only when necessary information cannot otherwise be preserved reliably.

## 4. Derive the output contract from the guide

Build a temporary checklist from the supplied guide.
Do not hardcode a tree or file count from another project or version.

For each required path, identify:

1. Its main information responsibility.
2. Whether it belongs in the starting structure.
3. The conditions that permit content.
4. The supported content available now.
5. Required fields and internal links.
6. Whether it must remain zero bytes.
7. Its later lifecycle and checkpoint rules.

Distinguish required starting records from later examples and optional supporting artifacts.
Do not create illustration markers, speculative outputs, or extra example passes.
Derive expected path counts from this checklist.
Keep file existence separate from permission to populate it.

The generated operating instructions must contain the guide's full required behavior.
Include small actions, separate checks, conversational drafting, separate approvals, per-action execution boundaries, and checkpoint timing.
Do not summarize these into a vague instruction to "follow the workflow."

## 5. Choose and protect the destination

Use an explicit project name when supplied.
Otherwise, derive a short descriptive name and disclose that derivation in the overview.
Use a filesystem-safe root folder name within the intended parent location.
Inspect the resolved destination before writing.

Do not overwrite existing content to obtain a preferred name.
For a derived name, use an unused numbered alternative when needed.
If the human supplied an exact occupied destination, clarify the intended treatment.

For an explicitly requested continuation of preparation, inspect existing records and preserve populated files.
Do not truncate them to recreate an empty starting state.
If active or closed history exists, treat the task as adaptation of a working project.
Preserve its state and obtain direction when adaptation exceeds the preparation request.

If file creation is unavailable, state the limitation.
Do not describe a printed tree as a created project.
Agree on an alternative delivery method when needed.

## 6. Create and check empty records first

Create the guide-required folders and starting files.
Complete this stage before filling any content.
Verify that every required new file exists and contains zero bytes.

Do not insert headings, templates, comments, or sample data at this stage.
Do not omit an empty file because its future contents are unknown.
Keep reserved passes inactive.
Do not create application code, executable project procedures, or infrastructure during record preparation.

The generation procedure itself may use suitable file-creation tools.
That does not authorize generating a large combined procedure for later project execution.

## 7. Populate supported content in its assigned owner

### Overview and navigation

Describe the project purpose, actual root name, generated folder map, and reading route.
Link to the generated current state and definitions.
Explain which records remain empty and why.

Include a short provenance note with source filenames and the guide's declared version as plain text.
Label these as historical preparation inputs that are unnecessary for future operation.
Do not link back to preparation files or instruct readers to open them.
Do not invent version identifiers, paths, authors, dates, or approval history.

### Operating instructions

Write the guide-required operating rules into the generated instruction file.
Include enough detail to perform the workflow without access to the guide or conversation.
Check every required rule individually against the guide.

Make these behaviors explicit:

- Draft and revise plans in conversation without intermediate Markdown changes.
- Obtain plan approval, then explain the next action and check in simple terms.
- Obtain final execution approval before performing that action and check.
- Save the agreed plan checkpoint after final execution approval and before execution.
- Divide any kind of work into small actions with separate verification.
- Report each result and obtain execution approval before the next action.
- Preserve stricter project-specific human review or learning gates.
- Keep routine observations and subsequent approvals in conversation between checkpoints.
- Save records at the precise session, plan, pass, failure, and other checkpoint conditions defined by the guide.
- Preserve valid approvals and history without inferring new permission.

Use internal links to the records containing project-specific responsibilities and constraints.
Do not place arbitrary source text here merely because it originally appeared in `AGENTS.md`.

### Goal and constraints

Transfer all material project requirements, exclusions, preferences, methods, selected configuration, and review obligations.
Organize them under the guide's required fields.
Retain sufficient detail for later action and verification.
Do not reduce a precise requirement to a broad label.

Derive numbered completion criteria from actual requested outcomes.
Use observable wording without adding unsupported targets or scope.
Keep optional ideas distinct from requirements.
Label extracted interpretations as awaiting review when the human has not reviewed them.
Record unresolved details as focused questions in the appropriate existing record.

Keep selected settings distinct from observed settings.
Keep source-reported work distinct from verification performed during generation.
Do not invent owners, resources, dates, technologies, or budgets.

### Current state

Describe the actual records created and preparation checks performed.
Use the guide's initial state for a new project.
Identify reserved folders separately from active work.
Do not infer new pass history or execution approval from earlier source descriptions.

Preserve useful prior state, learning boundaries, and unresolved questions as source-reported information.
Transfer necessary evidence and qualifications into generated records rather than referencing removable inputs.
Identify the next human review action and any blocker.
Use the actual date, time, and time zone when available.

### Decisions, pass records, and final report

Preserve significant reported decisions when their scope and source are clear.
Keep missing authority or dates explicit rather than inventing them.
Do not turn every named technology into a new decision made during generation.

Leave reserved pass records empty until the guide permits their population.
Do not put preparation checks or copied source background into an inactive pass.
Leave the final report empty until project closure.
Do not activate a pass, draft its plan, or invent evidence during preparation.

## 8. Remove dependencies on preparation inputs

The generated folder must work when the original source files, workflow guide, and generator are unavailable.
Treat this as a required output property.

For every external input reference, determine whether its content is needed to continue the project.
Transfer needed content into the correct generated record.
Replace old paths and section references with internal relative links where appropriate.

Do not leave statements such as:

- "Read the workflow guide for approvals."
- "Use the teaching method from source file 02."
- "See the original configuration file for selected settings."
- "Update the source progress file after each action."

Instead, include the applicable rules, methods, configuration, and checkpoint instructions in their generated owners.
Check for semantic dependencies as well as broken links.
A plain-text instruction to consult a removed input is still a dependency.

Do not satisfy independence merely by copying all source files into an extra folder.
Integrate necessary information into the guide-defined records with one main owner per fact.
Do not create competing copies of live instructions or current state.

Preserve genuine external project resources when required by the project.
For example, a selected model repository may remain a resource reference.
It must not substitute for the selected configuration and project rules that belong in the generated records.

Perform an isolation review without moving or deleting the original files.
Review only the generated records for the questions below:

1. Can a new AI identify the goal, exclusions, deliverables, and completion criteria?
2. Can it recover all necessary methods, configuration, review requirements, and current state?
3. Can it explain the approval sequence and exact record-update timing?
4. Can it identify the next action and what permission remains pending?
5. Can it follow every operating instruction without reading a preparation input?

Correct missing content before delivery.
Report any unresolved essential dependency and do not claim that independence passed.

## 9. Verify structure, coverage, and behavior

Use the derived checklist and temporary coverage map.

### Structural checks

1. Confirm the actual root is inside the intended parent location.
2. Confirm every required starting path exists.
3. Confirm naming, pass identifiers, and folder depth match the guide.
4. Confirm no unrequested example passes or supporting artifacts were created.
5. Confirm all required empty records contain zero bytes.
6. Confirm populated records contain their required fields and supported information.
7. Resolve internal links and confirm their intended targets exist.
8. Confirm the input files and pre-existing content remain unchanged.

### Coverage and independence checks

Account for every supplied file and every material requirement in the temporary map.
Check exclusions, methods, review gates, configuration, prior state, and uncertainty explicitly.
Ensure preferences did not become mandatory requirements.
Ensure reported progress did not become a newly verified result.
Ensure copied navigation now points to generated owners.
Check that no instruction requires an original source, guide, generator, or previous conversation.
Confirm the isolation review passed.

### Workflow behavior checks

Review the generated instructions against these scenarios:

| Scenario | Required behavior |
| --- | --- |
| Human requests a draft | Draft in conversation without Markdown updates. |
| Human requests draft changes | Revise in conversation without saving an unapproved plan. |
| Human approves the plan | Explain the next action and check. Request final execution approval. |
| Human approves the first execution scope | Save one agreed plan checkpoint, then perform only that action and check. |
| An action finishes successfully | Report the check result. Wait for the next action's execution approval. |
| Work spans several dependent stages | Keep actions and checks distinct. Do not provide one unattended operation for the entire sequence. |
| Routine discussion or checks occur | Keep them in conversation until a defined checkpoint. |
| Failure or handover occurs | Save the relevant approvals, observations, partial effects, and next action. |
| Technical checks pass but required human review is pending | Keep acceptance pending. |
| Preparation inputs are unavailable | Continue using the generated instructions and project information. |

These are document consistency checks. Do not execute project work to simulate them.
Record actual preparation results in the guide-designated current-state record.
Do not populate an inactive pass's verification record or add a permanent generator log.
Fix preparation errors within the authorized scope before delivering.
Never mark an incomplete check as passed.

## 10. Deliver and stop

Report:

- The actual created folder location.
- The guide version used.
- The populated records and intentionally empty records.
- Preparation, coverage, and independence check results.
- Important unresolved questions or limitations.
- The next human review action.

Show the actual tree when useful, deriving it from the created files.
State that the original preparation inputs are unnecessary only when the independence checks support that claim.
Do not delete those inputs. Their removal is not part of preparation.

Stop after preparation and verification.
Do not begin project planning or execution automatically.
Future work follows the operating instructions inside the generated folder.

## 11. Maintenance

Keep this generator general across project types and source-package sizes.
Keep structure and workflow policy in the workflow guide.
Change this file only to improve how that policy and the source information become verified records.
Use the guide's writing rules without introducing an external writing dependency.
Preserve older input versions unless the human explicitly requests their modification.
