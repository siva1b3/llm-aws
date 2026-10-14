# Project initialization instructions

Use these instructions to turn a human's project description into the project records defined by the workflow guide.
The workflow guide remains the authority for the output structure and working rules.

These instructions explain how to read, extract, create, populate, and check those records.
They do not define another folder structure or another workflow.

## 1. Inputs and responsibility

Supply these three files:

1. `finalgoal_01.md`, or the goal file named by the human.
2. `project-workflow-guide-v2.md`.
3. This instruction file.

The goal file can contain requirements, preferences, examples, background knowledge, and open questions.
It does not need to be complete or neatly organized.

| Input | What it determines |
| --- | --- |
| Goal file | The project purpose, requested outcomes, requirements, exclusions, preferences, and supplied constraints. |
| Workflow guide | Folder structure, filenames, file responsibilities, workflow, approval boundaries, record lifecycle, and writing rules. |
| These instructions | How to apply those sources to prepare this project's starting records. |

The dependency goes one way.
These instructions read the workflow guide.
The guide does not need to reference or know about these instructions.

Keep both source documents unchanged during project generation.
Do not revise the guide to make an initialization request easier to complete.
Do not maintain a second authoritative tree, file count, or rule list in this instruction file.

## 2. How the human starts

Provide the three files in the same conversation or accessible workspace.
Name the intended parent folder if its location matters.

A suitable request is:

> Read my goal file and the workflow guide. Use the initialization instructions to prepare a project-named folder. Derive its records and creation timing from the guide. Populate supported information and preserve unknowns. Stop at the next required human approval.

An explicit request authorizes preparation within the guide's existing rules.
It does not waive a planning or execution approval required by the guide.
Do not request an approval again when the human already granted it for the same scope.

If the guide requires a setup pass, follow that process.
If it permits empty records before pass activation, use that permission within its stated scope.
Do not invent either permission.

Stop before application implementation or any other work outside the approved preparation scope.

## 3. Read the actual sources

Read the supplied goal and guide before creating project files.
Do not substitute a remembered version of either document.

Ask for a missing or unreadable required input.
Do not create a generic project and claim that it reflects an unread goal.

Use the guide's own rules when linked background material is unavailable.
Do not claim to read an unavailable reference.
Request a reference only if a specific missing rule prevents faithful preparation.

Identify which version of the guide was supplied.
Use a declared version or a recoverable source reference when available.
Do not invent a version identifier.

The guide can contain a main structure, optional records, templates, and worked examples.
Read the related prose before deciding what each example permits.
A tree alone does not establish creation timing or approval.

## 4. Derive the output requirements from the guide

Build a temporary checklist from the supplied guide.
This checklist is a working interpretation of that guide.
It is not a new project record or an alternative specification.

For every output item, determine:

| Checklist field | Information to extract |
| --- | --- |
| Path | The location and filename specified by the guide. |
| Purpose | The responsibility assigned to that record. |
| Requirement | Whether the item is required, optional, illustrative, or conditional. |
| Creation timing | The stage or permission that allows its creation. |
| Content timing | When meaningful information belongs in the record. |
| Initial contents | Supported facts, required operating instructions, a permitted template, or an empty file. |
| Approval | Any human decision required before creating or populating it. |
| Links | Other records that it must reference. |
| Later lifecycle | When it changes, closes, remains preserved, or receives final results. |

Read the guide's sections on structure, file responsibilities, approvals, lifecycle, and required fields together.
Use their responsibilities to place information.
Do not assign a new purpose to an existing file.

Do not copy an illustrative second pass as an automatic requirement.
Do not create a file named after an ellipsis or another illustration marker.
Do not treat sample commands, logs, screenshots, or application outputs as required management files.

Determine the initial pass quantity from the guide and the human's compatible request.
Use the naming convention defined by the guide.
Do not reserve additional passes merely because the source includes several example attempts.

Derive the expected file count from the resulting checklist.
Do not use a fixed count stored in these instructions.

### Missing or conflicting rules

If two guide statements conflict, identify both statements and their effect.
Do not silently choose the interpretation that permits more work.

If the human requests earlier file creation than the guide permits, explain the conflict before that creation.
Ask whether to follow the guide or request a separate rule change.
Keep dependent actions pending until the conflict is resolved.

A request to change a rule is separate from ordinary project initialization.
Do not edit the guide as an automatic response to a conflict.

If a requested feature has no clear record location, use an existing applicable responsibility where possible.
Ask before inventing a permanent folder or file that the guide does not allow.

## 5. Extract the human's intent

Read the full goal file before summarizing individual requirements.
Identify the intended users, purpose, and desired outcome when the source provides them.

Separate these information types:

| Source information | Treatment |
| --- | --- |
| Explicit requirement | Preserve its meaning in the record responsible for goals or constraints. |
| Explicit exclusion | Preserve it as a scope boundary. |
| Preference | Keep it distinct from a mandatory requirement. |
| Optional idea | Preserve it for review without making it a completion criterion. |
| Background knowledge | Use it as context. Do not present it as a verified project result. |
| Existing work described by the human | Label it as human-reported until an authorized check verifies it. |
| Missing detail | Record a focused question in an appropriate existing record. |
| Conflicting requirements | Preserve both statements and identify the decision needed. |
| Prior decision | Preserve its reported scope and source when the guide provides a suitable record. |

Keep all material requirements and exclusions represented.
Combine repeated statements without changing their meaning.
Preserve uncertainty and qualifications.

Use supplied section names or verified line references when they help trace an important requirement.
Do not invent source references.

Do not add a feature because similar projects often include it.
Do not turn an example into an approved design.
Do not select an unspecified framework, programming language, database, provider, or deployment environment.
Do not add unsupported performance targets, budgets, or deadlines.

Keep credentials out of generated records.
Preserve the relevant access requirement without copying a secret value.

### Completion criteria

Use the guide's method for recording completion criteria.
Derive criteria from the human's requested outcomes.

Use observable wording when it preserves the source meaning.
Do not expand the scope to make a criterion easier to measure.

For example, a request for user sign-in can justify a sign-in criterion.
It does not establish a particular identity provider or password policy.

Record a question when the requirement is too unclear to assess.
Do not mark an extracted criterion as achieved.

### Assumptions and questions

Separate supplied facts from assumptions made during extraction.
State the assumption and the reason it matters.

Ask immediately when missing information prevents safe file creation or a reliable destination choice.
Keep other unresolved project details in the generated records for human review.
Do not delay all record preparation because the application design is incomplete.

## 6. Choose the project destination

Use the project name supplied by the human.
If no name is given, derive a short descriptive name from the stated purpose.
Record that the name was derived.

Use a filesystem-safe folder name.
Preserve the original display name when its folder form differs.
Exclude invalid characters, reserved names, and parent-directory traversal.
Resolve the destination inside the intended parent folder.

Use the human's requested parent location.
Otherwise, use the current available workspace when its use is clear.
State the actual location in the completion response.
Ask for a destination when no suitable workspace is available.

Check whether the destination already exists.
Do not overwrite an existing project to obtain the preferred name.

For a new project, use an unused, clearly numbered name when the preferred name is occupied.
Report the chosen name and reason.

If the human requests work in an existing folder, inspect it before writing.
Distinguish unfinished record preparation from an active or completed project.
Preserve existing approvals, decisions, populated records, and pass history.

Do not reset a working project to a new-project state.
Request the appropriate resumption or adaptation decision instead.

If file tools are unavailable, state that limitation.
Do not claim that a displayed tree is a created folder.
Ask before substituting another delivery method.

## 7. Create records at the permitted point

Use the checklist derived from the guide to determine what can exist now.
Create only the approved paths and currently permitted records.

If the guide permits preparation of the full record structure, create the permitted placeholders.
An empty placeholder does not establish approval, progress, or completion.

If the guide delays a file until a later stage, leave that path uncreated.
Identify the guide's reason in the delivery summary when the human expected that file immediately.

When the guide requires a setup plan, follow its drafting and execution approvals.
Use the conversation for an approval record until its designated file can exist.
Transfer the actual approval with its scope and reference when that record becomes available.

Do not change creation timing by calling a planned pass a placeholder.
Use placeholder status only when the guide permits it.

### Empty files

Leave a permitted record empty when its stage has not supplied meaningful content.
An empty file contains zero bytes.
Do not add a heading, comment, invented entry, or sample result merely to make it appear populated.

Use a template only when the guide permits it at that stage.
Mark unperformed checks according to the guide when a real approved plan requires them.
Do not create planned checks during unapproved planning.

Do not erase existing content to produce an empty initial state.
The empty-file rule applies to new records that have no justified contents.

## 8. Populate each record by its assigned responsibility

Resolve the actual filename for each role from the guide.
The following guidance explains extraction and adaptation.
It does not assign new paths or override the guide's field requirements.

### Project overview and navigation

Explain the project using the supplied purpose.
Include the actual generated folder map and the reading route required by the guide.

Identify the source filenames and their roles.
Use verified links where possible.
Use a plain source filename when a portable link is unavailable.

Distinguish a prepared project record from an implemented application.
Identify intentionally empty or deferred records without suggesting that work occurred.

### AI operating instructions

Derive the applicable operating instructions from the guide.
Preserve its approval boundaries, record duties, and decision authority.

Generate enough instructions for a later AI session to follow the project's agreed process.
Do not rely only on an earlier chat that the next session might not receive.

Use a verified, stable guide reference when the supplied environment provides one.
Otherwise, include the necessary guide-derived rules in the designated operating record.
Identify that record as an application of the supplied guide, not an independent rulebook.

Resolve project-specific responsibilities from the human's explicit information.
Do not grant authority that the source does not grant.

Do not copy arbitrary goal text into operating instructions.
Keep project requirements in the records responsible for those requirements.

### Goal and constraints

Use the guide's required fields.
Preserve requested outcomes, scope boundaries, deliverables, preferences, and supported criteria.

Identify the source goal file.
Mark the extracted definition as awaiting human review when that review has not occurred.
Do not invent agreement to newly inferred details.

Keep project-specific constraints distinct from process rules supplied by the guide.
Record missing technical or resource decisions as questions.

Do not invent team members, reviewers, operating owners, or approval roles.
Apply only roles supported by the guide and the human's information.

### Current state

Record what actually exists after the authorized preparation.
Use the guide's status values and required fields.

Distinguish these matters explicitly:

- Records created during this task.
- Existing work reported by the human.
- Checks actually performed during preparation.
- Implementation and project checks that did not occur.
- Actual approval status.
- Reserved, proposed, active, or closed passes, as the guide permits.
- The next human decision and its prerequisite.

If only empty pass records were prepared, do not mark a pass active.
If an approved setup pass was performed, record its actual state instead.
Do not force every initialization into the same pass status.

Describe structural checks as checks of the generated records.
Do not present them as verification of application behavior.

### Decisions, pass records, evidence, and reports

Use the guide's content conditions for these records.
Do not fill them merely because their files exist.

Preserve an explicitly supplied historical decision when the guide assigns it a suitable location.
Identify its source and distinguish reported history from a new approval.

Fill a pass plan only after the required drafting approval.
Record expected checks only when authorized planning supplies them.
Record observed results only after an authorized check produces them.

Keep background notes from the source distinct from pass investigation findings.
Do not create evidence entries for checks that never occurred.
Do not write a completion report merely because a report placeholder exists.

## 9. Check the generated result

Compare the result with the supplied guide and the derived checklist.
Do not verify it against a separate folder tree stored in these instructions.

### Structure and lifecycle

Check these conditions:

1. The root uses the intended project name and location.
2. Required paths exist for the approved stage.
3. Deferred paths follow the guide's timing rules.
4. Pass quantities, identifiers, and folder depth follow the guide.
5. No sample artifact became a real file without a requirement.
6. Required empty files contain zero bytes.
7. Populated records contain their required supported information.
8. Relative links resolve to the intended records.
9. Existing project content remains preserved.
10. Source files remain unchanged.

Use actual filesystem observations for these checks.
Derive expected counts from the guide's rules for this project and stage.

### Meaning and authority

Check that every material requirement and exclusion is represented.
Check that preferences remain distinct from mandatory requirements.
Check that unknowns and conflicts remain visible.

Check that the generated operating rules match the supplied guide.
Check that no approval, decision, test result, active pass, or completion status was invented.
Check that the current-state record agrees with the work actually performed.

Check that the guide does not acquire a dependency on these instructions.
Project generation must not modify its rules, references, or content.

Record preparation checks where the guide permits them.
If no record is available at the current stage, report the results in the conversation.
Do not create an unapproved evidence file merely to store the initialization check.

Correct mistakes within the authorized preparation scope.
Report any remaining issue that needs a human decision.
Do not mark an incomplete check as passed.

## 10. Deliver and stop at the correct boundary

Report the actual result in a short completion message.
Include:

- The created project folder or the exact point where approval is required.
- The supplied guide used for the structure and rules.
- The records populated from the goal.
- The records left empty or deferred under the guide.
- Important unresolved questions.
- Preparation checks and limitations.
- The next human action.

Include the actual folder tree when it helps the human inspect the output.
Generate that tree from the created result, not from a static template in this file.

Do not claim that all records exist when the guide required some to remain deferred.
Do not claim successful initialization if a required step is still pending.

Stop at the next boundary established by the guide.
Do not continue into application planning, implementation, or deployment without the required authority.

## 11. Writing and future maintenance

Use the controlled-language rules stated in the supplied guide.
Keep the writing simple, specific, and consistent.
Preserve exact names, identifiers, and source meaning.

The workflow guide owns structure and policy.
These instructions own only the procedure for applying that policy to the supplied goal.

If the guide changes, derive the output from its new rules.
Do not keep an obsolete structure alive by repeating it here.
If an application step becomes incompatible, report the mismatch without silently changing the guide.

These instructions can be improved independently when their procedure is unclear.
A policy change requires a separate, explicit decision about the guide.
