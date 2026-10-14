# Project records generator

Use these instructions to prepare project records from a human's goal and an existing workflow guide.

The method is:

**Read rules → Read goal → Create empty records → Fill supported contents → Verify → Stop.**

The workflow guide defines the folders, files, responsibilities, and working rules.
The goal supplies the project information.
This document explains how to apply those two sources.

## 1. Required inputs

Supply these three files:

1. The human's goal file, such as `finalgoal_01.md`.
2. `project-workflow-guide-v2.md`.
3. This generator instruction file.

Use the actual goal filename supplied by the human.
The goal can contain rough notes, requirements, preferences, examples, and unanswered questions.
It does not need to be complete before record preparation starts.

### Request to use the generator

The human can provide the files with this request:

> Read my goal file and the workflow guide. Follow the project records generator instructions. Create a project-named folder with the starting structure required by the guide. Create the files empty first. Then fill each file with relevant, supported information. Leave other files empty. Check the result and stop before pass planning.

Use a parent location named by the human when provided.
Otherwise, use the current available workspace when its use is clear.
Ask for a destination only when no suitable location is established.

An explicit record-preparation request authorizes the preparation permitted by the guide.
It does not authorize application planning, implementation, investigation, or external delivery.

## 2. Keep the responsibilities separate

| Source | What it supplies | How to use it |
| --- | --- | --- |
| Workflow guide | Output structure, record responsibilities, approval boundaries, lifecycle, required fields, and writing rules. | Read these requirements directly from the supplied guide. |
| Goal file | Project purpose, outcomes, requirements, preferences, exclusions, constraints, and background. | Put supported information into the records assigned by the guide. |
| Generator instructions | The procedure for extracting, creating, filling, and checking records. | Apply the two sources without changing their meaning or authority. |

The dependency goes one way.
This generator reads the guide.
The guide does not need to reference or know about this generator.

Do not edit either input document during generation.
Do not maintain another authoritative folder tree, file count, or workflow in this document.
Do not invent a missing rule to make generation easier.

If the supplied guide conflicts with the human's requested output, identify the conflict before dependent work.
Ask for the decision needed to resolve it.
Do not silently change the guide or omit a requested requirement.

## 3. Read the inputs before writing

Read the full goal and the relevant guide provisions before creating the project folder.
Do not rely on a remembered version or infer contents from a filename.

Read the guide's structure together with its explanations.
Also read its file responsibilities, preparation rules, approval rules, lifecycle, and required contents.
Worked examples and templates do not automatically create additional output requirements.

Ask for a required input if it is missing or unreadable.
Do not replace it with guessed content.

The supplied guide contains the operational rules needed for this task.
Use those rules when a linked background reference is unavailable.
Do not claim to inspect an unavailable reference.
Request further material only when a specific missing requirement prevents correct preparation.

Keep an identifiable reference to the supplied guide version when available.
Use its declared version or another recoverable reference.
Do not invent a version identifier.

## 4. Derive the starting structure

Build a temporary checklist from the guide.
Use it to prepare and check this project.
Do not save it as an additional permanent project file unless the guide requires one.

For each item, identify:

| Detail | Question to answer from the guide |
| --- | --- |
| Path | Which folder or file must exist, and where? |
| Responsibility | What information belongs there? |
| Starting requirement | Is the item part of the starting structure or a later example? |
| Content condition | What permits or requires the record to receive contents? |
| Initial treatment | Which contents are supported now, and which files must remain empty? |
| Links | Which other records must this item reference? |
| Lifecycle | How does the record change when work later begins or finishes? |

Create the starting structure that the guide actually requires.
Use its pass naming convention and its rules for the first reserved pass.
Respect any compatible human request for additional reserved folders.

Do not copy a later example pass into the starting structure automatically.
Do not create a file for an ellipsis or illustration marker.
Do not generate example logs, screenshots, or supporting artifacts without a requirement.

Derive the expected file count from the checklist.
Do not use a fixed count or copied tree from an earlier project.

The checklist must cover both file existence and file contents.
A file can be required to exist while its contents remain empty.

## 5. Extract the project information

Read the goal as the source of project intent.
Preserve its meaning when organizing that information.

| Information found | Treatment |
| --- | --- |
| Explicit requirement | Preserve it in the record responsible for that requirement. |
| Explicit exclusion | Keep it as a scope boundary. |
| Preference | Keep it distinct from a mandatory requirement. |
| Optional idea | Preserve it for review without making it a completion condition. |
| Example | Use it to explain intent unless the human makes it a requirement. |
| Background knowledge | Keep useful context without calling it a verified project result. |
| Existing work described by the human | Label it as human-reported until an approved check verifies it. |
| Missing detail | Record a focused question in an appropriate existing record. |
| Conflicting statements | Preserve the conflict and identify the decision needed. |
| Prior decision | Preserve its reported scope and source when a guide-defined record applies. |

Represent every material requirement and exclusion.
Combine repeated statements without changing their meaning.
Keep uncertainty and qualifications from the source.

Do not add features because similar projects often include them.
Do not select an unspecified technology, provider, deployment target, budget, or deadline.
Do not convert a suggested tool into an approved design.
Do not infer implementation approval from the contents of the goal file.

Use source headings or verified line references when they help trace important requirements.
Keep credentials out of generated records.
Preserve the relevant access requirement without copying a secret value.

### Completion criteria

Use the guide's method for expressing completion criteria.
Derive each criterion from a requested outcome.
Use observable wording without adding unsupported scope or numerical targets.

Record a clarification when the source cannot support a clear criterion.
Do not mark a criterion complete merely because it appears in the goal.

### Questions that can wait

Missing application details usually do not prevent preparation of the records.
Keep those questions visible for human review.

Ask immediately when a missing fact prevents a safe destination choice or a reliable reading of the required structure.
Do not delay the entire task to obtain every future implementation decision.

## 6. Choose and protect the destination

Use an explicit project name from the human when available.
Otherwise, derive a short descriptive name from the project's stated purpose.
Record that the name was derived.

Use a filesystem-safe form for the root folder.
Preserve the original display name in the appropriate overview record when the folder name differs.
Avoid reserved names, invalid characters, and parent-directory traversal.
Resolve the final destination inside the intended parent location.

Inspect whether the destination already exists before writing.
Do not overwrite an existing project to obtain the preferred name.

If only a display name was supplied, choose an unused numbered name when necessary.
Report the actual name used.
If the human specified an exact occupied destination, clarify how to proceed.

### Existing project content

This procedure normally creates a new project folder.
If the human asks to complete an existing preparation, inspect its records first.
Create missing paths without replacing populated records.
Do not truncate existing files to make them match an empty starting state.

If the folder contains active or closed pass history, treat it as an existing working project.
Do not reset its approvals, state, or history.
Request direction to resume or adapt it under the guide.

If tools cannot create files, state the limitation.
Do not present a printed tree as a created project.
Ask before substituting another delivery method.

## 7. Create all starting files empty

Create the root folder and the paths derived from the guide.
For a new project, create every required starting Markdown file as an empty file.

Complete this stage before filling project information.
Check that the required paths exist and the new files are empty.
This establishes that the structure came from the guide rather than from available goal content.

An empty file contains zero bytes.
Do not insert headings, templates, comments, or sample entries during the empty-file stage.
Do not omit a required file because its future contents are unknown.

Keep the reserved pass inactive.
Creating its folder and files does not grant planning or execution approval.
Do not create application source files or infrastructure during record preparation.

## 8. Fill only supported contents

Resolve each record's actual path from the guide.
Use its assigned responsibility and required fields to decide what belongs there.

Use project facts from the goal.
Use operating rules and record instructions from the guide.
Use actual observations for preparation status and structural checks.

The following role descriptions explain how to apply those sources.
They do not define additional filenames, paths, or policy.

### Overview and navigation records

Describe the supplied purpose and actual project name.
Include the generated folder map and reading route required by the guide.
Identify the source files without inventing their locations.

Use relative links to generated records.
Use a verified reference for an external source, or state its filename without a broken link.
Explain which records are intentionally empty and why.
Do not describe the unbuilt application as operational.

### AI operating instructions

Derive the generated operating instructions from the guide's rules for that record.
Preserve the required approval boundaries, record duties, and human authority.

Include enough guide-derived instructions for a later session to follow the process.
Do not rely solely on the previous conversation.
Use a verified guide reference when it remains available to the project.

Do not grant permissions that the guide or human did not grant.
Keep project requirements in their designated records instead of copying arbitrary goal text into operating instructions.

### Goal and constraint records

Use the required fields defined by the guide.
Preserve requested outcomes, deliverables, scope, relevant preferences, exclusions, and supported criteria.
Identify the source goal file.

Label extracted information as awaiting human review when that review has not occurred.
Separate supplied facts, source assumptions, and unresolved questions.
Do not imply acceptance of a newly inferred requirement.

Keep project-specific constraints distinct from workflow rules.
Record only supported owners, reviewers, deadlines, environments, and resource limits.
Do not invent a team or operating arrangement.

### Current-state record

Describe the records that now exist and the checks actually completed.
Use the status fields and terms defined by the guide.

Record no active or closed pass for a newly prepared project.
Identify the reserved pass separately.
Keep planning and execution approval ungranted unless the human actually supplied the applicable approval.
Do not exercise additional authority during this generator run beyond record preparation.

Distinguish preparation checks from application verification.
Do not claim that code, accounts, resources, or deployments exist merely because the goal mentions them.
Mark described existing work as human-reported when it was not inspected.

Record unresolved review questions and the next human action.
Use the actual update time and time zone when available.

### Decisions and work-history records

Follow the guide's conditions for meaningful entries.
Preserve a supplied historical decision only when its actual scope and source are clear.
Do not invent a decision merely because a requirement names a technology.

Leave pass records empty while planning and pass work remain unapproved or unperformed.
Keep source background separate from pass findings.
Do not write evidence for a check that never occurred.
Leave the final report empty until the guide's closure conditions apply.

### Preserve genuinely empty records

After populating justified contents, keep every remaining placeholder at zero bytes.
Do not add a heading merely to make a file appear used.
Explain empty files in the overview or current-state record where the guide permits it.

## 9. Verify the generated project

Compare the actual output with the supplied guide and the derived checklist.
Do not compare it with a remembered tree or fixed file count.

### Structure checks

1. Confirm the actual project root and intended parent location.
2. Confirm that every required starting path exists.
3. Confirm that names, pass identifiers, and folder depth follow the guide.
4. Confirm that no unrequested example pass or sample artifact was created.
5. Confirm that required empty files contain zero bytes.
6. Confirm that populated files contain their required supported information.
7. Confirm that generated links lead to the intended records.
8. Confirm that source files and existing project content remain unchanged.

### Content checks

Check that every material requirement and exclusion has an appropriate place.
Check that preferences remain distinct from mandatory requirements.
Check that questions and conflicts remain visible.
Check that completion criteria do not exceed the supplied goal.

Check that the generated operating instructions preserve the guide's rules.
Check that no pass activation, plan, execution result, or completion claim was invented.
Check that current state agrees with the work actually performed.

Record the preparation checks where the guide permits them.
Do not use an inactive pass's verification record as a log for generator activity.
Do not create a new permanent evidence file outside the guide's structure.

Correct preparation mistakes within the authorized scope before delivery.
Report unresolved problems and the human decision needed.
Do not report an incomplete check as passed.

## 10. Deliver the prepared records and stop

Provide a short completion message with:

- The actual project folder location.
- The source guide used for the structure and rules.
- The records populated from the goal and guide.
- The records intentionally left empty.
- Important unanswered questions.
- Preparation check results and any limitations.
- The next human review or approval required by the guide.

Show the actual created tree when it helps the human inspect the result.
Derive that display from the created files.

Stop after record preparation and verification.
Do not continue into pass planning, application implementation, or deployment.
Use the workflow guide when the human later chooses to start the first pass.

## 11. Writing and maintenance

Apply the guide's controlled-language principles to generated Markdown.
Use simple English, consistent terms, and precise statements.
Preserve uncertainty and exact technical names.
Do not claim full ASD-STE100 compliance without the required language review.

Keep the guide as the only authority for structure and workflow policy.
If its structure changes, derive the new output from the supplied version.
Do not preserve an obsolete layout by duplicating it in this generator.

Changes to this document improve the generation procedure.
Changes to project structure or working rules belong to a separate, explicitly authorized guide revision.
