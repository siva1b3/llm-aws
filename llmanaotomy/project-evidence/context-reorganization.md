# Source comparison and context corrections

**Date:** 2026-10-11.

## Question and authorization

Does the root context package preserve the important content and meaning of the supplied source documents?

The learner authorized comparison and corrections to `AGENTS.md` and every file in the three context folders.
The organizing and writing references must remain unchanged.
The learner then supplied six local source documents in `sources/`.

## Inputs and method

Read all six source files and all eight destination files with PowerShell `Get-Content`.
Compare each numbered source section with its owning destination document.
Use [project-context-guide.md](../project-context-guide.md) to organize rules, current state, and evidence.
Use [asd-ste100.md](../asd-ste100.md) for direct language, short sentences, consistent terms, and preservation of uncertainty.

Record SHA-256 hashes of the two references and six source files before editing.
Apply focused changes with `apply_patch`. Replace this comparison record with PowerShell `Set-Content`.
Check destination links, heading anchors, and protected-file hashes after editing.
Content coverage is a manual comparison of requirements and meaning. Link checks alone do not prove coverage.

## Observed gaps and corrections

| Finding before correction | Correction |
| --- | --- |
| Ten broken local file links | Corrected reference paths and removed obsolete dependencies on absent files |
| Evidence navigation depended on absent `investigations/README.md` | Restored the supplied investigation protocol in the project guide and linked it from evidence |
| Component record details were incomplete | Added state, reason, position, assets, read/write behavior, checks, limitations, confirmation, and questions |
| Asset and observation recording details were incomplete | Added resource distinctions, asset fields, observation state, and comparison fields |
| Input preparation was not explicit in evidence requirements | Required exact input, chat template, special-token preparation, caching, and inference mode |
| Teaching constraints were partly implicit | Made lesson size, minimum context, learner questions, and limits of simplifications explicit |
| Startup guidance did not explicitly prevent restarting completed work | Added guidance to resume at the last verified, learner-confirmed boundary |
| Technical reference omitted the baseline evidence route | Added the evidence route and a requirement to link a verified baseline record |
| Expected-property correction could be confused with changing an agreed choice | Distinguished factual corrections from changes to governing choices |
| Status described four sources and a separate `reorganized-context/` package | Replaced outdated status with the six-source comparison and actual root layout |
| Earlier history and verification counts lacked source support | Retained them below as unverified prior reports |

## Coverage by source section

All 49 numbered sections have a destination or an explicit treatment below.
These tables describe semantic coverage after correction. They do not claim identical wording or duplication of source layouts.

### 01 — Project overview

Source: [01_PROJECT_OVERVIEW.md](../sources/01_PROJECT_OVERVIEW.md).

| Source section | Destination or treatment |
| --- | --- |
| 1. Goal | [Purpose and scope](../project-rules/project-guide.md#purpose-and-scope) |
| 2. Exactly one forward pass | [Phase 1](../project-rules/project-guide.md#phase-1) |
| 3. Chosen reference system | [Technical reference](../project-rules/technical-reference.md) and Phase 1 |
| 4. Final learning outcomes | [Completion criteria](../project-rules/project-guide.md#completion-criteria) |
| 5. Completion standard | Completion criteria and the [verification gate](../project-rules/project-guide.md#verification-gate) |
| 6. Future phases | Phase 1 requires explicit authorization and an update to the project guide |
| 7. Where other rules live | [Root reading order](../AGENTS.md#reading-order), folder READMEs, and links between owning documents |

### 02 — Learning methodology

Source: [02_LEARNING_METHODOLOGY.md](../sources/02_LEARNING_METHODOLOGY.md).

| Source section | Destination or treatment |
| --- | --- |
| 1. Teacher responsibilities | [Responsibilities](../project-rules/project-guide.md#responsibilities) |
| 2. Progressive inspection protocol | [One black box at a time](../project-rules/project-guide.md#one-black-box-at-a-time) |
| 3. Build first and introduce prerequisites when needed | Inspection sequence and [lesson size](../project-rules/project-guide.md#lesson-size-and-stopping-conditions) |
| 4. Learner responsibilities | Responsibilities and explicit confirmation gates |
| 5. One-lesson interaction pattern | Lesson size, diagram, objective, action, observations, explanation, and understanding check |
| 6. Verification supports teaching | Lesson size and [reproduction and errors](../project-rules/project-guide.md#reproduction-and-errors) |
| 7. Mandatory stopping behavior | Lesson size and stopping conditions |
| 8. Phase-specific teaching constraint | Phase 1 and [execution discipline](../project-rules/project-guide.md#execution-discipline) |

### 03 — Implementation, inspection, and verification

Source: [03_IMPLEMENTATION_RULES.md](../sources/03_IMPLEMENTATION_RULES.md).

| Source section | Destination or treatment |
| --- | --- |
| 1. Implementation discipline | [Small changes](../project-rules/project-guide.md#small-changes) and execution discipline |
| 2. Phase-1 execution contract | Execution discipline and [initial runtime settings](../project-rules/technical-reference.md#initial-runtime-settings) |
| 3. Operation data contract | [Required observations](../project-rules/project-guide.md#required-observations) and the [component format](../project-rules/project-guide.md#component-record-format) |
| 4. Displays by data type | [Readable displays](../project-rules/project-guide.md#readable-displays) |
| 5. Bounded output | Readable displays, including shapes, dtypes, indices, units, and bounded samples |
| 6. Source inspection and reproduction | Reproduction and errors |
| 7. Verification gate | Verification gate and [distinct states](../project-rules/project-guide.md#distinct-states) |
| 8. File changes and tests | Small changes, reproduction and errors, [handoff](../project-rules/project-guide.md#handoff), and baseline evidence route |
| 9. Error handling | Reproduction and errors and [decision changes](../project-rules/technical-reference.md#decision-changes) |

### 04 — Technical configuration

Source: [04_TECHNICAL_CONFIGURATION.md](../sources/04_TECHNICAL_CONFIGURATION.md).

| Source section | Destination or treatment |
| --- | --- |
| 1. Selected AWS environment | [AWS environment](../project-rules/technical-reference.md#aws-environment), including source cautions |
| 2. Selected model | [Model](../project-rules/technical-reference.md#model), including expectations and reasons |
| 3. Software and runtime choices | [Software](../project-rules/technical-reference.md#software) and initial runtime settings |
| 4. Expected assets | [Expected assets](../project-rules/technical-reference.md#expected-assets) and evidence requirements |
| 5. Asset lifecycle | [Resource lifecycle](../project-rules/technical-reference.md#resource-lifecycle) |
| 6. Observed runtime record | [Runtime verification state](../project-status/current-progress.md#runtime-verification-state) and evidence records |
| 7. Decision protocol | Decision changes |

Observed versions, paths, runtime settings, and commands belong in evidence under the organizing guide.
The technical reference links to those records. Current verification state belongs in status.
This relocation preserves the source requirements without maintaining competing copies of changing facts.

### 05 — Inference investigation

Source: [05_INFERENCE_INVESTIGATION.md](../sources/05_INFERENCE_INVESTIGATION.md).

| Source section | Destination or treatment |
| --- | --- |
| 1. Investigation scope | Phase 1, required observations, and evidence requirements |
| 2. Initial learning boundary | [Learning boundary](../project-status/current-progress.md#learning-boundary) and expected model structure |
| 3. Fixed input and traceability | Initial runtime settings and [recording requirements](README.md#recording-requirements) |
| 4. Component template | Component record format, used only during actual investigation |
| 5. Asset investigation table | [Assets, observations, and comparisons](../project-rules/project-guide.md#assets-observations-and-comparisons), preserving fields and resource distinctions |
| 6. Observation register | Assets, observations, and comparisons, preserving operation, observation, evidence, and state fields |
| 7. Reproductions and comparisons | Assets, observations, and comparisons and the project guide's reproduction rules |
| 8. Notebook boundaries | Investigation instructions, evidence navigation, status separation, and progressive inspection rules |
| 9. First intended investigation | [First baseline record](../project-rules/project-guide.md#first-baseline-record) and the unchanged learning gate |

No empty model investigation records were created. The source tables contain no actual runtime findings to migrate.
Their recording fields remain available as instructions for future observed work.

### 06 — Progress and handoff

Source: [06_PROJECT_PROGRESS.md](../sources/06_PROJECT_PROGRESS.md).

| Source section | Destination or treatment |
| --- | --- |
| 1. Snapshot | [Current checkpoint](../project-status/current-progress.md#current-checkpoint) with selected choices linked through the technical reference |
| 2. Status meanings | Distinct states in the project guide |
| 3. Learning boundary | Learning boundary in current progress |
| 4. Milestone register | [Milestones](../project-status/current-progress.md#milestones), preserving inference's unverified state |
| 5. Documentation package | Root reading order and folder navigation. Source-era inventory is historical rather than current status. |
| 6. Exact next action | [Next single action](../project-status/current-progress.md#next-single-action), with later baseline work governed by the project guide |
| 7. Run and evidence record | Runtime verification state and evidence requirements. No inference command or result is invented. |
| 8. Session handoff | Handoff in the project guide |
| 9. New-session reading route | Root reading order and resume rules |

## Earlier record: unverified reports

The earlier version of this record described a four-source rewrite under `reorganized-context/`.
It named `AGENTS.md`, `CORE_GUIDE.md`, `CURRENT_PROGRESS.md`, and `TECHNICAL_REFERENCE.md` as its original inputs.
It reported a folder-creation failure followed by eight created files, 40 valid local links, and seven unchanged protected files.

That record also described 49 sections split into six folders, six indexes, and 97 checked links across 57 Markdown files.
It reported learner difficulty with fragmentation, later consolidation into four documents and `investigations/README.md`, and removal of fragmented files.
It reported a failed Git status check at `C:/main/llm-aws`.

The six supplied sources do not substantiate those historical counts or that consolidation sequence.
This review did not inspect the claimed earlier inputs, removed files, or prior validation interval.
Treat those statements as unverified prior reports. They do not describe the current workspace or this review's results.

The supplied progress source separately reports a seven-document specification with checked local links.
Only six source files are present here. Its separate `AGENTS.md` is referenced but was not supplied.
This review cannot verify that missing source's contents or reproduce the source-era seven-file check.

## Verification and limitations

Manual review found no unresolved material omission or change of meaning within the six supplied sources after correction.
The coverage tables provide the review route. They are a reasoned content assessment, not an automated proof of equivalence.

The final PowerShell validation checked the following:

| Check | Observed result |
| --- | --- |
| Destination inventory | Eight Markdown files reviewed, seven corrected, and the status README retained unchanged |
| Source inventory | Six files with 49 numbered sections |
| Coverage map | 49 source-section rows |
| Local Markdown targets | 98 checked, with no missing file targets |
| Destination heading anchors | 52 checked, with no missing anchors |
| Protected hashes | Both references and all six sources match their pre-edit SHA-256 hashes |
| Basic writing scan | No sentence exceeded the applicable 20-word or 25-word limit in the line-based scan. No prose semicolons appeared. |

The validation used `Get-ChildItem`, `Get-Content`, `Test-Path`, `Get-FileHash`, and regular expressions.
The writing scan excluded headings, tables, and fenced code. It does not verify every grammar rule or approved vocabulary.
The writing reference's `scripts/ste-lint.py` is absent from this workspace, so that linter was not run.

No external URL availability or Library identity check is claimed.
The local sources are the files supplied by the learner. Their identity with the earlier Library folder was not independently verified.

No EC2 environment, installed model runtime, model assets, tensors, or inference output was inspected.
Documentation authorization does not confirm an inference concept. Learner review of the corrected documents remains pending.
The rewrite uses the supplied clarity guidance. It does not claim certified ASD-STE100 compliance.

