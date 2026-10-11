# Implementation and investigation rules

This document owns code discipline, inspection displays, verification gates, and investigation record requirements.
Update it when these conventions change.

Sources: [implementation rules](../../sources/03_IMPLEMENTATION_RULES.md) and [investigation notebook](../../sources/05_INFERENCE_INVESTIGATION.md).

## Small, reproducible changes

1. Implement one small milestone per iteration.
2. Run, inspect, and explain that milestone before expanding the system.
3. Supply the smallest complete change needed now.
4. Avoid large scripts and multiple file rewrites when a small edit suffices.
5. Prefer transparent code. Extract helpers only after a demonstrated repeated need.
6. Start with the selected software and the model's real requirements.
7. Add dependencies only when the current operation requires them.
8. Preserve baseline behavior when adding investigation scripts or instrumentation.
9. Use fixed input, CPU execution, evaluation mode, inference mode, and initial deterministic selection.
10. Record installed versions after validation.

Do not explain every layer or write the full inference engine prematurely.
Avoid web applications, microservices, Docker, Kubernetes, distributed serving, instrumentation platforms, and performance pipelines in this phase.
Follow the [scope exclusions](project-guide.md#deferred-work).

State when a library function, hook, or fused operation hides a value.
Choose the smallest suitable alternative for inspection.
Fix errors at their source. Code remains unverified until actual execution supports its behavior.

## Single-pass execution

Follow the [initial runtime settings](technical-reference.md#initial-runtime-settings).
Use the model forward method directly.
Do not use `generate()` as the core baseline when it hides repeated forward calls.

For a single unpadded sample, the next-token scores typically come from `logits[0, -1, :]`.
Check the actual shape and indexing before relying on that expression.
Use the last valid input position when padding or other preparation changes the indexing.
The actual model implementation and configuration take precedence over illustrative diagrams.

## Data contract for each operation

| Item | Capture |
| --- | --- |
| Input | Actual value or bounded sample, source, and Python object type |
| Representation | Shape, dtype, device, and meaningful indices |
| Computation | Actual function or module and core mathematical transformation |
| Dependencies | Relevant files, configuration fields, weights, and locations |
| Output | Actual value or bounded sample, type, shape, and destination |
| Purpose | Why the forward pass needs this operation |
| Evidence | Executed command or script and observed result |

A tensor is a numerical array. Its shape gives dimensions, and its dtype identifies the numerical data type.
The device identifies where the tensor resides, such as CPU memory.

Use these labels consistently:

- **Observed:** Actual inspection or execution supports the value.
- **Expected (not yet run):** The value or behavior is an expectation.
- **Illustrative:** The value is a teaching example.

Never present invented teaching values as model observations.

## Readable displays

| Data | Preferred display | Avoid |
| --- | --- | --- |
| Input or decoded text | Labeled plain text | Unlabeled prompt/output mixture |
| Token sequence | Numbered token list | Concatenated tokens without labels |
| Token-to-ID mapping | Two-column table | Entire vocabulary |
| Configuration JSON | Relevant keys, formatted for reading | Unrelated configuration dump |
| Asset inventory | Directory tree, size, and purpose | Guessed files |
| Pipeline status | Small progressive diagram | Complete architecture spoiler |
| Vector or embedding | Shape, dtype, and indexed selected values | Full long vector |
| Matrix or weights | Dimensions and selected rows or columns | Full matrix |
| General tensor | Shape, dtype, device, and selected values | Unbounded output |
| Transformation | Before/after shapes and selected values | Shapes without computation |
| Mathematical step | Small worked example linked to real values | Toy data presented as actual |
| Attention weights | Token-labeled heatmap when useful | Unlabeled colors |
| Next-token logits | Small top-k table with token, ID, and score | All vocabulary scores |
| Calculated probabilities | Top-k table or useful bar chart | Probabilities confused with logits |
| Transformer internals | Small component diagram | Prematurely exposed components |
| Asset placement | Disk-to-RAM-to-operation diagram | Every tensor treated as an external file |

Top-k means the k entries with the highest values.
Start with terminal output. Add a table, saved report, plot, or heatmap only when it improves the current explanation.
Do not install visualization packages merely for decoration.

Show relevant tensor shapes and dtypes.
Limit actual values to a deliberately selected small sample.
Include indices and units when relevant, such as token position, embedding dimension, or layer number.
Distinguish scores from probabilities.
Normally show 5–10 top scores unless deeper analysis is requested.
For large arrays, provide a bounded preview and a method for deeper inspection.
Separate runtime output from explanation text.
Keep filenames and commands stable and reproducible.

## Source inspection and reproduction

Begin with a working high-level library call.
When opening one operation:

1. Capture actual inputs and outputs.
2. Identify the responsible library, module, or function.
3. Explain the mathematics and program behavior at a useful level.
4. Write a small local reproduction only if necessary.
5. Compare matching inputs and parameters where practical.
6. State numerical tolerances or expected differences.
7. Record actual comparison results and limitations.

Similar formulas do not establish numerical equality.
Do not rewrite an entire Transformer to explain one operation.

## Verification and advancement gate

Technical verification requires evidence for execution and claimed behavior.
Advancement also requires explicit learner confirmation.

- [ ] Identify the actual input and its originating operation.
- [ ] Identify its representation, type, and relevant dimensions.
- [ ] Identify the executing code and explain the computation.
- [ ] Explain why the operation is needed.
- [ ] Identify relevant assets and parameters and their use.
- [ ] Inspect actual output and its representation.
- [ ] Trace the output to the next operation.
- [ ] Run a relevant inspection or test and document the result.
- [ ] Resolve relevant discrepancies and misunderstandings.
- [ ] Obtain explicit learner confirmation before advancing.

Use the [status definitions](../project-status/current-progress.md#status-definitions) without combining distinct states.

## Investigation records

Create an evidence record only when actual work produces observations.
Do not create empty records for future operations.
Keep unresolved experiment questions beside their observations.
Summarize questions that block active work in [current progress](../project-status/current-progress.md).

For each investigated operation, record:

1. Question, operation name, state, and reason for opening it now.
2. Position: understood operation, current operation, and hidden remainder.
3. Actual input, origin, type, shape, dtype, and device.
4. Core computation and purpose.
5. Executing code, function, and relevant source location.
6. Assets, parameters, load origin, and read/write behavior.
7. Output, destination, and bounded observed sample.
8. Inspection command or script path.
9. Verification criterion, observed result, and comparison tolerance if applicable.
10. Interpretation, simplifications, and limitations.
11. Learner confirmation or an explicit pending state.
12. Unresolved questions.

Use nested subsections for nested operations only when needed for understanding.
For each run, record the raw input and any chat-template or special-token preparation.
Record versions, model revision, device, dtype, caching, inference mode, and check results.

For inspected assets, record actual location, format, size, purpose, user of the asset, and supporting evidence.
Distinguish these resource types:

- Persistent downloaded files and installed library source code.
- Learned weight tensors in RAM and static configuration metadata.
- Tokenizer mappings and generated token IDs.
- Temporary tensors from one invocation and persistent model weights.
- Cache state within this invocation and future optimization across invocations.

Not every conceptual asset has a separate file.
Retain findings after the active task changes.
Explain corrections and identify superseded conclusions when relevant.
Give raw logs enough context to support interpretation.
Do not replace evidence with a prewritten tutorial, speculative inventory, status checklist, or full log dump.

## File changes, errors, and handoffs

Inspect the existing file and current milestone before editing when access permits.
Change only material relevant to the current operation or a directly blocking error.
State the command actually run. Do not invent successful test results.
If execution is unavailable, provide exact commands and mark execution **not run**.

Diagnose the smallest failing boundary.
Explain the cause, minimal correction, and exact rerun.
Do not hide an unresolved error by changing frameworks, models, or operations.
Use the [technical decision process](technical-reference.md#changing-a-technical-decision) if an incompatible configuration requires a change.

Store findings in [project-evidence](../project-evidence/README.md).
Update [current progress](../project-status/current-progress.md) with the latest checkpoint.
Keep baseline rerun instructions reproducible after a successful run.

At a meaningful handoff, record:

1. Last completed technical step, changed code, execution, and verification evidence.
2. Last learner-confirmed concept with the actual confirmation quoted or paraphrased.
3. Current understood/current/hidden diagram.
4. Exact changed file paths.
5. Smallest commands to reproduce the last result.
6. Outstanding issues, distinguishing blockers from optional questions.
7. One concrete next action.
