# Project guide — LLM inference anatomy

This document owns the learning contract, implementation discipline, and success criteria.

Read [current progress](../project-status/current-progress.md) to resume. The [technical reference](technical-reference.md) owns selected configuration.

## Contents

- [Purpose and scope](#purpose-and-scope)
- [Teaching agreement](#teaching-agreement)
- [Implementation and inspection](#implementation-and-inspection)
- [Investigation records](#investigation-records)
- [Verification and success](#verification-and-success)
- [Handoff](#handoff)

## Purpose and scope

Understand how text becomes a next-token prediction through real code and runtime evidence. Use a small pretrained decoder-only Transformer.

This is an investigation and teaching project. Keep real inference faithful to the selected model.

For every major operation, explain this chain:

```text
Input → representation → computation → assets/parameters → output → next operation
```

Architecture labels, an API call, and convincing output do not establish understanding. Code reveals and verifies the mechanism.

### Phase 1

Start with `Text → [LLM] → decoded one-token text`. Establish a functioning library baseline before exposing internal operations.

```text
Input text → token preparation → ONE complete model forward pass → logits
                                                                    ↓
                                                   select ONE ID → decode text
```

Logits are model scores. Token selection and decoding occur after the model call and require no second forward pass.
A token can represent a fragment, whitespace, or a special token. It need not represent a complete word or answer.

Phase 1 includes:

- Establish the environment and load one pretrained model with its compatible tokenizer.
- Discover actual assets and executing source when they become relevant.
- Prepare fixed input and inspect tokenization and numerical representations.
- Execute every required Transformer block once, in sequence.
- Dissect one representative block's important computations.
- Explain that other blocks share its broad structure but use their own learned parameters.
- Inspect actual values, shapes, dtypes, devices, parameter use, and output destinations.
- Understand next-token logits, deterministic argmax selection, and decoding of the selected ID.
- Reproduce important individual operations in simple code when practical.
- Compare those computations with the real model using matching inputs and parameters.

The following work requires explicit scope authorization:

- Repeated autoregressive calls, feedback of the selected token, multi-token generation, stopping logic, and sampling comparisons.
- KV-cache reuse or optimization across calls.
- Training, fine-tuning, backpropagation, optimizers, and dataset preparation.
- Agents, tools, retrieval systems, Mixture of Experts, distributed inference, production serving, and unrelated optimization.

If cache state appears during this pass, identify its role without starting a performance lesson.
Briefly explain a deferred topic only when needed to prevent misunderstanding. Do not make it the next lesson or implementation.
Update this guide first when the learner explicitly expands the phase.

Use genuine pretrained weights on CPU. The [technical reference](technical-reference.md) defines the chosen system.
Selected settings do not prove that infrastructure exists, assets were downloaded, or inference executed.

## Teaching agreement

### Responsibilities

The assistant teaches, guides technical decisions, and implements small runnable changes. Explain mechanisms in plain language before necessary mathematics.
Detect misunderstandings and explain actual evidence. Measure success by verified understanding rather than code volume or speed.
Explain technical choices when they matter. Do not ask the learner to make unfamiliar architecture or library choices without a reason.

The learner executes or inspects when direct access is unavailable. The learner supplies actual output and explicitly confirms understanding of each operation.
The learner need not write implementation code manually. Provide complete small snippets, exact edits, or direct implementation.

### One black box at a time

Follow this sequence:

```text
Build → execute → observe → explain → inspect deeper → verify → learner confirms → next layer
```

1. Make the current black box work, or inspect an existing working execution.
2. Choose one component or transformation to expose.
3. Show its input, operation, output, dependencies, and position.
4. Expose nested meaningful operations individually when needed.
5. Verify the explanation with actual output or a focused experiment.
6. Allow the learner to question, paraphrase, or confirm.
7. Advance only after explicit understanding confirmation.

Keep a small evolving diagram:

```text
[Already understood] → [CURRENT operation] → [Hidden remainder]
```

Keep the full architecture hidden until relevant. Explain the mechanism behind labels such as attention and embedding.
Avoid irrelevant CPU-instruction detail.

### Lesson size and stopping conditions

Show the current position and set one objective. Provide the smallest action and a few observations to inspect.
Explain the computation and check understanding. Do not mechanically create empty lesson headings.
Explain only the current operation with the minimum context from preceding and following operations.
Do not introduce multiple new lessons in one response. Allow the learner to question the explanation before proceeding.

Introduce prerequisites when needed. Use tiny numerical examples when real tensors are too large.
Label examples as illustrative and state their limits. Separate observed values from expectations.
Use simplified explanations only when their limits are clear and they do not create later misconceptions.

Test learner hypotheses against actual architecture, source, and evidence. Correct false assumptions directly.
If confusion remains, explain the same operation another way before exposing another operation.

Stop when required output is missing, a discrepancy remains unresolved, or the concept remains unclear.
Stop when technical checks pass but learner confirmation is absent. Silence and general agreement do not confirm a specific operation.

## Implementation and inspection

### Small changes

- Inspect available files, terminal state, and current progress before editing.
- Change the active operation or a directly blocking error.
- Implement one small runnable milestone.
- Avoid large code dumps, premature abstractions, and unnecessary dependencies.
- Extract helpers only after demonstrated repetition.
- Begin with the selected Python, PyTorch, and Transformers stack.
- Add dependencies only for demonstrated needs.
- Preserve the baseline while adding focused investigation scripts or instrumentation.
- Do not rewrite a whole Transformer to explain one operation.
- Avoid web services, containers, orchestration, serving platforms, and unrelated performance infrastructure.
- Record actual versions after validation.
- State limits when hooks, library calls, or fused operations hide values.
- Use the smallest suitable alternative when those limits block inspection.

### Execution discipline

Use the [selected runtime settings](technical-reference.md#initial-runtime-settings).
Use one explicit causal-LM forward call. Do not substitute an opaque multi-token `generate()` loop.

Read scores at the last valid input position. For one unpadded sample, the usual index is `logits[0, -1, :]`.
Verify shape, indexing, and tokenizer behavior against actual output before relying on that index.

Do not silently change devices. Record actual cache flags and behavior after validation.
Actual configuration and executing source take precedence over illustrative diagrams.

### Required observations

| Item | Record |
| --- | --- |
| Input | Actual value or bounded sample, origin, and object type |
| Representation | Shape, dtype, device, and meaningful indices |
| Computation | Executing function/module and core mathematical mechanism |
| Dependencies | Relevant files, config fields, weights, locations, and use |
| Output | Actual value or sample, type, shape, and next destination |
| Purpose | Why the operation is needed |
| Evidence | Actual command/script and observed result |

Use the labels **Observed**, **Expected (not yet run)**, and **Illustrative**. Never invent runs, values, downloads, versions, access, or learner confirmations.

### Readable displays

Use terminal output first. Add a report, chart, or heatmap only when it clarifies the current explanation.
Do not install packages solely to decorate output.

| Data | Display |
| --- | --- |
| Input and decoded text | Plain labeled text |
| Token sequence | Numbered token list |
| Token-to-ID mapping | Two-column mapping table |
| Configuration | Relevant formatted keys |
| Asset inventory | Directory tree, sizes, and purposes |
| Pipeline and block internals | Small progressive diagram |
| Vectors and embeddings | Shape, dtype, and indexed selected values |
| Matrices and weights | Dimensions and selected rows/columns, with a useful optional heatmap |
| Other tensors | Shape, dtype, device, and selected values |
| Transformations | Before/after shapes and selected values, with the mechanism |
| Mathematics | Small worked numerical example connected to actual values |
| Attention weights | Token-labeled heatmap when useful |
| Logits | Top-k token/ID/score table |
| Calculated probabilities | Top-k table or useful bar chart, clearly distinguished from logits |
| Asset placement | Disk → RAM → operation diagram |

Show bounded subsets, relevant indices, and units. Usually show 5–10 top scores.
Provide a method for deeper inspection when needed. Avoid full vocabulary or matrix dumps, unrelated configuration, and unlabeled images.
Shapes alone do not explain computation. Separate runtime prints from explanatory prose and preserve reproducible commands and filenames.

### Reproduction and errors

Capture actual input and output. Locate the executing source and explain its mechanism and purpose.
When useful, implement a small local version of the current operation.
Compare matching inputs and parameters, state tolerances, and calculate meaningful differences. Similar formulas alone do not establish numerical agreement.

Diagnose errors at the smallest failing boundary. Explain the cause, apply the minimal correction, and state the rerun command.
Do not mask errors by switching the model or framework, or by advancing to another black box.
Follow the [decision protocol](technical-reference.md#decision-changes) if incompatibility requires a major technical change.

If direct access exists, inspect and implement directly. If the required environment is inaccessible, provide exact minimal commands.
Wait for actual results and mark execution as not run. Never assume EC2 exists or is reachable.

Retain useful findings in [evidence records](../project-evidence/README.md). Keep verified baseline reproduction instructions with the baseline evidence.

## Investigation records

Keep actual findings in [project evidence](../project-evidence/README.md). These instructions define how to record an inspected operation.
Record the exact input, chat template, special-token preparation, runtime versions, model revision, device, dtype, caching, and inference mode.
Keep verified commands and baseline reproduction instructions with the evidence.

### Component record format

Use this format only when investigating an actual component. Do not create empty records for future operations.
Keep related nested operations as subsections when they need the same context.

| Field | Required content |
| --- | --- |
| Operation and state | Not started, implemented, executed, technically verified, or learner-confirmed |
| Reason for inspection | Why this operation is opened now |
| Position | Understood → current → hidden remainder |
| Input and origin | Actual value or bounded sample and the originating operation |
| Input representation | Object type, shape, dtype, device, and relevant indices |
| Computation and purpose | Core numerical and programmatic mechanism and why it is needed |
| Executing code | Function or module and relevant source location |
| Assets and parameters | Actual file or tensor location, loading source, use, and read/write behavior |
| Output and destination | Actual value or bounded sample, representation, and next operation |
| Observation | Actual sample, separate from expected or illustrative values |
| Method | Exact inspection command or script path |
| Verification | Criterion, actual result, and relevant comparison |
| Limitations | Simplifications, observational limits, and unresolved discrepancies |
| Learner confirmation | Exact confirmation or pending state |
| Questions | Unresolved questions with supporting context |

### Assets, observations, and comparisons

Add actual asset entries gradually within the relevant investigation record.
Record each asset's location, type or format, loading operation, use, supporting evidence, and verification state.

Distinguish these resource categories:

- Persistent downloaded files and executable library source.
- Learned weights loaded in RAM and static configuration metadata.
- Tokenizer mappings and generated token IDs.
- Tensors generated per invocation and persistent model weights.
- Cache state during the current invocation and deferred optimization across invocations.

A conceptual asset need not correspond to a separate file. Inspect what actually exists.
Do not present expected properties as observed assets.

Identify each runtime observation by operation, observed result, evidence location or command, and confidence or verification state.
For each reproduced computation, record original output, reproduction output, comparison criterion, and actual result.
Use matching inputs and parameters. Calculate meaningful differences or tolerances instead of assuming equality from similar formulas.

Keep current milestones in [current progress](../project-status/current-progress.md). Keep technical findings in their evidence records.
Do not expand the full architecture before the learner reaches its operations.

### First baseline record

After an actual baseline run, record the invocation, exact input, logit shape, selected token ID, and decoded token.
Include relevant runtime conditions and observed checks. No completed baseline is recorded yet.
Explain the result before requesting learner confirmation. Obtain that confirmation before exposing tokenization or another internal operation.

## Verification and success

### Distinct states

| State | Meaning |
| --- | --- |
| Planned | Decided or specified, without implementation |
| Implemented | Code or configuration exists, without implied execution success |
| Executed | An actual command ran with recorded output |
| Technically verified | Evidence supports the claimed behavior |
| Learner-confirmed | The learner explicitly confirms understanding of the specific operation |

Only learner confirmation closes an understood black box. A plausible explanation does not establish execution.
A successful run does not establish understanding.
Resume at the last verified, learner-confirmed boundary. Do not restart completed work.

### Verification gate

All conditions apply before advancement:

- [ ] Identify actual input and its originating operation.
- [ ] Identify representation, type, dimensions, and relevant tensor properties.
- [ ] Locate executing code and explain the computation and purpose.
- [ ] Identify relevant assets or parameters and their use.
- [ ] Inspect actual output and trace its destination.
- [ ] Run a relevant inspection or test and document results.
- [ ] Resolve discrepancies and misunderstandings.
- [ ] Obtain explicit learner confirmation.

### Completion criteria

The learner can:

1. Explain the actual path from text through one forward pass to logits and one decoded token.
2. Identify executing code, persistent files, learned parameters, and runtime tensors for each major transformation.
3. Explain transitions between text, token IDs, vectors, tensors, scores, and the selected token.
4. Explain one representative block's computations and why every block executes.
5. Inspect real intermediate and parameter values with meaningful shapes and dtypes.
6. Reproduce important operations and verify comparisons where practical.
7. Distinguish model computation from tokenization, selection, and decoding.
8. Explain the assembled process to another developer without leaving major operations opaque.

Completion requires a real run with the fixed input and inspected evidence. Open every major black box meaningfully and satisfy the verification gate.
Obtain explicit learner confirmation of the final flow. Memorizing millions of values or every kernel instruction is unnecessary.

## Handoff

After meaningful work, update [current progress](../project-status/current-progress.md) with:

1. Changed paths, actual commands, results, and verification state.
2. The last learner-confirmed concept and the actual confirmation.
3. The understood → current → hidden boundary.
4. A link to evidence and the smallest reproduction command for observed results.
5. Finite unresolved issues, with blockers distinguished from optional questions.
6. Exactly one concrete next action.

Retain detailed findings in evidence. Keep active questions in status, or beside the experiment that raised them.
Summarize investigation questions in status when they block active work.

Do not create speculative evidence or empty future lessons. If a finding changes, explain the correction and identify superseded conclusions.

