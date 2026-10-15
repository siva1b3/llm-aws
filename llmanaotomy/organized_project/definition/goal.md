# Goal

Status: extracted from supplied sources; wording and derived criteria await human review.
Source: [project overview](../../sources/01_PROJECT_OVERVIEW.md), sections 1–6.

## Purpose and user

The learner wants to understand real LLM inference through code and runtime evidence.
The assistant teaches, implements small changes, and explains the observed mechanism.
The learner reviews results and explicitly confirms understanding.
This is a teaching investigation, not a production inference service.

For each major operation, explain:
**Input → representation → computation → assets/parameters → output → next operation.**

## Outcomes and deliverables

Produce a runnable one-forward-pass learning artifact, progressive investigation records, and reproducible observations.
Enable the learner to explain the complete path to another developer.
Code volume, plausible generated text, and memorized architecture labels do not establish success.
Deliverable locations beyond this starting record structure remain undecided.

Included scope:

- Set up the selected environment and load one small pretrained model and its compatible tokenizer.
- Discover relevant assets and files when the current operation requires them.
- Prepare the fixed input and inspect tokenization and numerical representations.
- Execute every required Transformer block once in the correct sequence.
- Dissect one representative block; explain that other blocks have their own learned parameters.
- Inspect actual values, shapes, dtypes, devices, parameter usage, outputs, and relevant source code.
- Interpret next-token logits, select one argmax token ID initially, and decode only that token.
- Reproduce selected important computations in simple code and compare against the model where practical.

## Exclusions and deferred scope

Do not feed the predicted token back into another forward invocation.
Repeated autoregression, multi-token generation, stopping logic, and sampling comparisons are deferred.
Cross-pass KV-cache reuse and optimization are deferred.
Identify caches encountered within the real pass only as needed to avoid misunderstanding.
Training, fine-tuning, backpropagation, optimizers, and dataset preparation are excluded.
Agents, tool use, retrieval, Mixture of Experts, distributed inference, and multi-GPU execution are excluded.
Production serving, scaling, and optimization unrelated to learning are excluded.
No web app, microservices, Docker, Kubernetes, complex instrumentation, or performance pipeline is required.
Brief clarification of a deferred topic does not authorize implementing or teaching it as the next milestone.
The learner need not memorize parameters or reproduce library kernels instruction by instruction.
A later generation phase requires an explicit scope change.

## Completion criteria

These new stable IDs express the source outcomes; they are not new approved numerical targets.
All criteria are pending. No implementation evidence or learner confirmation exists in this preparation.

| ID | Observable outcome | Source |
| --- | --- | --- |
| G-001 | For the fixed input, real code executes exactly one full model forward call, selects one next-token ID, and decodes it. Actual output is inspected. | Overview sections 2, 5 |
| G-002 | The learner traces and explains the full text-to-logits-to-one-token path, separating tokenizer, numerical forward pass, selection, and decoding. | Overview section 4, items 1, 3, 7 |
| G-003 | For every major operation, records identify input, representation, executing code, computation, purpose, assets/parameters, output, and destination. | Overview sections 1, 4; implementation sections 3, 7 |
| G-004 | Real intermediate values, shapes, types, devices, and selected parameters are inspected using bounded, readable evidence. | Overview section 4, item 5 |
| G-005 | The learner explains important computations in one representative Transformer block and why all blocks still execute. | Overview section 4, item 4 |
| G-006 | Important selected operations are reproduced and compared with matching model inputs and parameters where practical; limitations are explicit. | Overview section 4, item 6 |
| G-007 | Each major operation satisfies the technical inspection gate, resolves discrepancies, and receives explicit learner confirmation before advancement. | Overview section 5; implementation section 7 |
| G-008 | The learner explains the assembled process to another developer and explicitly confirms understanding of the final flow without unexplained major operations. | Overview section 4, item 8; section 5 |

A decoded token may be whitespace, a fragment, or a special token.
No criterion requires a complete response or the particular answer “Paris”.
Real inference must remain faithful to the chosen model.
Educational simplifications must identify their limits.

See [constraints](constraints.md) for runtime choices, teaching requirements, and the full inspection gate.
