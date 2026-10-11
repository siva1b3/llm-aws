# Project guide

This document owns project goals, phase boundaries, exclusions, and success criteria.
Update it when the user changes these agreements.

Source: [project overview](../../sources/01_PROJECT_OVERVIEW.md).

## Goal

Understand how input text passes through a small pretrained decoder-only Transformer and becomes a next-token prediction.
Use real code and runtime evidence to understand every major operation in one forward pass.
Inspect representations, computations, software, learned parameters, and supporting assets.

This is an investigation and teaching project.
Architecture labels, a high-level API call, or convincing output alone do not meet the goal.

For each major operation, explain:

```text
Input → representation → computation → assets/parameters → output → next operation
```

A forward pass is one model invocation that computes output scores from prepared input.
A token is a unit of text represented by an integer ID.
Logits are the model's numerical scores before conversion to probabilities.

## Phase 1: one complete forward pass

Begin with this conceptual view:

```text
Input text → [LLM black box] → one-token output text
```

The actual endpoint is:

```text
Input text → token preparation → ONE full model forward pass → logits
                                                               ↓
                                                select ONE ID → decode text
```

Selection and decoding occur after the model produces logits. They do not require another forward pass.
The decoded token may be a word fragment, whitespace, or a special token.

### Included work

- Set up the environment and load one small pretrained model with its tokenizer.
- Inspect relevant model files and assets when needed.
- Prepare a fixed prompt and inspect tokenization and numerical representations.
- Execute every required Transformer block once, in the correct sequence.
- Examine one representative block's important internal operations in detail.
- Recognize that other blocks share the broad structure and have their own learned parameters.
- Inspect real values, shapes, data types, parameters, outputs, and relevant source code.
- Select one token with deterministic argmax initially. Argmax selects the index with the highest score.
- Decode only that token.
- Reproduce important individual computations and compare them with the model where practical.

### Deferred work

- Feeding the selected token into another invocation.
- Repeated autoregressive passes, multi-token responses, stopping logic, and sampling comparisons.
- Reuse or optimization of key/value caches across forward passes.
- Training, fine-tuning, backpropagation, optimizers, and dataset preparation.
- Agents, tools, retrieval, Mixture of Experts, distributed inference, and multiple GPUs.
- Production serving, scaling, and unrelated optimization.

If caches appear in the actual one-pass implementation, identify their role without opening a separate performance topic.
Discuss a deferred topic briefly only when necessary to prevent a misunderstanding.
Do not implement or teach it as the next milestone without explicit user approval.

## Reference system

Use a small, dense, decoder-only Transformer on CPU with pretrained Hugging Face weights and model files.
The [technical reference](technical-reference.md) owns the selected model and environment.
Selected settings do not prove provisioning or execution.

A high-level library may establish the baseline.
Then examine its actual computations one black box at a time.
Keep real inference faithful to the selected model.
Label simplified educational examples and state their limits.

## Required learning outcomes

At completion, the learner can:

1. Draw and explain the single-pass path from text to logits and one decoded token.
2. Identify executable code, persistent files, parameters, and runtime tensors for each major transformation.
3. Explain changes among text, token IDs, vectors, tensors, scores, and the selected token.
4. Explain one representative block and why execution still uses all blocks.
5. Inspect real intermediate values, shapes, data types, and selected parameter values.
6. Reproduce important operations in readable code and check them against the model where practical.
7. Distinguish model computation from tokenization, selection, and decoding.
8. Explain the complete process to another developer without major unexplained operations.

The learner need not memorize millions of values or reproduce every library kernel instruction.

## Completion criteria

Use a fixed example such as the prompt selected in the [technical reference](technical-reference.md#initial-runtime-settings).
Trace the real request through one forward pass and the next-token prediction path.

Completion requires all of these conditions:

- The code ran and the evidence was inspected.
- Every major black box reached a meaningful computational explanation.
- The [implementation checks](implementation-rules.md#verification-and-advancement-gate) passed.
- The learner explicitly confirmed understanding of the assembled flow.

Do not treat Phase 1 output as an entire generated response.
A later phase may examine repeated passes after separate authorization.
Future interest does not expand the current scope.
