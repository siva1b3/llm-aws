# 01 — Project Overview

**Project:** Understanding LLM inference by building and dissecting a real model  
**Document role:** Authoritative project goals, phase scope, boundaries, and success criteria  
**Current phase:** Phase 1 — one complete forward pass  
**Update when:** The user changes the goal, the current phase, the boundaries, or the definition of success.

## 1. Goal

Understand, through real code and real runtime evidence, how input text travels through a small pretrained decoder-only Transformer and becomes a prediction for the next token. The learner should understand the representations, computations, software, parameters, and supporting assets at every **major operation** of a single forward pass.

This is an **investigation and teaching project**, not a production inference service. Memorizing architecture labels, calling a high-level API, or making a model produce convincing text is not sufficient.

The desired explanatory contract for every major operation is:

> **Input → representation → computation → assets/parameters → output → next operation.**

The code is a means of discovering and verifying that mechanism.

## 2. Current phase: exactly one forward pass

Conceptual starting view:

```text
Human-readable input text → [ BLACK BOX: LLM ] → Human-readable output
```

The actual single-pass endpoint is more precise:

```text
Input text → [token preparation + ONE full model forward pass] → logits
                                                      ↓
                                      select ONE token ID → decode to text
```

The decoder and token-selection operations are application-side steps after the model's logits. They do **not** require a second forward pass. The selected token may be a word fragment or whitespace, not a full answer.

### In scope now

- Environment setup and loading one small **pretrained** model and its tokenizer.
- Discovering and inspecting relevant model assets and files as they become necessary.
- Preparing one fixed input prompt; inspecting tokenization and numerical representations.
- Executing **all** of the model's required Transformer blocks once in the correct sequence.
- Dissecting **one representative Transformer block** down to its important internal operations; recognizing that the other blocks have the same broad structure but their own learned parameters.
- Inspecting actual tensor values, shapes, data types, parameter usage, model outputs, and relevant source code.
- Understanding next-token logits, selecting one token (initially deterministic argmax), and decoding that one token into readable text.
- Reproducing selected important computations in simple code and comparing them with the actual model, when practical.

### Explicitly deferred

- Feeding the newly selected token back for another model invocation.
- Repeated autoregressive forward passes, multi-token response generation, generation stopping logic, and sampling strategy comparisons.
- Reuse and optimization of KV caches **across** forward passes. If caches appear in the actual one-pass implementation, identify their role without turning them into a separate performance topic.
- Training, fine-tuning, backpropagation, optimizer implementation, and dataset preparation.
- Agents, tool use, retrieval systems, Mixture of Experts, distributed inference, multi-GPU execution, and production serving/scaling.
- Optimization techniques unrelated to learning the single-pass computation.

Discuss a deferred topic briefly only if necessary to prevent a misunderstanding. Do not implement or teach it as the next milestone without explicit user approval.

## 3. Chosen reference system

We use a **small, dense, decoder-only Transformer** running on CPU, with pretrained weights and model files from Hugging Face. Specific current choices live only in [`04_TECHNICAL_CONFIGURATION.md`](04_TECHNICAL_CONFIGURATION.md). These are **chosen settings**, not evidence that the machine is provisioned or that inference has executed.

We may use a high-level inference library to get a working baseline, but subsequently open its black boxes and examine actual computations. Real inference must remain faithful to the selected model; simplified educational examples must be clearly labeled as such.

## 4. Final learning outcomes for Phase 1

At the end, the learner should be able to:

1. Draw and explain the real single-pass path from text input to the model's output logits and one decoded token.
2. Show which executable code, persistent files, learned parameters, and runtime tensors participate in each major transformation.
3. Explain how representations change (text, token IDs, vectors, tensors, scores, and a selected token).
4. Explain the computations inside one representative Transformer block and why all the model's blocks are still executed.
5. Inspect real intermediate values, tensor shapes, data types, and selected parameter values.
6. Reproduce important individual operations in readable code and verify their behavior against the model where practical.
7. Distinguish the model's numerical **forward pass** from the surrounding tokenizer, token-selection, and decoder logic.
8. Explain the complete process to another developer without leaving major operations unexplained.

The learner does **not** need to memorize millions of parameter values or reproduce library-level kernels instruction by instruction.

## 5. Completion standard

Using a fixed example such as:

```text
The capital of France is
```

trace the real request through a **single forward pass** and show the next-token prediction path. The exercise is complete only when:

- The code has run and evidence has been inspected, not merely written.
- Every major black box in the phase has been opened to a meaningful computational level.
- Required explanations and checks in [`03_IMPLEMENTATION_RULES.md`](03_IMPLEMENTATION_RULES.md) have been satisfied.
- The learner explicitly confirms understanding of the final assembled flow.

**Do not reinterpret "output text" as an entire model-generated response during Phase 1.**

## 6. Future phases

A later, separately authorized phase may analyze repeated passes and complete autoregressive generation. That future interest does not enlarge the current scope. Change this file first if the user explicitly expands the phase.

## 7. Where other rules live

- Learning style and human-in-the-loop behavior → [`02_LEARNING_METHODOLOGY.md`](02_LEARNING_METHODOLOGY.md)
- Code size, observations, inspection displays, and verification gates → [`03_IMPLEMENTATION_RULES.md`](03_IMPLEMENTATION_RULES.md)
- EC2, model, software versions, and environment decisions → [`04_TECHNICAL_CONFIGURATION.md`](04_TECHNICAL_CONFIGURATION.md)
- Evidence-based inference anatomy notes → [`05_INFERENCE_INVESTIGATION.md`](05_INFERENCE_INVESTIGATION.md)
- Current checkpoint and next action → [`06_PROJECT_PROGRESS.md`](06_PROJECT_PROGRESS.md)
