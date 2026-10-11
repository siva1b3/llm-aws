# Current phase: exactly one forward pass

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
