# Final learning outcomes for Phase 1

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
