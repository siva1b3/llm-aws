# Phase-1 execution contract

The minimal baseline is one model forward call from tokenized input and one post-processing step that chooses/decodes **one** token:

```text
Text → token IDs → ONE model forward call → logits → one ID → decoded text
```

For the selected causal language model, compare next-token scores from the appropriate final input position (typically `logits[0, -1, :]` for a single unpadded sample). The shape and indexing must be verified against the actual runtime output. Use deterministic argmax initially, not a stochastic generation loop.

- Set `model.eval()` and use `torch.inference_mode()` (or equivalent no-grad inference context).
- Run on CPU; do not silently switch devices or introduce GPU code.
- Do not call `generate()` as the core baseline if it hides repeated forward calls. Use the model forward method directly.
- Do not assume all outputs are printable words: one decoded token may be a text fragment, leading space, or special token.
- If the model has cache-related options, choose settings consistent with one-pass inspection and record them in the technical configuration after real validation.

The actual model implementation and configuration are authoritative over illustrative architecture diagrams.
