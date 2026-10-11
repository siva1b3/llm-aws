# Minimal software decisions

| Component | Choice | Reason |
| --- | --- | --- |
| Implementation | Python 3 (record exact installed version later) | Direct access to model code and numeric introspection |
| Tensor library | PyTorch | Run/inspect tensor operations |
| Model integration | Hugging Face Transformers | Establish baseline with genuine pretrained model |
| Tokenizer | Model-compatible tokenizer from its repository | Use real model token mappings |
| Inspection | Simple terminal output first | Minimize setup and distraction |
| Optional charts | Add only at point of demonstrated need | Avoid premature dependencies |

Avoid arbitrary remote-model code execution. Do not enable `trust_remote_code=True` without a documented necessity and review. Record precise library versions **after installation**, preferably along with a reproducible dependency list/lockfile once the first baseline works.

### Runtime choices for initial single forward pass

- Fixed input string: `The capital of France is`
- One example at a time: batch size 1, unless a documented reason emerges.
- `model.eval()` and `torch.inference_mode()` or equivalent no-gradient context.
- CPU execution.
- One explicit causal-LM model forward call (not an opaque multi-token `generate()` loop).
- Read next-token logits from the model's last valid input position.
- Select the highest-scoring next-token ID (`argmax`) deterministically.
- Decode only that selected token ID; report special tokens or fragments honestly.
- Keep optional caching out of the learning path when it obscures the single-pass computation; if disabled, record the actual flag and behavior.

Any actual code should validate assumptions about tensor dimensions and tokenizer behavior before relying on them.
