# Selected Hugging Face model

| Setting | Value |
| --- | --- |
| Model ID | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| Model URL | https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct |
| Architecture family | Dense decoder-only causal Transformer |
| Parameter count | Approximately 135M |
| Number of Transformer blocks | 30 (verify against downloaded `config.json`) |
| Hidden size | 576 (verify against downloaded `config.json`) |
| Vocabulary size | 49,152 (verify against downloaded tokenizer/config) |
| Device | CPU |
| Initial parameter precision | Prefer ordinary FP32 CPU execution; inspect actual dtype |
| Priority | Learning and inspectability, not response quality |

The downloaded **actual config and source implementation take precedence** over a hardcoded property shown here. If any field differs, investigate and update this file with the observed value. The architecture may include grouped-query attention and rotary position embeddings; study them only when their current operation is opened.

### Why this model

It is small enough to be practical on CPU while retaining genuine pretrained weights, multiple Transformer blocks, tokenization, attention-related computation, and next-token logits. An instruction-tuned model may prefer its chat template for conversations; a simple fixed raw text prompt is adequate for the single-pass anatomy experiment, provided we document exactly what input was supplied.
