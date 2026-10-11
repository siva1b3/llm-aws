# Technical reference

This document owns selected infrastructure, model, software, and runtime choices.
Update it when a technical decision changes.
Store observed environment status in [current progress](../project-status/current-progress.md#runtime-record).
Store detailed runtime evidence in [project-evidence](../project-evidence/README.md).

Source: [technical configuration](../../sources/04_TECHNICAL_CONFIGURATION.md).
These are selected settings and recorded expectations. They do not prove provisioning, installation, downloading, or execution.

## Selected AWS environment

| Setting | Selected value | Qualification |
| --- | --- | --- |
| Provider | AWS EC2 | Provisioning not verified |
| Region | `ap-south-1`, Mumbai | Decision recorded |
| Instance | `t3.large` | Provisioning not verified |
| vCPUs | 2 | Source-recorded instance specification |
| Memory | 8 GiB | Source-recorded instance specification |
| Accelerator | No GPU | CPU-only by design |
| Operating system | Ubuntu Server 24.04 LTS | Intended machine image, not inspected |
| Architecture | x86-64 | Chosen platform |
| Storage | 30 GiB gp3 EBS | Intended provision, not inspected |
| Payment | On-Demand | Chosen approach |

The source records these operational cautions:

- T3 uses CPU credits. Sustained use may consume credits.
- Depending on mode, surplus credits can incur costs.
- Storage and public IPv4 may have separate costs.
- Stop the instance when unused. EBS typically continues to incur storage charges.
- Public web-serving ports are not required.
- Do not place secrets in project files or output transcripts.

This revision preserves those cautions without a new pricing review.
CPU-only execution favors transparency over throughput.
Do not add CUDA, GPU drivers, distributed execution, quantization, or deployment machinery for Phase 1.

## Selected model

| Setting | Selection or source-recorded expectation |
| --- | --- |
| Model ID | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| Model page | [Hugging Face model repository](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) |
| Architecture family | Dense decoder-only causal Transformer |
| Parameters | Approximately 135M |
| Transformer blocks | Expected 30, check downloaded `config.json` |
| Hidden size | Expected 576, check downloaded `config.json` |
| Vocabulary size | Expected 49,152, check downloaded tokenizer/configuration |
| Device | CPU |
| Initial precision | Prefer ordinary FP32, inspect actual dtype |
| Priority | Learning and inspectability |

FP32 means 32-bit floating-point representation.
The downloaded configuration and actual implementation take precedence over these recorded properties.
Investigate differences and update the affected owner with observed evidence.
The architecture may include grouped-query attention and rotary position embeddings.
Study them only when the learner reaches the relevant operation.

The source selects this model for practical CPU use with genuine pretrained weights and multiple Transformer blocks.
It retains tokenization, attention-related computation, and next-token logits.
An instruction-tuned model may prefer its chat template for conversation.
A fixed raw text prompt is adequate for this experiment if the exact preparation is documented.

## Selected software

| Component | Choice | Purpose |
| --- | --- | --- |
| Implementation | Python 3 | Model code and numerical inspection |
| Tensor library | PyTorch | Tensor execution and inspection |
| Model integration | Hugging Face Transformers | Baseline using pretrained weights |
| Tokenizer | Compatible tokenizer from the model repository | Real token mappings |
| Inspection | Terminal output first | Minimal setup |
| Charts | Only when needed | Support the current explanation |

Record precise installed versions after installation.
Prefer a reproducible dependency list or lockfile after the baseline works.
Avoid arbitrary execution of remote model code.
Do not enable `trust_remote_code=True` without a documented necessity and review.

## Initial runtime settings

The fixed input is:

```text
The capital of France is
```

- Use batch size 1 unless a documented reason requires a change.
- Set `model.eval()`.
- Use `torch.inference_mode()` or an equivalent context without gradient computation.
- Execute on CPU.
- Perform one explicit causal-language-model forward call.
- Read next-token logits from the last valid input position.
- Select the highest-scoring token ID with deterministic argmax.
- Decode only that ID. Report fragments, spaces, and special tokens accurately.
- Keep optional caching outside the learning path when it hides the single-pass computation.
- If caching is disabled, record the actual flag and behavior after validation.

Check tensor dimensions and tokenizer behavior before relying on assumptions.
Record any chat template or special-token preparation that changes the raw prompt.
Observed dtype, device, versions, cache paths, and model revision require runtime evidence.

## Expected asset categories

These categories are expectations, not an observed file inventory:

- Architecture and configuration JSON.
- Tokenizer JSON, configuration, special-token mappings, or vocabulary files.
- Parameter files, commonly safetensors, and an index if present.
- Optional generation configuration and repository documentation.
- Cached files and executable implementations from installed Python packages.

The actual set depends on revision and download method.
One category need not correspond to one file.
Learned embeddings typically reside in weight tensors rather than a token-to-vector text file.
After downloading, record the repository revision or commit and actual paths, sizes, and purposes.
Use the [investigation requirements](implementation-rules.md#investigation-records) for evidence capture.

## Planned resource lifecycle

```text
Downloaded/cached files on disk
  → loaded code, tokenizer/model objects, and parameters in host RAM
  → temporary tensors and logits from one model invocation
  → one selected token ID and decoded text
```

RAM is working memory used by the running program.
This diagram describes resource categories. It is not an observed numerical trace.
Open internal computations only through the [learning protocol](learning-method.md#mandatory-onion-peeling).

## Changing a technical decision

For changes to EC2, model, operating system, dependencies, device, or runtime behavior:

1. Explain the concrete reason and affected work.
2. Obtain learner approval before changing an agreed major choice.
3. Update this technical reference.
4. Correct affected evidence and mark invalidated tests for rerun.
5. Update current progress and its next action.

Do not replace the model or machine because another tool is more familiar or fashionable.
