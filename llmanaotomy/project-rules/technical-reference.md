# Technical reference — selected configuration

This document owns governing technical choices and expected resource categories. **Selected does not mean observed.**
Check [current progress](../project-status/current-progress.md) for verification state. Record actual environment details and reproduction commands in [project evidence](../project-evidence/README.md).

## Contents

- [AWS environment](#aws-environment)
- [Model](#model)
- [Software](#software)
- [Initial runtime settings](#initial-runtime-settings)
- [Expected assets](#expected-assets)
- [Resource lifecycle](#resource-lifecycle)
- [Decision changes](#decision-changes)

## AWS environment

| Setting | Selected value |
| --- | --- |
| Provider | AWS EC2 |
| Region | `ap-south-1` (Mumbai) |
| Instance | `t3.large` |
| Specified capacity | 2 vCPUs and 8 GiB memory |
| Accelerator | No GPU |
| Operating system | Ubuntu Server 24.04 LTS |
| Architecture | x86-64 |
| Storage | 30 GiB gp3 EBS |
| Payment | On-Demand |

These are recorded choices and specifications. They do not establish actual provision, reachability, capacity, or installed operating system.

CPU execution prioritizes transparency over throughput. Do not add CUDA, GPU drivers, distributed execution, quantization, or deployment machinery for Phase 1.

T3 uses CPU credits. Sustained use may consume credits and incur surplus-credit charges, depending on mode.
Storage and public IPv4 may incur separate charges. Stop the instance when unused. EBS typically continues to incur storage charges.
Public web-serving ports are unnecessary. Keep secrets out of project files and output transcripts.

## Model

| Setting | Selected value or expectation |
| --- | --- |
| Model | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| Repository | [Hugging Face model page](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) |
| Architecture family | Dense decoder-only causal Transformer |
| Approximate parameter count | 135M |
| Expected block count | 30, subject to downloaded configuration |
| Expected hidden size | 576, subject to downloaded configuration |
| Expected vocabulary size | 49,152, subject to actual tokenizer/configuration |
| Device | CPU |
| Initial precision | Prefer ordinary FP32, then inspect actual dtype |
| Priority | Learning and inspectability rather than response quality |

Actual downloaded configuration and executing source take precedence over expected properties. Investigate differences and record observed values in evidence.
Revise a governing choice here only through the decision protocol.
Correct an expected property here when actual configuration disproves it. Link to the supporting observation.

The model may use grouped-query attention and rotary position embeddings. Examine them only when the relevant operation becomes the current lesson.

This model offers genuine pretrained weights, multiple blocks, tokenization, attention computation, and next-token logits at a practical CPU scale.
An instruction-tuned model may prefer a chat template for conversations. Fixed raw text is adequate for this experiment if its exact form is recorded.

## Software

| Component | Choice | Purpose |
| --- | --- | --- |
| Implementation | Python 3 | Numeric inspection and direct source access |
| Tensor library | PyTorch | Execute and inspect tensor operations |
| Model integration | Hugging Face Transformers | Establish the pretrained-model baseline |
| Tokenizer | Compatible tokenizer from the model repository | Use actual token mappings |
| Inspection | Terminal output first | Minimize setup and distraction |
| Charts | Add only for demonstrated need | Clarify the current lesson |

Record precise installed versions after actual validation. Retain a reproducible dependency list or lockfile once the baseline works.
Avoid arbitrary remote-model code execution. Do not enable `trust_remote_code=True` without documented necessity and review.

Keep verified commands, versions, asset paths, and actual runtime settings with their [evidence records](../project-evidence/README.md#recording-requirements).
Link the baseline record here after a verified run. No baseline command or runtime evidence exists in the current project record.

## Initial runtime settings

- Fixed input: `The capital of France is`.
- Use batch size 1 unless a documented reason requires a change.
- Use `model.eval()` and `torch.inference_mode()` or equivalent no-gradient mode.
- Execute on CPU.
- Make exactly one explicit causal-LM forward call.
- Read logits from the last valid input position.
- Select one token ID with deterministic `argmax`.
- Decode only that ID and report fragments or special tokens honestly.
- Keep optional caching outside the lesson when it obscures the single-pass computation.
- If caching is disabled, record the actual flag and behavior.

Validate tensor dimensions and tokenizer behavior against actual output. The [project guide](project-guide.md#execution-discipline) defines inspection requirements.

## Expected assets

These categories are expectations, not confirmed local filenames:

- Architecture and configuration JSON.
- Tokenizer data, configuration, special-token mappings, and vocabulary-related files.
- Parameter files, commonly safetensors, and any index that exists.
- Optional generation configuration and repository documentation.
- Cached files and executable implementations from installed Python packages.

Inspect actual listings after download. Record exact paths, sizes, purposes, and repository revision or commit in an evidence record.
Connect each asset to its executing operation when that operation is inspected.

The exact set depends on revision and download method. A category need not correspond to a separate file.
Learned embeddings usually reside inside parameter tensors rather than a separate token-to-vector text table.

## Resource lifecycle

```text
Downloaded/cached files on disk
                ↓ loading
Python code + tokenizer/model objects + parameters in host RAM
                ↓ one real forward call
Temporary tensors + next-token logits in host RAM
                ↓ selection and decoding
Decoded one-token text
```

This diagram describes resource categories. It does not claim observed execution or explain every internal numerical transformation.
Inspect those transformations through the teaching sequence in the project guide.

## Decision changes

For changes to EC2, model, operating system, dependencies, device, or runtime behavior:

1. Explain the concrete reason and affected work.
2. Obtain learner approval for a change to an agreed major choice.
3. Update the affected choice in this document.
4. Correct invalidated evidence and mark affected tests for rerun.
5. Update current progress and its one next action.

Do not replace the chosen model or machine merely because another tool is more familiar or fashionable.
