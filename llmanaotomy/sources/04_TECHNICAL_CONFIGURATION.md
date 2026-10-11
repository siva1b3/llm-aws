# 04 — Technical Configuration and Decisions

**Document role:** Authoritative infrastructure, model, software, and reproducibility choices  
**Update when:** EC2/model/dependency choices change, installed versions are observed, asset paths are known, or actual runtime settings are verified.

> **Important:** This file records *selected configuration*, not evidence that an EC2 instance exists, dependencies have been installed, or the model has run. Record observed state separately from intended state.

## 1. Selected AWS environment

| Setting | Selected value | Verification state |
| --- | --- | --- |
| Provider | AWS EC2 | Decision recorded; provision not verified |
| Region | `ap-south-1` (Mumbai) | Decision recorded |
| Instance | `t3.large` | Decision recorded; provision not verified |
| vCPUs | 2 | Instance specification |
| Memory | 8 GiB | Instance specification |
| Accelerator | No GPU | CPU-only by design |
| Operating system | Ubuntu Server 24.04 LTS | Intended AMI/OS; not inspected |
| Architecture | x86-64 | Chosen platform |
| Storage | 30 GiB gp3 EBS | Intended provision; not inspected |
| Payment | On-Demand | Chosen approach |

**Operational cautions:** T3 is credit-based. Sustained CPU usage may consume credits and can incur surplus-credit costs depending on mode; storage and public IPv4 may have separate costs. Stop the instance when not in use; EBS typically continues to incur storage charges. There is no requirement for public web-serving ports. Do not put secrets in project files or output transcripts.

CPU-only is a deliberate learning choice: small-scale transparency matters more than throughput. Do not add CUDA, GPU drivers, distributed model execution, model quantization, or deployment machinery for Phase 1.

## 2. Selected Hugging Face model

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

## 3. Minimal software decisions

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

## 4. Asset inventory: expected categories, not confirmed filenames

Do **not** pretend the repository has been downloaded. When downloaded, inspect real local file listings and record exact paths, sizes, and purposes in [`05_INFERENCE_INVESTIGATION.md`](05_INFERENCE_INVESTIGATION.md) when each asset is used.

Potential assets include:

- Model architecture/configuration JSON.
- Tokenizer JSON, configuration, special-token mappings, or vocabulary-related files.
- Model parameter files (commonly safetensors) and any index, if present.
- Optional generation configuration or repository documentation.
- Cached files and executable implementations provided by installed Python packages.

The exact set depends on the model revision and download method. Not every named category must be a separate file. Learned embeddings typically live inside model weight tensors, not a standalone token-to-vector text table.

Record the repository revision/commit when actually downloaded for reproducibility.

## 5. Execution and asset lifecycle we will inspect

```text
Downloaded/cached files on disk
               ↓ loading
Python code + tokenizer/model objects + parameters in host RAM
               ↓ a real single forward invocation
Temporary tensors and next-token logits in host RAM
               ↓ one-token selection and decoding
Human-readable decoded token text
```

This is an **overview of resource categories**, not a claim to know every internal numerical transformation yet. Open and examine those internal steps only under the onion-peeling protocol.

## 6. Runtime record — fill only with observed facts

| Field | Current value |
| --- | --- |
| EC2 created / reachable | **Not verified** |
| Ubuntu release (`/etc/os-release`) | Not inspected |
| Python executable/version | Not inspected |
| Installed PyTorch version | Not installed/verified in this project record |
| Installed Transformers version | Not installed/verified in this project record |
| Model revision/commit | Not downloaded/verified |
| Model/tokenizer local cache path | Not observed |
| Real parameter dtype/device | Not observed |
| First baseline command | Not created |
| First baseline result | Not run |

Replace individual rows when new evidence arrives; preserve the distinction between **chosen** and **observed**.

## 7. Decision change protocol

If EC2, model, operating system, dependencies, device, or runtime behavior must change:

1. Explain the concrete reason and what it affects.
2. Obtain the learner's approval when changing an already agreed major choice.
3. Modify this file (single source for the technical decision).
4. Update any invalidated claims in investigation evidence and mark any affected tests for rerun.
5. Record the current next action and status in [`06_PROJECT_PROGRESS.md`](06_PROJECT_PROGRESS.md).

Do not replace the chosen model or machine merely because an unrelated tool is more fashionable or familiar.
