# LLM Internals — Learning Project

A hands-on study of how a transformer LLM works internally, by loading a real
model and inspecting it one concept at a time.

This is **a learning project, not a production system**. No KV cache, no
batching, no optimization. The goal is understanding, not speed.

---

## 1. Goal

Peel the model open in small, controlled steps. Start from a complete black
box ("text in → text out") and progressively expose internals — tokenization,
embedding, transformer blocks (attention + FFN), normalization, language model
head, sampling, decoding — until every tensor, every weight matrix, and every
operation in one forward pass is visible and named.

**Reference architecture:** Llama-3-style decoder-only transformer
(RMSNorm, RoPE, Grouped-Query Attention, SwiGLU FFN).

**Inference scope:** one prompt, one predicted token, batch size 1.
This keeps every shape simple and every step traceable.

---

## 2. Approach

- **Inspect a real loaded model.** Do not study a from-scratch reference
  implementation. The truth is what the actual weights and forward pass do.
- **PyTorch forward hooks + `print` / DataFrames** to capture inputs, outputs,
  shapes, and dtypes at each step.
- **Incremental peeling.** Each level adds exactly one new concept on top of
  the previous level. Nothing is skipped, nothing is bundled.
- **Real numbers, real shapes.** Every notebook prints actual tensor shapes,
  dtypes, and values from the loaded model — not symbolic placeholders.

---

## 3. Project structure

The project is organized into **phases**, and each phase contains multiple
**levels**. Each level is a single Jupyter notebook that builds directly on
the previous one.

```
phase<PP>_level_<LL>_<short_name>.ipynb
```

Examples:

```
phase01_level_00_opaque_box.ipynb
phase01_level_01_split_decode.ipynb
phase01_level_02_split_tokenize.ipynb
...
```

- `PP` — phase number, zero-padded to 2 digits
- `LL` — level number within the project, zero-padded to 2 digits
- Lowercase, underscores only, no spaces

Levels are numbered globally across phases, so file sort order matches
learning order.

More phases and more levels will be added over time. The structure is
designed to grow without renumbering existing files.

---

## 4. Standard notebook layout

Every level notebook follows the same four-part skeleton so the diff between
two consecutive levels is obvious.

| Part | Purpose |
|---|---|
| Part 0 | Setup — imports, RAM tracker, helper functions, model ID |
| Part 1 | Download stage — what files come from HuggingFace Hub and where they land on disk |
| Part 2 | Load stage — disk → RAM, tokenizer + config + weights, RAM accounting |
| Part 3 | Inference — the actual level content, broken into one cell per pipeline step |

Parts 0, 1, 2 stay essentially identical across levels (cache hits make them
fast). Part 3 is what changes each level — it grows by exactly one step or
one exposed weight matrix.

Each step in Part 3 is presented as three blocks:

- **IN** — input tensor name, shape, dtype, ndim, value summary
- **PROCESS** — the operation, and the static data it consumes
  (`{vocab}`, `{E}`, `{W_Q}`, etc.) with shapes
- **OUT** — output tensor name, shape, dtype, ndim, value summary

When a step is still opaque at a given level, the PROCESS block says so
explicitly. It is never silently skipped.

---

## 5. Hardware setup (AWS EC2)

| Item | Choice | Reason |
|---|---|---|
| Provider | AWS EC2 | Local laptop RAM is constrained by Chrome + VS Code |
| Region | `ap-south-1` (Mumbai) | Closest region, lowest network latency |
| Instance | `t3.xlarge` | 16 GB RAM, 4 vCPU — enough for a 1B-parameter model in fp16 |
| CPU/GPU | CPU-only | Speed is not the goal. CPU keeps cost low and tensors easy to read |
| OS | Ubuntu 22.04 LTS (plain AMI) | No Deep Learning AMI — keep the stack visible |
| Storage | EBS gp3, 30 GB | Enough for model cache (~5 GB) + system + workspace |
| Pricing | On-demand | No reserved or spot — simple to stop and start |
| Stop policy | Stop the instance when not in use | Avoid 24/7 charges |
| Billing safety | CloudWatch billing alarm at $20 | Hard ceiling against forgotten instances |
| Access | SSH + Jupyter port-forward | Jupyter runs on EC2; browser stays on laptop |

**Cost expectation:** ~$15–20/month with discipline about stopping the
instance. ~$130/month if left running 24/7.

---

## 6. Software setup

The full one-time setup, exactly as it was run:

```bash
# OS packages
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv python3-dev build-essential \
                    git curl wget

# Project directory + virtualenv
cd /home/ubuntu/llm-aws/code
python3 -m venv llm-env
source llm-env/bin/activate
python -V
which python

# Pip + PyTorch (CPU build)
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Core ML / HuggingFace stack
pip install transformers accelerate huggingface_hub jupyter ipywidgets
pip install sentencepiece protobuf
pip install jupyter ipykernel
pip install pandas matplotlib psutil

# Register the venv as a Jupyter kernel
python -m ipykernel install --user --name llm-env \
                            --display-name "Python (llm-env)"

# Sanity check
python -c "import torch, transformers, sentencepiece, accelerate; \
print('torch', torch.__version__); \
print('transformers', transformers.__version__); \
print('cuda', torch.cuda.is_available())"

# Disk / memory check
free -h
df -h

# HuggingFace login (required for gated Llama models)
hf auth login
hf auth whoami
hf auth list
hf auth token
```

**Package roles:**

| Package | Why it is needed |
|---|---|
| `torch` (CPU build) | Tensor library and model runtime |
| `transformers` | Model + tokenizer loading from HuggingFace Hub |
| `accelerate` | Device placement, dtype handling during load |
| `huggingface_hub` | Auth + `snapshot_download` for cache control |
| `sentencepiece`, `protobuf` | Tokenizer backends required by some models |
| `jupyter`, `ipykernel`, `ipywidgets` | Notebook environment and progress bars |
| `psutil`, `pandas`, `matplotlib` | RAM tracking, tables, and plots inside notebooks |

---

## 7. Model

| Item | Value |
|---|---|
| Model | `meta-llama/Llama-3.2-1B-Instruct` |
| Access | Gated — requires HuggingFace login and license acceptance |
| Parameters | ~1.24 B |
| Precision | fp16 (loaded with `torch_dtype=torch.float16`) |
| Theoretical weight size | ~2.3 GB |
| Resident RAM after load | ~2.6 GB |
| Architecture | `LlamaForCausalLM` (RMSNorm, RoPE, GQA, SwiGLU) |

A 1B model is small enough to fit comfortably in 16 GB RAM and large enough to
have all the real architectural pieces (multiple transformer blocks, GQA,
SwiGLU FFN, RoPE) — not a toy.

---

## 8. Filesystem layout on the EC2 instance

```
/home/ubuntu/
├── llm-aws/
│   └── code/
│       ├── llm-env/                     # Python virtualenv
│       └── notebooks/
│           ├── phase01_level_00_opaque_box.ipynb
│           ├── phase01_level_01_split_decode.ipynb
│           └── ...
└── .cache/
    └── huggingface/
        └── hub/
            └── models--meta-llama--Llama-3.2-1B-Instruct/
                ├── blobs/               # actual file contents (content-addressed)
                ├── refs/                # branch/tag pointers
                └── snapshots/           # human-readable view; symlinks into blobs/
                    └── <commit-sha>/
                        ├── config.json
                        ├── tokenizer.json
                        ├── tokenizer_config.json
                        ├── special_tokens_map.json
                        ├── generation_config.json
                        ├── model.safetensors
                        └── ...
```

The HuggingFace cache is content-addressed: blobs are stored once and
referenced from snapshot directories via symlinks. Re-downloading the same
revision is a no-op once cached.

---

## 9. Constraints kept simple on purpose

The following are **intentionally excluded** to keep each level small enough
to reason about end-to-end. They can be added later, once the unrolled forward
pass is fully understood.

- **No KV cache.** Every inference recomputes attention over all input tokens.
- **No batching.** Batch dimension is always 1.
- **No streaming generation.** Exactly one new token is predicted per run.
- **Greedy decoding (argmax) only.** No temperature, top-k, top-p.
- **No quantization beyond fp16.** No int8, no GPTQ, no AWQ.
- **No fine-tuning, no LoRA, no training.** Inference only.
- **No MoE, no chain-of-thought scaffolding, no RLHF concepts.**
  These belong to a later study.

---

## 10. Style / response preferences (for AI assistants reading this)

- Direct and technically rigorous. No filler, no praise, no motivational language.
- Start with the concise answer, then expand with reasoning.
- Challenge incorrect assumptions instead of agreeing. Explain why and propose
  a better approach.
- Explain clearly. The reader is new to LLM internals and English is a second
  language. Use precise, simple wording. Avoid idioms.
- Ask a clarifying question when the request is ambiguous instead of guessing.
- When code is involved, give clean examples and call out edge cases and
  important design decisions explicitly.

---

## 11. Status

Update this section per working session. Examples:

- `Phase 1 / Level 00 — complete. Opaque box runs, output verified.`
- `Phase 1 / Level 01 — complete. LLM call split from decode step.`
- `Phase 1 / Level XX — in progress.`
