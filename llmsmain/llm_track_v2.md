# Transformer LLM Internals — Learning Track v2

> Standalone learning track for understanding Llama 3.2 1B internals by inspecting a loaded model on EC2. This file does not depend on any prior file; references to `llm_levels_02.md` are reading aids only. Use this file alone to drive the work.

---

## Section 1 — Goal and constraints

**Goal.** Build a complete mental model of how Llama 3.2 1B Instruct converts input text into one predicted token, by manually reproducing every intermediate tensor from raw weights and elementary PyTorch operations.

**Target end state.** For any operation in the model's forward pass, you can name (a) the mathematical operation, (b) the input and output tensor shapes, (c) the static weights involved, (d) the reason for any shape change, (e) the PyTorch module path in the loaded model.

**Hard constraints.**

| # | Constraint | Choice |
|---|---|---|
| 1 | Environment | AWS EC2, t3.xlarge, ap-south-1, Ubuntu 22.04 LTS |
| 2 | Access | VS Code Remote-SSH; Jupyter inside VS Code |
| 3 | Device | CPU-only |
| 4 | Model | Llama 3.2 1B Instruct (HuggingFace) |
| 5 | Precision | bf16 |
| 6 | Library | HuggingFace `transformers` |
| 7 | Inspection | Direct submodule calls, forward hooks, `print`, `assert` |
| 8 | Sequence | One input, one predicted token, greedy (argmax) |
| 9 | Out of scope | KV cache, batching, sampling beyond argmax, training, MoE, RLHF |

**Default tolerance for value comparisons.** `atol=1e-4, rtol=1e-4`. If a comparison fails at this tolerance, investigate — do not loosen.

---

## Section 2 — How to read this file

Three structural levels:

- **Phase** — a coherent goal with a single verifiable milestone.
- **Level** — opens one concept inside a phase.
- **Step** — introduces exactly one tensor, operation, weight, or shape change, and verifies it.

Every step uses a six-block template (Section 5). Every step that can be value-checked against the live model is value-checked.

**The Predict block is written before running the code.** Reading it after seeing the answer destroys the only signal that your mental model and the model's behavior have diverged. If you find yourself reading the Predict block after running the code, treat that step as not-yet-done and redo it on a different input.

**On step counts.** The numbering inside this file is not significant. If you find a step that hides two concepts, split it. If you find two steps that share one concept, merge them. Counts are emergent; understanding is the deliverable.

---

## Section 3 — Notation reference

Concrete numbers are confirmed by code in Phase 1. Do not trust the table; verify against `model.config`.

| Symbol | Meaning | Value (Llama 3.2 1B) |
|---|---|---|
| `N_tok` | number of input tokens | varies per input |
| `d` | hidden size | 2048 |
| `N` | number of transformer blocks | 16 |
| `n_h` | number of query heads | 32 |
| `n_kv` | number of key/value heads | 8 |
| `d_h` | per-head dimension | 64 |
| `d_ff` | FFN intermediate dimension | 8192 |
| `vocab_size` | vocabulary size | 128,256 |
| `max_pos` | maximum positional index | 131,072 |

---

## Section 4 — Phase skeleton

| Phase | Name |
|---|---|
| 1 | Foundation: load model and map sizes |
| 2 | Outer pipeline: replace `generate()` |
| 3 | Open the embedding |
| 4 | Open the LM head and final-position decode |
| 5 | Open the final RMSNorm |
| 6 | Open the block stack |
| 7 | Open the block — four children, residual + pre-norm pattern |
| 8 | Open attention projections — Q, K, V, O |
| 9 | Open single-head attention (a correct special case) |
| 10 | Generalize to multi-head and GQA |
| 11 | Apply RoPE — attention matches the live model exactly |
| 12 | Open the FFN — projections, SwiGLU |
| 13 | Stitch the manual block and the full manual inference |
| 14 | Consolidation |

There is no separate "stitch the outer pipeline" phase. Outer-pipeline pieces (embedding, LM head, final norm) are stitched into helpers within the phases that open them; final composition into `manual_inference` lives in Phase 13.

There is no "intentionally wrong attention" phase. Phase 9 builds single-head attention as a correct special case (assuming `n_h = 1`, `n_kv = 1`, no RoPE) that matches the model *if* we run the model with the same restrictions in a separate experiment, then Phase 10 generalizes the head count and Phase 11 adds RoPE. At each phase boundary, the manual version matches a clearly-defined reference.

---

## Section 5 — Step template

Every step follows this structure. Six blocks. The order is fixed. Blocks may not be merged or omitted *unless explicitly noted in Section 6's instructions*.

```
### Step P.L.S — <short name>

**Delta:** <one sentence: what is new vs the previous step>

**Predict:** <one or two lines, written BEFORE running the code: expected shape, dtype,
              value range, and/or comparison-to-live-model result>

**Code:**
```python
# Jupyter-cell style. Prior variables assumed in kernel.
<minimal new code for THIS step>
```

**Observe:**
```python
print("...:", ...)
```

**Verify:**
```python
assert ...
```

**Reflect (one line):** <why the shape/value came out this way; not what>
```

### Permitted variants

- **Pure-binding step** (assigning a config value or pulling a submodule reference). Predict may collapse into one line, and Observe may collapse into the assertion. The template still has six blocks but the content is short.
- **Stitching step** (defining a helper). Code may include the function definition (more than one line) because the function *is* the step's one concept.

### Level header

```
### Level P.L — <name>

**Concept introduced:** <one sentence>
**Black boxes at start of level:** <list, each annotated with the phase where it opens>
**Black boxes opened by this level:** <list>
**Black boxes still deferred at end of level:** <list, each annotated with phase>

[steps here]

**End-of-level check:** <one short predict-then-verify exercise OR "without scrolling, ...">
```

### Phase header

```
## Phase P — <name>

**Goal:** <one sentence>
**Milestone:** <one verifiable condition>
**Builds on:** <prior phase milestone, or "starting point">
**Helpers stitched in this phase:** <names>

[levels here]

**End-of-phase milestone check:**
- <verifiable condition 1>
- <verifiable condition 2>
- Without scrolling: <recall exercise>
- What would break if this phase were skipped? (one paragraph, written from memory)
```
```
## Section 6 — Instructions for future authors

This section governs how new phases are written. Read it after reading at least Phases 1–5, so the rules have concrete examples to point to. Each rule is followed by *what it rejects* — the failure mode it exists to prevent.

### 6.1 Step granularity

A step opens exactly one concept. The test: can the step's *result* be named with one noun? If yes, one step. If you find yourself writing "and" to describe a step's output, split.

- Good: "Compute `Q = x_normed @ W_Q.T`" — result is `Q`, one noun.
- Bad: "Compute `Q` and reshape into heads" — two results, two steps.
- Bad: "Load the model and call eval" — two operations on the same object; either combine into one step framed as "instantiate the model in inference mode," or split into "load weights" then "switch to inference mode" if the eval call is conceptually important enough to deserve its own observation.

**Rejects:** steps that hide one concept inside another.

### 6.2 Predict block

The Predict block is a falsifiable claim. It must say something specific enough that the Verify block can disprove it.

- Required content: at minimum, the expected output shape and dtype. Add value range if values matter. Add a comparison expectation if the live model produces the same tensor.
- Required position: before the Code block, mentally and on the page. If you write the Code first and then the Predict, you've turned a hypothesis into a transcript.

When the prediction is the same as the verification (e.g., for a pure config-extraction step like `d = config.hidden_size`), the Predict block may state the value but should add one sentence of *reasoning* — why is `hidden_size` 2048 the right number here? Otherwise the step is ceremony.

**Rejects:** vague predictions ("a tensor will appear"), tautological predictions (the assertion restated), and after-the-fact predictions (no falsifiability).

### 6.3 Verify block — ground truth

Every step gets at least a shape/dtype assertion. Steps that produce a tensor the live model also produces get a *value* assertion in addition. Both, not one.

Ground truth, in priority order:

1. **Live submodule call** when the operation corresponds to a HuggingFace submodule. Use `torch.allclose` at default tolerance.
2. **Forward hook capture** when the intermediate has no submodule exposing it (e.g., attention scores after softmax). The pattern is documented in Phase 7.
3. **Structural property** (rows of a softmax sum to 1; RMSNorm output has unit per-row RMS) as a complement to 1 and 2, not a replacement.
4. **Hardcoded expected number** only for config values, and only with a fragility margin — never `assert eps == 1e-5` (exact float equality on an external constant); use `assert abs(eps - 1e-5) < 1e-9` or `assert eps < 1e-4`.

**Rejects:** assertions that only test what you typed (`d = 2048; assert d == 2048`), exact-float-equality against external constants, missing value comparisons when the live model could provide one.

### 6.4 Reflect block

One line. Answers *why*, not *what*. The line should be something that a reader could disagree with — a claim, not a restatement.

- Good: "The scaling by `1/sqrt(d_h)` keeps the variance of `scores` independent of head dimension, so softmax does not saturate as `d_h` grows."
- Bad: "Q has shape `[N_tok, d]`." (restates the Verify block; no claim)
- Bad: "This text is chosen because the prediction is 'Paris'." (metadata about the choice, not a reflection on the operation; belongs in a comment, not Reflect)

If you cannot write a one-line Reflect, the step is not understood. Re-derive, or split into smaller steps until the Reflect writes itself.

**Rejects:** restatements of the Verify block, metadata about why the step exists, multi-sentence reflections that drift.

### 6.5 Black box discipline

At the start of every level, list the black boxes entering the level and the black boxes still deferred at the end. Annotate each deferred black box with the phase that opens it. If a level uses something that is not listed in either set, the level is malformed.

**Rejects:** silent opacity. Every black-box dependency must be named.

### 6.6 Stitching discipline

When a piece is fully understood and verified across multiple inputs or multiple instances (e.g., across all 16 blocks), it is stitched into a helper function in a "stitch" step. The stitch step's Predict is "the helper produces the same output as the manual derivation." The Verify is `torch.allclose` (or `torch.equal` if no arithmetic happens).

The helper signature must match how callers will invoke it. If the helper takes `(x, weight, eps)`, the spec elsewhere in this file says `(x, weight, eps)` — not `(x, weight)`. Inconsistency between spec and implementation is a bug.

**Rejects:** helpers that exist only on the canonical input (untested generality); spec/implementation drift in arg lists.

### 6.7 End-of-level and end-of-phase checks

Every level ends with one short check. Every phase ends with a milestone check that has at least three items:

1. A code-verifiable condition (an assertion the reader can run).
2. A code-verifiable condition (a second one).
3. A from-memory recall exercise.

The "what would break if this phase were skipped" paragraph at end of phase is the highest-value writing in the file. It surfaces partial understanding that nothing else catches.

**Rejects:** phases that end without a self-check; phases that end with only "you understand the material now" (unverifiable).

### 6.8 Step numbering

`P.L.S`. Numbers within a level start at 1. New levels start at 1 within a phase. If a step is inserted between existing steps after the file is otherwise complete, use a letter suffix (`3.2.4a`) rather than renumbering downstream.

### 6.9 Cross-references

When a step references a future phase, state the *concept* and the *phase number*, in that order: "RoPE — opened in Phase 11," not "Phase 11 — RoPE." This makes scan-reads forward-compatible if phase numbers shift.

### 6.10 Common failure modes (do not repeat these)

The following bugs appeared in an earlier draft of this file. They are listed here so future authors don't re-introduce them.

| Failure | Cause | Correct alternative |
|---|---|---|
| `torch.set_default_dtype(bfloat16)` | Set default for *all* new tensors including int operations; breaks `torch.arange` and position-ID code later. | Do not set a global default. Load the model in bf16; let int tensors stay int64. |
| `assert input_ids[0,0] == 128000` | Hardcoded BOS ID; some tokenizer configs don't add BOS. | `assert input_ids[0, 0].item() == tokenizer.bos_token_id`. |
| `assert eps == 1e-5` | Exact float-equality on an external constant. | `assert abs(eps - 1e-5) < 1e-9`. |
| `assert probs.sum() == pytest_approx(1.0)` | Uses undefined helper, then adds a note saying "replace if pytest is not imported." | Just write `assert abs(probs.sum().item() - 1.0) < 1e-4` from the start. |
| Predict says `ms.shape == (1, N_tok)` but Verify checks `(1, N_tok, 1)` due to `keepdim=True` | Predict block written without checking PyTorch semantics. | Always trace the shape change through `keepdim` arguments before writing Predict. |
| "Intentionally wrong" attention | Reader who skims internalizes the wrong version. | Build attention as a correct special case (single-head, no RoPE), then generalize. Each phase ends with a manual implementation that matches a clearly-defined reference. |
| Two H1 headers per phase | `# PHASE N` followed by `## Phase N` is redundant. | One header per phase. Use `## Phase N — name`. |
| Hardcoded `assert N_tok == 6` | Brittle: breaks any time the input text changes. | Either compute `N_tok` from `input_ids.shape[1]` without asserting a number, or assert it inside a Predict-block discussion of why this specific text was chosen. |

### 6.11 Tone

- No motivational filler. No "great job."
- No essay-length reflections. One line means one line.
- No parenthetical asides that turn one sentence into three. If the parenthetical matters, make it its own sentence.
- No "we" when "you" or "the reader" is meant. The author is not in the room.

### 6.12 Length budget per phase

Rough guideline, not a rule: Phases that open a single submodule (embedding, LM head, RMSNorm) run 50–150 lines. Phases that open multi-operation structures (attention, FFN) run 200–400 lines. Phases that stitch and consolidate run 30–80 lines. If a phase exceeds 500 lines, it is doing too much — split.

---

# Phase 1 — Foundation: load model and map sizes

## Phase 1 — Foundation: load model and map sizes

**Goal:** Load Llama 3.2 1B Instruct on the EC2 instance, read its config, and bind every abstract size symbol used throughout this file to a concrete integer.

**Milestone:** Every symbol in Section 3 is bound to an integer derived from `model.config` and asserted, and the four top-level pieces of the model (embedding, block stack, final norm, LM head) are each pulled into a named variable.

**Builds on:** starting point.

**Helpers stitched in this phase:** none.

---

### Level 1.1 — Environment

**Concept introduced:** PyTorch and HuggingFace are imported, gradient tracking is disabled, CPU-only is confirmed. This level has no model math.

**Black boxes at start of level:** everything.
**Black boxes opened by this level:** none new (still everything).
**Black boxes still deferred at end of level:** everything; opens in Level 1.2 onward.

---

#### Step 1.1.1 — Import and confirm CPU-only

**Delta:** Bring `torch` and the HuggingFace classes into the kernel; confirm no GPU is present.

**Predict:** `torch.__version__` is a string ending `+cpu` (or similar). `torch.cuda.is_available()` is `False`. No exception raised. This matches the hard constraint of CPU-only execution.

**Code:**
```python
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
```

**Observe:**
```python
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("cuda available:", torch.cuda.is_available())
```

**Verify:**
```python
assert torch.cuda.is_available() is False
```

**Reflect (one line):** A silent GPU would change memory and speed expectations for every later step, so the CPU-only assertion is a tripwire that fires before any model load.

---

#### Step 1.1.2 — Disable gradient tracking

**Delta:** Turn autograd off globally — this is an inference-only session.

**Predict:** After the call, any tensor operation produces a result with `requires_grad=False`, regardless of input.

**Code:**
```python
torch.set_grad_enabled(False)
```

**Observe:**
```python
y = torch.randn(2, 2) @ torch.randn(2, 2)
print("requires_grad after disabling:", y.requires_grad)
```

**Verify:**
```python
assert y.requires_grad is False
```

**Reflect (one line):** Disabling grad globally avoids building a computation graph during long manual derivations, which would silently consume memory across phases.

---

**End-of-level check:**
- Without scrolling: name the two environment conditions established by Level 1.1 and why each one matters for the rest of the work.

---

### Level 1.2 — Load tokenizer and model

**Concept introduced:** The tokenizer and the model are two separate HuggingFace objects, each loaded with a single call.

**Black boxes at start of level:** everything inside model and tokenizer.
**Black boxes opened by this level:** the existence of the loaded objects and their top-level types.
**Black boxes still deferred at end of level:** the model's forward pass, all submodules, all weights, the tokenizer's BPE algorithm.

---

#### Step 1.2.1 — Name the model

**Delta:** Bind the model identifier as a constant.

**Predict:** `MODEL_NAME` is a string literal. No network activity. The constant centralizes the identifier so swapping to a different size later requires editing one line.

**Code:**
```python
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
```

**Observe:**
```python
print("model id:", MODEL_NAME)
```

**Verify:**
```python
assert "Llama-3.2-1B" in MODEL_NAME
```

**Reflect (one line):** A single named constant prevents identifier drift across cells and makes the work scriptable later.

---

#### Step 1.2.2 — Load the tokenizer

**Delta:** Load the tokenizer; this is the first HuggingFace network call (or cache hit).

**Predict:** `tokenizer` is an instance of a fast tokenizer class. `tokenizer.bos_token_id` is defined (Llama 3 uses `<|begin_of_text|>` as BOS). `tokenizer.vocab_size` is `128000` — note that this is the base vocabulary count; the model's `vocab_size` includes additional reserved slots (confirmed in Level 1.3).

**Code:**
```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
```

**Observe:**
```python
print("tokenizer class:", type(tokenizer).__name__)
print("vocab_size:", tokenizer.vocab_size)
print("bos_token_id:", tokenizer.bos_token_id)
print("eos_token_id:", tokenizer.eos_token_id)
```

**Verify:**
```python
assert tokenizer.vocab_size == 128000
assert tokenizer.bos_token_id is not None
```

**Reflect (one line):** Tokenizer `vocab_size` (128,000) and model `vocab_size` (128,256, confirmed in Level 1.3) differ because the model embedding allocates rows for reserved/special tokens the tokenizer does not count.

---

#### Step 1.2.3 — Load the model weights

**Delta:** Download (or cache-read) the 1B parameters as bf16 and instantiate the PyTorch module.

**Predict:** `model` is a `LlamaForCausalLM`. Loading takes ~30s on first run, near-instant from cache. RSS grows by approximately `1.24B × 2 bytes ≈ 2.5 GB`. All parameters are bf16.

**Code:**
```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
```

**Observe:**
```python
print("model class:", type(model).__name__)
print("first param dtype:", next(model.parameters()).dtype)
print("first param device:", next(model.parameters()).device)
```

**Verify:**
```python
assert type(model).__name__ == "LlamaForCausalLM"
assert next(model.parameters()).dtype == torch.bfloat16
assert next(model.parameters()).device.type == "cpu"
```

**Reflect (one line):** Passing `torch_dtype=torch.bfloat16` at load time keeps the weights bf16 in memory and on disk path — using fp32 would roughly double memory use and provide no benefit since the model was trained in bf16.

---

#### Step 1.2.4 — Switch to eval mode

**Delta:** Call `model.eval()` to put the module in inference mode.

**Predict:** `model.training` is `False`. Llama has neither dropout nor batchnorm so the call is a no-op for *this* model, but the convention is non-negotiable for inference work and protects against silently leaving training-mode behavior on in other architectures.

**Code:**
```python
model.eval()
```

**Observe:**
```python
print("training flag:", model.training)
```

**Verify:**
```python
assert model.training is False
```

**Reflect (one line):** `eval()` is one of two PyTorch conventions for inference (the other is the `torch.no_grad()` we already enabled globally); using both is defensive but the combined cost is zero.

---

#### Step 1.2.5 — Count parameters

**Delta:** Sum element counts across all learnable tensors as a sanity check on which model loaded.

**Predict:** Total is approximately `1.24 × 10⁹`. If a different model (e.g., 3B or 7B) had been pulled by mistake, the count would be off by 2× or more.

**Code:**
```python
total_params = sum(p.numel() for p in model.parameters())
```

**Observe:**
```python
print(f"total parameters: {total_params:,}")
print(f"approx GB at bf16: {total_params * 2 / 1e9:.2f}")
```

**Verify:**
```python
assert 1.0e9 < total_params < 1.5e9
```

**Reflect (one line):** A loose-bound assertion (1.0B to 1.5B) catches the catastrophic case of a wrong model load without breaking on legitimate minor variation between Llama 3.2 1B versions.

---

**End-of-level check:**
- Predict: what would be the symptom — at load time — if `torch_dtype=torch.bfloat16` had been omitted? Then read the HuggingFace `from_pretrained` defaults to verify.

---

### Level 1.3 — Read the config

**Concept introduced:** `model.config` is the single source of truth for every architectural size. The size symbols in Section 3 are bound by reading config fields.

**Black boxes at start of level:** model internals.
**Black boxes opened by this level:** the sizes the architecture is parameterized by — but not how those sizes are used in the computation.
**Black boxes still deferred at end of level:** all forward-pass operations.

---

#### Step 1.3.1 — Look at the raw config

**Delta:** Print the whole config to see what fields exist.

**Predict:** `model.config` is a `LlamaConfig`. The repr shows fields including `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `vocab_size`, `max_position_embeddings`.

**Code:**
```python
config = model.config
```

**Observe:**
```python
print(config)
```

**Verify:**
```python
assert type(config).__name__ == "LlamaConfig"
```

**Reflect (one line):** The config repr is the architecture's authors telling you which parameters they thought important enough to expose — skimming it once is worth more than memorizing field names.

---

#### Step 1.3.2 — Bind size symbols

**Delta:** Extract all eight size symbols in one operation. Pure-binding step (per Section 5 variant): the Predict and Observe are compact because there is no arithmetic to verify, only field reads. The shape and value sanity comes from the assertions.

**Predict:** All values match Section 3. `d_h = d // n_h = 64` is a derived value. The divisibility check `n_h % n_kv == 0` is a precondition for GQA.

**Code:**
```python
d          = config.hidden_size
N          = config.num_hidden_layers
n_h        = config.num_attention_heads
n_kv       = config.num_key_value_heads
d_h        = d // n_h
d_ff       = config.intermediate_size
vocab_size = config.vocab_size
max_pos    = config.max_position_embeddings
```

**Observe:**
```python
sizes = dict(d=d, N=N, n_h=n_h, n_kv=n_kv, d_h=d_h,
             d_ff=d_ff, vocab_size=vocab_size, max_pos=max_pos)
for k, v in sizes.items():
    print(f"  {k:>10} = {v:,}")
```

**Verify:**
```python
assert d == 2048
assert N == 16
assert n_h == 32
assert n_kv == 8
assert d_h == 64
assert d_ff == 8192
assert vocab_size == 128256
assert max_pos == 131072
assert n_h * d_h == d
assert n_h % n_kv == 0
```

**Reflect (one line):** The two derived assertions (`n_h * d_h == d` and `n_h % n_kv == 0`) encode architectural invariants the model silently assumes — explicit assertions surface those invariants before any later step depends on them.

---

**End-of-level check:**
- Without scrolling: write all eight symbols and their values. Then predict: if `n_h` were 16 with `d` unchanged, what would `d_h` be? Verify by recomputing.

---

### Level 1.4 — Inspect top-level module structure

**Concept introduced:** The loaded model has four top-level architectural pieces: embedding, block stack, final norm, LM head. Each piece is bound to a named variable.

**Black boxes at start of level:** the body of every submodule.
**Black boxes opened by this level:** the names and weight shapes of the four top-level pieces.
**Black boxes still deferred at end of level:** the forward computation of every piece. Embedding opens in Phase 3, LM head in Phase 4, final norm in Phase 5, blocks in Phases 6+.

---

#### Step 1.4.1 — Print the model

**Delta:** Print the full module tree to see the structure at a glance.

**Predict:** Top level shows `LlamaForCausalLM(model=LlamaModel(...), lm_head=Linear(2048, 128256, bias=False))`. Inside `LlamaModel`: `embed_tokens` (Embedding), `layers` (ModuleList of 16 `LlamaDecoderLayer`s), `norm` (LlamaRMSNorm), `rotary_emb` (LlamaRotaryEmbedding).

**Code:**
```python
print(model)
```

**Observe:** the `print` itself.

**Verify:**
```python
assert hasattr(model, "model")
assert hasattr(model, "lm_head")
assert hasattr(model.model, "embed_tokens")
assert hasattr(model.model, "layers")
assert hasattr(model.model, "norm")
```

**Reflect (one line):** The two-level nesting (`model.model.X`) is a HuggingFace convention where `model` is the language-modeling wrapper and `model.model` is the bare transformer; only `lm_head` lives outside the inner wrapper.

---

#### Step 1.4.2 — Bind the four top-level pieces

**Delta:** Pull each top-level piece into a named variable for use in later phases.

**Predict:** Four references created. `embed.weight.shape == (128256, 2048)`, `len(layers) == 16`, `final_norm.weight.shape == (2048,)`, `lm_head.weight.shape == (128256, 2048)`, `lm_head.bias is None`.

**Code:**
```python
embed      = model.model.embed_tokens
layers     = model.model.layers
final_norm = model.model.norm
lm_head    = model.lm_head
```

**Observe:**
```python
print("embed.weight.shape     :", tuple(embed.weight.shape))
print("len(layers)            :", len(layers))
print("final_norm.weight.shape:", tuple(final_norm.weight.shape))
print("lm_head.weight.shape   :", tuple(lm_head.weight.shape))
print("lm_head.bias           :", lm_head.bias)
```

**Verify:**
```python
assert tuple(embed.weight.shape) == (vocab_size, d)
assert len(layers) == N
assert all(type(L).__name__ == "LlamaDecoderLayer" for L in layers)
assert tuple(final_norm.weight.shape) == (d,)
assert tuple(lm_head.weight.shape) == (vocab_size, d)
assert lm_head.bias is None
```

**Reflect (one line):** The embedding and LM head share the same shape `[vocab_size, d]` because both map between the per-token hidden vector and the vocabulary — the symmetry is structural even when (as here) the two are not weight-tied.

---

#### Step 1.4.3 — Check whether embedding and LM head are weight-tied

**Delta:** Compare the two weight tensors by identity and by value.

**Predict:** In Llama 3.2 1B Instruct, the two are not tied. `embed.weight is lm_head.weight` is `False`. `torch.equal(embed.weight, lm_head.weight)` is also `False`.

**Code:**
```python
tied_id    = embed.weight is lm_head.weight
tied_value = torch.equal(embed.weight, lm_head.weight)
```

**Observe:**
```python
print("tied by identity:", tied_id)
print("tied by value   :", tied_value)
```

**Verify:**
```python
assert tied_id is False
assert tied_value is False
```

**Reflect (one line):** Knowing they are separate matters for Phase 4 — manual code that mistakenly used `embed.weight.T` instead of `lm_head.weight.T` would silently produce wrong logits if the two had been tied, so we verify *here* that they aren't.

---

**End-of-level check:**
- Without scrolling: draw the top-level module tree (four pieces). Then check by re-running `print(model)`.

---

**End-of-phase milestone check (Phase 1):**
- `model` and `tokenizer` loaded, isinstance-checked.
- All eight size symbols bound, asserted against expected values, with derived invariants (`n_h * d_h == d`, `n_h % n_kv == 0`) asserted.
- Four top-level pieces (`embed`, `layers`, `final_norm`, `lm_head`) bound, weight shapes asserted.
- Without scrolling: list the four top-level pieces and the shape of each piece's primary weight tensor.
- What would break if Phase 1 were skipped? Write one paragraph from memory.

---

# Phase 2 — Outer pipeline: replace `generate()`

## Phase 2 — Outer pipeline: replace `generate()`

**Goal:** Produce one predicted token end-to-end without calling `model.generate()`. The model's `__call__` is treated as a black box returning logits; everything around it (tokenize, slice, argmax, decode) is done manually.

**Milestone:** A function `predict_one_token(text)` exists, produces a Paris-containing string for the canonical input, and matches `model.generate(..., max_new_tokens=1, do_sample=False)`.

**Builds on:** Phase 1 milestone.

**Helpers stitched in this phase:** `predict_one_token(text)`.

---

### Level 2.1 — Canonical input and tokenization

**Concept introduced:** A fixed input text is bound once and reused across all later experiments. Tokenization converts the text to a tensor of integer IDs.

**Black boxes at start of level:** the entire model forward pass; the tokenizer's BPE algorithm.
**Black boxes opened by this level:** the shape and content of the tokenized output.
**Black boxes still deferred at end of level:** the model forward pass (opens partially in Level 2.2); BPE internals (out of scope for this track).

---

#### Step 2.1.1 — Define the canonical input

**Delta:** Bind a fixed input string used by every later phase.

**Predict:** A short, deterministic string with an overwhelmingly likely next-token continuation. `len(TEXT)` is 24 characters.

**Code:**
```python
TEXT = "The capital of France is"
```

**Observe:**
```python
print("input text:", repr(TEXT))
print("char count:", len(TEXT))
```

**Verify:**
```python
assert TEXT == "The capital of France is"
```

**Reflect (one line):** A fixed input with a predictable next token ("Paris") gives every later assertion a sanity check — shape-correct logits that produce nonsense would still pass shape assertions but would fail the Paris check.

---

#### Step 2.1.2 — Tokenize

**Delta:** Convert the text to a tensor of integer token IDs.

**Predict:** `input_ids.shape == (1, N_tok)` where `N_tok` is small (likely 6 for this input including BOS). Dtype `int64`. `input_ids[0, 0]` equals `tokenizer.bos_token_id` because the Llama tokenizer prepends BOS by default.

**Code:**
```python
input_ids = tokenizer(TEXT, return_tensors="pt").input_ids
N_tok     = input_ids.shape[1]
```

**Observe:**
```python
print("input_ids   :", input_ids)
print("shape       :", input_ids.shape)
print("dtype       :", input_ids.dtype)
print("N_tok       :", N_tok)
print("first id    :", input_ids[0, 0].item(), "(bos_token_id =", tokenizer.bos_token_id, ")")
```

**Verify:**
```python
assert input_ids.ndim == 2
assert input_ids.shape[0] == 1
assert input_ids.dtype == torch.int64
assert input_ids[0, 0].item() == tokenizer.bos_token_id
```

**Reflect (one line):** The leading batch dim is kept everywhere because HuggingFace modules expect it, even though batch size is fixed at 1 throughout this track.

---

#### Step 2.1.3 — Decode each token for sanity

**Delta:** Print each token ID alongside the substring it decodes to.

**Predict:** First token is BOS. Remaining tokens spell the input piece by piece: `'The'`, `' capital'`, `' of'`, `' France'`, `' is'` — BPE preserves leading spaces.

**Code:**
```python
per_token = [(int(tid), tokenizer.decode([tid])) for tid in input_ids[0]]
```

**Observe:**
```python
for i, (tid, piece) in enumerate(per_token):
    print(f"  pos {i}: id={tid:>6}  piece={piece!r}")
```

**Verify:**
```python
assert per_token[0][0] == tokenizer.bos_token_id
joined = "".join(p for _, p in per_token[1:])
assert joined == TEXT
```

**Reflect (one line):** Tokenization is reversible at the byte level — concatenating per-token decodes (skipping BOS) reproduces the original text exactly, because BPE retains all whitespace.

---

**End-of-level check:**
- Predict: how many tokens are added if the input becomes `TEXT + " Paris"`? Verify by tokenizing and comparing lengths.

---

### Level 2.2 — Call the model and extract logits

**Concept introduced:** The model is a callable that takes integer IDs and returns logits over the vocabulary, for every input position.

**Black boxes at start of level:** the entire forward pass.
**Black boxes opened by this level:** the I/O contract of the forward pass.
**Black boxes still deferred at end of level:** the internal computation (Phases 3+).

---

#### Step 2.2.1 — Forward pass

**Delta:** Call the model on `input_ids` and capture the output object.

**Predict:** `outputs` is a `CausalLMOutputWithPast`. Its `.logits` attribute has shape `[1, N_tok, vocab_size]` and dtype bf16.

**Code:**
```python
outputs = model(input_ids=input_ids)
```

**Observe:**
```python
print("outputs type:", type(outputs).__name__)
print("logits shape:", outputs.logits.shape)
print("logits dtype:", outputs.logits.dtype)
```

**Verify:**
```python
assert type(outputs).__name__ == "CausalLMOutputWithPast"
assert tuple(outputs.logits.shape) == (1, N_tok, vocab_size)
assert outputs.logits.dtype == torch.bfloat16
```

**Reflect (one line):** The model returns logits for *every* input position — for one-token inference we use only the last row, so the first `N_tok - 1` rows are computational waste that a KV-cached generation loop would reuse, but excluding the cache keeps the work simpler.

---

#### Step 2.2.2 — Bind `logits` for convenience

**Delta:** Drop the `.logits` chain into a clean variable name.

**Predict:** `logits is outputs.logits`. Same shape, same dtype, same object identity.

**Code:**
```python
logits = outputs.logits
```

**Observe:**
```python
print("logits shape:", logits.shape)
print("logits sample at (0, 0, :5):", logits[0, 0, :5].float().tolist())
```

**Verify:**
```python
assert logits is outputs.logits
assert tuple(logits.shape) == (1, N_tok, vocab_size)
```

**Reflect (one line):** Logits are unnormalized scores — any real number — and the spread between the largest and the rest determines how peaked the softmax distribution becomes.

---

### Level 2.3 — Slice, softmax, argmax, decode

**Concept introduced:** The next-token prediction comes from the *last* input position's logits. Softmax converts logits to probabilities; argmax picks the predicted token ID. Decode converts ID back to text.

**Black boxes at start of level:** the model interior.
**Black boxes opened by this level:** the four post-forward operations (slice, softmax, argmax, decode).
**Black boxes still deferred at end of level:** the model interior.

---

#### Step 2.3.1 — Last-position slice

**Delta:** Take `logits[0, -1, :]`.

**Predict:** `last_logits.shape == (vocab_size,)`, 1D. Dtype bf16.

**Code:**
```python
last_logits = logits[0, -1, :]
```

**Observe:**
```python
print("last_logits shape:", last_logits.shape)
print("min/max:", last_logits.float().min().item(), last_logits.float().max().item())
```

**Verify:**
```python
assert last_logits.shape == (vocab_size,)
assert last_logits.dtype == torch.bfloat16
```

**Reflect (one line):** Dropping the batch dim (index 0) and selecting the last position (index -1) collapses a 3D tensor to 1D — the prediction is one row of vocab-size scores.

---

#### Step 2.3.2 — Argmax

**Delta:** Pick the token ID with the highest score.

**Predict:** `pred_id` is a 0-D int64 tensor. Its decode contains "Paris" (likely with a leading space).

**Code:**
```python
pred_id = last_logits.argmax()
```

**Observe:**
```python
print("pred_id:", pred_id.item())
print("decoded:", repr(tokenizer.decode([pred_id.item()])))
```

**Verify:**
```python
assert pred_id.dtype == torch.int64
assert pred_id.ndim == 0
assert "Paris" in tokenizer.decode([pred_id.item()])
```

**Reflect (one line):** Argmax is deterministic, which is what makes it suitable for verifying that our manual reproductions in later phases match the model exactly, value for value.

---

#### Step 2.3.3 — Softmax for interpretability

**Delta:** Compute the probability distribution to read off the model's confidence. For greedy decoding this is unnecessary (softmax is monotonic) — but the probability gives the human reader a confidence number.

**Predict:** `probs.shape == (vocab_size,)`. All values in `[0, 1]`. Sum is approximately 1.0 within `1e-4`. `probs.argmax() == pred_id`.

**Code:**
```python
probs = torch.softmax(last_logits.float(), dim=-1)
```

**Observe:**
```python
print("probs shape   :", probs.shape)
print("sum           :", probs.sum().item())
print("top-1 prob    :", probs.max().item())
print("top-5 probs   :", probs.topk(5).values.tolist())
print("argmax matches:", probs.argmax().item() == pred_id.item())
```

**Verify:**
```python
assert probs.shape == (vocab_size,)
assert probs.min().item() >= 0.0
assert probs.max().item() <= 1.0
assert abs(probs.sum().item() - 1.0) < 1e-4
assert probs.argmax().item() == pred_id.item()
```

**Reflect (one line):** Casting to `.float()` before softmax avoids bf16 precision loss in the `exp(x)` and subsequent sum — bf16 has wide exponent range but only 7 mantissa bits, which is too few for accurate probability normalization.

---

#### Step 2.3.4 — Decode

**Delta:** Convert the predicted ID back to a string.

**Predict:** `pred_text` contains "Paris", likely with leading space.

**Code:**
```python
pred_text = tokenizer.decode([pred_id.item()])
```

**Observe:**
```python
print("predicted text:", repr(pred_text))
print("input + pred  :", repr(TEXT + pred_text))
```

**Verify:**
```python
assert "Paris" in pred_text
```

**Reflect (one line):** The leading-space convention in the decoded token (`' Paris'` not `'Paris'`) is BPE preserving the corpus statistic that "Paris" almost always follows a space.

---

### Level 2.4 — Stitch and cross-check

**Concept introduced:** The five outer-pipeline operations are bundled into a single helper; a cross-check against `model.generate` confirms equivalence to HuggingFace's own greedy decoding.

**Black boxes at start of level:** the model interior.
**Black boxes opened by this level:** none.
**Black boxes still deferred at end of level:** the model interior.

---

#### Step 2.4.1 — Define `predict_one_token`

**Delta:** Bundle tokenize → forward → slice → argmax → decode into one function.

**Predict:** `predict_one_token(TEXT) == pred_text` from the prior level.

**Code:**
```python
def predict_one_token(text):
    ids   = tokenizer(text, return_tensors="pt").input_ids
    lgts  = model(input_ids=ids).logits[0, -1, :]
    pid   = lgts.argmax().item()
    return tokenizer.decode([pid])
```

**Observe:**
```python
result = predict_one_token(TEXT)
print("predict_one_token:", repr(result))
```

**Verify:**
```python
assert predict_one_token(TEXT) == pred_text
```

**Reflect (one line):** Wrapping the pipeline into one function turns five concepts into one callable, so later phases can invoke "outer pipeline" by name without re-deriving its operations.

---

#### Step 2.4.2 — Cross-check against `model.generate`

**Delta:** Independently produce one token via `model.generate(max_new_tokens=1, do_sample=False)` and confirm equality.

**Predict:** Same token ID, same decoded text.

**Code:**
```python
gen_ids    = model.generate(input_ids=input_ids, max_new_tokens=1, do_sample=False)
gen_new_id = gen_ids[0, -1].item()
gen_text   = tokenizer.decode([gen_new_id])
```

**Observe:**
```python
print("generate() id  :", gen_new_id)
print("generate() text:", repr(gen_text))
print("manual id      :", pred_id.item())
print("manual text    :", repr(pred_text))
```

**Verify:**
```python
assert gen_new_id == pred_id.item()
assert gen_text == pred_text
```

**Reflect (one line):** Matching `generate()` token-for-token confirms the manual pipeline reproduces HuggingFace's greedy decoding exactly — any later assertion that fires this check has detected real drift.

---

**End-of-level check:**
- Without scrolling: list the five operations inside `predict_one_token`, in order, and the shape of the tensor flowing between each pair of operations.

---

**End-of-phase milestone check (Phase 2):**
- `predict_one_token(TEXT)` exists, returns Paris-containing text, matches `model.generate`.
- The five outer-pipeline operations are each understood as discrete with named I/O.
- Without scrolling: list the five operations and the shape between each pair.
- What would break if Phase 2 were skipped? Write one paragraph from memory.

---

# Phase 3 — Open the embedding

## Phase 3 — Open the embedding

**Goal:** Replace the call `embed(input_ids)` with a manual row-lookup from the embedding weight tensor.

**Milestone:** A helper `manual_embedding(ids, model)` exists and matches `embed(ids)` bit-for-bit (`torch.equal`) on the canonical input and at least one additional input.

**Builds on:** Phase 2 milestone.

**Helpers stitched in this phase:** `manual_embedding(ids, model)`.

---

### Level 3.1 — Look at the embedding weight

**Concept introduced:** The embedding submodule is a thin wrapper around a 2D weight tensor `[vocab_size, d]`. Each row is the vector for one token ID.

**Black boxes at start of level:** the `__call__` of `nn.Embedding`.
**Black boxes opened by this level:** the shape of the embedding weight, and the structure of one row.
**Black boxes still deferred at end of level:** the row-selection mechanism (opens in Level 3.2).

---

#### Step 3.1.1 — Bind the weight as `E`

**Delta:** Pull `embed.weight` out as `E` so manual operations use the same symbol the architecture conventions use.

**Predict:** `E.shape == (vocab_size, d)`. Dtype bf16. `E is embed.weight` (same object).

**Code:**
```python
E = embed.weight
```

**Observe:**
```python
print("E.shape:", E.shape)
print("E.dtype:", E.dtype)
print("E is embed.weight:", E is embed.weight)
```

**Verify:**
```python
assert tuple(E.shape) == (vocab_size, d)
assert E.dtype == torch.bfloat16
assert E is embed.weight
```

**Reflect (one line):** Binding `E` as the same object (not a clone) means later assertions test the actual model weight, not a snapshot.

---

#### Step 3.1.2 — Look at one row

**Delta:** Slice `E[tokenizer.bos_token_id]` to see the embedding for BOS.

**Predict:** Shape `(d,)`. Values are small floats; trained embeddings typically have per-coordinate magnitude in single digits (rough bound; exact range depends on training).

**Code:**
```python
bos_vec = E[tokenizer.bos_token_id]
```

**Observe:**
```python
print("bos_vec.shape:", bos_vec.shape)
print("first 5 vals :", bos_vec[:5].float().tolist())
print("abs max      :", bos_vec.float().abs().max().item())
```

**Verify:**
```python
assert bos_vec.shape == (d,)
assert bos_vec.dtype == torch.bfloat16
```

**Reflect (one line):** A single row of `E` is a `d`-dimensional point in the model's representation space — every token starts as exactly one such point before any block has shaped it with context.

---

### Level 3.2 — Manual row lookup

**Concept introduced:** `nn.Embedding`'s `__call__` is exactly fancy indexing into the weight tensor with the input IDs.

**Black boxes at start of level:** what `embed(input_ids)` does internally.
**Black boxes opened by this level:** that operation is `E[input_ids]`.
**Black boxes still deferred at end of level:** none for the embedding.

---

#### Step 3.2.1 — Fancy-index `E` with `input_ids`

**Delta:** Compute the manual embedding.

**Predict:** `x_manual.shape == (1, N_tok, d)`. The leading dim is preserved because `input_ids.shape == (1, N_tok)` and PyTorch fancy indexing preserves the indexing tensor's shape.

**Code:**
```python
x_manual = E[input_ids]
```

**Observe:**
```python
print("x_manual.shape:", x_manual.shape)
print("x_manual.dtype:", x_manual.dtype)
print("first 5 at (0, 0):", x_manual[0, 0, :5].float().tolist())
```

**Verify:**
```python
assert tuple(x_manual.shape) == (1, N_tok, d)
assert x_manual.dtype == torch.bfloat16
```

**Reflect (one line):** Fancy indexing replaces each integer with the corresponding row, producing a shape that is the indexing tensor's shape with the row's shape appended.

---

#### Step 3.2.2 — Compare against `embed(input_ids)`

**Delta:** Call the submodule and check exact equality.

**Predict:** `torch.equal(x_manual, embed(input_ids))` is `True`. Exact equality is achievable because no floating-point arithmetic happens — only index-and-copy.

**Code:**
```python
x_real = embed(input_ids)
```

**Observe:**
```python
print("equal?     :", torch.equal(x_manual, x_real))
print("max abs diff:", (x_manual.float() - x_real.float()).abs().max().item())
```

**Verify:**
```python
assert x_real.shape == x_manual.shape
assert torch.equal(x_manual, x_real)
```

**Reflect (one line):** Exact equality (not `allclose`) is achievable here because the operation involves no arithmetic — there is nothing to round.

---

#### Step 3.2.3 — Confirm position-independence

**Delta:** Swap two positions in `input_ids`; verify the embedding output for those positions also swaps.

**Predict:** If positions `i` and `j` swap, the corresponding rows in the output swap; all other rows are unchanged.

**Code:**
```python
swapped = input_ids.clone()
swapped[0, 1], swapped[0, 2] = swapped[0, 2].clone(), swapped[0, 1].clone()
x_swapped = E[swapped]
```

**Observe:**
```python
print("row 1 ↔ row 2:", torch.equal(x_manual[0, 1], x_swapped[0, 2]),
                          torch.equal(x_manual[0, 2], x_swapped[0, 1]))
print("row 0 unchanged:", torch.equal(x_manual[0, 0], x_swapped[0, 0]))
```

**Verify:**
```python
assert torch.equal(x_manual[0, 1], x_swapped[0, 2])
assert torch.equal(x_manual[0, 2], x_swapped[0, 1])
assert torch.equal(x_manual[0, 0], x_swapped[0, 0])
```

**Reflect (one line):** Embedding has no positional awareness — that is added later by RoPE inside attention (Phase 11); at the embedding stage, identical IDs produce identical vectors regardless of position.

---

### Level 3.3 — Stitch `manual_embedding`

**Concept introduced:** The row-lookup pattern is wrapped into a helper for use in later phases.

**Black boxes at start of level:** none.
**Black boxes opened by this level:** none.
**Black boxes still deferred at end of level:** none.

---

#### Step 3.3.1 — Define and verify the helper

**Delta:** Bundle row lookup into a function; test on two inputs.

**Predict:** `manual_embedding(ids, model)` matches `model.model.embed_tokens(ids)` via `torch.equal` for any token-ID tensor.

**Code:**
```python
def manual_embedding(ids, model):
    return model.model.embed_tokens.weight[ids]
```

**Observe:**
```python
out1 = manual_embedding(input_ids, model)
other_ids = tokenizer("Hello world", return_tensors="pt").input_ids
out2 = manual_embedding(other_ids, model)
print("canonical input shape :", out1.shape, " equal:", torch.equal(out1, embed(input_ids)))
print("other input shape     :", out2.shape, " equal:", torch.equal(out2, embed(other_ids)))
```

**Verify:**
```python
assert torch.equal(out1, embed(input_ids))
assert torch.equal(out2, embed(other_ids))
```

**Reflect (one line):** Testing on a second input is what turns "works on the canonical input" into "works for any token-ID tensor of the right type."

---

**End-of-level check:**
- Without scrolling: write the body of `manual_embedding` from memory. One line. Verify.

---

**End-of-phase milestone check (Phase 3):**
- `E` bound; shape `[128256, 2048]`, dtype bf16.
- `manual_embedding(ids, model)` exists; matches `embed(ids)` exactly on two inputs.
- Without scrolling: input shape, output shape, weight shape; why does the output have one more dimension than the input?
- What would break if Phase 3 were skipped? Write one paragraph from memory.

---

# Phase 4 — Open the LM head and final-position decode

## Phase 4 — Open the LM head and final-position decode

**Goal:** Replace `lm_head(x)` with a manual `x @ H.T`. The last-position slice, softmax, and argmax already covered in Phase 2 are not re-opened — they are unchanged operations.

**Milestone:** A helper `manual_lm_head(x, model)` exists and matches `lm_head(x)` within `atol=1e-4` on two different inputs. The chain `manual_embedding → blocks (opaque) → final_norm (opaque) → manual_lm_head → slice → argmax → decode` produces the same token as `predict_one_token`.

**Builds on:** Phase 3 milestone. Blocks and final norm remain opaque (open in Phases 6 and 5 respectively).

**Helpers stitched in this phase:** `manual_lm_head(x, model)`.

---

### Level 4.1 — Get the hidden state that feeds the LM head

**Concept introduced:** To verify the LM head, we need the tensor it consumes — the post-final-norm hidden state.

**Black boxes at start of level:** blocks and final norm.
**Black boxes opened by this level:** *which* tensor feeds the LM head and what shape it has.
**Black boxes still deferred at end of level:** how that tensor was computed (blocks → Phase 6; final norm → Phase 5).

---

#### Step 4.1.1 — Re-run the model exposing hidden states

**Delta:** Call with `output_hidden_states=True`.

**Predict:** `hidden_states` is a tuple of length `N + 1 = 17` (one before any block, plus one after each of 16 blocks). Each element has shape `[1, N_tok, d]`. Critically, `hidden_states[-1]` is the output of the *last block*, **before** the final norm — so it is not what the LM head consumes directly.

**Code:**
```python
outputs_hs    = model(input_ids=input_ids, output_hidden_states=True)
hidden_states = outputs_hs.hidden_states
```

**Observe:**
```python
print("num hidden_states :", len(hidden_states))
print("each shape        :", hidden_states[0].shape)
print("first dtype       :", hidden_states[0].dtype)
print("last  dtype       :", hidden_states[-1].dtype)
```

**Verify:**
```python
assert len(hidden_states) == N + 1
for hs in hidden_states:
    assert tuple(hs.shape) == (1, N_tok, d)
```

**Reflect (one line):** `hidden_states[-1]` is pre-final-norm; the LM head consumes post-final-norm, so we cannot use `hidden_states[-1]` directly — we must apply the final norm in the next step.

---

#### Step 4.1.2 — Apply the final norm (still treating its internals as opaque)

**Delta:** Run the final norm submodule on `hidden_states[-1]` to get the actual tensor the LM head receives. The norm submodule is called as a black box here; its internals open in Phase 5.

**Predict:** `x_final.shape == (1, N_tok, d)`. Dtype bf16. Values differ from `hidden_states[-1]` because RMSNorm rescales each row and applies a per-channel scale.

**Code:**
```python
x_final = final_norm(hidden_states[-1])
```

**Observe:**
```python
print("x_final shape    :", x_final.shape)
print("pre-norm sample  :", hidden_states[-1][0, -1, :5].float().tolist())
print("post-norm sample :", x_final[0, -1, :5].float().tolist())
```

**Verify:**
```python
assert tuple(x_final.shape) == (1, N_tok, d)
assert x_final.dtype == torch.bfloat16
```

**Reflect (one line):** Shape is preserved across the norm because normalization is purely per-token — the operation rescales values within each row but never moves data across rows or across channels-and-rows together.

---

### Level 4.2 — Open the LM head

**Concept introduced:** The LM head is a bias-less `Linear(d → vocab_size)`. Its weight `H` has shape `[vocab_size, d]`. The operation is `x @ H.T`.

**Black boxes at start of level:** what `lm_head(x)` does internally.
**Black boxes opened by this level:** the weight `H` and the matmul.
**Black boxes still deferred at end of level:** none for the LM head.

---

#### Step 4.2.1 — Bind `H`

**Delta:** Pull `lm_head.weight` out as `H`.

**Predict:** `H.shape == (vocab_size, d)`. Dtype bf16. `H is lm_head.weight`. Same shape as `E` but separate tensor (verified in Step 1.4.3).

**Code:**
```python
H = lm_head.weight
```

**Observe:**
```python
print("H.shape:", H.shape)
print("H.dtype:", H.dtype)
print("H is lm_head.weight:", H is lm_head.weight)
```

**Verify:**
```python
assert tuple(H.shape) == (vocab_size, d)
assert H.dtype == torch.bfloat16
assert H is lm_head.weight
```

**Reflect (one line):** `H` and `E` have the same shape but are separate tensors in this model — Step 1.4.3 already established that, so manual code using `H.T` for projection is guaranteed not to silently use embedding weights.

---

#### Step 4.2.2 — Manual matmul

**Delta:** Compute `logits_manual = x_final @ H.T`.

**Predict:** `logits_manual.shape == (1, N_tok, vocab_size)`. Matches `lm_head(x_final)` within `atol=1e-4` (small bf16 matmul rounding differences are expected).

**Code:**
```python
logits_manual = x_final @ H.T
```

**Observe:**
```python
logits_real = lm_head(x_final)
max_diff    = (logits_manual.float() - logits_real.float()).abs().max().item()
print("logits_manual shape:", logits_manual.shape)
print("max diff to lm_head:", max_diff)
```

**Verify:**
```python
assert logits_manual.shape == logits_real.shape
assert torch.allclose(logits_manual.float(), logits_real.float(), atol=1e-4, rtol=1e-4)
```

**Reflect (one line):** The `.T` is needed because PyTorch's `Linear` stores its weight as `[out_features, in_features]` but performs `x @ W.T + b` — the transpose is implicit in the module call but must be explicit in the manual matmul.

---

#### Step 4.2.3 — Confirm against the original forward-pass logits

**Delta:** Verify `logits_manual` matches the `logits` tensor from Phase 2 — closing the loop on the outer pipeline.

**Predict:** `logits_manual` matches `logits` within the same tolerance, because both were computed from the same model on the same input via paths that overlap from the final norm onward.

**Code:**
```python
diff_to_original = (logits_manual.float() - logits.float()).abs().max().item()
```

**Observe:**
```python
print("max diff to original logits:", diff_to_original)
```

**Verify:**
```python
assert torch.allclose(logits_manual.float(), logits.float(), atol=1e-4, rtol=1e-4)
```

**Reflect (one line):** Matching the original forward-pass `logits` confirms the chain `embed → blocks → final_norm → manual_lm_head` reproduces the full model output — even with blocks and final norm still opaque, the LM head's standalone correctness is now established.

---

### Level 4.3 — Stitch `manual_lm_head`

**Concept introduced:** The matmul is wrapped into a helper for use in later phases.

**Black boxes at start of level:** none new.
**Black boxes opened by this level:** none.
**Black boxes still deferred at end of level:** none.

---

#### Step 4.3.1 — Define and verify the helper on two inputs

**Delta:** Bundle the matmul; test on `x_final` and on a separate `[1, N_tok, d]` tensor (the embedding output).

**Predict:** `manual_lm_head(x, model)` matches `lm_head(x)` within tolerance for any tensor of shape `[..., d]`.

**Code:**
```python
def manual_lm_head(x, model):
    return x @ model.lm_head.weight.T
```

**Observe:**
```python
out1 = manual_lm_head(x_final, model)
test_input = embed(input_ids)
out2 = manual_lm_head(test_input, model)
print("on x_final  : max diff =", (out1.float() - lm_head(x_final).float()).abs().max().item())
print("on embedding: max diff =", (out2.float() - lm_head(test_input).float()).abs().max().item())
```

**Verify:**
```python
assert tuple(out1.shape) == (1, N_tok, vocab_size)
assert torch.allclose(out1.float(), lm_head(x_final).float(), atol=1e-4, rtol=1e-4)
assert torch.allclose(out2.float(), lm_head(test_input).float(), atol=1e-4, rtol=1e-4)
```

**Reflect (one line):** Testing on a semantically different tensor (raw embedding output) confirms the helper is a pure function of its input shape and dtype, not of what produced the input.

---

**End-of-level check:**
- Without scrolling: write `manual_lm_head` from memory. One line.

---

**End-of-phase milestone check (Phase 4):**
- `H` bound; shape `[128256, 2048]`.
- `manual_lm_head(x, model)` matches `lm_head(x)` on two inputs within `atol=1e-4`.
- The chain `manual_embedding → blocks (opaque) → final_norm (opaque) → manual_lm_head → slice → argmax → decode` produces the same token as `predict_one_token`.
- Without scrolling: why is the LM head operation `x @ H.T` rather than `x @ H`? What is the shape of `H` and what does each dim mean?
- What would break if Phase 4 were skipped? Write one paragraph from memory.

---

# Phase 5 — Open the final RMSNorm

## Phase 5 — Open the final RMSNorm

**Goal:** Replace `final_norm(x)` with manual computation in raw tensor operations. Each numerical sub-step of RMSNorm is observable and verified.

**Milestone:** A helper `manual_rmsnorm(x, weight, eps)` exists and matches all 33 RMSNorm submodules in the model (1 final + 16 × 2 per-block) within `atol=1e-4`.

**Builds on:** Phase 4 milestone.

**Helpers stitched in this phase:** `manual_rmsnorm(x, weight, eps)`.

---

### Level 5.1 — Inputs to RMSNorm

**Concept introduced:** RMSNorm is parameterized by a single weight vector and a small epsilon. Both are sourced from the submodule.

**Black boxes at start of level:** the RMSNorm formula.
**Black boxes opened by this level:** the inputs to that formula.
**Black boxes still deferred at end of level:** the formula itself (opens in Level 5.2).

---

#### Step 5.1.1 — Bind the weight and epsilon

**Delta:** Pull `final_norm.weight` and `final_norm.variance_epsilon`.

**Predict:** `g_final.shape == (d,)`. Dtype bf16. `eps` is a small positive float (Llama uses `1e-5`).

**Code:**
```python
g_final = final_norm.weight
eps     = final_norm.variance_epsilon
```

**Observe:**
```python
print("g_final.shape:", g_final.shape)
print("g_final.dtype:", g_final.dtype)
print("g_final[:5]  :", g_final[:5].float().tolist())
print("eps          :", eps)
```

**Verify:**
```python
assert tuple(g_final.shape) == (d,)
assert g_final.dtype == torch.bfloat16
assert abs(eps - 1e-5) < 1e-9
```

**Reflect (one line):** RMSNorm has only a per-channel scale `g` — no bias, no centering — which distinguishes it from LayerNorm and makes it cheaper, the single reason it is the standard normalization in modern LLMs.

---

### Level 5.2 — Build RMSNorm step by step

**Concept introduced:** The RMSNorm formula `output = x * rsqrt(mean(x^2) + eps) * g` is built in eight pedagogical sub-steps so each intermediate tensor is observable.

**Black boxes at start of level:** the formula.
**Black boxes opened by this level:** all eight sub-steps.
**Black boxes still deferred at end of level:** none for RMSNorm.

---

#### Step 5.2.1 — Cast input to fp32

**Delta:** Following HuggingFace's Llama implementation, do RMSNorm math in fp32 to avoid bf16 precision loss in the `mean(x^2)` reduction.

**Predict:** `x_fp32.shape == x_in.shape`. Dtype fp32. Values bit-equal to `x_in.float()` (bf16-to-fp32 widening is exact).

**Code:**
```python
x_in   = hidden_states[-1]   # pre-final-norm hidden state, bf16
x_fp32 = x_in.float()
```

**Observe:**
```python
print("x_in.dtype  :", x_in.dtype)
print("x_fp32.dtype:", x_fp32.dtype)
print("equal?      :", torch.equal(x_in.float(), x_fp32))
```

**Verify:**
```python
assert x_fp32.dtype == torch.float32
assert x_fp32.shape == x_in.shape
assert torch.equal(x_in.float(), x_fp32)
```

**Reflect (one line):** bf16 → fp32 is exact because every bf16 bit pattern is representable in fp32 — what matters is that subsequent arithmetic in fp32 keeps mantissa bits that bf16 would round away.

---

#### Step 5.2.2 — Square element-wise

**Delta:** Compute `x^2`.

**Predict:** Shape unchanged. All values `>= 0`.

**Code:**
```python
sq = x_fp32 * x_fp32
```

**Observe:**
```python
print("sq.shape:", sq.shape)
print("sq.min  :", sq.min().item())
```

**Verify:**
```python
assert sq.shape == x_fp32.shape
assert sq.min().item() >= 0.0
```

**Reflect (one line):** Squaring before averaging is what makes this a *root-mean-square* normalization; absolute value would produce a different (and worse) normalization.

---

#### Step 5.2.3 — Mean over the last dim, with `keepdim=True`

**Delta:** Average `sq` across the channel axis to get one scalar per token. `keepdim=True` preserves the reduced dim as size 1.

**Predict:** `ms.shape == (1, N_tok, 1)` — the last dim is preserved with size 1 because `keepdim=True`. Values are positive (mean of squares).

**Code:**
```python
ms = sq.mean(dim=-1, keepdim=True)
```

**Observe:**
```python
print("ms.shape :", ms.shape)
print("ms values:", ms.squeeze().tolist())
```

**Verify:**
```python
assert ms.shape == (1, N_tok, 1)
assert (ms > 0).all().item()
```

**Reflect (one line):** `keepdim=True` keeps the last axis as size 1 so the upcoming broadcast against `x_fp32` of shape `[1, N_tok, d]` works automatically — without it, an explicit `unsqueeze(-1)` would be needed.

---

#### Step 5.2.4 — Add epsilon

**Delta:** Add `eps` for numerical stability before the upcoming reciprocal-sqrt.

**Predict:** Shape unchanged. Values are `ms + eps`, essentially unchanged at the resolution of the final output (since `eps ≈ 1e-5` and `ms` is `O(1)`).

**Code:**
```python
ms_eps = ms + eps
```

**Observe:**
```python
print("ms_eps   :", ms_eps.squeeze().tolist())
print("diff to ms:", (ms_eps - ms).squeeze().tolist())
```

**Verify:**
```python
assert ms_eps.shape == ms.shape
assert (ms_eps > ms).all().item()
```

**Reflect (one line):** `eps` matters only in pathological cases (all-zero rows during training), but it must be present in inference too — the trained model expects it, and omitting it breaks bit-reproducibility against the submodule.

---

#### Step 5.2.5 — Reciprocal square root

**Delta:** Compute `1 / sqrt(ms_eps)` using `torch.rsqrt`.

**Predict:** Shape unchanged. Values positive. Sanity: `ms_eps * inv_rms * inv_rms` is approximately 1.

**Code:**
```python
inv_rms = torch.rsqrt(ms_eps)
```

**Observe:**
```python
print("inv_rms          :", inv_rms.squeeze().tolist())
print("ms_eps * inv_rms²:", (ms_eps * inv_rms * inv_rms).squeeze().tolist())
```

**Verify:**
```python
assert inv_rms.shape == ms_eps.shape
assert (inv_rms > 0).all().item()
ones = ms_eps * inv_rms * inv_rms
assert torch.allclose(ones, torch.ones_like(ones), atol=1e-5)
```

**Reflect (one line):** The reciprocal-then-multiply form `x * (1/rms)` is faster than divide on most hardware because reciprocals vectorize cleanly while division does not.

---

#### Step 5.2.6 — Multiply input by `inv_rms`

**Delta:** Scale the input by the per-token reciprocal RMS.

**Predict:** Shape unchanged. Per-row RMS of `x_normed` is approximately 1.

**Code:**
```python
x_normed = x_fp32 * inv_rms
```

**Observe:**
```python
print("x_normed.shape:", x_normed.shape)
rms_per_row = (x_normed * x_normed).mean(dim=-1).sqrt()
print("per-row RMS   :", rms_per_row.squeeze().tolist())
```

**Verify:**
```python
assert x_normed.shape == x_fp32.shape
assert torch.allclose(rms_per_row, torch.ones_like(rms_per_row), atol=1e-3)
```

**Reflect (one line):** After this step every row has RMS ≈ 1 regardless of the input's original magnitude — that consistent scale is what downstream layers depend on.

---

#### Step 5.2.7 — Multiply by `g_final`

**Delta:** Apply the learned per-channel scale.

**Predict:** Shape unchanged. Channel `c` is `x_normed[..., c] * g_final[c]`.

**Code:**
```python
x_scaled = x_normed * g_final.float()
```

**Observe:**
```python
print("x_scaled.shape:", x_scaled.shape)
print("sample        :", x_scaled[0, -1, :5].tolist())
```

**Verify:**
```python
assert x_scaled.shape == x_normed.shape
ch0_manual = x_normed[..., 0] * g_final[0].float()
assert torch.allclose(x_scaled[..., 0], ch0_manual, atol=1e-5)
```

**Reflect (one line):** The per-channel scale `g` lets the model learn to amplify some channels and dampen others — without it, normalization would force every channel to the same average magnitude.

---

#### Step 5.2.8 — Cast back to bf16

**Delta:** Return to bf16 to rejoin the rest of the model.

**Predict:** Shape unchanged. Dtype bf16. Values match `x_scaled` modulo bf16 rounding.

**Code:**
```python
x_out = x_scaled.to(torch.bfloat16)
```

**Observe:**
```python
print("x_out.dtype     :", x_out.dtype)
print("max rounding gap:", (x_out.float() - x_scaled).abs().max().item())
```

**Verify:**
```python
assert x_out.dtype == torch.bfloat16
assert x_out.shape == x_scaled.shape
```

**Reflect (one line):** The bf16 cast is the only lossy step — all math happened in fp32 precision; the output rejoins the network at the network's bf16 dtype.

---

#### Step 5.2.9 — Verify against the live submodule

**Delta:** Compare manually-built `x_out` against `final_norm(x_in)`.

**Predict:** Match within `atol=1e-4`.

**Code:**
```python
x_norm_real = final_norm(x_in)
max_diff    = (x_out.float() - x_norm_real.float()).abs().max().item()
```

**Observe:**
```python
print("max diff to final_norm:", max_diff)
```

**Verify:**
```python
assert x_out.shape == x_norm_real.shape
assert torch.allclose(x_out.float(), x_norm_real.float(), atol=1e-4, rtol=1e-4)
```

**Reflect (one line):** Matching within `1e-4` confirms our eight-step decomposition reproduces the submodule; a larger gap would indicate a missing step (most commonly: forgotten `eps` or skipped fp32 cast).

---

**End-of-level check:**
- Without scrolling: list the eight sub-steps in order.
- Predict: if `eps` were `0` instead of `1e-5`, would the output for our specific input differ visibly (outside `1e-4`)? Verify by replacing the `+ eps` step.

---

### Level 5.3 — Stitch `manual_rmsnorm`

**Concept introduced:** The eight sub-steps are wrapped into one helper.

**Black boxes at start of level:** none.
**Black boxes opened by this level:** none.
**Black boxes still deferred at end of level:** none.

---

#### Step 5.3.1 — Define the helper

**Delta:** Bundle the formula. Arg list is `(x, weight, eps)`.

**Predict:** `manual_rmsnorm(x_in, g_final, eps)` matches `final_norm(x_in)` within tolerance.

**Code:**
```python
def manual_rmsnorm(x, weight, eps):
    orig_dtype = x.dtype
    x_f = x.float()
    inv_rms = torch.rsqrt((x_f * x_f).mean(dim=-1, keepdim=True) + eps)
    return (x_f * inv_rms * weight.float()).to(orig_dtype)
```

**Observe:**
```python
out_helper = manual_rmsnorm(x_in, g_final, eps)
print("helper shape   :", out_helper.shape)
print("max diff to ref:", (out_helper.float() - final_norm(x_in).float()).abs().max().item())
```

**Verify:**
```python
assert out_helper.shape == x_in.shape
assert torch.allclose(out_helper.float(), final_norm(x_in).float(), atol=1e-4, rtol=1e-4)
```

**Reflect (one line):** Eight pedagogical sub-steps compress into four computational expressions — the eight existed for understanding; the four are sufficient for computation.

---

#### Step 5.3.2 — Verify across all 33 norm submodules

**Delta:** Test the helper against every RMSNorm in the model.

**Predict:** Zero mismatches across one final norm and 32 per-block norms.

**Code:**
```python
mismatches = []
for i, block in enumerate(layers):
    for name in ("input_layernorm", "post_attention_layernorm"):
        sub  = getattr(block, name)
        real = sub(x_in)
        mine = manual_rmsnorm(x_in, sub.weight, sub.variance_epsilon)
        gap  = (real.float() - mine.float()).abs().max().item()
        if gap > 1e-4:
            mismatches.append((i, name, gap))
# also test the final norm
final_real = final_norm(x_in)
final_mine = manual_rmsnorm(x_in, final_norm.weight, final_norm.variance_epsilon)
final_gap  = (final_real.float() - final_mine.float()).abs().max().item()
if final_gap > 1e-4:
    mismatches.append(("final", "norm", final_gap))
```

**Observe:**
```python
print("mismatches:", mismatches)
print("total norms tested:", 2 * len(layers) + 1)
```

**Verify:**
```python
assert mismatches == []
```

**Reflect (one line):** A single helper matches 33 different norm submodules — same code, different weights — confirming RMSNorm is one operation parameterized by `(weight, eps)`, not 33 different operations.

---

**End-of-level check:**
- Without scrolling: write `manual_rmsnorm` from scratch. Confirm the structure (cast → square → mean → eps → rsqrt → multiply → scale → cast back).

---

**End-of-phase milestone check (Phase 5):**
- Eight sub-steps of RMSNorm each have a named tensor and a value-checked assertion.
- `manual_rmsnorm(x, weight, eps)` matches all 33 norm submodules within `atol=1e-4`.
- Without scrolling: what does RMSNorm do, in plain English, in one paragraph? What three things distinguish it from LayerNorm?
- What would break if Phase 5 were skipped? Write one paragraph from memory.
