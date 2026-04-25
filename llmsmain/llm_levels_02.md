# LLM Inference — 50-level Incremental Build

> **Target:** one inference, one token predicted. No KV cache, no batching. Each level adds exactly one new concept. Yellow-highlighted rows mark what changed vs. the previous level.

**Column meaning**

- **Step** — unique label of the step (e.g., `3.i.a.q : produce Q`).
- **Op** — the actual operation: row lookup, matmul, elementwise add, softmax, reshape, RMSNorm, etc.
- **In** — runtime tensor name(s) entering the step.
- **In shape** — tensor shape(s) of the inputs.
- **Static** — fixed data used (learned weights or pre-built tables), with shape annotated inline as `{name : shape}`.
- **Out** — runtime tensor name leaving the step.
- **Out shape** — tensor shape of the output.
- **Dim out** — scalar / 1D vector / 2D matrix / 3D tensor / 4D tensor.

### Sizes (declared once)

```
N_tok       = number of input tokens             (variable per inference)
d           = per-token vector dim                (e.g., 4096)
d_ff        = FFN hidden dim                      (e.g., 14336)
n_h         = number of query heads               (e.g., 32)
n_kv        = number of K/V heads (GQA)           (e.g., 8)
d_h         = per-head dim = d / n_h              (e.g., 128)
vocab_size  = vocabulary size                     (e.g., 128256)
N           = number of transformer blocks        (e.g., 32)
max_pos     = max supported positions             (e.g., 8192)
```

### Static data catalog (revealed progressively)

```
{vocab}            : [vocab_size]                    BPE table (pre-built)
{E}                : [vocab_size × d]                embedding matrix (learned)
{attn_i.norm}      : [d]                             pre-attn RMSNorm scale (learned)
{attn_i.W_Q}       : [d × d]                         Q projection (learned)
{attn_i.W_K}       : [d × (n_kv · d_h)]              K projection (learned)
{attn_i.W_V}       : [d × (n_kv · d_h)]              V projection (learned)
{attn_i.W_O}       : [d × d]                         O projection (learned)
{RoPE}             : [max_pos × d_h]                 sin/cos table (pre-built)
{causal_mask}      : [N_tok × N_tok]                 lower-triangular mask (pre-built)
{ffn_i.norm}       : [d]                             pre-FFN RMSNorm scale (learned)
{ffn_i.W_gate}     : [d × d_ff]                      gate projection (learned)
{ffn_i.W_up}       : [d × d_ff]                      up projection (learned)
{ffn_i.W_down}     : [d_ff × d]                      down projection (learned)
{final_norm}       : [d]                             final RMSNorm scale (learned)
{H}                : [d × vocab_size]                LM head (learned)
```

### Table of contents (50 levels)

- [Level 0 — opaque box](#L0)
- [Level 1 — split model call from decode](#L1)
- [Level 2 — split tokenize from the model](#L2)
- [Level 3 — expose embed as a separate step (internals opaque)](#L3)
- [Level 4 — expose {E} as the embedding matrix](#L4)
- [Level 5 — LLM core outputs vectors, not a token ID](#L5)
- [Level 6 — split to token into LM head + pick](#L6)
- [Level 7 — expose {H} as the LM head matrix](#L7)
- [Level 8 — take only the last position's logits](#L8)
- [Level 9 — softmax converts logits to probabilities](#L9)
- [Level 10 — pick = argmax (for one-inference target)](#L10)
- [Level 11 — add final norm before LM head (weights opaque)](#L11)
- [Level 12 — expose {final_norm}](#L12)
- [Level 13 — core is N sequential blocks (each opaque)](#L13)
- [Level 14 — each block has two sub-stages A then B (both opaque)](#L14)
- [Level 15 — name sub-stage A as attention](#L15)
- [Level 16 — name sub-stage B as FFN](#L16)
- [Level 17 — residual connection around attention](#L17)
- [Level 18 — pre-attention norm (weights opaque)](#L18)
- [Level 19 — expose {attn_i.norm}](#L19)
- [Level 20 — attention core ends with output projection (opaque)](#L20)
- [Level 21 — expose {W_O}](#L21)
- [Level 22 — attention inner consumes three derived tensors Q, K, V](#L22)
- [Level 23 — expose {W_Q}](#L23)
- [Level 24 — expose {W_K}](#L24)
- [Level 25 — expose {W_V}](#L25)
- [Level 26 — op begins with Q·Kᵀ scores](#L26)
- [Level 27 — scale by 1/√d_h](#L27)
- [Level 28 — apply causal mask](#L28)
- [Level 29 — softmax over keys](#L29)
- [Level 30 — weighted sum with V](#L30)
- [Level 31 — RoPE on Q](#L31)
- [Level 32 — RoPE on K](#L32)
- [Level 33 — split Q into heads](#L33)
- [Level 34 — split K into heads](#L34)
- [Level 35 — split V into heads](#L35)
- [Level 36 — concat heads before output projection](#L36)
- [Level 37 — GQA: K and V have fewer heads than Q](#L37)
- [Level 38 — residual connection around FFN](#L38)
- [Level 39 — pre-FFN norm (weights opaque)](#L39)
- [Level 40 — expose {ffn_i.norm}](#L40)
- [Level 41 — FFN core ends with down projection (opaque)](#L41)
- [Level 42 — expose {W_down}](#L42)
- [Level 43 — FFN inner has an up projection (opaque)](#L43)
- [Level 44 — expose {W_up}](#L44)
- [Level 45 — parallel gate projection (opaque)](#L45)
- [Level 46 — expose {W_gate}](#L46)
- [Level 47 — SiLU on gate](#L47)
- [Level 48 — combine = elementwise multiply](#L48)
- [Level 49 — complete reference](#L49)

## Phase 1 — Outer pipeline (around the core)

### Level 0 — opaque box <a id="L0"></a>

**New:** Entire pipeline is one box. `{model}` bundles vocab plus all learned weights.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| **1** | LLM system | opaque | text | `—` | {model} | text piece | `—` | — |

### Level 1 — split model call from decode <a id="L1"></a>

**New:** The model's true output is a token ID. Converting that integer to readable text uses `{vocab}`.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| **1** | LLM | opaque | text | `—` | {model internals} | 1 token ID | `scalar` | scalar |
| **2** | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

### Level 2 — split tokenize from the model <a id="L2"></a>

**New:** Text can't enter the model as a raw string. It's first split into integer token IDs using the same `{vocab}`.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| **1** | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| **2** | LLM | opaque | token IDs | `[N_tok]` | {model internals} | 1 token ID | `scalar` | scalar |
| 3 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

### Level 3 — expose embed as a separate step (internals opaque) <a id="L3"></a>

**New:** The LLM core doesn't consume integer IDs. Before it runs, each token ID is converted to a vector by an embed step. Internals not yet exposed.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| **2** | embed | row lookup in {E} | token IDs | `[N_tok]` | {embed internals} | vectors | `[N_tok × d]` | 2D matrix |
| **3** | LLM core | opaque | vectors | `[N_tok × d]` | {core internals} | 1 token ID | `scalar` | scalar |
| 4 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

### Level 4 — expose `{E}` as the embedding matrix <a id="L4"></a>

**New:** Embed is a row-lookup in `{E}`. Row `i` is the vector for token ID `i`. Learned matrix, but the operation is indexing, not matmul.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| **2** | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | LLM core | opaque | vectors | `[N_tok × d]` | {core internals} | 1 token ID | `scalar` | scalar |
| 4 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

### Level 5 — LLM core outputs vectors, not a token ID <a id="L5"></a>

**New:** The core's real output is vectors. A new downstream `to token` step converts those vectors into a token ID.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| **3** | LLM core | opaque | vectors | `[N_tok × d]` | {core internals} | vectors | `[N_tok × d]` | 2D matrix |
| **4** | to token | opaque | vectors | `[N_tok × d]` | {to-token internals} | 1 token ID | `scalar` | scalar |
| 5 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

### Level 6 — split `to token` into LM head + pick <a id="L6"></a>

**New:** `to token`'s first sub-step is a linear projection — the LM head — producing raw scores (logits). The remaining `pick` step is stateless.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | LLM core | opaque | vectors | `[N_tok × d]` | {core internals} | vectors | `[N_tok × d]` | 2D matrix |
| **4** | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {head internals} | logits | `[N_tok × vocab_size]` | 2D matrix |
| **5** | pick | argmax / sample | logits | `[N_tok × vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 6 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

### Level 7 — expose `{H}` as the LM head matrix <a id="L7"></a>

**New:** LM head is a matmul with learned `{H}` of shape `[d × vocab_size]`.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | LLM core | opaque | vectors | `[N_tok × d]` | {core internals} | vectors | `[N_tok × d]` | 2D matrix |
| **4** | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits | `[N_tok × vocab_size]` | 2D matrix |
| 5 | pick | argmax / sample | logits | `[N_tok × vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 6 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

### Level 8 — take only the last position's logits <a id="L8"></a>

**New:** LM head produces one logits row per input token. For predicting one next token, only the last position's row is used.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | LLM core | opaque | vectors | `[N_tok × d]` | {core internals} | vectors | `[N_tok × d]` | 2D matrix |
| **4** | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| **5** | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| **6** | pick | argmax / sample | logits (last position) | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 7 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

### Level 9 — softmax converts logits to probabilities <a id="L9"></a>

**New:** `pick` doesn't consume raw logits. Softmax normalizes them into a distribution that sums to 1.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | LLM core | opaque | vectors | `[N_tok × d]` | {core internals} | vectors | `[N_tok × d]` | 2D matrix |
| 4 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 5 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| **6** | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| **7** | pick | argmax / sample | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 8 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

### Level 10 — pick = argmax (for one-inference target) <a id="L10"></a>

**New:** Pick is a sampling policy. Simplest is argmax. Alternatives: temperature, top-k, top-p. For one-token prediction, argmax is sufficient.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | LLM core | opaque | vectors | `[N_tok × d]` | {core internals} | vectors | `[N_tok × d]` | 2D matrix |
| 4 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 5 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 6 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| **7** | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 8 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

## Phase 2 — Around the transformer core

### Level 11 — add final norm before LM head (weights opaque) <a id="L11"></a>

**New:** Between core output and LM head, a normalization step stabilizes vector magnitudes. Parameters not yet exposed.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | LLM core | opaque | vectors | `[N_tok × d]` | {core internals} | vectors | `[N_tok × d]` | 2D matrix |
| **4** | final norm | RMSNorm | vectors | `[N_tok × d]` | {final-norm internals} | vectors | `[N_tok × d]` | 2D matrix |
| 5 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 6 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 7 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 8 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 9 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

### Level 12 — expose `{final_norm}` <a id="L12"></a>

**New:** Final-norm's learned scale weights are `{final_norm}` of shape `[d]`. Operation is RMSNorm.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | LLM core | opaque | vectors | `[N_tok × d]` | {core internals} | vectors | `[N_tok × d]` | 2D matrix |
| **4** | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 5 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 6 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 7 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 8 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 9 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

## Phase 3 — Inside the core

### Level 13 — core is N sequential blocks (each opaque) <a id="L13"></a>

**New:** The core is `N` blocks in series. Each has its own independent weights. Each preserves shape `[N_tok × d]`.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| **3** | block 1 | opaque | vectors | `[N_tok × d]` | {block 1 internals} | vectors | `[N_tok × d]` | 2D matrix |
| **4** | block 2 | opaque | vectors | `[N_tok × d]` | {block 2 internals} | vectors | `[N_tok × d]` | 2D matrix |
| **5** | ... | ... | ... | `...` | ... | ... | `?` | ? |
| **6** | block N | opaque | vectors | `[N_tok × d]` | {block N internals} | vectors | `[N_tok × d]` | 2D matrix |
| 7 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 8 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 9 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 10 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 11 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 12 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

### Level 14 — each block has two sub-stages A then B (both opaque) <a id="L14"></a>

**New:** A block is not atomic. It contains two internal sub-stages in sequence with different roles. Roles not yet named.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| **3** | 3.i.a : stage A (block i) | opaque | vectors | `[N_tok × d]` | {A internals of block i} | vectors | `[N_tok × d]` | 2D matrix |
| **4** | 3.i.b : stage B (block i) | opaque | vectors | `[N_tok × d]` | {B internals of block i} | vectors | `[N_tok × d]` | 2D matrix |
| 5 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 6 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 7 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 8 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 9 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 10 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

### Level 15 — name sub-stage A as attention <a id="L15"></a>

**New:** The first sub-stage is called attention. Still opaque internally.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| **3** | 3.i.a : attention (block i) | opaque | vectors | `[N_tok × d]` | {attn_i internals} | vectors | `[N_tok × d]` | 2D matrix |
| 4 | 3.i.b : stage B (block i) | opaque | vectors | `[N_tok × d]` | {B internals of block i} | vectors | `[N_tok × d]` | 2D matrix |
| 5 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 6 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 7 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 8 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 9 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 10 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

### Level 16 — name sub-stage B as FFN <a id="L16"></a>

**New:** The second sub-stage is called the feedforward network (FFN). Still opaque internally.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a : attention (block i) | opaque | vectors | `[N_tok × d]` | {attn_i internals} | vectors | `[N_tok × d]` | 2D matrix |
| **4** | 3.i.b : FFN (block i) | opaque | vectors | `[N_tok × d]` | {ffn_i internals} | vectors | `[N_tok × d]` | 2D matrix |
| 5 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 6 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 7 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 8 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 9 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 10 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

## Phase 4 — Inside attention (outer wrapping)

### Level 17 — residual connection around attention <a id="L17"></a>

**New:** Attention doesn't replace x; it produces a correction added back to x. Save and add have no static data — pure plumbing.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| **3** | 3.i.a.save : save x | identity (hold copy of x) | x (vectors) | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| **4** | 3.i.a.core : attention core | opaque | x | `[N_tok × d]` | {attn_i core internals} | attn delta | `[N_tok × d]` | 2D matrix |
| **5** | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 6 | 3.i.b : FFN (block i) | opaque | vectors | `[N_tok × d]` | {ffn_i internals} | vectors | `[N_tok × d]` | 2D matrix |
| 7 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 8 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 9 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 10 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 11 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 12 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

### Level 18 — pre-attention norm (weights opaque) <a id="L18"></a>

**New:** Before the attention core, input is normalized. Residual is added to the *un-normed* x — pre-norm architecture.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a.save : save x | identity (hold copy of x) | x | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| **4** | 3.i.a.norm : pre-attn norm | RMSNorm | x | `[N_tok × d]` | {attn_i.norm internals} | x_normed | `[N_tok × d]` | 2D matrix |
| **5** | 3.i.a.core : attention core | opaque | x_normed | `[N_tok × d]` | {attn_i core internals} | attn delta | `[N_tok × d]` | 2D matrix |
| 6 | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 7 | 3.i.b : FFN (block i) | opaque | vectors | `[N_tok × d]` | {ffn_i internals} | vectors | `[N_tok × d]` | 2D matrix |
| 8 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 9 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 10 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 11 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 12 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 13 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

### Level 19 — expose `{attn_i.norm}` <a id="L19"></a>

**New:** Pre-attn norm weights are `{attn_i.norm}` of shape `[d]`. RMSNorm, per-block.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a.save : save x | identity (hold copy of x) | x | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| **4** | 3.i.a.norm : pre-attn norm | RMSNorm | x | `[N_tok × d]` | {attn_i.norm : d} | x_normed | `[N_tok × d]` | 2D matrix |
| 5 | 3.i.a.core : attention core | opaque | x_normed | `[N_tok × d]` | {attn_i core internals} | attn delta | `[N_tok × d]` | 2D matrix |
| 6 | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 7 | 3.i.b : FFN (block i) | opaque | vectors | `[N_tok × d]` | {ffn_i internals} | vectors | `[N_tok × d]` | 2D matrix |
| 8 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 9 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 10 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 11 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 12 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 13 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

### Level 20 — attention core ends with output projection (opaque) <a id="L20"></a>

**New:** Last step of the attention core is a linear projection mapping an internal result back into the block's vector space.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a.save : save x | identity (hold copy of x) | x | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| 4 | 3.i.a.norm : pre-attn norm | RMSNorm | x | `[N_tok × d]` | {attn_i.norm : d} | x_normed | `[N_tok × d]` | 2D matrix |
| **5** | 3.i.a.inner : attention inner | opaque | x_normed | `[N_tok × d]` | {attn_i inner internals} | inner result | `[N_tok × d]` | 2D matrix |
| **6** | 3.i.a.proj : output projection | matmul (· W_O) | inner result | `[N_tok × d]` | {W_O internals} | attn delta | `[N_tok × d]` | 2D matrix |
| 7 | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 8 | 3.i.b : FFN (block i) | opaque | vectors | `[N_tok × d]` | {ffn_i internals} | vectors | `[N_tok × d]` | 2D matrix |
| 9 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 10 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 11 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 12 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 13 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 14 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

### Level 21 — expose `{W_O}` <a id="L21"></a>

**New:** Output projection uses `{attn_i.W_O}` of shape `[d × d]`.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a.save : save x | identity (hold copy of x) | x | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| 4 | 3.i.a.norm : pre-attn norm | RMSNorm | x | `[N_tok × d]` | {attn_i.norm : d} | x_normed | `[N_tok × d]` | 2D matrix |
| 5 | 3.i.a.inner : attention inner | opaque | x_normed | `[N_tok × d]` | {attn_i inner internals} | inner result | `[N_tok × d]` | 2D matrix |
| **6** | 3.i.a.proj : output projection | matmul (· W_O) | inner result | `[N_tok × d]` | {attn_i.W_O : d × d} | attn delta | `[N_tok × d]` | 2D matrix |
| 7 | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 8 | 3.i.b : FFN (block i) | opaque | vectors | `[N_tok × d]` | {ffn_i internals} | vectors | `[N_tok × d]` | 2D matrix |
| 9 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 10 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 11 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 12 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 13 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 14 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

### Level 22 — attention inner consumes three derived tensors Q, K, V <a id="L22"></a>

**New:** The attention core doesn't use x_normed directly. Three separate projections produce Q, K, V. The attention op combines them.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a.save : save x | identity (hold copy of x) | x | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| 4 | 3.i.a.norm : pre-attn norm | RMSNorm | x | `[N_tok × d]` | {attn_i.norm : d} | x_normed | `[N_tok × d]` | 2D matrix |
| **5** | 3.i.a.q : produce Q | matmul (x_normed · W_Q) | x_normed | `[N_tok × d]` | {W_Q internals} | Q | `[N_tok × d]` | 2D matrix |
| **6** | 3.i.a.k : produce K | matmul (x_normed · W_K) | x_normed | `[N_tok × d]` | {W_K internals} | K | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| **7** | 3.i.a.v : produce V | matmul (x_normed · W_V) | x_normed | `[N_tok × d]` | {W_V internals} | V | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| **8** | 3.i.a.op : attention op | opaque | Q, K, V | `[N_tok × d], [N_tok × (n_kv·d_h)], [N_tok × (n_kv·d_h)]` | — | inner result | `[N_tok × d]` | 2D matrix |
| 9 | 3.i.a.proj : output projection | matmul (· W_O) | inner result | `[N_tok × d]` | {attn_i.W_O : d × d} | attn delta | `[N_tok × d]` | 2D matrix |
| 10 | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 11 | 3.i.b : FFN (block i) | opaque | vectors | `[N_tok × d]` | {ffn_i internals} | vectors | `[N_tok × d]` | 2D matrix |
| 12 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 13 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 14 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 15 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 16 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 17 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

### Level 23 — expose `{W_Q}` <a id="L23"></a>

**New:** `Q = x_normed · W_Q`, with `W_Q` of shape `[d × d]`.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a.save : save x | identity (hold copy of x) | x | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| 4 | 3.i.a.norm : pre-attn norm | RMSNorm | x | `[N_tok × d]` | {attn_i.norm : d} | x_normed | `[N_tok × d]` | 2D matrix |
| **5** | 3.i.a.q : produce Q | matmul (x_normed · W_Q) | x_normed | `[N_tok × d]` | {attn_i.W_Q : d × d} | Q | `[N_tok × d]` | 2D matrix |
| 6 | 3.i.a.k : produce K | matmul (x_normed · W_K) | x_normed | `[N_tok × d]` | {W_K internals} | K | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 7 | 3.i.a.v : produce V | matmul (x_normed · W_V) | x_normed | `[N_tok × d]` | {W_V internals} | V | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 8 | 3.i.a.op : attention op | opaque | Q, K, V | `[N_tok × d], [N_tok × (n_kv·d_h)], [N_tok × (n_kv·d_h)]` | — | inner result | `[N_tok × d]` | 2D matrix |
| 9 | 3.i.a.proj : output projection | matmul (· W_O) | inner result | `[N_tok × d]` | {attn_i.W_O : d × d} | attn delta | `[N_tok × d]` | 2D matrix |
| 10 | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 11 | 3.i.b : FFN (block i) | opaque | vectors | `[N_tok × d]` | {ffn_i internals} | vectors | `[N_tok × d]` | 2D matrix |
| 12 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 13 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 14 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 15 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 16 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 17 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

### Level 24 — expose `{W_K}` <a id="L24"></a>

**New:** `K = x_normed · W_K`, with `W_K` of shape `[d × (n_kv · d_h)]`. Second dim differs from W_Q because of grouped-query attention (see Level 37).

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a.save : save x | identity (hold copy of x) | x | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| 4 | 3.i.a.norm : pre-attn norm | RMSNorm | x | `[N_tok × d]` | {attn_i.norm : d} | x_normed | `[N_tok × d]` | 2D matrix |
| 5 | 3.i.a.q : produce Q | matmul (x_normed · W_Q) | x_normed | `[N_tok × d]` | {attn_i.W_Q : d × d} | Q | `[N_tok × d]` | 2D matrix |
| **6** | 3.i.a.k : produce K | matmul (x_normed · W_K) | x_normed | `[N_tok × d]` | {attn_i.W_K : d × (n_kv·d_h)} | K | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 7 | 3.i.a.v : produce V | matmul (x_normed · W_V) | x_normed | `[N_tok × d]` | {W_V internals} | V | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 8 | 3.i.a.op : attention op | opaque | Q, K, V | `[N_tok × d], [N_tok × (n_kv·d_h)], [N_tok × (n_kv·d_h)]` | — | inner result | `[N_tok × d]` | 2D matrix |
| 9 | 3.i.a.proj : output projection | matmul (· W_O) | inner result | `[N_tok × d]` | {attn_i.W_O : d × d} | attn delta | `[N_tok × d]` | 2D matrix |
| 10 | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 11 | 3.i.b : FFN (block i) | opaque | vectors | `[N_tok × d]` | {ffn_i internals} | vectors | `[N_tok × d]` | 2D matrix |
| 12 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 13 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 14 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 15 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 16 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 17 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

### Level 25 — expose `{W_V}` <a id="L25"></a>

**New:** `V = x_normed · W_V`, with `W_V` of shape `[d × (n_kv · d_h)]`.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a.save : save x | identity (hold copy of x) | x | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| 4 | 3.i.a.norm : pre-attn norm | RMSNorm | x | `[N_tok × d]` | {attn_i.norm : d} | x_normed | `[N_tok × d]` | 2D matrix |
| 5 | 3.i.a.q : produce Q | matmul (x_normed · W_Q) | x_normed | `[N_tok × d]` | {attn_i.W_Q : d × d} | Q | `[N_tok × d]` | 2D matrix |
| 6 | 3.i.a.k : produce K | matmul (x_normed · W_K) | x_normed | `[N_tok × d]` | {attn_i.W_K : d × (n_kv·d_h)} | K | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| **7** | 3.i.a.v : produce V | matmul (x_normed · W_V) | x_normed | `[N_tok × d]` | {attn_i.W_V : d × (n_kv·d_h)} | V | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 8 | 3.i.a.op : attention op | opaque | Q, K, V | `[N_tok × d], [N_tok × (n_kv·d_h)], [N_tok × (n_kv·d_h)]` | — | inner result | `[N_tok × d]` | 2D matrix |
| 9 | 3.i.a.proj : output projection | matmul (· W_O) | inner result | `[N_tok × d]` | {attn_i.W_O : d × d} | attn delta | `[N_tok × d]` | 2D matrix |
| 10 | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 11 | 3.i.b : FFN (block i) | opaque | vectors | `[N_tok × d]` | {ffn_i internals} | vectors | `[N_tok × d]` | 2D matrix |
| 12 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 13 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 14 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 15 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 16 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 17 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

## Phase 5 — Inside the attention op

### Level 26 — op begins with Q·Kᵀ scores <a id="L26"></a>

**New:** The attention op's first step computes similarity scores via `Q · Kᵀ`.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a.save : save x | identity (hold copy of x) | x | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| 4 | 3.i.a.norm : pre-attn norm | RMSNorm | x | `[N_tok × d]` | {attn_i.norm : d} | x_normed | `[N_tok × d]` | 2D matrix |
| 5 | 3.i.a.q : produce Q | matmul (x_normed · W_Q) | x_normed | `[N_tok × d]` | {attn_i.W_Q : d × d} | Q | `[N_tok × d]` | 2D matrix |
| 6 | 3.i.a.k : produce K | matmul (x_normed · W_K) | x_normed | `[N_tok × d]` | {attn_i.W_K : d × (n_kv·d_h)} | K | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 7 | 3.i.a.v : produce V | matmul (x_normed · W_V) | x_normed | `[N_tok × d]` | {attn_i.W_V : d × (n_kv·d_h)} | V | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| **8** | 3.i.a.op.scores : Q·Kᵀ | batched matmul (Q · Kᵀ) | Q, K | `[N_tok × d], [N_tok × (n_kv·d_h)]` | — | raw scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| **9** | 3.i.a.op.rest : rest of op | opaque | raw scores, V | `[n_h × N_tok × N_tok], [N_tok × (n_kv·d_h)]` | — | inner result | `[N_tok × d]` | 2D matrix |
| 10 | 3.i.a.proj : output projection | matmul (· W_O) | inner result | `[N_tok × d]` | {attn_i.W_O : d × d} | attn delta | `[N_tok × d]` | 2D matrix |
| 11 | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 12 | 3.i.b : FFN (block i) | opaque | vectors | `[N_tok × d]` | {ffn_i internals} | vectors | `[N_tok × d]` | 2D matrix |
| 13 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 14 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 15 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 16 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 17 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 18 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

### Level 27 — scale by 1/√d_h <a id="L27"></a>

**New:** Raw scores are divided by `√d_h`. Prevents softmax saturation as d_h grows.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a.save : save x | identity (hold copy of x) | x | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| 4 | 3.i.a.norm : pre-attn norm | RMSNorm | x | `[N_tok × d]` | {attn_i.norm : d} | x_normed | `[N_tok × d]` | 2D matrix |
| 5 | 3.i.a.q : produce Q | matmul (x_normed · W_Q) | x_normed | `[N_tok × d]` | {attn_i.W_Q : d × d} | Q | `[N_tok × d]` | 2D matrix |
| 6 | 3.i.a.k : produce K | matmul (x_normed · W_K) | x_normed | `[N_tok × d]` | {attn_i.W_K : d × (n_kv·d_h)} | K | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 7 | 3.i.a.v : produce V | matmul (x_normed · W_V) | x_normed | `[N_tok × d]` | {attn_i.W_V : d × (n_kv·d_h)} | V | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 8 | 3.i.a.op.scores : Q·Kᵀ | batched matmul (Q · Kᵀ) | Q, K | `[N_tok × d], [N_tok × (n_kv·d_h)]` | — | raw scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| **9** | 3.i.a.op.scale : scale by 1/√d_h | scalar divide (/ √d_h) | raw scores | `[n_h × N_tok × N_tok]` | — | scaled scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| **10** | 3.i.a.op.rest : rest of op | opaque | scaled scores, V | `[n_h × N_tok × N_tok], [N_tok × (n_kv·d_h)]` | — | inner result | `[N_tok × d]` | 2D matrix |
| 11 | 3.i.a.proj : output projection | matmul (· W_O) | inner result | `[N_tok × d]` | {attn_i.W_O : d × d} | attn delta | `[N_tok × d]` | 2D matrix |
| 12 | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 13 | 3.i.b : FFN (block i) | opaque | vectors | `[N_tok × d]` | {ffn_i internals} | vectors | `[N_tok × d]` | 2D matrix |
| 14 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 15 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 16 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 17 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 18 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 19 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

### Level 28 — apply causal mask <a id="L28"></a>

**New:** Each query position may only attend to itself and earlier ones. Mask sets future-position scores to `-∞`.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a.save : save x | identity (hold copy of x) | x | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| 4 | 3.i.a.norm : pre-attn norm | RMSNorm | x | `[N_tok × d]` | {attn_i.norm : d} | x_normed | `[N_tok × d]` | 2D matrix |
| 5 | 3.i.a.q : produce Q | matmul (x_normed · W_Q) | x_normed | `[N_tok × d]` | {attn_i.W_Q : d × d} | Q | `[N_tok × d]` | 2D matrix |
| 6 | 3.i.a.k : produce K | matmul (x_normed · W_K) | x_normed | `[N_tok × d]` | {attn_i.W_K : d × (n_kv·d_h)} | K | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 7 | 3.i.a.v : produce V | matmul (x_normed · W_V) | x_normed | `[N_tok × d]` | {attn_i.W_V : d × (n_kv·d_h)} | V | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 8 | 3.i.a.op.scores : Q·Kᵀ | batched matmul (Q · Kᵀ) | Q, K | `[N_tok × d], [N_tok × (n_kv·d_h)]` | — | raw scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 9 | 3.i.a.op.scale : scale by 1/√d_h | scalar divide (/ √d_h) | raw scores | `[n_h × N_tok × N_tok]` | — | scaled scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| **10** | 3.i.a.op.mask : causal mask | elementwise add (−∞ to future) | scaled scores | `[n_h × N_tok × N_tok]` | {causal_mask : N_tok × N_tok} | masked scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| **11** | 3.i.a.op.rest : rest of op | opaque | masked scores, V | `[n_h × N_tok × N_tok], [N_tok × (n_kv·d_h)]` | — | inner result | `[N_tok × d]` | 2D matrix |
| 12 | 3.i.a.proj : output projection | matmul (· W_O) | inner result | `[N_tok × d]` | {attn_i.W_O : d × d} | attn delta | `[N_tok × d]` | 2D matrix |
| 13 | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 14 | 3.i.b : FFN (block i) | opaque | vectors | `[N_tok × d]` | {ffn_i internals} | vectors | `[N_tok × d]` | 2D matrix |
| 15 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 16 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 17 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 18 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 19 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 20 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

### Level 29 — softmax over keys <a id="L29"></a>

**New:** Softmax runs along the key axis. For each query position, scores across keys become a distribution.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a.save : save x | identity (hold copy of x) | x | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| 4 | 3.i.a.norm : pre-attn norm | RMSNorm | x | `[N_tok × d]` | {attn_i.norm : d} | x_normed | `[N_tok × d]` | 2D matrix |
| 5 | 3.i.a.q : produce Q | matmul (x_normed · W_Q) | x_normed | `[N_tok × d]` | {attn_i.W_Q : d × d} | Q | `[N_tok × d]` | 2D matrix |
| 6 | 3.i.a.k : produce K | matmul (x_normed · W_K) | x_normed | `[N_tok × d]` | {attn_i.W_K : d × (n_kv·d_h)} | K | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 7 | 3.i.a.v : produce V | matmul (x_normed · W_V) | x_normed | `[N_tok × d]` | {attn_i.W_V : d × (n_kv·d_h)} | V | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 8 | 3.i.a.op.scores : Q·Kᵀ | batched matmul (Q · Kᵀ) | Q, K | `[N_tok × d], [N_tok × (n_kv·d_h)]` | — | raw scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 9 | 3.i.a.op.scale : scale by 1/√d_h | scalar divide (/ √d_h) | raw scores | `[n_h × N_tok × N_tok]` | — | scaled scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 10 | 3.i.a.op.mask : causal mask | elementwise add (−∞ to future) | scaled scores | `[n_h × N_tok × N_tok]` | {causal_mask : N_tok × N_tok} | masked scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| **11** | 3.i.a.op.smax : softmax (per row) | softmax (over key axis) | masked scores | `[n_h × N_tok × N_tok]` | — | attn weights | `[n_h × N_tok × N_tok]` | 3D tensor |
| **12** | 3.i.a.op.rest : rest of op | opaque | attn weights, V | `[n_h × N_tok × N_tok], [N_tok × (n_kv·d_h)]` | — | inner result | `[N_tok × d]` | 2D matrix |
| 13 | 3.i.a.proj : output projection | matmul (· W_O) | inner result | `[N_tok × d]` | {attn_i.W_O : d × d} | attn delta | `[N_tok × d]` | 2D matrix |
| 14 | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 15 | 3.i.b : FFN (block i) | opaque | vectors | `[N_tok × d]` | {ffn_i internals} | vectors | `[N_tok × d]` | 2D matrix |
| 16 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 17 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 18 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 19 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 20 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 21 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

### Level 30 — weighted sum with V <a id="L30"></a>

**New:** For each query position, inner result is a weighted sum of V rows using that row's attention weights.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a.save : save x | identity (hold copy of x) | x | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| 4 | 3.i.a.norm : pre-attn norm | RMSNorm | x | `[N_tok × d]` | {attn_i.norm : d} | x_normed | `[N_tok × d]` | 2D matrix |
| 5 | 3.i.a.q : produce Q | matmul (x_normed · W_Q) | x_normed | `[N_tok × d]` | {attn_i.W_Q : d × d} | Q | `[N_tok × d]` | 2D matrix |
| 6 | 3.i.a.k : produce K | matmul (x_normed · W_K) | x_normed | `[N_tok × d]` | {attn_i.W_K : d × (n_kv·d_h)} | K | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 7 | 3.i.a.v : produce V | matmul (x_normed · W_V) | x_normed | `[N_tok × d]` | {attn_i.W_V : d × (n_kv·d_h)} | V | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 8 | 3.i.a.op.scores : Q·Kᵀ | batched matmul (Q · Kᵀ) | Q, K | `[N_tok × d], [N_tok × (n_kv·d_h)]` | — | raw scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 9 | 3.i.a.op.scale : scale by 1/√d_h | scalar divide (/ √d_h) | raw scores | `[n_h × N_tok × N_tok]` | — | scaled scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 10 | 3.i.a.op.mask : causal mask | elementwise add (−∞ to future) | scaled scores | `[n_h × N_tok × N_tok]` | {causal_mask : N_tok × N_tok} | masked scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 11 | 3.i.a.op.smax : softmax (per row) | softmax (over key axis) | masked scores | `[n_h × N_tok × N_tok]` | — | attn weights | `[n_h × N_tok × N_tok]` | 3D tensor |
| **12** | 3.i.a.op.weighted : weighted sum with V | batched matmul (weights · V) | attn weights, V | `[n_h × N_tok × N_tok], [N_tok × (n_kv·d_h)]` | — | inner result | `[N_tok × d]` | 2D matrix |
| 13 | 3.i.a.proj : output projection | matmul (· W_O) | inner result | `[N_tok × d]` | {attn_i.W_O : d × d} | attn delta | `[N_tok × d]` | 2D matrix |
| 14 | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 15 | 3.i.b : FFN (block i) | opaque | vectors | `[N_tok × d]` | {ffn_i internals} | vectors | `[N_tok × d]` | 2D matrix |
| 16 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 17 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 18 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 19 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 20 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 21 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

## Phase 6 — Positional information (RoPE)

### Level 31 — RoPE on Q <a id="L31"></a>

**New:** Without positional info, scores are order-insensitive. RoPE rotates each Q vector by a position-dependent angle. `{RoPE}` is pre-built, not learned.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a.save : save x | identity (hold copy of x) | x | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| 4 | 3.i.a.norm : pre-attn norm | RMSNorm | x | `[N_tok × d]` | {attn_i.norm : d} | x_normed | `[N_tok × d]` | 2D matrix |
| 5 | 3.i.a.q : produce Q | matmul (x_normed · W_Q) | x_normed | `[N_tok × d]` | {attn_i.W_Q : d × d} | Q | `[N_tok × d]` | 2D matrix |
| 6 | 3.i.a.k : produce K | matmul (x_normed · W_K) | x_normed | `[N_tok × d]` | {attn_i.W_K : d × (n_kv·d_h)} | K | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| **7** | 3.i.a.v : produce V | matmul (x_normed · W_V) | x_normed | `[N_tok × d]` | {attn_i.W_V : d × (n_kv·d_h)} | V | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 8 | 3.i.a.rope_q : rotate Q | elementwise rotate (RoPE) | Q | `[N_tok × d]` | {RoPE : max_pos × d_h} | Q_rot | `[n_h × N_tok × d_h]` | 3D tensor |
| 9 | 3.i.a.op.scores : Q·Kᵀ | batched matmul (Q · Kᵀ) | Q_rot, K | `[n_h × N_tok × d_h], [N_tok × (n_kv·d_h)]` | — | raw scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| **10** | 3.i.a.op.scale : scale by 1/√d_h | scalar divide (/ √d_h) | raw scores | `[n_h × N_tok × N_tok]` | — | scaled scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 11 | 3.i.a.op.mask : causal mask | elementwise add (−∞ to future) | scaled scores | `[n_h × N_tok × N_tok]` | {causal_mask : N_tok × N_tok} | masked scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 12 | 3.i.a.op.smax : softmax (per row) | softmax (over key axis) | masked scores | `[n_h × N_tok × N_tok]` | — | attn weights | `[n_h × N_tok × N_tok]` | 3D tensor |
| 13 | 3.i.a.op.weighted : weighted sum with V | batched matmul (weights · V) | attn weights, V | `[n_h × N_tok × N_tok], [N_tok × (n_kv·d_h)]` | — | inner result | `[N_tok × d]` | 2D matrix |
| 14 | 3.i.a.proj : output projection | matmul (· W_O) | inner result | `[N_tok × d]` | {attn_i.W_O : d × d} | attn delta | `[N_tok × d]` | 2D matrix |
| 15 | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 16 | 3.i.b : FFN (block i) | opaque | vectors | `[N_tok × d]` | {ffn_i internals} | vectors | `[N_tok × d]` | 2D matrix |
| 17 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 18 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 19 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 20 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 21 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 22 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

### Level 32 — RoPE on K <a id="L32"></a>

**New:** K is also rotated by `{RoPE}`. V is never rotated. Result: `Q_rot · K_rotᵀ` depends only on relative position.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a.save : save x | identity (hold copy of x) | x | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| 4 | 3.i.a.norm : pre-attn norm | RMSNorm | x | `[N_tok × d]` | {attn_i.norm : d} | x_normed | `[N_tok × d]` | 2D matrix |
| 5 | 3.i.a.q : produce Q | matmul (x_normed · W_Q) | x_normed | `[N_tok × d]` | {attn_i.W_Q : d × d} | Q | `[N_tok × d]` | 2D matrix |
| 6 | 3.i.a.k : produce K | matmul (x_normed · W_K) | x_normed | `[N_tok × d]` | {attn_i.W_K : d × (n_kv·d_h)} | K | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 7 | 3.i.a.v : produce V | matmul (x_normed · W_V) | x_normed | `[N_tok × d]` | {attn_i.W_V : d × (n_kv·d_h)} | V | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| **8** | 3.i.a.rope_q : rotate Q | elementwise rotate (RoPE) | Q | `[N_tok × d]` | {RoPE : max_pos × d_h} | Q_rot | `[n_h × N_tok × d_h]` | 3D tensor |
| 9 | 3.i.a.rope_k : rotate K | elementwise rotate (RoPE) | K | `[N_tok × (n_kv·d_h)]` | {RoPE : max_pos × d_h} | K_rot | `[n_kv × N_tok × d_h]` | 3D tensor |
| **10** | 3.i.a.op.scores : Q·Kᵀ | batched matmul (Q · Kᵀ) | Q_rot, K_rot | `[n_h × N_tok × d_h], [n_kv × N_tok × d_h]` | — | raw scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 11 | 3.i.a.op.scale : scale by 1/√d_h | scalar divide (/ √d_h) | raw scores | `[n_h × N_tok × N_tok]` | — | scaled scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 12 | 3.i.a.op.mask : causal mask | elementwise add (−∞ to future) | scaled scores | `[n_h × N_tok × N_tok]` | {causal_mask : N_tok × N_tok} | masked scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 13 | 3.i.a.op.smax : softmax (per row) | softmax (over key axis) | masked scores | `[n_h × N_tok × N_tok]` | — | attn weights | `[n_h × N_tok × N_tok]` | 3D tensor |
| 14 | 3.i.a.op.weighted : weighted sum with V | batched matmul (weights · V) | attn weights, V | `[n_h × N_tok × N_tok], [N_tok × (n_kv·d_h)]` | — | inner result | `[N_tok × d]` | 2D matrix |
| 15 | 3.i.a.proj : output projection | matmul (· W_O) | inner result | `[N_tok × d]` | {attn_i.W_O : d × d} | attn delta | `[N_tok × d]` | 2D matrix |
| 16 | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 17 | 3.i.b : FFN (block i) | opaque | vectors | `[N_tok × d]` | {ffn_i internals} | vectors | `[N_tok × d]` | 2D matrix |
| 18 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 19 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 20 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 21 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 22 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 23 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

## Phase 7 — Multi-head (and GQA)

### Level 33 — split Q into heads <a id="L33"></a>

**New:** Q is reshaped so `n_h` heads each get a `d_h`-dim slice. Downstream ops run per-head.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a.save : save x | identity (hold copy of x) | x | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| **4** | 3.i.a.norm : pre-attn norm | RMSNorm | x | `[N_tok × d]` | {attn_i.norm : d} | x_normed | `[N_tok × d]` | 2D matrix |
| **5** | 3.i.a.q : produce Q | matmul (x_normed · W_Q) | x_normed | `[N_tok × d]` | {attn_i.W_Q : d × d} | Q flat | `[N_tok × d]` | 2D matrix |
| 6 | 3.i.a.q_split : split Q into heads | reshape (split heads) | Q flat | `[N_tok × d]` | — | Q per-head | `[n_h × N_tok × d_h]` | 3D tensor |
| 7 | 3.i.a.k : produce K | matmul (x_normed · W_K) | x_normed | `[N_tok × d]` | {attn_i.W_K : d × (n_kv·d_h)} | K | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| **8** | 3.i.a.v : produce V | matmul (x_normed · W_V) | x_normed | `[N_tok × d]` | {attn_i.W_V : d × (n_kv·d_h)} | V | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 9 | 3.i.a.rope_q : rotate Q | elementwise rotate (RoPE) | Q per-head | `[n_h × N_tok × d_h]` | {RoPE : max_pos × d_h} | Q_rot | `[n_h × N_tok × d_h]` | 3D tensor |
| 10 | 3.i.a.rope_k : rotate K | elementwise rotate (RoPE) | K | `[N_tok × (n_kv·d_h)]` | {RoPE : max_pos × d_h} | K_rot | `[n_kv × N_tok × d_h]` | 3D tensor |
| 11 | 3.i.a.op.scores : Q·Kᵀ | batched matmul (Q · Kᵀ) | Q_rot, K_rot | `[n_h × N_tok × d_h], [n_kv × N_tok × d_h]` | — | raw scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 12 | 3.i.a.op.scale : scale by 1/√d_h | scalar divide (/ √d_h) | raw scores | `[n_h × N_tok × N_tok]` | — | scaled scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 13 | 3.i.a.op.mask : causal mask | elementwise add (−∞ to future) | scaled scores | `[n_h × N_tok × N_tok]` | {causal_mask : N_tok × N_tok} | masked scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 14 | 3.i.a.op.smax : softmax (per row) | softmax (over key axis) | masked scores | `[n_h × N_tok × N_tok]` | — | attn weights | `[n_h × N_tok × N_tok]` | 3D tensor |
| 15 | 3.i.a.op.weighted : weighted sum with V | batched matmul (weights · V) | attn weights, V | `[n_h × N_tok × N_tok], [N_tok × (n_kv·d_h)]` | — | inner result | `[N_tok × d]` | 2D matrix |
| 16 | 3.i.a.proj : output projection | matmul (· W_O) | inner result | `[N_tok × d]` | {attn_i.W_O : d × d} | attn delta | `[N_tok × d]` | 2D matrix |
| 17 | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 18 | 3.i.b : FFN (block i) | opaque | vectors | `[N_tok × d]` | {ffn_i internals} | vectors | `[N_tok × d]` | 2D matrix |
| 19 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 20 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 21 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 22 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 23 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 24 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

### Level 34 — split K into heads <a id="L34"></a>

**New:** K is reshaped into `n_kv` heads.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a.save : save x | identity (hold copy of x) | x | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| 4 | 3.i.a.norm : pre-attn norm | RMSNorm | x | `[N_tok × d]` | {attn_i.norm : d} | x_normed | `[N_tok × d]` | 2D matrix |
| 5 | 3.i.a.q : produce Q | matmul (x_normed · W_Q) | x_normed | `[N_tok × d]` | {attn_i.W_Q : d × d} | Q flat | `[N_tok × d]` | 2D matrix |
| **6** | 3.i.a.q_split : split Q into heads | reshape (split heads) | Q flat | `[N_tok × d]` | — | Q per-head | `[n_h × N_tok × d_h]` | 3D tensor |
| **7** | 3.i.a.k : produce K | matmul (x_normed · W_K) | x_normed | `[N_tok × d]` | {attn_i.W_K : d × (n_kv·d_h)} | K flat | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 8 | 3.i.a.k_split : split K into heads | reshape (split heads) | K flat | `[N_tok × (n_kv·d_h)]` | — | K per-head | `[n_kv × N_tok × d_h]` | 3D tensor |
| 9 | 3.i.a.v : produce V | matmul (x_normed · W_V) | x_normed | `[N_tok × d]` | {attn_i.W_V : d × (n_kv·d_h)} | V | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| **10** | 3.i.a.rope_q : rotate Q | elementwise rotate (RoPE) | Q per-head | `[n_h × N_tok × d_h]` | {RoPE : max_pos × d_h} | Q_rot | `[n_h × N_tok × d_h]` | 3D tensor |
| 11 | 3.i.a.rope_k : rotate K | elementwise rotate (RoPE) | K per-head | `[n_kv × N_tok × d_h]` | {RoPE : max_pos × d_h} | K_rot | `[n_kv × N_tok × d_h]` | 3D tensor |
| 12 | 3.i.a.op.scores : Q·Kᵀ | batched matmul (Q · Kᵀ) | Q_rot, K_rot | `[n_h × N_tok × d_h], [n_kv × N_tok × d_h]` | — | raw scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 13 | 3.i.a.op.scale : scale by 1/√d_h | scalar divide (/ √d_h) | raw scores | `[n_h × N_tok × N_tok]` | — | scaled scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 14 | 3.i.a.op.mask : causal mask | elementwise add (−∞ to future) | scaled scores | `[n_h × N_tok × N_tok]` | {causal_mask : N_tok × N_tok} | masked scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 15 | 3.i.a.op.smax : softmax (per row) | softmax (over key axis) | masked scores | `[n_h × N_tok × N_tok]` | — | attn weights | `[n_h × N_tok × N_tok]` | 3D tensor |
| 16 | 3.i.a.op.weighted : weighted sum with V | batched matmul (weights · V) | attn weights, V | `[n_h × N_tok × N_tok], [N_tok × (n_kv·d_h)]` | — | inner result | `[N_tok × d]` | 2D matrix |
| 17 | 3.i.a.proj : output projection | matmul (· W_O) | inner result | `[N_tok × d]` | {attn_i.W_O : d × d} | attn delta | `[N_tok × d]` | 2D matrix |
| 18 | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 19 | 3.i.b : FFN (block i) | opaque | vectors | `[N_tok × d]` | {ffn_i internals} | vectors | `[N_tok × d]` | 2D matrix |
| 20 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 21 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 22 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 23 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 24 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 25 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

### Level 35 — split V into heads <a id="L35"></a>

**New:** V is reshaped into `n_kv` heads. Weighted sum runs per-head.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a.save : save x | identity (hold copy of x) | x | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| 4 | 3.i.a.norm : pre-attn norm | RMSNorm | x | `[N_tok × d]` | {attn_i.norm : d} | x_normed | `[N_tok × d]` | 2D matrix |
| 5 | 3.i.a.q : produce Q | matmul (x_normed · W_Q) | x_normed | `[N_tok × d]` | {attn_i.W_Q : d × d} | Q flat | `[N_tok × d]` | 2D matrix |
| 6 | 3.i.a.q_split : split Q into heads | reshape (split heads) | Q flat | `[N_tok × d]` | — | Q per-head | `[n_h × N_tok × d_h]` | 3D tensor |
| 7 | 3.i.a.k : produce K | matmul (x_normed · W_K) | x_normed | `[N_tok × d]` | {attn_i.W_K : d × (n_kv·d_h)} | K flat | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| **8** | 3.i.a.k_split : split K into heads | reshape (split heads) | K flat | `[N_tok × (n_kv·d_h)]` | — | K per-head | `[n_kv × N_tok × d_h]` | 3D tensor |
| **9** | 3.i.a.v : produce V | matmul (x_normed · W_V) | x_normed | `[N_tok × d]` | {attn_i.W_V : d × (n_kv·d_h)} | V flat | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 10 | 3.i.a.v_split : split V into heads | reshape (split heads) | V flat | `[N_tok × (n_kv·d_h)]` | — | V per-head | `[n_kv × N_tok × d_h]` | 3D tensor |
| 11 | 3.i.a.rope_q : rotate Q | elementwise rotate (RoPE) | Q per-head | `[n_h × N_tok × d_h]` | {RoPE : max_pos × d_h} | Q_rot | `[n_h × N_tok × d_h]` | 3D tensor |
| 12 | 3.i.a.rope_k : rotate K | elementwise rotate (RoPE) | K per-head | `[n_kv × N_tok × d_h]` | {RoPE : max_pos × d_h} | K_rot | `[n_kv × N_tok × d_h]` | 3D tensor |
| 13 | 3.i.a.op.scores : Q·Kᵀ | batched matmul (Q · Kᵀ) | Q_rot, K_rot | `[n_h × N_tok × d_h], [n_kv × N_tok × d_h]` | — | raw scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 14 | 3.i.a.op.scale : scale by 1/√d_h | scalar divide (/ √d_h) | raw scores | `[n_h × N_tok × N_tok]` | — | scaled scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| **15** | 3.i.a.op.mask : causal mask | elementwise add (−∞ to future) | scaled scores | `[n_h × N_tok × N_tok]` | {causal_mask : N_tok × N_tok} | masked scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 16 | 3.i.a.op.smax : softmax (per row) | softmax (over key axis) | masked scores | `[n_h × N_tok × N_tok]` | — | attn weights | `[n_h × N_tok × N_tok]` | 3D tensor |
| 17 | 3.i.a.op.weighted : weighted sum with V | batched matmul (weights · V) | attn weights, V per-head | `[n_h × N_tok × N_tok], [n_kv × N_tok × d_h]` | — | inner result | `[N_tok × d]` | 2D matrix |
| 18 | 3.i.a.proj : output projection | matmul (· W_O) | inner result | `[N_tok × d]` | {attn_i.W_O : d × d} | attn delta | `[N_tok × d]` | 2D matrix |
| 19 | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 20 | 3.i.b : FFN (block i) | opaque | vectors | `[N_tok × d]` | {ffn_i internals} | vectors | `[N_tok × d]` | 2D matrix |
| 21 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 22 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 23 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 24 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 25 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 26 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

### Level 36 — concat heads before output projection <a id="L36"></a>

**New:** Per-head results are concatenated back into one `d`-dim tensor per token. `W_O` then mixes across heads.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a.save : save x | identity (hold copy of x) | x | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| 4 | 3.i.a.norm : pre-attn norm | RMSNorm | x | `[N_tok × d]` | {attn_i.norm : d} | x_normed | `[N_tok × d]` | 2D matrix |
| 5 | 3.i.a.q : produce Q | matmul (x_normed · W_Q) | x_normed | `[N_tok × d]` | {attn_i.W_Q : d × d} | Q flat | `[N_tok × d]` | 2D matrix |
| 6 | 3.i.a.q_split : split Q into heads | reshape (split heads) | Q flat | `[N_tok × d]` | — | Q per-head | `[n_h × N_tok × d_h]` | 3D tensor |
| 7 | 3.i.a.k : produce K | matmul (x_normed · W_K) | x_normed | `[N_tok × d]` | {attn_i.W_K : d × (n_kv·d_h)} | K flat | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 8 | 3.i.a.k_split : split K into heads | reshape (split heads) | K flat | `[N_tok × (n_kv·d_h)]` | — | K per-head | `[n_kv × N_tok × d_h]` | 3D tensor |
| 9 | 3.i.a.v : produce V | matmul (x_normed · W_V) | x_normed | `[N_tok × d]` | {attn_i.W_V : d × (n_kv·d_h)} | V flat | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 10 | 3.i.a.v_split : split V into heads | reshape (split heads) | V flat | `[N_tok × (n_kv·d_h)]` | — | V per-head | `[n_kv × N_tok × d_h]` | 3D tensor |
| 11 | 3.i.a.rope_q : rotate Q | elementwise rotate (RoPE) | Q per-head | `[n_h × N_tok × d_h]` | {RoPE : max_pos × d_h} | Q_rot | `[n_h × N_tok × d_h]` | 3D tensor |
| 12 | 3.i.a.rope_k : rotate K | elementwise rotate (RoPE) | K per-head | `[n_kv × N_tok × d_h]` | {RoPE : max_pos × d_h} | K_rot | `[n_kv × N_tok × d_h]` | 3D tensor |
| 13 | 3.i.a.op.scores : Q·Kᵀ | batched matmul (Q · Kᵀ) | Q_rot, K_rot | `[n_h × N_tok × d_h], [n_kv × N_tok × d_h]` | — | raw scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 14 | 3.i.a.op.scale : scale by 1/√d_h | scalar divide (/ √d_h) | raw scores | `[n_h × N_tok × N_tok]` | — | scaled scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| **15** | 3.i.a.op.mask : causal mask | elementwise add (−∞ to future) | scaled scores | `[n_h × N_tok × N_tok]` | {causal_mask : N_tok × N_tok} | masked scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| **16** | 3.i.a.op.smax : softmax (per row) | softmax (over key axis) | masked scores | `[n_h × N_tok × N_tok]` | — | attn weights | `[n_h × N_tok × N_tok]` | 3D tensor |
| **17** | 3.i.a.op.weighted : weighted sum with V | batched matmul (weights · V) | attn weights, V per-head | `[n_h × N_tok × N_tok], [n_kv × N_tok × d_h]` | — | per-head results | `[n_h × N_tok × d_h]` | 3D tensor |
| 18 | 3.i.a.concat : concat heads | reshape (concat heads) | per-head results | `[n_h × N_tok × d_h]` | — | merged result | `[N_tok × d]` | 2D matrix |
| 19 | 3.i.a.proj : output projection | matmul (· W_O) | merged result | `[N_tok × d]` | {attn_i.W_O : d × d} | attn delta | `[N_tok × d]` | 2D matrix |
| 20 | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 21 | 3.i.b : FFN (block i) | opaque | vectors | `[N_tok × d]` | {ffn_i internals} | vectors | `[N_tok × d]` | 2D matrix |
| 22 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 23 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 24 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 25 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 26 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 27 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

### Level 37 — GQA: K and V have fewer heads than Q <a id="L37"></a>

**New:** Q has `n_h` heads; K/V have `n_kv < n_h`. Each K/V head serves a group of `n_h / n_kv` Q heads (repeated before scoring). This is why `W_K, W_V` have second dim `n_kv · d_h`. *Table identical to Level 36.*

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a.save : save x | identity (hold copy of x) | x | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| 4 | 3.i.a.norm : pre-attn norm | RMSNorm | x | `[N_tok × d]` | {attn_i.norm : d} | x_normed | `[N_tok × d]` | 2D matrix |
| 5 | 3.i.a.q : produce Q | matmul (x_normed · W_Q) | x_normed | `[N_tok × d]` | {attn_i.W_Q : d × d} | Q flat | `[N_tok × d]` | 2D matrix |
| 6 | 3.i.a.q_split : split Q into heads | reshape (split heads) | Q flat | `[N_tok × d]` | — | Q per-head | `[n_h × N_tok × d_h]` | 3D tensor |
| 7 | 3.i.a.k : produce K | matmul (x_normed · W_K) | x_normed | `[N_tok × d]` | {attn_i.W_K : d × (n_kv·d_h)} | K flat | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 8 | 3.i.a.k_split : split K into heads | reshape (split heads) | K flat | `[N_tok × (n_kv·d_h)]` | — | K per-head | `[n_kv × N_tok × d_h]` | 3D tensor |
| 9 | 3.i.a.v : produce V | matmul (x_normed · W_V) | x_normed | `[N_tok × d]` | {attn_i.W_V : d × (n_kv·d_h)} | V flat | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 10 | 3.i.a.v_split : split V into heads | reshape (split heads) | V flat | `[N_tok × (n_kv·d_h)]` | — | V per-head | `[n_kv × N_tok × d_h]` | 3D tensor |
| 11 | 3.i.a.rope_q : rotate Q | elementwise rotate (RoPE) | Q per-head | `[n_h × N_tok × d_h]` | {RoPE : max_pos × d_h} | Q_rot | `[n_h × N_tok × d_h]` | 3D tensor |
| 12 | 3.i.a.rope_k : rotate K | elementwise rotate (RoPE) | K per-head | `[n_kv × N_tok × d_h]` | {RoPE : max_pos × d_h} | K_rot | `[n_kv × N_tok × d_h]` | 3D tensor |
| 13 | 3.i.a.op.scores : Q·Kᵀ | batched matmul (Q · Kᵀ) | Q_rot, K_rot | `[n_h × N_tok × d_h], [n_kv × N_tok × d_h]` | — | raw scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 14 | 3.i.a.op.scale : scale by 1/√d_h | scalar divide (/ √d_h) | raw scores | `[n_h × N_tok × N_tok]` | — | scaled scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 15 | 3.i.a.op.mask : causal mask | elementwise add (−∞ to future) | scaled scores | `[n_h × N_tok × N_tok]` | {causal_mask : N_tok × N_tok} | masked scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 16 | 3.i.a.op.smax : softmax (per row) | softmax (over key axis) | masked scores | `[n_h × N_tok × N_tok]` | — | attn weights | `[n_h × N_tok × N_tok]` | 3D tensor |
| 17 | 3.i.a.op.weighted : weighted sum with V | batched matmul (weights · V) | attn weights, V per-head | `[n_h × N_tok × N_tok], [n_kv × N_tok × d_h]` | — | per-head results | `[n_h × N_tok × d_h]` | 3D tensor |
| 18 | 3.i.a.concat : concat heads | reshape (concat heads) | per-head results | `[n_h × N_tok × d_h]` | — | merged result | `[N_tok × d]` | 2D matrix |
| 19 | 3.i.a.proj : output projection | matmul (· W_O) | merged result | `[N_tok × d]` | {attn_i.W_O : d × d} | attn delta | `[N_tok × d]` | 2D matrix |
| 20 | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 21 | 3.i.b : FFN (block i) | opaque | vectors | `[N_tok × d]` | {ffn_i internals} | vectors | `[N_tok × d]` | 2D matrix |
| 22 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 23 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 24 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 25 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 26 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 27 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

## Phase 8 — Inside FFN

### Level 38 — residual connection around FFN <a id="L38"></a>

**New:** FFN produces a correction added to y, not a replacement.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a.save : save x | identity (hold copy of x) | x | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| 4 | 3.i.a.norm : pre-attn norm | RMSNorm | x | `[N_tok × d]` | {attn_i.norm : d} | x_normed | `[N_tok × d]` | 2D matrix |
| 5 | 3.i.a.q : produce Q | matmul (x_normed · W_Q) | x_normed | `[N_tok × d]` | {attn_i.W_Q : d × d} | Q flat | `[N_tok × d]` | 2D matrix |
| 6 | 3.i.a.q_split : split Q into heads | reshape (split heads) | Q flat | `[N_tok × d]` | — | Q per-head | `[n_h × N_tok × d_h]` | 3D tensor |
| 7 | 3.i.a.k : produce K | matmul (x_normed · W_K) | x_normed | `[N_tok × d]` | {attn_i.W_K : d × (n_kv·d_h)} | K flat | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 8 | 3.i.a.k_split : split K into heads | reshape (split heads) | K flat | `[N_tok × (n_kv·d_h)]` | — | K per-head | `[n_kv × N_tok × d_h]` | 3D tensor |
| 9 | 3.i.a.v : produce V | matmul (x_normed · W_V) | x_normed | `[N_tok × d]` | {attn_i.W_V : d × (n_kv·d_h)} | V flat | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 10 | 3.i.a.v_split : split V into heads | reshape (split heads) | V flat | `[N_tok × (n_kv·d_h)]` | — | V per-head | `[n_kv × N_tok × d_h]` | 3D tensor |
| 11 | 3.i.a.rope_q : rotate Q | elementwise rotate (RoPE) | Q per-head | `[n_h × N_tok × d_h]` | {RoPE : max_pos × d_h} | Q_rot | `[n_h × N_tok × d_h]` | 3D tensor |
| 12 | 3.i.a.rope_k : rotate K | elementwise rotate (RoPE) | K per-head | `[n_kv × N_tok × d_h]` | {RoPE : max_pos × d_h} | K_rot | `[n_kv × N_tok × d_h]` | 3D tensor |
| 13 | 3.i.a.op.scores : Q·Kᵀ | batched matmul (Q · Kᵀ) | Q_rot, K_rot | `[n_h × N_tok × d_h], [n_kv × N_tok × d_h]` | — | raw scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 14 | 3.i.a.op.scale : scale by 1/√d_h | scalar divide (/ √d_h) | raw scores | `[n_h × N_tok × N_tok]` | — | scaled scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 15 | 3.i.a.op.mask : causal mask | elementwise add (−∞ to future) | scaled scores | `[n_h × N_tok × N_tok]` | {causal_mask : N_tok × N_tok} | masked scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 16 | 3.i.a.op.smax : softmax (per row) | softmax (over key axis) | masked scores | `[n_h × N_tok × N_tok]` | — | attn weights | `[n_h × N_tok × N_tok]` | 3D tensor |
| 17 | 3.i.a.op.weighted : weighted sum with V | batched matmul (weights · V) | attn weights, V per-head | `[n_h × N_tok × N_tok], [n_kv × N_tok × d_h]` | — | per-head results | `[n_h × N_tok × d_h]` | 3D tensor |
| 18 | 3.i.a.concat : concat heads | reshape (concat heads) | per-head results | `[n_h × N_tok × d_h]` | — | merged result | `[N_tok × d]` | 2D matrix |
| 19 | 3.i.a.proj : output projection | matmul (· W_O) | merged result | `[N_tok × d]` | {attn_i.W_O : d × d} | attn delta | `[N_tok × d]` | 2D matrix |
| 20 | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| **21** | 3.i.b.save : save y | identity (hold copy of y) | y (vectors) | `[N_tok × d]` | — | y (held) | `[N_tok × d]` | 2D matrix |
| **22** | 3.i.b.core : FFN core | opaque | y | `[N_tok × d]` | {ffn_i core internals} | ffn delta | `[N_tok × d]` | 2D matrix |
| **23** | 3.i.b.add : add residual | elementwise add (residual) | y (held), ffn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 24 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 25 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 26 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 27 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 28 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 29 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

### Level 39 — pre-FFN norm (weights opaque) <a id="L39"></a>

**New:** Before FFN core, input is normalized. Residual added to un-normed y.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a.save : save x | identity (hold copy of x) | x | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| 4 | 3.i.a.norm : pre-attn norm | RMSNorm | x | `[N_tok × d]` | {attn_i.norm : d} | x_normed | `[N_tok × d]` | 2D matrix |
| 5 | 3.i.a.q : produce Q | matmul (x_normed · W_Q) | x_normed | `[N_tok × d]` | {attn_i.W_Q : d × d} | Q flat | `[N_tok × d]` | 2D matrix |
| 6 | 3.i.a.q_split : split Q into heads | reshape (split heads) | Q flat | `[N_tok × d]` | — | Q per-head | `[n_h × N_tok × d_h]` | 3D tensor |
| 7 | 3.i.a.k : produce K | matmul (x_normed · W_K) | x_normed | `[N_tok × d]` | {attn_i.W_K : d × (n_kv·d_h)} | K flat | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 8 | 3.i.a.k_split : split K into heads | reshape (split heads) | K flat | `[N_tok × (n_kv·d_h)]` | — | K per-head | `[n_kv × N_tok × d_h]` | 3D tensor |
| 9 | 3.i.a.v : produce V | matmul (x_normed · W_V) | x_normed | `[N_tok × d]` | {attn_i.W_V : d × (n_kv·d_h)} | V flat | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 10 | 3.i.a.v_split : split V into heads | reshape (split heads) | V flat | `[N_tok × (n_kv·d_h)]` | — | V per-head | `[n_kv × N_tok × d_h]` | 3D tensor |
| 11 | 3.i.a.rope_q : rotate Q | elementwise rotate (RoPE) | Q per-head | `[n_h × N_tok × d_h]` | {RoPE : max_pos × d_h} | Q_rot | `[n_h × N_tok × d_h]` | 3D tensor |
| 12 | 3.i.a.rope_k : rotate K | elementwise rotate (RoPE) | K per-head | `[n_kv × N_tok × d_h]` | {RoPE : max_pos × d_h} | K_rot | `[n_kv × N_tok × d_h]` | 3D tensor |
| 13 | 3.i.a.op.scores : Q·Kᵀ | batched matmul (Q · Kᵀ) | Q_rot, K_rot | `[n_h × N_tok × d_h], [n_kv × N_tok × d_h]` | — | raw scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 14 | 3.i.a.op.scale : scale by 1/√d_h | scalar divide (/ √d_h) | raw scores | `[n_h × N_tok × N_tok]` | — | scaled scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 15 | 3.i.a.op.mask : causal mask | elementwise add (−∞ to future) | scaled scores | `[n_h × N_tok × N_tok]` | {causal_mask : N_tok × N_tok} | masked scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 16 | 3.i.a.op.smax : softmax (per row) | softmax (over key axis) | masked scores | `[n_h × N_tok × N_tok]` | — | attn weights | `[n_h × N_tok × N_tok]` | 3D tensor |
| 17 | 3.i.a.op.weighted : weighted sum with V | batched matmul (weights · V) | attn weights, V per-head | `[n_h × N_tok × N_tok], [n_kv × N_tok × d_h]` | — | per-head results | `[n_h × N_tok × d_h]` | 3D tensor |
| 18 | 3.i.a.concat : concat heads | reshape (concat heads) | per-head results | `[n_h × N_tok × d_h]` | — | merged result | `[N_tok × d]` | 2D matrix |
| 19 | 3.i.a.proj : output projection | matmul (· W_O) | merged result | `[N_tok × d]` | {attn_i.W_O : d × d} | attn delta | `[N_tok × d]` | 2D matrix |
| 20 | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 21 | 3.i.b.save : save y | identity (hold copy of y) | y (vectors) | `[N_tok × d]` | — | y (held) | `[N_tok × d]` | 2D matrix |
| **22** | 3.i.b.norm : pre-FFN norm | RMSNorm | y | `[N_tok × d]` | {ffn_i.norm internals} | y_normed | `[N_tok × d]` | 2D matrix |
| **23** | 3.i.b.core : FFN core | opaque | y_normed | `[N_tok × d]` | {ffn_i core internals} | ffn delta | `[N_tok × d]` | 2D matrix |
| 24 | 3.i.b.add : add residual | elementwise add (residual) | y (held), ffn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 25 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 26 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 27 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 28 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 29 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 30 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

### Level 40 — expose `{ffn_i.norm}` <a id="L40"></a>

**New:** Pre-FFN norm weights are `{ffn_i.norm}` of shape `[d]`.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a.save : save x | identity (hold copy of x) | x | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| 4 | 3.i.a.norm : pre-attn norm | RMSNorm | x | `[N_tok × d]` | {attn_i.norm : d} | x_normed | `[N_tok × d]` | 2D matrix |
| 5 | 3.i.a.q : produce Q | matmul (x_normed · W_Q) | x_normed | `[N_tok × d]` | {attn_i.W_Q : d × d} | Q flat | `[N_tok × d]` | 2D matrix |
| 6 | 3.i.a.q_split : split Q into heads | reshape (split heads) | Q flat | `[N_tok × d]` | — | Q per-head | `[n_h × N_tok × d_h]` | 3D tensor |
| 7 | 3.i.a.k : produce K | matmul (x_normed · W_K) | x_normed | `[N_tok × d]` | {attn_i.W_K : d × (n_kv·d_h)} | K flat | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 8 | 3.i.a.k_split : split K into heads | reshape (split heads) | K flat | `[N_tok × (n_kv·d_h)]` | — | K per-head | `[n_kv × N_tok × d_h]` | 3D tensor |
| 9 | 3.i.a.v : produce V | matmul (x_normed · W_V) | x_normed | `[N_tok × d]` | {attn_i.W_V : d × (n_kv·d_h)} | V flat | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 10 | 3.i.a.v_split : split V into heads | reshape (split heads) | V flat | `[N_tok × (n_kv·d_h)]` | — | V per-head | `[n_kv × N_tok × d_h]` | 3D tensor |
| 11 | 3.i.a.rope_q : rotate Q | elementwise rotate (RoPE) | Q per-head | `[n_h × N_tok × d_h]` | {RoPE : max_pos × d_h} | Q_rot | `[n_h × N_tok × d_h]` | 3D tensor |
| 12 | 3.i.a.rope_k : rotate K | elementwise rotate (RoPE) | K per-head | `[n_kv × N_tok × d_h]` | {RoPE : max_pos × d_h} | K_rot | `[n_kv × N_tok × d_h]` | 3D tensor |
| 13 | 3.i.a.op.scores : Q·Kᵀ | batched matmul (Q · Kᵀ) | Q_rot, K_rot | `[n_h × N_tok × d_h], [n_kv × N_tok × d_h]` | — | raw scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 14 | 3.i.a.op.scale : scale by 1/√d_h | scalar divide (/ √d_h) | raw scores | `[n_h × N_tok × N_tok]` | — | scaled scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 15 | 3.i.a.op.mask : causal mask | elementwise add (−∞ to future) | scaled scores | `[n_h × N_tok × N_tok]` | {causal_mask : N_tok × N_tok} | masked scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 16 | 3.i.a.op.smax : softmax (per row) | softmax (over key axis) | masked scores | `[n_h × N_tok × N_tok]` | — | attn weights | `[n_h × N_tok × N_tok]` | 3D tensor |
| 17 | 3.i.a.op.weighted : weighted sum with V | batched matmul (weights · V) | attn weights, V per-head | `[n_h × N_tok × N_tok], [n_kv × N_tok × d_h]` | — | per-head results | `[n_h × N_tok × d_h]` | 3D tensor |
| 18 | 3.i.a.concat : concat heads | reshape (concat heads) | per-head results | `[n_h × N_tok × d_h]` | — | merged result | `[N_tok × d]` | 2D matrix |
| 19 | 3.i.a.proj : output projection | matmul (· W_O) | merged result | `[N_tok × d]` | {attn_i.W_O : d × d} | attn delta | `[N_tok × d]` | 2D matrix |
| 20 | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 21 | 3.i.b.save : save y | identity (hold copy of y) | y (vectors) | `[N_tok × d]` | — | y (held) | `[N_tok × d]` | 2D matrix |
| **22** | 3.i.b.norm : pre-FFN norm | RMSNorm | y | `[N_tok × d]` | {ffn_i.norm : d} | y_normed | `[N_tok × d]` | 2D matrix |
| 23 | 3.i.b.core : FFN core | opaque | y_normed | `[N_tok × d]` | {ffn_i core internals} | ffn delta | `[N_tok × d]` | 2D matrix |
| 24 | 3.i.b.add : add residual | elementwise add (residual) | y (held), ffn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 25 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 26 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 27 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 28 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 29 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 30 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

### Level 41 — FFN core ends with down projection (opaque) <a id="L41"></a>

**New:** Last step of FFN core is a linear projection from `d_ff` back to `d`.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a.save : save x | identity (hold copy of x) | x | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| 4 | 3.i.a.norm : pre-attn norm | RMSNorm | x | `[N_tok × d]` | {attn_i.norm : d} | x_normed | `[N_tok × d]` | 2D matrix |
| 5 | 3.i.a.q : produce Q | matmul (x_normed · W_Q) | x_normed | `[N_tok × d]` | {attn_i.W_Q : d × d} | Q flat | `[N_tok × d]` | 2D matrix |
| 6 | 3.i.a.q_split : split Q into heads | reshape (split heads) | Q flat | `[N_tok × d]` | — | Q per-head | `[n_h × N_tok × d_h]` | 3D tensor |
| 7 | 3.i.a.k : produce K | matmul (x_normed · W_K) | x_normed | `[N_tok × d]` | {attn_i.W_K : d × (n_kv·d_h)} | K flat | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 8 | 3.i.a.k_split : split K into heads | reshape (split heads) | K flat | `[N_tok × (n_kv·d_h)]` | — | K per-head | `[n_kv × N_tok × d_h]` | 3D tensor |
| 9 | 3.i.a.v : produce V | matmul (x_normed · W_V) | x_normed | `[N_tok × d]` | {attn_i.W_V : d × (n_kv·d_h)} | V flat | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 10 | 3.i.a.v_split : split V into heads | reshape (split heads) | V flat | `[N_tok × (n_kv·d_h)]` | — | V per-head | `[n_kv × N_tok × d_h]` | 3D tensor |
| 11 | 3.i.a.rope_q : rotate Q | elementwise rotate (RoPE) | Q per-head | `[n_h × N_tok × d_h]` | {RoPE : max_pos × d_h} | Q_rot | `[n_h × N_tok × d_h]` | 3D tensor |
| 12 | 3.i.a.rope_k : rotate K | elementwise rotate (RoPE) | K per-head | `[n_kv × N_tok × d_h]` | {RoPE : max_pos × d_h} | K_rot | `[n_kv × N_tok × d_h]` | 3D tensor |
| 13 | 3.i.a.op.scores : Q·Kᵀ | batched matmul (Q · Kᵀ) | Q_rot, K_rot | `[n_h × N_tok × d_h], [n_kv × N_tok × d_h]` | — | raw scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 14 | 3.i.a.op.scale : scale by 1/√d_h | scalar divide (/ √d_h) | raw scores | `[n_h × N_tok × N_tok]` | — | scaled scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 15 | 3.i.a.op.mask : causal mask | elementwise add (−∞ to future) | scaled scores | `[n_h × N_tok × N_tok]` | {causal_mask : N_tok × N_tok} | masked scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 16 | 3.i.a.op.smax : softmax (per row) | softmax (over key axis) | masked scores | `[n_h × N_tok × N_tok]` | — | attn weights | `[n_h × N_tok × N_tok]` | 3D tensor |
| 17 | 3.i.a.op.weighted : weighted sum with V | batched matmul (weights · V) | attn weights, V per-head | `[n_h × N_tok × N_tok], [n_kv × N_tok × d_h]` | — | per-head results | `[n_h × N_tok × d_h]` | 3D tensor |
| 18 | 3.i.a.concat : concat heads | reshape (concat heads) | per-head results | `[n_h × N_tok × d_h]` | — | merged result | `[N_tok × d]` | 2D matrix |
| 19 | 3.i.a.proj : output projection | matmul (· W_O) | merged result | `[N_tok × d]` | {attn_i.W_O : d × d} | attn delta | `[N_tok × d]` | 2D matrix |
| 20 | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 21 | 3.i.b.save : save y | identity (hold copy of y) | y (vectors) | `[N_tok × d]` | — | y (held) | `[N_tok × d]` | 2D matrix |
| 22 | 3.i.b.norm : pre-FFN norm | RMSNorm | y | `[N_tok × d]` | {ffn_i.norm : d} | y_normed | `[N_tok × d]` | 2D matrix |
| **23** | 3.i.b.inner : FFN inner | opaque | y_normed | `[N_tok × d]` | {ffn_i inner internals} | hidden | `[N_tok × d_ff]` | 2D matrix |
| **24** | 3.i.b.down : down projection | matmul (hidden · W_down) | hidden | `[N_tok × d_ff]` | {W_down internals} | ffn delta | `[N_tok × d]` | 2D matrix |
| 25 | 3.i.b.add : add residual | elementwise add (residual) | y (held), ffn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 26 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 27 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 28 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 29 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 30 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 31 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

### Level 42 — expose `{W_down}` <a id="L42"></a>

**New:** `ffn_i.W_down` has shape `[d_ff × d]`.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a.save : save x | identity (hold copy of x) | x | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| 4 | 3.i.a.norm : pre-attn norm | RMSNorm | x | `[N_tok × d]` | {attn_i.norm : d} | x_normed | `[N_tok × d]` | 2D matrix |
| 5 | 3.i.a.q : produce Q | matmul (x_normed · W_Q) | x_normed | `[N_tok × d]` | {attn_i.W_Q : d × d} | Q flat | `[N_tok × d]` | 2D matrix |
| 6 | 3.i.a.q_split : split Q into heads | reshape (split heads) | Q flat | `[N_tok × d]` | — | Q per-head | `[n_h × N_tok × d_h]` | 3D tensor |
| 7 | 3.i.a.k : produce K | matmul (x_normed · W_K) | x_normed | `[N_tok × d]` | {attn_i.W_K : d × (n_kv·d_h)} | K flat | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 8 | 3.i.a.k_split : split K into heads | reshape (split heads) | K flat | `[N_tok × (n_kv·d_h)]` | — | K per-head | `[n_kv × N_tok × d_h]` | 3D tensor |
| 9 | 3.i.a.v : produce V | matmul (x_normed · W_V) | x_normed | `[N_tok × d]` | {attn_i.W_V : d × (n_kv·d_h)} | V flat | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 10 | 3.i.a.v_split : split V into heads | reshape (split heads) | V flat | `[N_tok × (n_kv·d_h)]` | — | V per-head | `[n_kv × N_tok × d_h]` | 3D tensor |
| 11 | 3.i.a.rope_q : rotate Q | elementwise rotate (RoPE) | Q per-head | `[n_h × N_tok × d_h]` | {RoPE : max_pos × d_h} | Q_rot | `[n_h × N_tok × d_h]` | 3D tensor |
| 12 | 3.i.a.rope_k : rotate K | elementwise rotate (RoPE) | K per-head | `[n_kv × N_tok × d_h]` | {RoPE : max_pos × d_h} | K_rot | `[n_kv × N_tok × d_h]` | 3D tensor |
| 13 | 3.i.a.op.scores : Q·Kᵀ | batched matmul (Q · Kᵀ) | Q_rot, K_rot | `[n_h × N_tok × d_h], [n_kv × N_tok × d_h]` | — | raw scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 14 | 3.i.a.op.scale : scale by 1/√d_h | scalar divide (/ √d_h) | raw scores | `[n_h × N_tok × N_tok]` | — | scaled scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 15 | 3.i.a.op.mask : causal mask | elementwise add (−∞ to future) | scaled scores | `[n_h × N_tok × N_tok]` | {causal_mask : N_tok × N_tok} | masked scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 16 | 3.i.a.op.smax : softmax (per row) | softmax (over key axis) | masked scores | `[n_h × N_tok × N_tok]` | — | attn weights | `[n_h × N_tok × N_tok]` | 3D tensor |
| 17 | 3.i.a.op.weighted : weighted sum with V | batched matmul (weights · V) | attn weights, V per-head | `[n_h × N_tok × N_tok], [n_kv × N_tok × d_h]` | — | per-head results | `[n_h × N_tok × d_h]` | 3D tensor |
| 18 | 3.i.a.concat : concat heads | reshape (concat heads) | per-head results | `[n_h × N_tok × d_h]` | — | merged result | `[N_tok × d]` | 2D matrix |
| 19 | 3.i.a.proj : output projection | matmul (· W_O) | merged result | `[N_tok × d]` | {attn_i.W_O : d × d} | attn delta | `[N_tok × d]` | 2D matrix |
| 20 | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 21 | 3.i.b.save : save y | identity (hold copy of y) | y (vectors) | `[N_tok × d]` | — | y (held) | `[N_tok × d]` | 2D matrix |
| 22 | 3.i.b.norm : pre-FFN norm | RMSNorm | y | `[N_tok × d]` | {ffn_i.norm : d} | y_normed | `[N_tok × d]` | 2D matrix |
| 23 | 3.i.b.inner : FFN inner | opaque | y_normed | `[N_tok × d]` | {ffn_i inner internals} | hidden | `[N_tok × d_ff]` | 2D matrix |
| **24** | 3.i.b.down : down projection | matmul (hidden · W_down) | hidden | `[N_tok × d_ff]` | {ffn_i.W_down : d_ff × d} | ffn delta | `[N_tok × d]` | 2D matrix |
| 25 | 3.i.b.add : add residual | elementwise add (residual) | y (held), ffn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 26 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 27 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 28 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 29 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 30 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 31 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

### Level 43 — FFN inner has an up projection (opaque) <a id="L43"></a>

**New:** A projection lifts y_normed from `d` to `d_ff`. Rest of inner still opaque.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a.save : save x | identity (hold copy of x) | x | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| 4 | 3.i.a.norm : pre-attn norm | RMSNorm | x | `[N_tok × d]` | {attn_i.norm : d} | x_normed | `[N_tok × d]` | 2D matrix |
| 5 | 3.i.a.q : produce Q | matmul (x_normed · W_Q) | x_normed | `[N_tok × d]` | {attn_i.W_Q : d × d} | Q flat | `[N_tok × d]` | 2D matrix |
| 6 | 3.i.a.q_split : split Q into heads | reshape (split heads) | Q flat | `[N_tok × d]` | — | Q per-head | `[n_h × N_tok × d_h]` | 3D tensor |
| 7 | 3.i.a.k : produce K | matmul (x_normed · W_K) | x_normed | `[N_tok × d]` | {attn_i.W_K : d × (n_kv·d_h)} | K flat | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 8 | 3.i.a.k_split : split K into heads | reshape (split heads) | K flat | `[N_tok × (n_kv·d_h)]` | — | K per-head | `[n_kv × N_tok × d_h]` | 3D tensor |
| 9 | 3.i.a.v : produce V | matmul (x_normed · W_V) | x_normed | `[N_tok × d]` | {attn_i.W_V : d × (n_kv·d_h)} | V flat | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 10 | 3.i.a.v_split : split V into heads | reshape (split heads) | V flat | `[N_tok × (n_kv·d_h)]` | — | V per-head | `[n_kv × N_tok × d_h]` | 3D tensor |
| 11 | 3.i.a.rope_q : rotate Q | elementwise rotate (RoPE) | Q per-head | `[n_h × N_tok × d_h]` | {RoPE : max_pos × d_h} | Q_rot | `[n_h × N_tok × d_h]` | 3D tensor |
| 12 | 3.i.a.rope_k : rotate K | elementwise rotate (RoPE) | K per-head | `[n_kv × N_tok × d_h]` | {RoPE : max_pos × d_h} | K_rot | `[n_kv × N_tok × d_h]` | 3D tensor |
| 13 | 3.i.a.op.scores : Q·Kᵀ | batched matmul (Q · Kᵀ) | Q_rot, K_rot | `[n_h × N_tok × d_h], [n_kv × N_tok × d_h]` | — | raw scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 14 | 3.i.a.op.scale : scale by 1/√d_h | scalar divide (/ √d_h) | raw scores | `[n_h × N_tok × N_tok]` | — | scaled scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 15 | 3.i.a.op.mask : causal mask | elementwise add (−∞ to future) | scaled scores | `[n_h × N_tok × N_tok]` | {causal_mask : N_tok × N_tok} | masked scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 16 | 3.i.a.op.smax : softmax (per row) | softmax (over key axis) | masked scores | `[n_h × N_tok × N_tok]` | — | attn weights | `[n_h × N_tok × N_tok]` | 3D tensor |
| 17 | 3.i.a.op.weighted : weighted sum with V | batched matmul (weights · V) | attn weights, V per-head | `[n_h × N_tok × N_tok], [n_kv × N_tok × d_h]` | — | per-head results | `[n_h × N_tok × d_h]` | 3D tensor |
| 18 | 3.i.a.concat : concat heads | reshape (concat heads) | per-head results | `[n_h × N_tok × d_h]` | — | merged result | `[N_tok × d]` | 2D matrix |
| 19 | 3.i.a.proj : output projection | matmul (· W_O) | merged result | `[N_tok × d]` | {attn_i.W_O : d × d} | attn delta | `[N_tok × d]` | 2D matrix |
| 20 | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 21 | 3.i.b.save : save y | identity (hold copy of y) | y (vectors) | `[N_tok × d]` | — | y (held) | `[N_tok × d]` | 2D matrix |
| 22 | 3.i.b.norm : pre-FFN norm | RMSNorm | y | `[N_tok × d]` | {ffn_i.norm : d} | y_normed | `[N_tok × d]` | 2D matrix |
| **23** | 3.i.b.up : up projection | matmul (y_normed · W_up) | y_normed | `[N_tok × d]` | {W_up internals} | up | `[N_tok × d_ff]` | 2D matrix |
| **24** | 3.i.b.inner_rest : rest of inner | opaque | up | `[N_tok × d_ff]` | {ffn_i inner-rest internals} | hidden | `[N_tok × d_ff]` | 2D matrix |
| 25 | 3.i.b.down : down projection | matmul (hidden · W_down) | hidden | `[N_tok × d_ff]` | {ffn_i.W_down : d_ff × d} | ffn delta | `[N_tok × d]` | 2D matrix |
| 26 | 3.i.b.add : add residual | elementwise add (residual) | y (held), ffn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 27 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 28 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 29 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 30 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 31 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 32 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

### Level 44 — expose `{W_up}` <a id="L44"></a>

**New:** `ffn_i.W_up` has shape `[d × d_ff]`.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a.save : save x | identity (hold copy of x) | x | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| 4 | 3.i.a.norm : pre-attn norm | RMSNorm | x | `[N_tok × d]` | {attn_i.norm : d} | x_normed | `[N_tok × d]` | 2D matrix |
| 5 | 3.i.a.q : produce Q | matmul (x_normed · W_Q) | x_normed | `[N_tok × d]` | {attn_i.W_Q : d × d} | Q flat | `[N_tok × d]` | 2D matrix |
| 6 | 3.i.a.q_split : split Q into heads | reshape (split heads) | Q flat | `[N_tok × d]` | — | Q per-head | `[n_h × N_tok × d_h]` | 3D tensor |
| 7 | 3.i.a.k : produce K | matmul (x_normed · W_K) | x_normed | `[N_tok × d]` | {attn_i.W_K : d × (n_kv·d_h)} | K flat | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 8 | 3.i.a.k_split : split K into heads | reshape (split heads) | K flat | `[N_tok × (n_kv·d_h)]` | — | K per-head | `[n_kv × N_tok × d_h]` | 3D tensor |
| 9 | 3.i.a.v : produce V | matmul (x_normed · W_V) | x_normed | `[N_tok × d]` | {attn_i.W_V : d × (n_kv·d_h)} | V flat | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 10 | 3.i.a.v_split : split V into heads | reshape (split heads) | V flat | `[N_tok × (n_kv·d_h)]` | — | V per-head | `[n_kv × N_tok × d_h]` | 3D tensor |
| 11 | 3.i.a.rope_q : rotate Q | elementwise rotate (RoPE) | Q per-head | `[n_h × N_tok × d_h]` | {RoPE : max_pos × d_h} | Q_rot | `[n_h × N_tok × d_h]` | 3D tensor |
| 12 | 3.i.a.rope_k : rotate K | elementwise rotate (RoPE) | K per-head | `[n_kv × N_tok × d_h]` | {RoPE : max_pos × d_h} | K_rot | `[n_kv × N_tok × d_h]` | 3D tensor |
| 13 | 3.i.a.op.scores : Q·Kᵀ | batched matmul (Q · Kᵀ) | Q_rot, K_rot | `[n_h × N_tok × d_h], [n_kv × N_tok × d_h]` | — | raw scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 14 | 3.i.a.op.scale : scale by 1/√d_h | scalar divide (/ √d_h) | raw scores | `[n_h × N_tok × N_tok]` | — | scaled scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 15 | 3.i.a.op.mask : causal mask | elementwise add (−∞ to future) | scaled scores | `[n_h × N_tok × N_tok]` | {causal_mask : N_tok × N_tok} | masked scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 16 | 3.i.a.op.smax : softmax (per row) | softmax (over key axis) | masked scores | `[n_h × N_tok × N_tok]` | — | attn weights | `[n_h × N_tok × N_tok]` | 3D tensor |
| 17 | 3.i.a.op.weighted : weighted sum with V | batched matmul (weights · V) | attn weights, V per-head | `[n_h × N_tok × N_tok], [n_kv × N_tok × d_h]` | — | per-head results | `[n_h × N_tok × d_h]` | 3D tensor |
| 18 | 3.i.a.concat : concat heads | reshape (concat heads) | per-head results | `[n_h × N_tok × d_h]` | — | merged result | `[N_tok × d]` | 2D matrix |
| 19 | 3.i.a.proj : output projection | matmul (· W_O) | merged result | `[N_tok × d]` | {attn_i.W_O : d × d} | attn delta | `[N_tok × d]` | 2D matrix |
| 20 | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 21 | 3.i.b.save : save y | identity (hold copy of y) | y (vectors) | `[N_tok × d]` | — | y (held) | `[N_tok × d]` | 2D matrix |
| 22 | 3.i.b.norm : pre-FFN norm | RMSNorm | y | `[N_tok × d]` | {ffn_i.norm : d} | y_normed | `[N_tok × d]` | 2D matrix |
| **23** | 3.i.b.up : up projection | matmul (y_normed · W_up) | y_normed | `[N_tok × d]` | {ffn_i.W_up : d × d_ff} | up | `[N_tok × d_ff]` | 2D matrix |
| 24 | 3.i.b.inner_rest : rest of inner | opaque | up | `[N_tok × d_ff]` | {ffn_i inner-rest internals} | hidden | `[N_tok × d_ff]` | 2D matrix |
| 25 | 3.i.b.down : down projection | matmul (hidden · W_down) | hidden | `[N_tok × d_ff]` | {ffn_i.W_down : d_ff × d} | ffn delta | `[N_tok × d]` | 2D matrix |
| 26 | 3.i.b.add : add residual | elementwise add (residual) | y (held), ffn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 27 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 28 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 29 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 30 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 31 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 32 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

### Level 45 — parallel gate projection (opaque) <a id="L45"></a>

**New:** A second projection of y_normed produces `gate`, parallel to up.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a.save : save x | identity (hold copy of x) | x | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| 4 | 3.i.a.norm : pre-attn norm | RMSNorm | x | `[N_tok × d]` | {attn_i.norm : d} | x_normed | `[N_tok × d]` | 2D matrix |
| 5 | 3.i.a.q : produce Q | matmul (x_normed · W_Q) | x_normed | `[N_tok × d]` | {attn_i.W_Q : d × d} | Q flat | `[N_tok × d]` | 2D matrix |
| 6 | 3.i.a.q_split : split Q into heads | reshape (split heads) | Q flat | `[N_tok × d]` | — | Q per-head | `[n_h × N_tok × d_h]` | 3D tensor |
| 7 | 3.i.a.k : produce K | matmul (x_normed · W_K) | x_normed | `[N_tok × d]` | {attn_i.W_K : d × (n_kv·d_h)} | K flat | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 8 | 3.i.a.k_split : split K into heads | reshape (split heads) | K flat | `[N_tok × (n_kv·d_h)]` | — | K per-head | `[n_kv × N_tok × d_h]` | 3D tensor |
| 9 | 3.i.a.v : produce V | matmul (x_normed · W_V) | x_normed | `[N_tok × d]` | {attn_i.W_V : d × (n_kv·d_h)} | V flat | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 10 | 3.i.a.v_split : split V into heads | reshape (split heads) | V flat | `[N_tok × (n_kv·d_h)]` | — | V per-head | `[n_kv × N_tok × d_h]` | 3D tensor |
| 11 | 3.i.a.rope_q : rotate Q | elementwise rotate (RoPE) | Q per-head | `[n_h × N_tok × d_h]` | {RoPE : max_pos × d_h} | Q_rot | `[n_h × N_tok × d_h]` | 3D tensor |
| 12 | 3.i.a.rope_k : rotate K | elementwise rotate (RoPE) | K per-head | `[n_kv × N_tok × d_h]` | {RoPE : max_pos × d_h} | K_rot | `[n_kv × N_tok × d_h]` | 3D tensor |
| 13 | 3.i.a.op.scores : Q·Kᵀ | batched matmul (Q · Kᵀ) | Q_rot, K_rot | `[n_h × N_tok × d_h], [n_kv × N_tok × d_h]` | — | raw scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 14 | 3.i.a.op.scale : scale by 1/√d_h | scalar divide (/ √d_h) | raw scores | `[n_h × N_tok × N_tok]` | — | scaled scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 15 | 3.i.a.op.mask : causal mask | elementwise add (−∞ to future) | scaled scores | `[n_h × N_tok × N_tok]` | {causal_mask : N_tok × N_tok} | masked scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 16 | 3.i.a.op.smax : softmax (per row) | softmax (over key axis) | masked scores | `[n_h × N_tok × N_tok]` | — | attn weights | `[n_h × N_tok × N_tok]` | 3D tensor |
| 17 | 3.i.a.op.weighted : weighted sum with V | batched matmul (weights · V) | attn weights, V per-head | `[n_h × N_tok × N_tok], [n_kv × N_tok × d_h]` | — | per-head results | `[n_h × N_tok × d_h]` | 3D tensor |
| 18 | 3.i.a.concat : concat heads | reshape (concat heads) | per-head results | `[n_h × N_tok × d_h]` | — | merged result | `[N_tok × d]` | 2D matrix |
| 19 | 3.i.a.proj : output projection | matmul (· W_O) | merged result | `[N_tok × d]` | {attn_i.W_O : d × d} | attn delta | `[N_tok × d]` | 2D matrix |
| 20 | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 21 | 3.i.b.save : save y | identity (hold copy of y) | y (vectors) | `[N_tok × d]` | — | y (held) | `[N_tok × d]` | 2D matrix |
| 22 | 3.i.b.norm : pre-FFN norm | RMSNorm | y | `[N_tok × d]` | {ffn_i.norm : d} | y_normed | `[N_tok × d]` | 2D matrix |
| **23** | 3.i.b.gate : gate projection | matmul (y_normed · W_gate) | y_normed | `[N_tok × d]` | {W_gate internals} | gate | `[N_tok × d_ff]` | 2D matrix |
| 24 | 3.i.b.up : up projection | matmul (y_normed · W_up) | y_normed | `[N_tok × d]` | {ffn_i.W_up : d × d_ff} | up | `[N_tok × d_ff]` | 2D matrix |
| **25** | 3.i.b.combine : combine gate, up | opaque (combine gate & up) | gate, up | `[N_tok × d_ff], [N_tok × d_ff]` | — | hidden | `[N_tok × d_ff]` | 2D matrix |
| 26 | 3.i.b.down : down projection | matmul (hidden · W_down) | hidden | `[N_tok × d_ff]` | {ffn_i.W_down : d_ff × d} | ffn delta | `[N_tok × d]` | 2D matrix |
| 27 | 3.i.b.add : add residual | elementwise add (residual) | y (held), ffn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 28 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 29 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 30 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 31 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 32 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 33 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

### Level 46 — expose `{W_gate}` <a id="L46"></a>

**New:** `ffn_i.W_gate` has shape `[d × d_ff]`.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a.save : save x | identity (hold copy of x) | x | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| 4 | 3.i.a.norm : pre-attn norm | RMSNorm | x | `[N_tok × d]` | {attn_i.norm : d} | x_normed | `[N_tok × d]` | 2D matrix |
| 5 | 3.i.a.q : produce Q | matmul (x_normed · W_Q) | x_normed | `[N_tok × d]` | {attn_i.W_Q : d × d} | Q flat | `[N_tok × d]` | 2D matrix |
| 6 | 3.i.a.q_split : split Q into heads | reshape (split heads) | Q flat | `[N_tok × d]` | — | Q per-head | `[n_h × N_tok × d_h]` | 3D tensor |
| 7 | 3.i.a.k : produce K | matmul (x_normed · W_K) | x_normed | `[N_tok × d]` | {attn_i.W_K : d × (n_kv·d_h)} | K flat | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 8 | 3.i.a.k_split : split K into heads | reshape (split heads) | K flat | `[N_tok × (n_kv·d_h)]` | — | K per-head | `[n_kv × N_tok × d_h]` | 3D tensor |
| 9 | 3.i.a.v : produce V | matmul (x_normed · W_V) | x_normed | `[N_tok × d]` | {attn_i.W_V : d × (n_kv·d_h)} | V flat | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 10 | 3.i.a.v_split : split V into heads | reshape (split heads) | V flat | `[N_tok × (n_kv·d_h)]` | — | V per-head | `[n_kv × N_tok × d_h]` | 3D tensor |
| 11 | 3.i.a.rope_q : rotate Q | elementwise rotate (RoPE) | Q per-head | `[n_h × N_tok × d_h]` | {RoPE : max_pos × d_h} | Q_rot | `[n_h × N_tok × d_h]` | 3D tensor |
| 12 | 3.i.a.rope_k : rotate K | elementwise rotate (RoPE) | K per-head | `[n_kv × N_tok × d_h]` | {RoPE : max_pos × d_h} | K_rot | `[n_kv × N_tok × d_h]` | 3D tensor |
| 13 | 3.i.a.op.scores : Q·Kᵀ | batched matmul (Q · Kᵀ) | Q_rot, K_rot | `[n_h × N_tok × d_h], [n_kv × N_tok × d_h]` | — | raw scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 14 | 3.i.a.op.scale : scale by 1/√d_h | scalar divide (/ √d_h) | raw scores | `[n_h × N_tok × N_tok]` | — | scaled scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 15 | 3.i.a.op.mask : causal mask | elementwise add (−∞ to future) | scaled scores | `[n_h × N_tok × N_tok]` | {causal_mask : N_tok × N_tok} | masked scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 16 | 3.i.a.op.smax : softmax (per row) | softmax (over key axis) | masked scores | `[n_h × N_tok × N_tok]` | — | attn weights | `[n_h × N_tok × N_tok]` | 3D tensor |
| 17 | 3.i.a.op.weighted : weighted sum with V | batched matmul (weights · V) | attn weights, V per-head | `[n_h × N_tok × N_tok], [n_kv × N_tok × d_h]` | — | per-head results | `[n_h × N_tok × d_h]` | 3D tensor |
| 18 | 3.i.a.concat : concat heads | reshape (concat heads) | per-head results | `[n_h × N_tok × d_h]` | — | merged result | `[N_tok × d]` | 2D matrix |
| 19 | 3.i.a.proj : output projection | matmul (· W_O) | merged result | `[N_tok × d]` | {attn_i.W_O : d × d} | attn delta | `[N_tok × d]` | 2D matrix |
| 20 | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 21 | 3.i.b.save : save y | identity (hold copy of y) | y (vectors) | `[N_tok × d]` | — | y (held) | `[N_tok × d]` | 2D matrix |
| 22 | 3.i.b.norm : pre-FFN norm | RMSNorm | y | `[N_tok × d]` | {ffn_i.norm : d} | y_normed | `[N_tok × d]` | 2D matrix |
| **23** | 3.i.b.gate : gate projection | matmul (y_normed · W_gate) | y_normed | `[N_tok × d]` | {ffn_i.W_gate : d × d_ff} | gate | `[N_tok × d_ff]` | 2D matrix |
| 24 | 3.i.b.up : up projection | matmul (y_normed · W_up) | y_normed | `[N_tok × d]` | {ffn_i.W_up : d × d_ff} | up | `[N_tok × d_ff]` | 2D matrix |
| 25 | 3.i.b.combine : combine gate, up | opaque (combine gate & up) | gate, up | `[N_tok × d_ff], [N_tok × d_ff]` | — | hidden | `[N_tok × d_ff]` | 2D matrix |
| 26 | 3.i.b.down : down projection | matmul (hidden · W_down) | hidden | `[N_tok × d_ff]` | {ffn_i.W_down : d_ff × d} | ffn delta | `[N_tok × d]` | 2D matrix |
| 27 | 3.i.b.add : add residual | elementwise add (residual) | y (held), ffn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 28 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 29 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 30 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 31 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 32 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 33 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

### Level 47 — SiLU on gate <a id="L47"></a>

**New:** `SiLU(x) = x · sigmoid(x)`, applied elementwise to gate only. Up is not activated.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a.save : save x | identity (hold copy of x) | x | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| 4 | 3.i.a.norm : pre-attn norm | RMSNorm | x | `[N_tok × d]` | {attn_i.norm : d} | x_normed | `[N_tok × d]` | 2D matrix |
| 5 | 3.i.a.q : produce Q | matmul (x_normed · W_Q) | x_normed | `[N_tok × d]` | {attn_i.W_Q : d × d} | Q flat | `[N_tok × d]` | 2D matrix |
| 6 | 3.i.a.q_split : split Q into heads | reshape (split heads) | Q flat | `[N_tok × d]` | — | Q per-head | `[n_h × N_tok × d_h]` | 3D tensor |
| 7 | 3.i.a.k : produce K | matmul (x_normed · W_K) | x_normed | `[N_tok × d]` | {attn_i.W_K : d × (n_kv·d_h)} | K flat | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 8 | 3.i.a.k_split : split K into heads | reshape (split heads) | K flat | `[N_tok × (n_kv·d_h)]` | — | K per-head | `[n_kv × N_tok × d_h]` | 3D tensor |
| 9 | 3.i.a.v : produce V | matmul (x_normed · W_V) | x_normed | `[N_tok × d]` | {attn_i.W_V : d × (n_kv·d_h)} | V flat | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 10 | 3.i.a.v_split : split V into heads | reshape (split heads) | V flat | `[N_tok × (n_kv·d_h)]` | — | V per-head | `[n_kv × N_tok × d_h]` | 3D tensor |
| 11 | 3.i.a.rope_q : rotate Q | elementwise rotate (RoPE) | Q per-head | `[n_h × N_tok × d_h]` | {RoPE : max_pos × d_h} | Q_rot | `[n_h × N_tok × d_h]` | 3D tensor |
| 12 | 3.i.a.rope_k : rotate K | elementwise rotate (RoPE) | K per-head | `[n_kv × N_tok × d_h]` | {RoPE : max_pos × d_h} | K_rot | `[n_kv × N_tok × d_h]` | 3D tensor |
| 13 | 3.i.a.op.scores : Q·Kᵀ | batched matmul (Q · Kᵀ) | Q_rot, K_rot | `[n_h × N_tok × d_h], [n_kv × N_tok × d_h]` | — | raw scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 14 | 3.i.a.op.scale : scale by 1/√d_h | scalar divide (/ √d_h) | raw scores | `[n_h × N_tok × N_tok]` | — | scaled scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 15 | 3.i.a.op.mask : causal mask | elementwise add (−∞ to future) | scaled scores | `[n_h × N_tok × N_tok]` | {causal_mask : N_tok × N_tok} | masked scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 16 | 3.i.a.op.smax : softmax (per row) | softmax (over key axis) | masked scores | `[n_h × N_tok × N_tok]` | — | attn weights | `[n_h × N_tok × N_tok]` | 3D tensor |
| 17 | 3.i.a.op.weighted : weighted sum with V | batched matmul (weights · V) | attn weights, V per-head | `[n_h × N_tok × N_tok], [n_kv × N_tok × d_h]` | — | per-head results | `[n_h × N_tok × d_h]` | 3D tensor |
| 18 | 3.i.a.concat : concat heads | reshape (concat heads) | per-head results | `[n_h × N_tok × d_h]` | — | merged result | `[N_tok × d]` | 2D matrix |
| 19 | 3.i.a.proj : output projection | matmul (· W_O) | merged result | `[N_tok × d]` | {attn_i.W_O : d × d} | attn delta | `[N_tok × d]` | 2D matrix |
| 20 | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 21 | 3.i.b.save : save y | identity (hold copy of y) | y (vectors) | `[N_tok × d]` | — | y (held) | `[N_tok × d]` | 2D matrix |
| 22 | 3.i.b.norm : pre-FFN norm | RMSNorm | y | `[N_tok × d]` | {ffn_i.norm : d} | y_normed | `[N_tok × d]` | 2D matrix |
| 23 | 3.i.b.gate : gate projection | matmul (y_normed · W_gate) | y_normed | `[N_tok × d]` | {ffn_i.W_gate : d × d_ff} | gate | `[N_tok × d_ff]` | 2D matrix |
| **24** | 3.i.b.silu : SiLU | elementwise SiLU | gate | `[N_tok × d_ff]` | — | gate_act | `[N_tok × d_ff]` | 2D matrix |
| 25 | 3.i.b.up : up projection | matmul (y_normed · W_up) | y_normed | `[N_tok × d]` | {ffn_i.W_up : d × d_ff} | up | `[N_tok × d_ff]` | 2D matrix |
| **26** | 3.i.b.combine : combine gate_act, up | opaque (combine gate & up) | gate_act, up | `[N_tok × d_ff], [N_tok × d_ff]` | — | hidden | `[N_tok × d_ff]` | 2D matrix |
| 27 | 3.i.b.down : down projection | matmul (hidden · W_down) | hidden | `[N_tok × d_ff]` | {ffn_i.W_down : d_ff × d} | ffn delta | `[N_tok × d]` | 2D matrix |
| 28 | 3.i.b.add : add residual | elementwise add (residual) | y (held), ffn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 29 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 30 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 31 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 32 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 33 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 34 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

### Level 48 — combine = elementwise multiply <a id="L48"></a>

**New:** `hidden = gate_act ⊙ up`. SwiGLU gating: gate modulates up dimensions.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a.save : save x | identity (hold copy of x) | x | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| 4 | 3.i.a.norm : pre-attn norm | RMSNorm | x | `[N_tok × d]` | {attn_i.norm : d} | x_normed | `[N_tok × d]` | 2D matrix |
| 5 | 3.i.a.q : produce Q | matmul (x_normed · W_Q) | x_normed | `[N_tok × d]` | {attn_i.W_Q : d × d} | Q flat | `[N_tok × d]` | 2D matrix |
| 6 | 3.i.a.q_split : split Q into heads | reshape (split heads) | Q flat | `[N_tok × d]` | — | Q per-head | `[n_h × N_tok × d_h]` | 3D tensor |
| 7 | 3.i.a.k : produce K | matmul (x_normed · W_K) | x_normed | `[N_tok × d]` | {attn_i.W_K : d × (n_kv·d_h)} | K flat | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 8 | 3.i.a.k_split : split K into heads | reshape (split heads) | K flat | `[N_tok × (n_kv·d_h)]` | — | K per-head | `[n_kv × N_tok × d_h]` | 3D tensor |
| 9 | 3.i.a.v : produce V | matmul (x_normed · W_V) | x_normed | `[N_tok × d]` | {attn_i.W_V : d × (n_kv·d_h)} | V flat | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 10 | 3.i.a.v_split : split V into heads | reshape (split heads) | V flat | `[N_tok × (n_kv·d_h)]` | — | V per-head | `[n_kv × N_tok × d_h]` | 3D tensor |
| 11 | 3.i.a.rope_q : rotate Q | elementwise rotate (RoPE) | Q per-head | `[n_h × N_tok × d_h]` | {RoPE : max_pos × d_h} | Q_rot | `[n_h × N_tok × d_h]` | 3D tensor |
| 12 | 3.i.a.rope_k : rotate K | elementwise rotate (RoPE) | K per-head | `[n_kv × N_tok × d_h]` | {RoPE : max_pos × d_h} | K_rot | `[n_kv × N_tok × d_h]` | 3D tensor |
| 13 | 3.i.a.op.scores : Q·Kᵀ | batched matmul (Q · Kᵀ) | Q_rot, K_rot | `[n_h × N_tok × d_h], [n_kv × N_tok × d_h]` | — | raw scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 14 | 3.i.a.op.scale : scale by 1/√d_h | scalar divide (/ √d_h) | raw scores | `[n_h × N_tok × N_tok]` | — | scaled scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 15 | 3.i.a.op.mask : causal mask | elementwise add (−∞ to future) | scaled scores | `[n_h × N_tok × N_tok]` | {causal_mask : N_tok × N_tok} | masked scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 16 | 3.i.a.op.smax : softmax (per row) | softmax (over key axis) | masked scores | `[n_h × N_tok × N_tok]` | — | attn weights | `[n_h × N_tok × N_tok]` | 3D tensor |
| 17 | 3.i.a.op.weighted : weighted sum with V | batched matmul (weights · V) | attn weights, V per-head | `[n_h × N_tok × N_tok], [n_kv × N_tok × d_h]` | — | per-head results | `[n_h × N_tok × d_h]` | 3D tensor |
| 18 | 3.i.a.concat : concat heads | reshape (concat heads) | per-head results | `[n_h × N_tok × d_h]` | — | merged result | `[N_tok × d]` | 2D matrix |
| 19 | 3.i.a.proj : output projection | matmul (· W_O) | merged result | `[N_tok × d]` | {attn_i.W_O : d × d} | attn delta | `[N_tok × d]` | 2D matrix |
| 20 | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 21 | 3.i.b.save : save y | identity (hold copy of y) | y (vectors) | `[N_tok × d]` | — | y (held) | `[N_tok × d]` | 2D matrix |
| 22 | 3.i.b.norm : pre-FFN norm | RMSNorm | y | `[N_tok × d]` | {ffn_i.norm : d} | y_normed | `[N_tok × d]` | 2D matrix |
| 23 | 3.i.b.gate : gate projection | matmul (y_normed · W_gate) | y_normed | `[N_tok × d]` | {ffn_i.W_gate : d × d_ff} | gate | `[N_tok × d_ff]` | 2D matrix |
| 24 | 3.i.b.silu : SiLU | elementwise SiLU | gate | `[N_tok × d_ff]` | — | gate_act | `[N_tok × d_ff]` | 2D matrix |
| 25 | 3.i.b.up : up projection | matmul (y_normed · W_up) | y_normed | `[N_tok × d]` | {ffn_i.W_up : d × d_ff} | up | `[N_tok × d_ff]` | 2D matrix |
| **26** | 3.i.b.combine : elementwise multiply | elementwise multiply (⊙) | gate_act, up | `[N_tok × d_ff], [N_tok × d_ff]` | — | hidden | `[N_tok × d_ff]` | 2D matrix |
| 27 | 3.i.b.down : down projection | matmul (hidden · W_down) | hidden | `[N_tok × d_ff]` | {ffn_i.W_down : d_ff × d} | ffn delta | `[N_tok × d]` | 2D matrix |
| 28 | 3.i.b.add : add residual | elementwise add (residual) | y (held), ffn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 29 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 30 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 31 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 32 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 33 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 34 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*

## Phase 9 — Complete reference

### Level 49 — complete reference <a id="L49"></a>

**New:** Full pipeline visible. No further concepts to add for one-inference-one-token. Backbone shape is `[N_tok × d]`. Attention scores `[n_h × N_tok × N_tok]` are the quadratic-in-sequence-length intermediate. FFN widens to `[N_tok × d_ff]`. LM head is the only step that changes per-token dim.

| # | Step | Op | In | In shape | Static | Out | Out shape | Dim out |
|---|---|---|---|---|---|---|---|---|
| 1 | tokenize | BPE segment + ID map | text | `—` | {vocab} | token IDs | `[N_tok]` | 1D vector |
| 2 | embed | row lookup in {E} | token IDs | `[N_tok]` | {E : vocab_size × d} | vectors | `[N_tok × d]` | 2D matrix |
| 3 | 3.i.a.save : save x | identity (hold copy of x) | x | `[N_tok × d]` | — | x (held) | `[N_tok × d]` | 2D matrix |
| 4 | 3.i.a.norm : pre-attn norm | RMSNorm | x | `[N_tok × d]` | {attn_i.norm : d} | x_normed | `[N_tok × d]` | 2D matrix |
| 5 | 3.i.a.q : produce Q | matmul (x_normed · W_Q) | x_normed | `[N_tok × d]` | {attn_i.W_Q : d × d} | Q flat | `[N_tok × d]` | 2D matrix |
| 6 | 3.i.a.q_split : split Q into heads | reshape (split heads) | Q flat | `[N_tok × d]` | — | Q per-head | `[n_h × N_tok × d_h]` | 3D tensor |
| 7 | 3.i.a.k : produce K | matmul (x_normed · W_K) | x_normed | `[N_tok × d]` | {attn_i.W_K : d × (n_kv·d_h)} | K flat | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 8 | 3.i.a.k_split : split K into heads | reshape (split heads) | K flat | `[N_tok × (n_kv·d_h)]` | — | K per-head | `[n_kv × N_tok × d_h]` | 3D tensor |
| 9 | 3.i.a.v : produce V | matmul (x_normed · W_V) | x_normed | `[N_tok × d]` | {attn_i.W_V : d × (n_kv·d_h)} | V flat | `[N_tok × (n_kv·d_h)]` | 2D matrix |
| 10 | 3.i.a.v_split : split V into heads | reshape (split heads) | V flat | `[N_tok × (n_kv·d_h)]` | — | V per-head | `[n_kv × N_tok × d_h]` | 3D tensor |
| 11 | 3.i.a.rope_q : rotate Q | elementwise rotate (RoPE) | Q per-head | `[n_h × N_tok × d_h]` | {RoPE : max_pos × d_h} | Q_rot | `[n_h × N_tok × d_h]` | 3D tensor |
| 12 | 3.i.a.rope_k : rotate K | elementwise rotate (RoPE) | K per-head | `[n_kv × N_tok × d_h]` | {RoPE : max_pos × d_h} | K_rot | `[n_kv × N_tok × d_h]` | 3D tensor |
| 13 | 3.i.a.op.scores : Q·Kᵀ | batched matmul (Q · Kᵀ) | Q_rot, K_rot | `[n_h × N_tok × d_h], [n_kv × N_tok × d_h]` | — | raw scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 14 | 3.i.a.op.scale : scale by 1/√d_h | scalar divide (/ √d_h) | raw scores | `[n_h × N_tok × N_tok]` | — | scaled scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 15 | 3.i.a.op.mask : causal mask | elementwise add (−∞ to future) | scaled scores | `[n_h × N_tok × N_tok]` | {causal_mask : N_tok × N_tok} | masked scores | `[n_h × N_tok × N_tok]` | 3D tensor |
| 16 | 3.i.a.op.smax : softmax (per row) | softmax (over key axis) | masked scores | `[n_h × N_tok × N_tok]` | — | attn weights | `[n_h × N_tok × N_tok]` | 3D tensor |
| 17 | 3.i.a.op.weighted : weighted sum with V | batched matmul (weights · V) | attn weights, V per-head | `[n_h × N_tok × N_tok], [n_kv × N_tok × d_h]` | — | per-head results | `[n_h × N_tok × d_h]` | 3D tensor |
| 18 | 3.i.a.concat : concat heads | reshape (concat heads) | per-head results | `[n_h × N_tok × d_h]` | — | merged result | `[N_tok × d]` | 2D matrix |
| 19 | 3.i.a.proj : output projection | matmul (· W_O) | merged result | `[N_tok × d]` | {attn_i.W_O : d × d} | attn delta | `[N_tok × d]` | 2D matrix |
| 20 | 3.i.a.add : add residual | elementwise add (residual) | x (held), attn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 21 | 3.i.b.save : save y | identity (hold copy of y) | y (vectors) | `[N_tok × d]` | — | y (held) | `[N_tok × d]` | 2D matrix |
| 22 | 3.i.b.norm : pre-FFN norm | RMSNorm | y | `[N_tok × d]` | {ffn_i.norm : d} | y_normed | `[N_tok × d]` | 2D matrix |
| 23 | 3.i.b.gate : gate projection | matmul (y_normed · W_gate) | y_normed | `[N_tok × d]` | {ffn_i.W_gate : d × d_ff} | gate | `[N_tok × d_ff]` | 2D matrix |
| 24 | 3.i.b.silu : SiLU | elementwise SiLU | gate | `[N_tok × d_ff]` | — | gate_act | `[N_tok × d_ff]` | 2D matrix |
| 25 | 3.i.b.up : up projection | matmul (y_normed · W_up) | y_normed | `[N_tok × d]` | {ffn_i.W_up : d × d_ff} | up | `[N_tok × d_ff]` | 2D matrix |
| 26 | 3.i.b.combine : elementwise multiply | elementwise multiply (⊙) | gate_act, up | `[N_tok × d_ff], [N_tok × d_ff]` | — | hidden | `[N_tok × d_ff]` | 2D matrix |
| 27 | 3.i.b.down : down projection | matmul (hidden · W_down) | hidden | `[N_tok × d_ff]` | {ffn_i.W_down : d_ff × d} | ffn delta | `[N_tok × d]` | 2D matrix |
| 28 | 3.i.b.add : add residual | elementwise add (residual) | y (held), ffn delta | `[N_tok × d], [N_tok × d]` | — | vectors | `[N_tok × d]` | 2D matrix |
| 29 | final norm | RMSNorm | vectors | `[N_tok × d]` | {final_norm : d} | vectors | `[N_tok × d]` | 2D matrix |
| 30 | LM head | matmul (vectors · {H}) | vectors | `[N_tok × d]` | {H : d × vocab_size} | logits (all positions) | `[N_tok × vocab_size]` | 2D matrix |
| 31 | take last row | slice (last position) | logits (all positions) | `[N_tok × vocab_size]` | — | logits (last position) | `[vocab_size]` | 1D vector |
| 32 | softmax | softmax over vocab | logits (last position) | `[vocab_size]` | — | probabilities | `[vocab_size]` | 1D vector |
| 33 | pick (argmax) | argmax | probabilities | `[vocab_size]` | — | 1 token ID | `scalar` | scalar |
| 34 | decode | inverse vocab lookup | 1 token ID | `scalar` | {vocab} | text piece | `—` | — |

*Block rows `3.i.*` repeat for each block `i = 1..N`.*
