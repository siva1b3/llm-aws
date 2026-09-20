# Transformer LLM Internals — Learning Track v3

> Standalone learning track for understanding Llama 3.2 1B Instruct internals by inspecting a loaded model on EC2. This file does not depend on any prior file. It supersedes v2 by completing the back half of the track — phases that open attention and FFN.

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

| Phase | Name | Status in this file |
|---|---|---|
| 1 | Foundation: load model and map sizes | fleshed (in v2) |
| 2 | Outer pipeline: replace `generate()` | fleshed (in v2) |
| 3 | Open the embedding | fleshed (in v2) |
| 4 | Open the LM head and final-position decode | fleshed (in v2) |
| 5 | Open the final RMSNorm | fleshed (in v2) |
| 6 | Open the block stack | **fleshed below** |
| 7 | Open the block — four children, residual + pre-norm pattern | **fleshed below** |
| 8 | Open attention projections — Q, K, V, O | **fleshed below** |
| 9 | Open single-head attention (a correct special case) | **fleshed below** |
| 10 | Generalize to multi-head and GQA | detailed stub |
| 11 | Apply RoPE — attention matches the live model exactly | detailed stub |
| 12 | Open the FFN — projections, SwiGLU | detailed stub |
| 13 | Stitch the manual block and the full manual inference | detailed stub |
| 14 | Consolidation | detailed stub |

**Note on file split.** v2 contains Phases 1–5 in full. v3 (this file) contains Phases 6–9 in full and detailed stubs for 10–14. The two files together form the complete track. The instruction sections (Section 1–5 above) are duplicated across both files so each is standalone-readable; if conflict ever arises, v3 is authoritative.

There is no separate "stitch the outer pipeline" phase. Outer-pipeline pieces (embedding, LM head, final norm) were stitched into helpers within the phases that opened them (Phases 3–5 in v2); final composition into `manual_inference` lives in Phase 13.

There is no "intentionally wrong attention" phase. Phase 9 builds single-head attention as a correct special case (forcing `n_h = 1` semantics by working with a single head's `Q`/`K`/`V` slice, no RoPE) that matches a clearly-defined reference; Phase 10 generalizes the head count; Phase 11 adds RoPE. At each phase boundary, the manual version matches a precise live reference.

---

## Section 5 — Step template

Every step follows this structure. Six blocks. The order is fixed.

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
- **Forward-hook step** (capturing an intermediate from a live submodule). Code includes the hook registration and one model call; the hook handle is removed at the end of the step.

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

---

# Phase 6 — Open the block stack

## Phase 6 — Open the block stack

**Goal:** Replace the implicit "16 blocks run in sequence" with an explicit loop over `model.model.layers`. Each individual block is still opaque (its internals open in Phase 7); this phase opens only the stack structure.

**Milestone:** A manual loop calling `layers[i](x)[0]` for `i in 0..N-1` reproduces the model's pre-final-norm hidden state (`hidden_states[-1]` from `output_hidden_states=True`) within `atol=1e-4`.

**Builds on:** Phase 5 milestone — embedding, LM head, and final RMSNorm are open and stitched.

**Helpers stitched in this phase:** `manual_block_stack(x, layers)`.

---

### Level 6.1 — Inspect the block stack as a uniform structure

**Concept introduced:** `model.model.layers` is a `ModuleList` of `N=16` identical-type, different-weights blocks. The blocks are independent — each has its own complete set of weights — but every block has the same module class and the same call signature.

**Black boxes at start of level:** everything inside each block (opens in Phase 7).
**Black boxes opened by this level:** the list-of-blocks structure; the call convention for an individual block.
**Black boxes still deferred at end of level:** all four children of each block (opens in Phase 7).

---

#### Step 6.1.1 — Bind `layers`

**Delta:** Pull `model.model.layers` into a clean variable.

**Predict:** `layers` is a `torch.nn.ModuleList` of length `N=16`. Every element has type `LlamaDecoderLayer`.

**Code:**
```python
layers = model.model.layers
```

**Observe:**
```python
print("layers type:", type(layers).__name__)
print("len(layers):", len(layers))
print("first element type:", type(layers[0]).__name__)
```

**Verify:**
```python
assert isinstance(layers, torch.nn.ModuleList)
assert len(layers) == N
assert all(type(L).__name__ == "LlamaDecoderLayer" for L in layers)
```

**Reflect (one line):** All 16 blocks share a type — same forward code, different weights — which is what makes "understand one block, understand all" a valid claim for the rest of the track.

---

#### Step 6.1.2 — Confirm weight independence across blocks

**Delta:** Verify that two different blocks do not share parameters by identity (each block has its own tensor objects).

**Predict:** For any pair of blocks `i != j`, no parameter tensor is shared by identity (`is`). The check passes against a small sample (blocks 0 and 1).

**Code:**
```python
params_0 = dict(layers[0].named_parameters())
params_1 = dict(layers[1].named_parameters())
shared_by_identity = [k for k in params_0 if params_0[k] is params_1.get(k)]
```

**Observe:**
```python
print("param names per block:", len(params_0))
print("any shared by identity:", shared_by_identity)
```

**Verify:**
```python
assert shared_by_identity == []
```

**Reflect (one line):** Independent weights are what makes depth useful — if blocks shared parameters, 16 blocks would compute the same function as 1 block applied 16 times, losing expressive power.

---

#### Step 6.1.3 — Inspect one block's call signature

**Delta:** Run a single block and examine its return value's structure to learn the calling convention.

**Predict:** `layers[0](x_embed)` returns a tuple. The first element is the new hidden state of shape `[1, N_tok, d]`. There may be additional elements (None for non-cached inference, or empty in some HuggingFace versions).

**Code:**
```python
x_embed = manual_embedding(input_ids, model)
block0_out = layers[0](x_embed)
```

**Observe:**
```python
print("return type:", type(block0_out).__name__)
print("len:", len(block0_out) if hasattr(block0_out, "__len__") else "n/a")
print("first element shape:", block0_out[0].shape)
print("first element dtype:", block0_out[0].dtype)
```

**Verify:**
```python
assert tuple(block0_out[0].shape) == (1, N_tok, d)
assert block0_out[0].dtype == torch.bfloat16
```

**Reflect (one line):** A block returns a tuple even when only the hidden state is wanted because HuggingFace's interface accommodates optional outputs (attention weights, present key-value cache) — indexing `[0]` pulls only the hidden state.

---

### Level 6.2 — Manual loop reproduces the block stack output

**Concept introduced:** The block stack is exactly a `for` loop. The output of block `i` is the input of block `i+1`. Threading the hidden state through this loop produces the pre-final-norm tensor.

**Black boxes at start of level:** individual block internals.
**Black boxes opened by this level:** the *stack* operation as a loop.
**Black boxes still deferred at end of level:** individual block internals (opens in Phase 7).

---

#### Step 6.2.1 — Capture the reference target

**Delta:** Re-run the model with `output_hidden_states=True` to capture the per-block reference tensors, so each iteration of the manual loop can be checked against the model's own intermediate.

**Predict:** `hidden_states` is a tuple of length `N + 1 = 17`. `hidden_states[0]` is the embedding output (input to block 0). `hidden_states[i+1]` is the output of block `i` for `i in 0..N-1`. All shapes are `[1, N_tok, d]`.

**Code:**
```python
outputs_hs = model(input_ids=input_ids, output_hidden_states=True)
hidden_states = outputs_hs.hidden_states
```

**Observe:**
```python
print("len(hidden_states):", len(hidden_states))
print("hidden_states[0].shape:", hidden_states[0].shape)
print("hidden_states[-1].shape:", hidden_states[-1].shape)
print("hidden_states[0] equals manual_embedding(input_ids):",
      torch.equal(hidden_states[0], manual_embedding(input_ids, model)))
```

**Verify:**
```python
assert len(hidden_states) == N + 1
for hs in hidden_states:
    assert tuple(hs.shape) == (1, N_tok, d)
assert torch.equal(hidden_states[0], manual_embedding(input_ids, model))
```

**Reflect (one line):** `hidden_states[0]` is the *input* to block 0, not its output — the indexing convention is "snapshot before block `i`," which means `hidden_states[i+1]` is the snapshot after block `i`.

---

#### Step 6.2.2 — Manual loop, one block at a time, verified per step

**Delta:** Thread `x` through all 16 blocks. After each iteration, assert that `x` matches the corresponding entry in `hidden_states`.

**Predict:** For each `i in 0..N-1`, `layers[i](x)[0]` matches `hidden_states[i+1]` exactly (`torch.equal`) because we are calling the same block on the same input that the model itself called it on.

**Code:**
```python
x = manual_embedding(input_ids, model)
mismatches = []
for i in range(N):
    x = layers[i](x)[0]
    if not torch.equal(x, hidden_states[i + 1]):
        gap = (x.float() - hidden_states[i + 1].float()).abs().max().item()
        mismatches.append((i, gap))
```

**Observe:**
```python
print("final x shape:", x.shape)
print("mismatches:", mismatches)
print("final x equals hidden_states[-1]:", torch.equal(x, hidden_states[-1]))
```

**Verify:**
```python
assert mismatches == []
assert torch.equal(x, hidden_states[-1])
```

**Reflect (one line):** Exact equality (not `allclose`) passes here because we are running the same blocks on the same inputs in the same dtype — any inequality would indicate the loop has drifted from the model's own forward pass, not a precision issue.

---

#### Step 6.2.3 — Stitch `manual_block_stack`

**Delta:** Wrap the loop into a helper.

**Predict:** `manual_block_stack(x_embed, layers)` produces a tensor equal to `hidden_states[-1]`.

**Code:**
```python
def manual_block_stack(x, layers):
    for block in layers:
        x = block(x)[0]
    return x
```

**Observe:**
```python
out = manual_block_stack(manual_embedding(input_ids, model), layers)
print("helper output shape:", out.shape)
print("equals hidden_states[-1]:", torch.equal(out, hidden_states[-1]))
```

**Verify:**
```python
assert out.shape == hidden_states[-1].shape
assert torch.equal(out, hidden_states[-1])
```

**Reflect (one line):** The helper exposes the architectural insight in two lines of code — depth is iteration — while hiding none of the per-block opacity that the next phase will dismantle.

---

#### Step 6.2.4 — Verify the helper on a different input

**Delta:** Confirm `manual_block_stack` generalizes beyond the canonical input.

**Predict:** For any tokenized text, `manual_block_stack(manual_embedding(ids, model), layers)` equals `model(ids, output_hidden_states=True).hidden_states[-1]`.

**Code:**
```python
other_ids = tokenizer("Hello world", return_tensors="pt").input_ids
other_ref = model(input_ids=other_ids, output_hidden_states=True).hidden_states[-1]
other_mine = manual_block_stack(manual_embedding(other_ids, model), layers)
```

**Observe:**
```python
print("other_ids.shape:", other_ids.shape)
print("equal:", torch.equal(other_mine, other_ref))
```

**Verify:**
```python
assert torch.equal(other_mine, other_ref)
```

**Reflect (one line):** Generalization-on-a-different-input is the line between "coincidence" and "property" — the helper is now a tool, not a snapshot.

---

**End-of-level check:**
- Without scrolling: write the body of `manual_block_stack`. Two lines. Confirm.
- Predict: if `layers[i](x)` were replaced with `layers[i](x)[0]` swapped for `layers[(i+1) % N](x)[0]` at any single iteration, would the final output match `hidden_states[-1]`? Predict, then test on iteration 0.

---

**End-of-phase milestone check (Phase 6):**
- `layers` bound; length `N`; all elements are `LlamaDecoderLayer`.
- `manual_block_stack(x, layers)` exists; matches `hidden_states[-1]` exactly for two different inputs.
- Without scrolling: state the call convention for a single block. Why is the return value a tuple and not a tensor? What does indexing `[0]` extract?
- What would break if Phase 6 were skipped? (one paragraph from memory)

---

# Phase 7 — Open the block — four children, residual + pre-norm pattern

## Phase 7 — Open the block — four children, residual + pre-norm pattern

**Goal:** Open a single `LlamaDecoderLayer` to identify its four children (`input_layernorm`, `self_attn`, `post_attention_layernorm`, `mlp`) and reproduce the block's forward pass as two pre-norm-plus-residual sub-stages, with `self_attn` and `mlp` still opaque.

**Milestone:** A function `manual_block(x, block)` reproduces `block(x)[0]` for all 16 blocks within `atol=1e-4`, using `manual_rmsnorm` for both norms and calling `block.self_attn`, `block.mlp` as opaque submodules.

**Builds on:** Phase 6 milestone — block stack is open; individual blocks are opaque.

**Helpers stitched in this phase:** `manual_block(x, block)` (with `self_attn` and `mlp` still opaque inside).

---

### Level 7.1 — Identify the four children

**Concept introduced:** A `LlamaDecoderLayer` is composed of exactly four submodules: two RMSNorms and two transformations (attention, MLP).

**Black boxes at start of level:** block forward pass.
**Black boxes opened by this level:** the *names* and *order* of the four children.
**Black boxes still deferred at end of level:** the dataflow that wires them; `self_attn` internals (Phases 8–11); `mlp` internals (Phase 12).

---

#### Step 7.1.1 — Print one block

**Delta:** Use the module repr to see the block's children.

**Predict:** The repr shows four children: `self_attn` (a `LlamaAttention` or `LlamaSdpaAttention`), `mlp` (a `LlamaMLP`), `input_layernorm` (a `LlamaRMSNorm`), `post_attention_layernorm` (a `LlamaRMSNorm`).

**Code:**
```python
block = layers[0]
```

**Observe:**
```python
print(block)
```

**Verify:**
```python
assert hasattr(block, "self_attn")
assert hasattr(block, "mlp")
assert hasattr(block, "input_layernorm")
assert hasattr(block, "post_attention_layernorm")
```

**Reflect (one line):** Naming convention encodes dataflow: `input_layernorm` is *before* attention, `post_attention_layernorm` is *after* attention and *before* MLP — the names describe position in the forward pass, not the operation.

---

#### Step 7.1.2 — Confirm the same four children exist in every block

**Delta:** Loop over all 16 blocks; assert each has the same four named children.

**Predict:** Every block has all four attributes. No block is missing one. No block has an extra named child of a different type.

**Code:**
```python
expected_children = {"self_attn", "mlp", "input_layernorm", "post_attention_layernorm"}
missing_per_block = []
for i, b in enumerate(layers):
    children = {name for name, _ in b.named_children()}
    if children != expected_children:
        missing_per_block.append((i, children))
```

**Observe:**
```python
print("blocks with non-matching children:", missing_per_block)
```

**Verify:**
```python
assert missing_per_block == []
```

**Reflect (one line):** Structural uniformity across blocks means a single helper (`manual_block`) can be written once and trusted across all 16 blocks — the work in Phase 7 onward applies block-wide, not block-by-block.

---

### Level 7.2 — Establish the dataflow by reading the live forward

**Concept introduced:** A single block computes two sub-stages, each shaped as `residual + transform(norm(residual))`. To verify the dataflow without reading HuggingFace source code, use forward hooks to capture intermediate tensors and reconstruct the topology from observed values.

**Black boxes at start of level:** internal dataflow of `block.forward`.
**Black boxes opened by this level:** the two-sub-stage residual+pre-norm structure.
**Black boxes still deferred at end of level:** `self_attn` and `mlp` internals.

---

#### Step 7.2.1 — Register hooks on the four children

**Delta:** Attach forward hooks that capture each child's input and output during a single `block(x)` call.

**Predict:** After running `block(x_embed)`, the hooks have captured eight tensors total: input and output for each of the four children. The dataflow can be inferred from how these tensors connect.

**Code:**
```python
captured = {}

def make_hook(name):
    def hook(module, args, output):
        captured[name] = {
            "input": args[0].detach().clone() if isinstance(args, tuple) else args.detach().clone(),
            "output": output[0].detach().clone() if isinstance(output, tuple) else output.detach().clone(),
        }
    return hook

handles = []
for name in ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp"):
    h = getattr(block, name).register_forward_hook(make_hook(name))
    handles.append(h)

_ = block(x_embed)

for h in handles:
    h.remove()
```

**Observe:**
```python
for name, tensors in captured.items():
    print(f"{name:>28}: in {tuple(tensors['input'].shape)}  out {tuple(tensors['output'].shape)}")
```

**Verify:**
```python
assert set(captured.keys()) == {"input_layernorm", "self_attn", "post_attention_layernorm", "mlp"}
for name in captured:
    assert tuple(captured[name]["input"].shape) == (1, N_tok, d)
    assert tuple(captured[name]["output"].shape) == (1, N_tok, d)
```

**Reflect (one line):** Every child sees and emits `[1, N_tok, d]` — none of them change the shape — which is what makes residual addition trivially shape-compatible at every junction.

---

#### Step 7.2.2 — Verify the attention sub-stage topology: `attn_out = x + self_attn(input_layernorm(x))`

**Delta:** Reconstruct the attention sub-stage from captured tensors and confirm `x_embed + captured["self_attn"]["output"]` equals the input to `post_attention_layernorm`.

**Predict:** The post-attention residual stream equals `x_embed + self_attn_output`, which also equals `captured["post_attention_layernorm"]["input"]`. The two should be equal exactly (same tensor, captured twice via different paths).

**Code:**
```python
attn_residual_manual = x_embed + captured["self_attn"]["output"]
attn_residual_real = captured["post_attention_layernorm"]["input"]
```

**Observe:**
```python
print("manual shape:", attn_residual_manual.shape)
print("real shape  :", attn_residual_real.shape)
print("max abs diff:", (attn_residual_manual.float() - attn_residual_real.float()).abs().max().item())
```

**Verify:**
```python
assert attn_residual_manual.shape == attn_residual_real.shape
assert torch.allclose(attn_residual_manual.float(), attn_residual_real.float(), atol=1e-4, rtol=1e-4)
```

**Reflect (one line):** Confirming the attention sub-stage topology by observation (not by reading source code) is the strongest possible evidence of the dataflow — if the formula `x + self_attn(input_layernorm(x))` were wrong, this assertion would fire.

---

#### Step 7.2.3 — Verify the MLP sub-stage topology: `block_out = y + mlp(post_attention_layernorm(y))`

**Delta:** Reconstruct the MLP sub-stage where `y` is the attention sub-stage output, and confirm the final block output equals `y + mlp_output`.

**Predict:** The block's returned hidden state equals `attn_residual_real + captured["mlp"]["output"]` within tolerance.

**Code:**
```python
y = attn_residual_real
block_out_manual = y + captured["mlp"]["output"]
block_out_real = block(x_embed)[0]
```

**Observe:**
```python
print("manual shape:", block_out_manual.shape)
print("real shape  :", block_out_real.shape)
print("max abs diff:", (block_out_manual.float() - block_out_real.float()).abs().max().item())
```

**Verify:**
```python
assert block_out_manual.shape == block_out_real.shape
assert torch.allclose(block_out_manual.float(), block_out_real.float(), atol=1e-4, rtol=1e-4)
```

**Reflect (one line):** The block's output is *not* the MLP's output — it is the MLP's output *added back to the residual stream* — which is why the residual stream survives all 16 blocks: every block writes a *delta*, never an overwrite.

---

#### Step 7.2.4 — Verify the pre-norm pattern: `self_attn` consumes `input_layernorm(x)`, not `x`

**Delta:** Confirm the attention input is the *normed* tensor by comparing `captured["self_attn"]["input"]` against `captured["input_layernorm"]["output"]`.

**Predict:** `captured["self_attn"]["input"]` equals `captured["input_layernorm"]["output"]` exactly (they are the same tensor in the dataflow).

**Code:**
```python
attn_input = captured["self_attn"]["input"]
norm_output = captured["input_layernorm"]["output"]
```

**Observe:**
```python
print("equal:", torch.equal(attn_input, norm_output))
print("attn_input first values  :", attn_input[0, -1, :5].float().tolist())
print("norm_output first values :", norm_output[0, -1, :5].float().tolist())
```

**Verify:**
```python
assert torch.equal(attn_input, norm_output)
```

**Reflect (one line):** This is the *pre-norm* pattern — norm before transform, residual added outside the norm — which keeps the residual stream itself unnormalized so that downstream blocks always read the raw accumulated signal, not a renormalized snapshot.

---

#### Step 7.2.5 — Mirror for the MLP: `mlp` consumes `post_attention_layernorm(y)`, not `y`

**Delta:** Same check for the second sub-stage.

**Predict:** `captured["mlp"]["input"]` equals `captured["post_attention_layernorm"]["output"]` exactly.

**Code:**
```python
mlp_input = captured["mlp"]["input"]
norm2_output = captured["post_attention_layernorm"]["output"]
```

**Observe:**
```python
print("equal:", torch.equal(mlp_input, norm2_output))
```

**Verify:**
```python
assert torch.equal(mlp_input, norm2_output)
```

**Reflect (one line):** Symmetry between the two sub-stages — each is exactly `residual + transform(norm(residual))` — is what makes the block a composition of two structurally identical pieces, not a one-off design.

---

### Level 7.3 — Reproduce the block manually using `manual_rmsnorm`

**Concept introduced:** With the dataflow confirmed, the block can be written as four operations: norm, attention (opaque), residual; norm, MLP (opaque), residual. The two norms are now `manual_rmsnorm` calls; `self_attn` and `mlp` remain submodule calls.

**Black boxes at start of level:** none new.
**Black boxes opened by this level:** the block's forward pass as composition.
**Black boxes still deferred at end of level:** `self_attn` (Phases 8–11); `mlp` (Phase 12).

---

#### Step 7.3.1 — Build the attention sub-stage manually

**Delta:** Compute `x_normed = manual_rmsnorm(x_embed, ...)`, call `block.self_attn(x_normed)`, add to residual.

**Predict:** The result equals `captured["post_attention_layernorm"]["input"]` (the attention sub-stage output, captured in Step 7.2.2) within tolerance.

**Code:**
```python
x = x_embed
ln1 = block.input_layernorm
x_normed_1 = manual_rmsnorm(x, ln1.weight, ln1.variance_epsilon)
attn_out = block.self_attn(x_normed_1)[0]
y = x + attn_out
```

**Observe:**
```python
print("y.shape:", y.shape)
print("max diff to captured attn_residual:",
      (y.float() - attn_residual_real.float()).abs().max().item())
```

**Verify:**
```python
assert y.shape == attn_residual_real.shape
assert torch.allclose(y.float(), attn_residual_real.float(), atol=1e-4, rtol=1e-4)
```

**Reflect (one line):** The attention sub-stage output `y` is what the second sub-stage sees as its residual stream — it carries the original `x` *and* the attention's contribution, both equally weighted by the simple addition.

---

#### Step 7.3.2 — Build the MLP sub-stage manually

**Delta:** Compute `y_normed = manual_rmsnorm(y, ...)`, call `block.mlp(y_normed)`, add to residual.

**Predict:** The result equals `block(x_embed)[0]` within tolerance.

**Code:**
```python
ln2 = block.post_attention_layernorm
y_normed = manual_rmsnorm(y, ln2.weight, ln2.variance_epsilon)
mlp_out = block.mlp(y_normed)
block_out = y + mlp_out
```

**Observe:**
```python
real = block(x_embed)[0]
print("block_out shape:", block_out.shape)
print("max diff to block(x_embed)[0]:", (block_out.float() - real.float()).abs().max().item())
```

**Verify:**
```python
assert block_out.shape == real.shape
assert torch.allclose(block_out.float(), real.float(), atol=1e-4, rtol=1e-4)
```

**Reflect (one line):** The block's output is the residual stream after two transforms have written into it; the original input `x_embed` is still present in the output, just with two corrections added.

---

#### Step 7.3.3 — Stitch `manual_block`

**Delta:** Wrap the four-operation sequence into a helper.

**Predict:** `manual_block(x_embed, layers[0])` equals `layers[0](x_embed)[0]` within tolerance.

**Code:**
```python
def manual_block(x, block):
    ln1 = block.input_layernorm
    x_normed = manual_rmsnorm(x, ln1.weight, ln1.variance_epsilon)
    attn_out = block.self_attn(x_normed)[0]
    y = x + attn_out

    ln2 = block.post_attention_layernorm
    y_normed = manual_rmsnorm(y, ln2.weight, ln2.variance_epsilon)
    mlp_out = block.mlp(y_normed)
    return y + mlp_out
```

**Observe:**
```python
out = manual_block(x_embed, layers[0])
real = layers[0](x_embed)[0]
print("max diff:", (out.float() - real.float()).abs().max().item())
```

**Verify:**
```python
assert out.shape == real.shape
assert torch.allclose(out.float(), real.float(), atol=1e-4, rtol=1e-4)
```

**Reflect (one line):** The helper exposes the block's full structure (norm, attention, residual, norm, MLP, residual) while keeping the two heavy transforms opaque — exactly the level of openness Phase 7 was meant to deliver.

---

#### Step 7.3.4 — Verify `manual_block` across all 16 blocks, threaded through the stack

**Delta:** Replace `manual_block_stack`'s body with a loop calling `manual_block` on each block; assert the final hidden state still matches `hidden_states[-1]`.

**Predict:** Threading `manual_block` through all 16 blocks produces a tensor equal to `hidden_states[-1]` within tolerance (loose `allclose` because of accumulated bf16 rounding across the chain).

**Code:**
```python
x = manual_embedding(input_ids, model)
for b in layers:
    x = manual_block(x, b)
```

**Observe:**
```python
print("final x shape:", x.shape)
print("max diff to hidden_states[-1]:",
      (x.float() - hidden_states[-1].float()).abs().max().item())
```

**Verify:**
```python
assert x.shape == hidden_states[-1].shape
assert torch.allclose(x.float(), hidden_states[-1].float(), atol=1e-3, rtol=1e-3)
```

**Reflect (one line):** Tolerance is loosened to `1e-3` here because `manual_rmsnorm` does its math in fp32 while the original block does the same — but small bf16 rounding accumulates across 32 norm calls (2 per block × 16 blocks); inside any single block, `1e-4` still holds.

---

**End-of-level check:**
- Without scrolling: write the body of `manual_block`. Confirm the order (norm, attn, residual, norm, mlp, residual).
- Predict: if you swapped `input_layernorm` and `post_attention_layernorm`, would the block's output still match? Predict the kind of failure (small drift vs. complete mismatch), then test.

---

**End-of-phase milestone check (Phase 7):**
- Four children of every block confirmed by name and type.
- Block dataflow confirmed by forward-hook observation, not by reading source code.
- `manual_block(x, block)` matches `block(x)[0]` for all 16 blocks within `atol=1e-4` per-block.
- Stack-wide manual loop matches `hidden_states[-1]` within `atol=1e-3` (accumulated rounding noted).
- Without scrolling: state the two sub-stages of a block. What is the pre-norm pattern? Why is the norm before the transform, and the residual added outside the norm?
- What would break if Phase 7 were skipped? (one paragraph from memory)

---

# Phase 8 — Open attention projections — Q, K, V, O

## Phase 8 — Open attention projections — Q, K, V, O

**Goal:** Inside `self_attn`, identify the four `Linear` projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`), read their weight shapes, and reproduce each projection manually with `x @ W.T`. This phase opens only the projections — the attention math itself (Q-K matmul, scaling, masking, softmax, weighted sum with V) stays opaque until Phase 9.

**Milestone:** Manual `Q = x_normed @ q_proj.weight.T`, and analogous K, V, O computations, match the live submodule calls within `atol=1e-4` for all 16 blocks. The shape asymmetry between Q (full size) and K/V (smaller, due to GQA) is observed and named.

**Builds on:** Phase 7 milestone — block dataflow open; `self_attn` is the next opaque box.

**Helpers stitched in this phase:** none yet. Helpers come in Phases 9–11.

---

### Level 8.1 — Identify the four projections

**Concept introduced:** A `LlamaAttention` (or `LlamaSdpaAttention`) module contains four `Linear` submodules — three input projections and one output projection. Their weight shapes reveal GQA: K and V are smaller than Q.

**Black boxes at start of level:** `self_attn.forward`.
**Black boxes opened by this level:** the names and shapes of `{q_proj, k_proj, v_proj, o_proj}`.
**Black boxes still deferred at end of level:** the attention math (Phases 9–11).

---

#### Step 8.1.1 — List `self_attn`'s children

**Delta:** Identify what submodules attention contains.

**Predict:** Four `Linear` submodules: `q_proj`, `k_proj`, `v_proj`, `o_proj`. All have `bias=None`. There may be one or more non-Linear children too (e.g., a rotary embedding helper in some versions).

**Code:**
```python
attn = block.self_attn
attn_children = dict(attn.named_children())
```

**Observe:**
```python
print("self_attn type:", type(attn).__name__)
for name, child in attn_children.items():
    print(f"  {name:>10}: {type(child).__name__}")
```

**Verify:**
```python
for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
    assert name in attn_children
    assert isinstance(attn_children[name], torch.nn.Linear)
    assert attn_children[name].bias is None
```

**Reflect (one line):** Three input projections (Q, K, V) and one output projection (O) is the canonical attention structure — every transformer attention module has this shape, whether multi-head, GQA, or otherwise.

---

#### Step 8.1.2 — Read Q's weight shape

**Delta:** Bind `W_Q` and confirm its shape.

**Predict:** `q_proj.weight.shape == (d, d) == (2048, 2048)`. The output dim equals the input dim because Q has `n_h * d_h = 32 * 64 = 2048 = d` total features across all query heads.

**Code:**
```python
W_Q = attn.q_proj.weight
```

**Observe:**
```python
print("W_Q.shape:", W_Q.shape)
print("W_Q.dtype:", W_Q.dtype)
print("n_h * d_h:", n_h * d_h)
```

**Verify:**
```python
assert tuple(W_Q.shape) == (d, d)
assert tuple(W_Q.shape) == (n_h * d_h, d)
assert W_Q.dtype == torch.bfloat16
```

**Reflect (one line):** Q's output dimension `n_h * d_h = d` is a coincidence of head-count choice — if the model had chosen `n_h = 16, d_h = 128`, the product would still be `d`, but the structural identity `output_dim == d` would still hold.

---

#### Step 8.1.3 — Read K's weight shape — discover the GQA asymmetry

**Delta:** Bind `W_K` and observe its output dimension is *smaller* than `d`.

**Predict:** `k_proj.weight.shape == (n_kv * d_h, d) == (8 * 64, 2048) == (512, 2048)`. K's output is one-quarter of Q's because only 8 K-heads exist (compared to 32 Q-heads); the four-to-one ratio is GQA.

**Code:**
```python
W_K = attn.k_proj.weight
```

**Observe:**
```python
print("W_K.shape:", W_K.shape)
print("n_kv * d_h:", n_kv * d_h)
print("Q output / K output ratio:", W_Q.shape[0] / W_K.shape[0])
```

**Verify:**
```python
assert tuple(W_K.shape) == (n_kv * d_h, d)
assert W_Q.shape[0] // W_K.shape[0] == n_h // n_kv
```

**Reflect (one line):** K's smaller output dimension is what makes Grouped Query Attention "grouped" — multiple query heads will share the same K head, opened mechanically in Phase 10.

---

#### Step 8.1.4 — Read V's weight shape — same as K

**Delta:** Bind `W_V` and confirm its shape matches K (both K and V are reduced by the same factor under GQA).

**Predict:** `v_proj.weight.shape == (n_kv * d_h, d) == (512, 2048)`. Same as K.

**Code:**
```python
W_V = attn.v_proj.weight
```

**Observe:**
```python
print("W_V.shape:", W_V.shape)
print("K and V same shape:", W_K.shape == W_V.shape)
```

**Verify:**
```python
assert tuple(W_V.shape) == (n_kv * d_h, d)
assert W_K.shape == W_V.shape
```

**Reflect (one line):** K and V are reduced together because the GQA grouping is symmetric in K and V — every grouped query head reads from the same K-head *and* the same V-head; reducing one without the other would break the pairing.

---

#### Step 8.1.5 — Read O's weight shape

**Delta:** Bind `W_O` and confirm it projects back from `d` to `d`.

**Predict:** `o_proj.weight.shape == (d, d) == (2048, 2048)`. The output projection takes the *concatenated* per-head outputs (`n_h * d_h = d`) and projects to `d`.

**Code:**
```python
W_O = attn.o_proj.weight
```

**Observe:**
```python
print("W_O.shape:", W_O.shape)
print("input dim (n_h * d_h):", n_h * d_h)
print("output dim (d):", d)
```

**Verify:**
```python
assert tuple(W_O.shape) == (d, d)
assert W_O.shape[1] == n_h * d_h
assert W_O.shape[0] == d
```

**Reflect (one line):** `o_proj` is the only attention projection whose *input* dimension reflects the multi-head structure (it eats per-head outputs concatenated together) while its *output* matches the residual stream's `d` — it is the bridge from "per-head space" back to "residual stream space."

---

### Level 8.2 — Manually compute each projection

**Concept introduced:** Each `Linear` submodule with `bias=None` is exactly `x @ W.T`. Verifying this for each of Q, K, V, O confirms the projection mechanism without yet involving the attention math.

**Black boxes at start of level:** the attention math (Q-K matmul, scaling, masking, softmax, weighted sum, concat, o_proj). The opacity is *only* the math; the projections themselves are now opened.
**Black boxes opened by this level:** the projection formulas.
**Black boxes still deferred at end of level:** the attention math itself.

---

#### Step 8.2.1 — Set up the test input

**Delta:** Use the same `x_normed_1` from Phase 7 (the pre-attention-normed hidden state for block 0).

**Predict:** `x_normed_1` already exists in the kernel from Step 7.3.1, shape `[1, N_tok, d]`, dtype `bf16`.

**Code:**
```python
x_attn_in = x_normed_1   # alias for clarity
```

**Observe:**
```python
print("x_attn_in.shape:", x_attn_in.shape)
print("x_attn_in.dtype:", x_attn_in.dtype)
```

**Verify:**
```python
assert tuple(x_attn_in.shape) == (1, N_tok, d)
assert x_attn_in.dtype == torch.bfloat16
```

**Reflect (one line):** Reusing the same input across all four projection checks lets later steps compare projections side-by-side without re-running the norm — and matches what the real attention module sees (one normed input, four projections out).

---

#### Step 8.2.2 — Manual Q projection

**Delta:** Compute `Q_manual = x_attn_in @ W_Q.T`; verify against `attn.q_proj(x_attn_in)`.

**Predict:** `Q_manual.shape == (1, N_tok, d) == (1, 6, 2048)`. Equals `attn.q_proj(x_attn_in)` within `atol=1e-4`.

**Code:**
```python
Q_manual = x_attn_in @ W_Q.T
Q_real = attn.q_proj(x_attn_in)
```

**Observe:**
```python
print("Q_manual.shape:", Q_manual.shape)
print("max diff:", (Q_manual.float() - Q_real.float()).abs().max().item())
```

**Verify:**
```python
assert tuple(Q_manual.shape) == (1, N_tok, d)
assert torch.allclose(Q_manual.float(), Q_real.float(), atol=1e-4, rtol=1e-4)
```

**Reflect (one line):** Q's shape `[1, N_tok, d]` is shape-identical to the residual stream's `[1, N_tok, d]` — only the *meaning* of each dimension changes; the residual stream represents "this token's evolving meaning," Q represents "what this token is querying for."

---

#### Step 8.2.3 — Manual K projection

**Delta:** Same operation, smaller output.

**Predict:** `K_manual.shape == (1, N_tok, n_kv * d_h) == (1, 6, 512)`. Equals `attn.k_proj(x_attn_in)` within `atol=1e-4`.

**Code:**
```python
K_manual = x_attn_in @ W_K.T
K_real = attn.k_proj(x_attn_in)
```

**Observe:**
```python
print("K_manual.shape:", K_manual.shape)
print("max diff:", (K_manual.float() - K_real.float()).abs().max().item())
```

**Verify:**
```python
assert tuple(K_manual.shape) == (1, N_tok, n_kv * d_h)
assert torch.allclose(K_manual.float(), K_real.float(), atol=1e-4, rtol=1e-4)
```

**Reflect (one line):** K's smaller third dimension (`512` vs `2048`) is the first place in the manual derivation where the residual stream's shape is *not preserved* — and this asymmetry is what every later step about GQA has to reconcile.

---

#### Step 8.2.4 — Manual V projection

**Delta:** Same operation as K.

**Predict:** `V_manual.shape == (1, N_tok, n_kv * d_h) == (1, 6, 512)`. Equals `attn.v_proj(x_attn_in)` within `atol=1e-4`.

**Code:**
```python
V_manual = x_attn_in @ W_V.T
V_real = attn.v_proj(x_attn_in)
```

**Observe:**
```python
print("V_manual.shape:", V_manual.shape)
print("max diff:", (V_manual.float() - V_real.float()).abs().max().item())
```

**Verify:**
```python
assert tuple(V_manual.shape) == (1, N_tok, n_kv * d_h)
assert torch.allclose(V_manual.float(), V_real.float(), atol=1e-4, rtol=1e-4)
```

**Reflect (one line):** Identical structure to K means `q_proj`, `k_proj`, `v_proj` are *three independent linear maps* that all read from the same `x_attn_in` — they are parallel, not sequential, and could (and on GPU often do) run as a single fused operation.

---

#### Step 8.2.5 — Manual O projection (input from a forward-hook capture)

**Delta:** O projects from a tensor we cannot yet compute (the concatenated per-head attention output, opened in Phase 10). Use a forward hook on `o_proj` to capture its input, then verify the manual matmul matches.

**Predict:** Hooking `o_proj`'s input gives a tensor of shape `[1, N_tok, d] == [1, 6, 2048]` — the concatenated per-head outputs reshaped flat. Manual `o_input @ W_O.T` matches `o_proj(o_input)` within `atol=1e-4`.

**Code:**
```python
o_input_holder = {}

def grab_o_input(module, args, output):
    o_input_holder["x"] = args[0].detach().clone()

h = attn.o_proj.register_forward_hook(grab_o_input)
_ = block.self_attn(x_attn_in)
h.remove()

o_input = o_input_holder["x"]
O_manual = o_input @ W_O.T
O_real = attn.o_proj(o_input)
```

**Observe:**
```python
print("o_input.shape:", o_input.shape)
print("O_manual.shape:", O_manual.shape)
print("max diff:", (O_manual.float() - O_real.float()).abs().max().item())
```

**Verify:**
```python
assert tuple(o_input.shape) == (1, N_tok, d)
assert tuple(O_manual.shape) == (1, N_tok, d)
assert torch.allclose(O_manual.float(), O_real.float(), atol=1e-4, rtol=1e-4)
```

**Reflect (one line):** O's input shape `[1, N_tok, d]` confirms what Phase 10 will derive — the per-head outputs are concatenated back to a `d`-wide tensor before this projection, restoring residual-stream-compatible shape.

---

### Level 8.3 — Verify projections across all 16 blocks

**Concept introduced:** The projection mechanism is uniform across blocks. The same `x @ W.T` formula works for every block's Q, K, V, O — only the weights differ.

**Black boxes at start of level:** none new.
**Black boxes opened by this level:** confirmation of cross-block uniformity.
**Black boxes still deferred at end of level:** the attention math.

---

#### Step 8.3.1 — Loop over blocks, verify Q

**Delta:** For each block, project `x_normed` (computed per-block using that block's `input_layernorm` and the corresponding hidden state from `hidden_states`) and confirm Q matches.

**Predict:** Zero mismatches across all 16 blocks.

**Code:**
```python
mismatches = []
for i, b in enumerate(layers):
    x_in_block = hidden_states[i]
    x_normed = manual_rmsnorm(x_in_block, b.input_layernorm.weight, b.input_layernorm.variance_epsilon)
    Q_m = x_normed @ b.self_attn.q_proj.weight.T
    Q_r = b.self_attn.q_proj(x_normed)
    gap = (Q_m.float() - Q_r.float()).abs().max().item()
    if gap > 1e-4:
        mismatches.append((i, gap))
```

**Observe:**
```python
print("Q mismatches:", mismatches)
```

**Verify:**
```python
assert mismatches == []
```

**Reflect (one line):** A single formula with 16 different weight sets matches 16 different submodules — confirming Q-projection is one operation parameterized by `W_Q`, not 16 different operations.

---

#### Step 8.3.2 — Loop over blocks, verify K, V, O together

**Delta:** Same as Step 8.3.1 but for K, V, and O simultaneously. For O, hook each block's `o_proj` to capture its input.

**Predict:** Zero mismatches for K, V, and O across all 16 blocks.

**Code:**
```python
mismatches = {"K": [], "V": [], "O": []}
for i, b in enumerate(layers):
    x_in_block = hidden_states[i]
    x_normed = manual_rmsnorm(x_in_block, b.input_layernorm.weight, b.input_layernorm.variance_epsilon)

    K_m = x_normed @ b.self_attn.k_proj.weight.T
    K_r = b.self_attn.k_proj(x_normed)
    gap = (K_m.float() - K_r.float()).abs().max().item()
    if gap > 1e-4: mismatches["K"].append((i, gap))

    V_m = x_normed @ b.self_attn.v_proj.weight.T
    V_r = b.self_attn.v_proj(x_normed)
    gap = (V_m.float() - V_r.float()).abs().max().item()
    if gap > 1e-4: mismatches["V"].append((i, gap))

    o_holder = {}
    h = b.self_attn.o_proj.register_forward_hook(
        lambda m, a, o, h=o_holder: h.update(x=a[0].detach().clone()))
    _ = b.self_attn(x_normed)
    h.remove()
    o_in = o_holder["x"]
    O_m = o_in @ b.self_attn.o_proj.weight.T
    O_r = b.self_attn.o_proj(o_in)
    gap = (O_m.float() - O_r.float()).abs().max().item()
    if gap > 1e-4: mismatches["O"].append((i, gap))
```

**Observe:**
```python
print("mismatches:", mismatches)
```

**Verify:**
```python
assert mismatches == {"K": [], "V": [], "O": []}
```

**Reflect (one line):** Verifying across all blocks at once is cheap and high-value — it converts a per-block hypothesis ("this block's projections work") into a structural property ("attention's projection mechanism is uniform across the architecture").

---

**End-of-level check:**
- Without scrolling: write the shape of `W_Q`, `W_K`, `W_V`, `W_O`. State which two are equal and why.
- Predict: if you swapped `W_K` with `W_V` for one block, would `attention(x_normed)` still match? Predict the kind of failure (shape error vs. value mismatch), then test.

---

**End-of-phase milestone check (Phase 8):**
- Four projections identified by name in `self_attn`; types confirmed (`Linear`, no bias).
- Weight shapes: Q `[d, d]`, K and V `[n_kv * d_h, d]`, O `[d, d]`. The GQA asymmetry (K, V smaller than Q) is observed and named.
- Manual `x @ W.T` matches each submodule for all 16 blocks within `atol=1e-4`.
- Without scrolling: state the four projection shapes. Why are K and V smaller than Q? What is the input shape that `o_proj` consumes and where does that shape come from?
- What would break if Phase 8 were skipped? (one paragraph from memory)

---

# Phase 9 — Open single-head attention (a correct special case)

## Phase 9 — Open single-head attention (a correct special case)

**Goal:** Open the attention math — Q-K matmul, scaling, causal mask, softmax, weighted sum with V — by treating *one head's slice* of Q, K, V as a stand-alone single-head attention. This produces a *correct* attention output for that one head, which can be checked against a head-slice of the live model's attention output. Phase 10 generalizes the head count; Phase 11 adds RoPE.

**Important framing.** This phase is not "single-head Llama" — Llama is multi-head with GQA. This phase opens the *math operations* (matmul, scale, mask, softmax, weighted sum) on data shaped as a single head. The math is exactly what the live model runs *per head*; what is missing is the per-head reshape (Phase 10) and the positional rotation (Phase 11).

**The reference.** Verifying single-head attention against the live model requires a per-head reference. We capture this via a forward hook on the **attention output before `o_proj`**, then slice out one head's contribution. The hook captures the multi-head output; we compare to head 0 of that capture.

**Milestone:** A function `single_head_attention(Q_h, K_h, V_h)` exists. For head 0 of block 0, when given the appropriate Q, K, V slices (without RoPE applied), it produces a tensor whose shape and qualitative behavior match the reference, with caveats about RoPE absence noted explicitly.

**Note on RoPE.** Because Llama applies RoPE inside attention, the manual single-head output will *not* match the live single-head output value-for-value yet. This is acknowledged at every relevant step; the precise value match comes in Phase 11. Phase 9's verification is on shape, structural properties (causal masking, softmax-rows-sum-to-1), and the principled gap to the live model.

**Builds on:** Phase 8 milestone — Q, K, V, O projections are open.

**Helpers stitched in this phase:** `single_head_attention(Q_h, K_h, V_h, mask)`.

---

### Level 9.1 — Slice one head out of Q, K, V

**Concept introduced:** To work with one head, take Q, K, V and slice along the channel dimension. For head 0, that means dimensions `[0:d_h]` of Q, and `[0:d_h]` of K and V. (For K and V under GQA, head 0 corresponds to K/V-head 0, which is the same K/V-head that Q-heads 0, 1, 2, 3 all share — but we are working with just Q-head 0 in this phase.)

**Black boxes at start of level:** the attention math.
**Black boxes opened by this level:** the per-head slicing convention.
**Black boxes still deferred at end of level:** the matmul/scale/mask/softmax/weighted-sum operations (later in this phase); the multi-head reshape (Phase 10); RoPE (Phase 11).

---

#### Step 9.1.1 — Recompute Q, K, V for block 0 cleanly

**Delta:** Re-derive Q, K, V from `x_attn_in` so the variables are unambiguous (the earlier ones in the kernel are from Phase 8 but with possibly different names).

**Predict:** `Q.shape == (1, N_tok, d)`. `K.shape == V.shape == (1, N_tok, n_kv * d_h)`.

**Code:**
```python
attn = layers[0].self_attn
x_attn_in = manual_rmsnorm(hidden_states[0], layers[0].input_layernorm.weight,
                           layers[0].input_layernorm.variance_epsilon)
Q = x_attn_in @ attn.q_proj.weight.T
K = x_attn_in @ attn.k_proj.weight.T
V = x_attn_in @ attn.v_proj.weight.T
```

**Observe:**
```python
print("Q.shape:", Q.shape)
print("K.shape:", K.shape)
print("V.shape:", V.shape)
```

**Verify:**
```python
assert tuple(Q.shape) == (1, N_tok, d)
assert tuple(K.shape) == (1, N_tok, n_kv * d_h)
assert tuple(V.shape) == (1, N_tok, n_kv * d_h)
```

**Reflect (one line):** Re-deriving the projections at the top of Phase 9 keeps the phase self-contained — every variable used here has a definition visible in this phase, not buried in Phase 8.

---

#### Step 9.1.2 — Slice Q for head 0

**Delta:** Take dimensions `[0:d_h]` of Q's last axis to get head 0's queries.

**Predict:** `Q_h.shape == (1, N_tok, d_h) == (1, 6, 64)`. The first 64 channels of Q are head 0's per-token query vectors.

**Code:**
```python
Q_h = Q[:, :, 0:d_h]
```

**Observe:**
```python
print("Q_h.shape:", Q_h.shape)
print("Q_h dtype:", Q_h.dtype)
```

**Verify:**
```python
assert tuple(Q_h.shape) == (1, N_tok, d_h)
assert Q_h.dtype == torch.bfloat16
```

**Reflect (one line):** The convention "head `h` occupies channels `[h*d_h : (h+1)*d_h]`" is a *convention* embedded in how the projection weights were trained — there is no mechanical reason for this slicing to be correct other than that Llama's `W_Q` was trained with that layout.

---

#### Step 9.1.3 — Slice K and V for K/V-head 0

**Delta:** Same idea, but for K and V the slicing is per K/V-head, and there are only `n_kv = 8` of those.

**Predict:** `K_h.shape == V_h.shape == (1, N_tok, d_h) == (1, 6, 64)`. Q-head 0 maps to K/V-head 0 because under GQA, Q-heads `0..3` all share K/V-head 0 (group size = `n_h // n_kv = 4`).

**Code:**
```python
kv_head_index = 0   # for Q-head 0, the corresponding K/V-head index is 0 // (n_h // n_kv) = 0
K_h = K[:, :, kv_head_index * d_h : (kv_head_index + 1) * d_h]
V_h = V[:, :, kv_head_index * d_h : (kv_head_index + 1) * d_h]
```

**Observe:**
```python
print("K_h.shape:", K_h.shape)
print("V_h.shape:", V_h.shape)
```

**Verify:**
```python
assert tuple(K_h.shape) == (1, N_tok, d_h)
assert tuple(V_h.shape) == (1, N_tok, d_h)
```

**Reflect (one line):** The mapping `kv_head_index = q_head_index // group_size` is the *only* place GQA differs from standard multi-head attention — once K_h and V_h are sliced, the rest of single-head math is identical regardless of GQA.

---

### Level 9.2 — Build the attention math

**Concept introduced:** Single-head attention is five operations: Q @ K.T, scale, causal mask, softmax, weighted sum with V. Each is one step.

**Black boxes at start of level:** the five math operations.
**Black boxes opened by this level:** all five.
**Black boxes still deferred at end of level:** RoPE (Phase 11); multi-head generalization (Phase 10).

---

#### Step 9.2.1 — Compute raw attention scores

**Delta:** `scores = Q_h @ K_h.transpose(-2, -1)`. For each query token, score every key token.

**Predict:** `scores.shape == (1, N_tok, N_tok) == (1, 6, 6)`. Diagonal entries are dot products of a token's Q with its own K — typically the largest values in absence of positional encoding.

**Code:**
```python
scores_raw = Q_h @ K_h.transpose(-2, -1)
```

**Observe:**
```python
print("scores_raw.shape:", scores_raw.shape)
print("scores_raw[0]:\n", scores_raw[0].float())
print("diagonal:", scores_raw[0].diag().float().tolist())
```

**Verify:**
```python
assert tuple(scores_raw.shape) == (1, N_tok, N_tok)
```

**Reflect (one line):** Scores are a token-to-token relevance matrix — entry `[i, j]` is "how strongly token `i`'s query is attracted to token `j`'s key" — but unlike a probability, scores can be any real number, positive or negative.

---

#### Step 9.2.2 — Scale by `1/sqrt(d_h)`

**Delta:** Divide scores by `sqrt(d_h)` to control variance.

**Predict:** `scores_scaled.shape == scores_raw.shape`. The scaling factor for `d_h=64` is `1/8`. Values shrink by 8×, which prevents softmax from saturating when `d_h` is large.

**Code:**
```python
import math
scale = 1.0 / math.sqrt(d_h)
scores_scaled = scores_raw * scale
```

**Observe:**
```python
print("scale:", scale)
print("scores_scaled max abs:", scores_scaled.float().abs().max().item())
print("scores_raw max abs   :", scores_raw.float().abs().max().item())
```

**Verify:**
```python
assert scores_scaled.shape == scores_raw.shape
assert abs(scale - 0.125) < 1e-9
```

**Reflect (one line):** The `1/sqrt(d_h)` factor exists because Q @ K.T has variance proportional to `d_h` (each output is a sum of `d_h` products); dividing by `sqrt(d_h)` restores unit variance, keeping softmax outputs from collapsing to one-hot as head dimensions grow.

---

#### Step 9.2.3 — Build the causal mask

**Delta:** Construct a `[N_tok, N_tok]` mask that is `0` where token `i` may attend to token `j` (i.e., `j <= i`) and `-inf` elsewhere.

**Predict:** `mask.shape == (N_tok, N_tok) == (6, 6)`. Lower triangle (including diagonal) is `0`; upper triangle is `-inf`. After adding to scores, the upper-triangle scores become `-inf`, which softmax sends to `0`.

**Code:**
```python
mask = torch.full((N_tok, N_tok), float("-inf"))
mask = torch.triu(mask, diagonal=1)   # zero on and below diagonal, -inf strictly above
```

**Observe:**
```python
print("mask:\n", mask)
```

**Verify:**
```python
assert mask.shape == (N_tok, N_tok)
# lower triangle (including diag) is 0
for i in range(N_tok):
    for j in range(i + 1):
        assert mask[i, j].item() == 0.0
# strict upper triangle is -inf
for i in range(N_tok):
    for j in range(i + 1, N_tok):
        assert mask[i, j].item() == float("-inf")
```

**Reflect (one line):** Causal masking enforces autoregressive generation — when predicting token `i+1`, the attention can only see tokens `0..i`, not the future — which is the structural reason a transformer LLM can be used for next-token generation in the first place.

---

#### Step 9.2.4 — Apply the mask

**Delta:** Add the mask to the scaled scores.

**Predict:** `scores_masked.shape == scores_scaled.shape`. Lower-triangular positions unchanged. Strict upper-triangular positions become `-inf`.

**Code:**
```python
scores_masked = scores_scaled + mask
```

**Observe:**
```python
print("scores_masked[0]:\n", scores_masked[0].float())
```

**Verify:**
```python
assert scores_masked.shape == scores_scaled.shape
# at position [0, last_row, 0:last_col], finite values
assert torch.isfinite(scores_masked[0, -1, :]).all()
# at position [0, 0, 1:], -inf values (token 0 cannot attend to future tokens)
assert torch.isinf(scores_masked[0, 0, 1:]).all()
```

**Reflect (one line):** Adding `-inf` (rather than multiplying or selecting) is the right way to mask before a softmax because `exp(-inf) = 0` — the masked positions contribute zero to the distribution without disturbing the relative weights of unmasked ones.

---

#### Step 9.2.5 — Softmax across keys

**Delta:** Apply softmax along the *key* axis (last dim) so each query's attention weights sum to 1.

**Predict:** `attn_weights.shape == scores_masked.shape == (1, N_tok, N_tok)`. Each row sums to `1.0` within `1e-4`. Row `i` has zeros for columns `> i` (masked positions).

**Code:**
```python
attn_weights = torch.softmax(scores_masked.float(), dim=-1)
```

**Observe:**
```python
print("attn_weights[0]:\n", attn_weights[0])
print("row sums:", attn_weights[0].sum(dim=-1).tolist())
```

**Verify:**
```python
assert attn_weights.shape == scores_masked.shape
row_sums = attn_weights[0].sum(dim=-1)
assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
# row 0 must put all weight on token 0
assert abs(attn_weights[0, 0, 0].item() - 1.0) < 1e-6
# row 0 must have zero weight on tokens 1..N_tok-1
assert (attn_weights[0, 0, 1:].abs() < 1e-9).all()
```

**Reflect (one line):** Computing softmax in fp32 (the `.float()` cast) avoids bf16 mantissa loss in the exponential — bf16 has 8 bits of exponent (same as fp32) but only 7 bits of mantissa, which is too few when normalizing a sum that may span orders of magnitude.

---

#### Step 9.2.6 — Weighted sum with V

**Delta:** Each output token is a weighted sum of value vectors, with weights from `attn_weights`.

**Predict:** `attn_out_h.shape == (1, N_tok, d_h) == (1, 6, 64)`. Token 0's output equals `V_h[0, 0, :]` exactly (because token 0 attends only to itself with weight 1).

**Code:**
```python
attn_out_h = attn_weights @ V_h.float()
```

**Observe:**
```python
print("attn_out_h.shape:", attn_out_h.shape)
print("token 0 output equals V_h[0,0]:",
      torch.allclose(attn_out_h[0, 0].float(), V_h[0, 0].float(), atol=1e-4))
```

**Verify:**
```python
assert tuple(attn_out_h.shape) == (1, N_tok, d_h)
assert torch.allclose(attn_out_h[0, 0].float(), V_h[0, 0].float(), atol=1e-4)
```

**Reflect (one line):** Token 0's output equaling its own V vector is the strongest possible structural check on causal masking and softmax-row-sums-to-1 — together they force the first token's output to be exactly its own value, with no contribution from anywhere else.

---

### Level 9.3 — Verify against the live model (with RoPE caveat)

**Concept introduced:** The live model includes RoPE inside attention, so Phase 9's output will not match the live single-head reference *exactly*. The verification at this stage is on (a) shape, (b) structural properties (mask zeroing, softmax rows summing to 1), and (c) qualitative similarity to the live reference. Precise value match comes in Phase 11.

**Black boxes at start of level:** the gap between manual no-RoPE attention and live with-RoPE attention.
**Black boxes opened by this level:** explicit measurement of that gap.
**Black boxes still deferred at end of level:** the gap is *closed* in Phase 11; here it is only *measured*.

---

#### Step 9.3.1 — Capture the live multi-head attention output (pre-`o_proj`)

**Delta:** Hook `o_proj`'s input to capture the concatenated multi-head output. Slice head 0's contribution: dimensions `[0:d_h]`.

**Predict:** `live_attn_out` shape `[1, N_tok, d] == [1, 6, 2048]`. Head 0's slice is `[:, :, 0:d_h]` with shape `[1, 6, 64]`.

**Code:**
```python
o_in_holder = {}
def grab(m, a, o, holder=o_in_holder):
    holder["x"] = a[0].detach().clone()
h = attn.o_proj.register_forward_hook(grab)
_ = block.self_attn(x_attn_in)
h.remove()

live_attn_out = o_in_holder["x"]
live_head0 = live_attn_out[:, :, 0:d_h]
```

**Observe:**
```python
print("live_attn_out.shape:", live_attn_out.shape)
print("live_head0.shape:   ", live_head0.shape)
print("attn_out_h.shape:   ", attn_out_h.shape)
```

**Verify:**
```python
assert tuple(live_attn_out.shape) == (1, N_tok, d)
assert tuple(live_head0.shape) == (1, N_tok, d_h)
```

**Reflect (one line):** Hooking `o_proj`'s input captures the multi-head attention output *before* projection, which is the only place the per-head structure is visible end-to-end — after `o_proj`, the head boundaries are mixed by `W_O` and unrecoverable.

---

#### Step 9.3.2 — Measure the gap

**Delta:** Compute `(attn_out_h - live_head0).abs().max()`. Expect a significant gap (much larger than `1e-4`), because the live computation applies RoPE inside attention.

**Predict:** The gap is much larger than `1e-4` — likely in the range `0.1` to `5.0` — because RoPE rotates Q and K before the matmul, and our manual computation skips this. The gap is *not* a bug; it is a measurement.

**Code:**
```python
gap = (attn_out_h.float() - live_head0.float()).abs().max().item()
mean_gap = (attn_out_h.float() - live_head0.float()).abs().mean().item()
```

**Observe:**
```python
print("max abs gap:", gap)
print("mean abs gap:", mean_gap)
print("gap is much larger than 1e-4:", gap > 0.01)
```

**Verify:**
```python
# Structural assertion only: token 0 should still match exactly,
# because RoPE at position 0 is the identity rotation.
assert torch.allclose(attn_out_h[0, 0].float(), live_head0[0, 0].float(), atol=1e-3)
# The full sequence's gap should be large enough to confirm RoPE is the culprit.
assert gap > 0.01
```

**Reflect (one line):** Token 0's match (within `1e-3`) and the rest of the sequence's mismatch is itself diagnostic — RoPE at position 0 applies an identity rotation, so position 0 is unaffected; the larger error grows with position, exactly as RoPE's effect grows with position.

---

### Level 9.4 — Stitch `single_head_attention`

**Concept introduced:** Wrap the five-operation math into a helper that takes one head's Q, K, V and a mask, and produces that head's attention output. The helper will be called per-head in Phase 10.

**Black boxes at start of level:** none new.
**Black boxes opened by this level:** the reusable form of single-head attention.
**Black boxes still deferred at end of level:** RoPE (Phase 11).

---

#### Step 9.4.1 — Define `single_head_attention(Q_h, K_h, V_h, mask)`

**Delta:** Bundle the math.

**Predict:** Given the same head-0 inputs from this phase, the helper produces a tensor equal to `attn_out_h` from Step 9.2.6.

**Code:**
```python
import math

def single_head_attention(Q_h, K_h, V_h, mask):
    """
    Q_h, K_h, V_h: shape [..., N_tok, d_h]
    mask: shape [N_tok, N_tok]; 0 where allowed, -inf where masked
    Returns: shape [..., N_tok, d_h]
    """
    d_h_local = Q_h.shape[-1]
    scale = 1.0 / math.sqrt(d_h_local)
    scores = (Q_h @ K_h.transpose(-2, -1)) * scale
    scores = scores + mask
    weights = torch.softmax(scores.float(), dim=-1)
    return weights @ V_h.float()
```

**Observe:**
```python
out_h = single_head_attention(Q_h, K_h, V_h, mask)
print("helper out shape:", out_h.shape)
print("matches attn_out_h:", torch.allclose(out_h, attn_out_h, atol=1e-5))
```

**Verify:**
```python
assert tuple(out_h.shape) == (1, N_tok, d_h)
assert torch.allclose(out_h, attn_out_h, atol=1e-5)
```

**Reflect (one line):** The helper reads `d_h` from the input shape rather than the global, which makes it usable for tensors of any head dimension — Phase 10 will call it on tensors that include a head axis, and the math is unchanged.

---

#### Step 9.4.2 — Verify the helper on a different head's slice

**Delta:** Slice Q and K/V for a different head (e.g., Q-head 5, which maps to K/V-head `5 // 4 = 1`) and confirm the helper produces a shape-correct result that also matches the corresponding live head's slice at token 0.

**Predict:** For head 5, `single_head_attention` returns shape `[1, N_tok, d_h]`. Token 0's output equals V's token-0 vector for K/V-head 1 within tolerance (same reasoning as Step 9.2.6: RoPE is identity at position 0).

**Code:**
```python
q_head_index = 5
kv_head_index = q_head_index // (n_h // n_kv)  # = 1

Q_h5 = Q[:, :, q_head_index * d_h : (q_head_index + 1) * d_h]
K_h5 = K[:, :, kv_head_index * d_h : (kv_head_index + 1) * d_h]
V_h5 = V[:, :, kv_head_index * d_h : (kv_head_index + 1) * d_h]

out_h5 = single_head_attention(Q_h5, K_h5, V_h5, mask)

live_head5 = live_attn_out[:, :, q_head_index * d_h : (q_head_index + 1) * d_h]
```

**Observe:**
```python
print("out_h5.shape:", out_h5.shape)
print("token-0 diff to live head 5:",
      (out_h5[0, 0].float() - live_head5[0, 0].float()).abs().max().item())
```

**Verify:**
```python
assert tuple(out_h5.shape) == (1, N_tok, d_h)
# Token 0 matches within 1e-3 (RoPE identity at position 0)
assert torch.allclose(out_h5[0, 0].float(), live_head5[0, 0].float(), atol=1e-3)
```

**Reflect (one line):** The helper works unchanged for a different head — the only per-head specifics are the *slices* of Q, K, V; the math is universal, which is exactly the property Phase 10 will exploit to vectorize across all heads.

---

**End-of-level check:**
- Without scrolling: write the body of `single_head_attention` from scratch. Five operations.
- Predict: for the canonical input, what is `attn_weights[0, 2, 5]` (token 2 attending to token 5)? Predict, then check. (Hint: token 5 is *future* relative to token 2.)

---

**End-of-phase milestone check (Phase 9):**
- Five attention math operations (scores, scale, mask, softmax, weighted sum) each have their own step with a value- or structural-check.
- `single_head_attention(Q_h, K_h, V_h, mask)` exists and matches the manual derivation for head 0 of block 0.
- Token-0 of any head's manual output matches the live model's corresponding head-0 slice within `1e-3` (RoPE identity at position 0).
- The gap to the live model at later positions is *measured* and named as the RoPE gap, to be closed in Phase 11.
- Without scrolling: state the five operations of attention math, in order. Why is the scaling factor `1/sqrt(d_h)` and not `1/d_h`? Why does causal masking use `-inf` addition and not multiplication?
- What would break if Phase 9 were skipped? (one paragraph from memory)

---

# Phase 10 — Generalize to multi-head and GQA (detailed stub)

## Phase 10 — Generalize to multi-head and GQA

**Goal:** Open the per-head reshape and the GQA grouping. Replace the per-head loop with a vectorized computation across all heads, producing the full pre-`o_proj` attention output. Verify against the live model's pre-`o_proj` output for all heads — *except* for the RoPE gap, which Phase 11 closes.

**Milestone:** A function `multi_head_attention(x_normed, attn_module, mask)` produces a tensor of shape `[1, N_tok, d]` that matches the live `o_proj` input for token 0 within `1e-3` (RoPE identity at position 0). At later positions, the gap is non-zero but matches the gap measured per-head in Phase 9, confirming the multi-head generalization is correct and RoPE remains the only missing piece.

**Builds on:** Phase 9 milestone — single-head math works and is stitched.

**Helpers stitched in this phase:** `multi_head_attention(x_normed, attn_module, mask)` (still missing RoPE; Phase 11 adds it).

**Reference target:** `o_proj`-input hook (captured in Phase 9 Step 9.3.1) — call this `live_attn_out`. Compare token-0 of `multi_head_attention(...)` against `live_attn_out[0, 0, :]`.

**Suggested level breakdown:**

- **Level 10.1 — Reshape Q to expose the head axis.**
  - Step: Q from `[1, N_tok, d]` reshaped to `[1, N_tok, n_h, d_h]` then transposed to `[1, n_h, N_tok, d_h]`. Verify shape; verify head 0's slice equals the per-head slice from Phase 9.
  - Step: same for K and V, but with `n_kv` instead of `n_h`: `[1, N_tok, n_kv * d_h]` → `[1, N_tok, n_kv, d_h]` → `[1, n_kv, N_tok, d_h]`.

- **Level 10.2 — Repeat K and V along the head axis to match `n_h`.**
  - Step: K from `[1, n_kv, N_tok, d_h]` repeated 4× to `[1, n_h, N_tok, d_h]`. Same for V.
  - Step: confirm the K/V indexing convention by checking that K head 0, 1, 2, 3 all come from K/V-head 0 (the GQA grouping is verified by observation, not assumption).
  - Suggested PyTorch operation: `K.repeat_interleave(n_h // n_kv, dim=1)`. HuggingFace internally uses a function called `repeat_kv`; the result is the same. Confirm whichever choice is used matches the live model by hooking the K used inside attention.

- **Level 10.3 — Vectorized attention across all heads.**
  - Step: scores = `Q @ K.transpose(-2, -1)` with shape `[1, n_h, N_tok, N_tok]`.
  - Step: scale.
  - Step: causal mask broadcast across the head axis. Mask has shape `[N_tok, N_tok]`; it broadcasts automatically against `[1, n_h, N_tok, N_tok]`. Verify by structural property: at position `[*, *, 0, 1:]`, scores are `-inf` for every head.
  - Step: softmax. Verify rows sum to 1 per head per token.
  - Step: weighted sum with V, shape `[1, n_h, N_tok, d_h]`.

- **Level 10.4 — Concatenate heads back.**
  - Step: transpose from `[1, n_h, N_tok, d_h]` back to `[1, N_tok, n_h, d_h]`.
  - Step: reshape to `[1, N_tok, n_h * d_h] == [1, N_tok, d]`. This is the input that the live `o_proj` consumes.
  - Step: verify against `live_attn_out` from Phase 9 Step 9.3.1, *for token 0 only* (rest has RoPE gap). Predict the gap pattern: token-0 matches within `1e-3`, later tokens diverge by the same per-head margin Phase 9 measured.

- **Level 10.5 — Stitch `multi_head_attention(x_normed, attn_module, mask)`.**
  - Step: define the helper. The helper computes Q/K/V, reshapes, repeats K/V, runs the math, concats. No RoPE yet.
  - Step: verify on block 0; verify on block 5; verify on block 15 (different blocks to confirm cross-block uniformity).
  - End-of-phase milestone: helper exists, matches live at token 0 across multiple blocks, RoPE gap acknowledged as the next phase's target.

**Patterns to follow (see Section 6.13 — "Patterns for attention/FFN-class phases"):**

- Every reshape gets its own step; every transpose gets its own step. Shape changes are first-class concepts.
- After any shape rearrangement, a verification step that checks the rearrangement is equivalence-preserving (a slice from the rearranged tensor matches the same slice from the pre-rearrangement tensor).
- The GQA repeat is conceptually distinct from the head reshape — they get separate levels.
- Final concat/reshape back to `[1, N_tok, d]` always gets its own step, with a note about why this shape is what `o_proj` consumes.

---

# Phase 11 — Apply RoPE — attention matches the live model exactly (detailed stub)

## Phase 11 — Apply RoPE — attention matches the live model exactly

**Goal:** Apply Rotary Position Embedding to Q and K (not V) *after* the per-head reshape and *before* the Q @ K matmul. With RoPE in place, the manual attention output matches the live model's `o_proj` input within `atol=1e-4` at every position.

**Milestone:** `manual_attention(x_normed, attn_module, position_ids, mask)` matches `live_attn_out` (the `o_proj` input captured by hook) within `atol=1e-4` for all 16 blocks.

**Builds on:** Phase 10 milestone — multi-head + GQA work; only RoPE missing.

**Helpers stitched in this phase:** `apply_rope(x, cos, sin)`; `manual_attention(x_normed, attn_module, position_ids, mask)`.

**Reference target:** Same as Phase 10 — `o_proj`-input hook for the relevant block.

**Inspecting RoPE in the live model.** Different HuggingFace versions implement RoPE differently. Some precompute `cos`/`sin` tables; others compute them on the fly inside `LlamaRotaryEmbedding.__call__`. Do not assume a specific table shape. Instead, inspect `model.model.rotary_emb` directly and read its `forward` signature. The two values it returns (typically `cos` and `sin`) are what RoPE rotation uses.

**Suggested level breakdown:**

- **Level 11.1 — Inspect the rotary embedding module.**
  - Step: print `model.model.rotary_emb`; identify its type and call signature.
  - Step: call it with a dummy input matching what attention would pass; observe the shape and dtype of the returned `cos` and `sin` tensors.
  - Step: confirm by hook that this exact module is called inside each block's attention.

- **Level 11.2 — Generate `position_ids`.**
  - Step: `position_ids = torch.arange(N_tok).unsqueeze(0)` — shape `[1, N_tok]`.
  - Step: confirm this matches what the model passes internally (hook on attention's call signature).

- **Level 11.3 — Compute `cos` and `sin` for the input positions.**
  - Step: call `model.model.rotary_emb(K, position_ids)` (or whatever the version requires) to get `cos`, `sin`.
  - Step: inspect shape; the values are typically `[1, N_tok, d_h]` or `[1, N_tok, d_h//2]` depending on implementation — verify.
  - Step: confirm that `cos` at position 0 is all `1.0` and `sin` at position 0 is all `0.0` — this is what makes Phase 9's token-0 match work.

- **Level 11.4 — Implement RoPE rotation manually.**
  - Step: the rotation interleaves pairs of dimensions or treats the first/second half separately, depending on convention. Implement the convention HuggingFace uses (typically "split into two halves, rotate, recombine," with `rotate_half(x) = cat([-x2, x1], dim=-1)` where `x1, x2 = x.chunk(2, dim=-1)`).
  - Step: write `apply_rope(x, cos, sin) = x * cos + rotate_half(x) * sin`.
  - Step: verify against the model's internal `apply_rotary_pos_emb` (hook the Q-after-RoPE somehow — typically by hooking the input to the SDPA call inside attention, or replicating the function from the HuggingFace source after reading it).

- **Level 11.5 — Apply RoPE to Q and K (not V); re-run multi-head attention.**
  - Step: call `apply_rope(Q, cos, sin)` and `apply_rope(K, cos, sin)`. V is not rotated.
  - Step: feed the rotated Q and K into the math from Phase 10. The output should now match `live_attn_out` within `atol=1e-4` at every position.
  - Step: measure the gap. It should be `< 1e-4`. If larger, the most likely cause is a RoPE convention mismatch (split-halves vs. interleaved).

- **Level 11.6 — Stitch `manual_attention(x_normed, attn_module, position_ids, mask)`.**
  - Step: the helper computes Q/K/V, reshapes, applies RoPE to Q and K, repeats K/V, runs attention math, concats.
  - Step: verify on all 16 blocks within `atol=1e-4`.

- **Level 11.7 — Apply `o_proj` and verify the full attention sub-stage.**
  - Step: `manual_attention_output = manual_attention(x_normed, attn, position_ids, mask) @ W_O.T`.
  - Step: verify against `block.self_attn(x_normed)[0]` within `atol=1e-4` for all 16 blocks. This is the closing assertion of Phase 11.

**Pattern note:** Phase 11 has a unique risk — the RoPE convention can differ between implementations (split-halves rotation vs. interleaved-pairs rotation; absolute frequencies vs. relative). The verification step that catches a convention mismatch is "does token 0 match within `1e-6`?" If yes, the conventions agree at position 0 (where both are identity); if subsequent tokens diverge, the convention used for frequencies or pairing is wrong.

---

# Phase 12 — Open the FFN — projections, SwiGLU (detailed stub)

## Phase 12 — Open the FFN — projections, SwiGLU

**Goal:** Open `block.mlp` to identify its three projections (`gate_proj`, `up_proj`, `down_proj`) and the SwiGLU activation pattern. Reproduce `block.mlp(y_normed)` manually with raw tensor operations.

**Milestone:** `manual_ffn(y_normed, mlp_module)` matches `block.mlp(y_normed)` within `atol=1e-4` for all 16 blocks.

**Builds on:** Phase 11 milestone — attention is fully open and stitched.

**Helpers stitched in this phase:** `manual_ffn(y_normed, mlp_module)`.

**Reference target:** `block.mlp(y_normed)` directly — no hook needed because `mlp` is a clean submodule.

**Suggested level breakdown:**

- **Level 12.1 — Identify the three projections.**
  - Step: print `block.mlp`; identify `gate_proj`, `up_proj`, `down_proj`. All `Linear`, no bias.
  - Step: read weight shapes. `gate_proj.weight: [d_ff, d]`. `up_proj.weight: [d_ff, d]`. `down_proj.weight: [d, d_ff]`.
  - Step: note that `gate` and `up` are *parallel* — both consume the same input — while `down` is sequential (consumes the elementwise product).

- **Level 12.2 — Compute `gate` and `up` projections.**
  - Step: `gate_pre = y_normed @ gate_proj.weight.T`. Shape `[1, N_tok, d_ff]`. Verify against submodule call.
  - Step: `up_out = y_normed @ up_proj.weight.T`. Shape `[1, N_tok, d_ff]`. Verify.

- **Level 12.3 — Apply SiLU to `gate_pre`.**
  - Step: SiLU (also called Swish) is `x * sigmoid(x)`. Compute `gate_act = gate_pre * torch.sigmoid(gate_pre)`.
  - Step: structural checks. SiLU(0) = 0. SiLU is monotonic on x > 0. SiLU(large negative) ≈ 0. Verify with specific values.
  - Step: alternative form. `gate_act` can also be computed via `torch.nn.functional.silu(gate_pre)`. Verify both forms agree exactly.

- **Level 12.4 — Elementwise multiply (the "gate" in SwiGLU).**
  - Step: `hidden = gate_act * up_out`. Shape `[1, N_tok, d_ff]`. The naming "gated linear unit" comes from `gate_act` modulating `up_out` — when `gate_act` is near zero for a given channel, that channel is suppressed.

- **Level 12.5 — Apply `down_proj`.**
  - Step: `ffn_out = hidden @ down_proj.weight.T`. Shape `[1, N_tok, d] = [1, N_tok, 2048]`. This restores residual-stream-compatible shape.
  - Step: verify against `block.mlp(y_normed)` within `atol=1e-4`.

- **Level 12.6 — Stitch `manual_ffn(y_normed, mlp_module)`.**
  - Step: define the helper. Five operations: gate_proj, up_proj, silu, multiply, down_proj.
  - Step: verify on block 0; verify on block 5; verify on block 15.
  - Step: verify across all 16 blocks in a loop.

**Pattern note:** SwiGLU's two-projections-and-multiply structure (parallel gate and up, then elementwise product) is structurally similar to attention's three-projections-and-matmul structure (parallel Q, K, V, then Q-K-matmul-then-V-multiply). The structural similarity is worth a Reflect line at the end of Level 12.4: both attention and FFN compute two intermediate tensors and combine them with a learned-or-fixed coupling — attention's coupling is data-dependent (the softmax weights), FFN's coupling is also data-dependent (the SiLU gate). The architectural insight is "this transformer's two sub-stages are structurally homologous."

---

# Phase 13 — Stitch the manual block and the full manual inference (detailed stub)

## Phase 13 — Stitch the manual block and the full manual inference

**Goal:** Replace the opaque `block.self_attn` and `block.mlp` calls inside `manual_block` (from Phase 7) with `manual_attention` (from Phase 11) and `manual_ffn` (from Phase 12). Then compose: embedding → block stack → final norm → LM head → slice → argmax → decode into `manual_inference(text)`.

**Milestone:** `manual_inference(text)` produces the same predicted token as `predict_one_token(text)` and `model.generate(..., max_new_tokens=1, do_sample=False)` for at least four different inputs spanning different lengths and domains.

**Builds on:** Phase 12 milestone — attention and FFN both fully open.

**Helpers stitched in this phase:** the final `manual_block(x, block, position_ids, mask)` (now fully manual inside); `manual_inference(text, model, tokenizer)`.

**Suggested level breakdown:**

- **Level 13.1 — Rewrite `manual_block` with no opaque submodule calls.**
  - Step: replace `block.self_attn(x_normed)[0]` with `manual_attention(x_normed, block.self_attn, position_ids, mask) @ block.self_attn.o_proj.weight.T`. (Or fold `o_proj` into `manual_attention`'s return — choose one convention and justify.)
  - Step: replace `block.mlp(y_normed)` with `manual_ffn(y_normed, block.mlp)`.
  - Step: verify on block 0 against `block(x_embed)[0]` within `atol=1e-4`.

- **Level 13.2 — Verify across all 16 blocks individually.**
  - Step: loop over blocks; per-block, run `manual_block` and compare to `block(x)[0]` where `x` is the corresponding hidden state.

- **Level 13.3 — Thread through the full stack.**
  - Step: run `manual_block` for all 16 blocks in sequence; compare final output to `hidden_states[-1]`. Tolerance may need to loosen to `1e-3` due to accumulated rounding across 16 blocks.

- **Level 13.4 — Compose `manual_inference`.**
  - Step: write the function as embedding → manual_block × 16 → manual_rmsnorm → manual_lm_head → last-slice → argmax → decode.
  - Step: verify on the canonical input matches `predict_one_token`.

- **Level 13.5 — Verify on multiple inputs.**
  - Step: pick four inputs of varying length and domain (e.g., "The capital of France is", "1 + 1 =", "Once upon a time", a longer multi-sentence input). For each, confirm `manual_inference(text) == predict_one_token(text)`.

- **Level 13.6 — Reflect on what is now true.**
  - End-of-phase milestone: state the full forward pass from text to predicted token, every operation named, every shape stated, every weight tensor named. Repeat from memory without notes.

---

# Phase 14 — Consolidation (detailed stub)

## Phase 14 — Consolidation

**Goal:** Verify durable understanding. This phase contains no new code. Its outputs are notes, drawings, and recall exercises.

**Milestone:** All three exercises in Section "End-of-track acceptance" can be completed from memory.

**Builds on:** Phase 13 milestone — full manual inference works.

**Helpers stitched:** none.

**Suggested levels:**

- **Level 14.1 — Tensor-shape recall.**
  - Without looking at the notebook or this file, write down every tensor shape from `input_ids` (entering the model) to the predicted token ID (leaving the model). Include shapes inside attention (Q, K, V before reshape, Q, K, V after reshape, K/V after GQA repeat, scores, weights, per-head output, concat) and inside FFN (gate, up, gate after SiLU, gated hidden, ffn out). Approximately 20 named shapes.

- **Level 14.2 — Weight-tensor recall.**
  - Without looking, write down every learned weight tensor by name and shape:
    - `embed_tokens.weight: [vocab_size, d]`
    - per block (× 16): `input_layernorm.weight: [d]`, `q_proj.weight: [d, d]`, `k_proj.weight: [n_kv*d_h, d]`, `v_proj.weight: [n_kv*d_h, d]`, `o_proj.weight: [d, d]`, `post_attention_layernorm.weight: [d]`, `gate_proj.weight: [d_ff, d]`, `up_proj.weight: [d_ff, d]`, `down_proj.weight: [d, d_ff]`
    - `final_norm.weight: [d]`
    - `lm_head.weight: [vocab_size, d]`
  - Then check.

- **Level 14.3 — Random-row drill.**
  - Pick three random rows from the original `llm_levels_02.md` table at level 49 (the full reference). For each, state: the operation, input shapes, output shape, static weights used, PyTorch module path. Check.
  - Repeat daily for one week. Tracking metric: number of rows answered correctly without notes on first try.

- **Level 14.4 — What-would-break drill.**
  - Pick any one phase. Without scrolling, write one paragraph: what would break if that phase were skipped? Repeat for two more phases.

- **Level 14.5 — Teaching test.**
  - Explain, in writing, to an imaginary reader who knows Python and linear algebra but not transformers: how does Llama 3.2 1B convert "The capital of France is" into the next token "Paris"? Aim for 800 words. Include shapes. The test of understanding is whether this can be written without scrolling.

**End-of-track acceptance:**

1. Without scrolling, list every operation from `input_ids` to predicted token, with shapes.
2. Without scrolling, list every learned weight tensor with shape.
3. Without scrolling, write one paragraph each on (a) why pre-norm not post-norm, (b) why GQA, (c) why scaled dot-product attention scales by `1/sqrt(d_h)`, (d) why RoPE applies to Q and K but not V.

---

# Section 6 — Instructions for future authors

> This section governs how new phases are written and how existing phases should be revised. Place these instructions *after* the phases — they are reference material to consult while writing, not a preface to read first. Each rule is followed by *what it rejects* — the failure mode it exists to prevent.

## 6.1 Step granularity

A step opens exactly one concept. The test: can the step's *result* be named with one noun? If yes, one step. If you find yourself writing "and" to describe a step's output, split.

- Good: "Compute `Q = x_normed @ W_Q.T`" — result is `Q`, one noun.
- Bad: "Compute `Q` and reshape into heads" — two results, two steps.
- Bad: "Load the model and call eval" — combine into one step framed as "instantiate the model in inference mode," or split if the eval call deserves its own observation.

**Rejects:** steps that hide one concept inside another.

## 6.2 Predict block

The Predict block is a falsifiable claim. It must say something specific enough that the Verify block can disprove it.

- Required content: at minimum, the expected output shape and dtype. Add value range if values matter. Add a comparison expectation if the live model produces the same tensor.
- Required position: before the Code block, mentally and on the page. If you write the Code first and then the Predict, you've turned a hypothesis into a transcript.

When the prediction is the same as the verification (e.g., for a pure config-extraction step like `d = config.hidden_size`), the Predict block may state the value but should add one sentence of *reasoning* — otherwise the step is ceremony.

**Rejects:** vague predictions, tautological predictions (Predict restates Verify), after-the-fact predictions.

## 6.3 Verify block — ground truth

Every step gets at least a shape/dtype assertion. Steps that produce a tensor the live model also produces get a *value* assertion in addition. Both, not one.

Ground truth, in priority order:

1. **Live submodule call** when the operation corresponds to a HuggingFace submodule (`block.self_attn.q_proj`, `block.input_layernorm`). Use `torch.allclose` at default tolerance.
2. **Forward hook capture** when the intermediate has no submodule exposing it (e.g., attention scores after softmax). The pattern is documented in Phase 7 Step 7.2.1.
3. **Structural property** (rows of a softmax sum to 1; RMSNorm output has unit per-row RMS) as a complement to 1 and 2, not a replacement.
4. **Hardcoded expected number** only for config values, and only with a fragility margin — never `assert eps == 1e-5` (exact float equality on an external constant); use `assert abs(eps - 1e-5) < 1e-9` or `assert eps < 1e-4`.

**Rejects:** assertions that only test what you typed; exact-float-equality against external constants; missing value comparisons when the live model could provide one.

## 6.4 Reflect block

One line. Answers *why*, not *what*. The line should be something a reader could disagree with — a claim, not a restatement.

- Good: "The scaling by `1/sqrt(d_h)` keeps the variance of `scores` independent of head dimension, so softmax does not saturate as `d_h` grows."
- Bad: "Q has shape `[N_tok, d]`." (restates the Verify block)
- Bad: "This text is chosen because the prediction is 'Paris'." (metadata about the choice, not a reflection on the operation)

If you cannot write a one-line Reflect, the step is not understood. Re-derive, or split into smaller steps.

**Rejects:** restatements of the Verify block; metadata about why the step exists; multi-sentence reflections that drift.

## 6.5 Black box discipline

At the start of every level, list the black boxes entering the level and the black boxes still deferred at the end. Annotate each deferred black box with the phase that opens it. If a level uses something that is not listed, the level is malformed.

**Rejects:** silent opacity.

## 6.6 Stitching discipline

When a piece is verified across multiple inputs or multiple instances, stitch it into a helper function in a "stitch" step. The stitch step's Predict is "the helper produces the same output as the manual derivation." The Verify is `torch.allclose` (or `torch.equal` if no arithmetic happens).

The helper signature must match how callers will invoke it. Inconsistency between the spec elsewhere in the file and the implementation is a bug.

**Rejects:** helpers tested only on one input; spec/implementation drift in arg lists.

## 6.7 End-of-level and end-of-phase checks

Every level ends with one short check. Every phase ends with a milestone check with at least three items:

1. A code-verifiable condition.
2. A second code-verifiable condition.
3. A from-memory recall exercise.

The "what would break if this phase were skipped" paragraph is the highest-value writing in the file.

**Rejects:** phases that end without a self-check; phases that end with "you understand it now."

## 6.8 Step numbering

`P.L.S`. Numbers within a level start at 1. New levels start at 1 within a phase. If a step is inserted between existing steps, use a letter suffix (`3.2.4a`) rather than renumbering downstream.

## 6.9 Cross-references

When referencing a future phase, state the *concept* and the *phase number*, in that order: "RoPE — opened in Phase 11," not "Phase 11 — RoPE." This keeps scan-reads forward-compatible if phase numbers shift.

## 6.10 Common failure modes (do not repeat these)

| Failure | Cause | Correct alternative |
|---|---|---|
| `torch.set_default_dtype(bfloat16)` | Breaks int operations like `torch.arange`. | Do not set a global default; load the model in bf16 and let int tensors stay int64. |
| `assert input_ids[0,0] == 128000` | Hardcoded BOS ID; some tokenizer configs don't add BOS. | `assert input_ids[0, 0].item() == tokenizer.bos_token_id`. |
| `assert eps == 1e-5` | Exact float-equality on an external constant. | `assert abs(eps - 1e-5) < 1e-9`. |
| `assert probs.sum() == pytest_approx(1.0)` | Undefined helper. | `assert abs(probs.sum().item() - 1.0) < 1e-4`. |
| Predict says `ms.shape == (1, N_tok)` but Verify checks `(1, N_tok, 1)` | Predict written without tracing `keepdim`. | Always trace shape changes through `keepdim` before writing Predict. |
| "Intentionally wrong" attention as a teaching device | Skim-readers internalize the wrong version. | Build single-head attention as a correct special case; later phases generalize. |
| Two H1 headers per phase | `# PHASE N` followed by `## Phase N` is redundant. | One header per phase. |
| Hardcoded `assert N_tok == 6` | Breaks any time the input text changes. | Compute `N_tok` from `input_ids.shape[1]`; assert only inside a Predict discussion of the canonical input. |
| Skipping the value assertion when the live model exposes the tensor | Only the document predicted the shape; the value could be wrong silently. | Always add the live-model value comparison when one is available. |
| Hardcoding RoPE table shape (`[max_pos, d_h]`) | HuggingFace versions vary; some compute on the fly. | Inspect `model.model.rotary_emb` in the kernel; read the actual return shape. |

## 6.11 Tone

- No motivational filler.
- One-line reflections are one line. If it grew to three, split the step.
- No parenthetical asides that turn one sentence into three.
- No "we" when "you" is meant.

## 6.12 Length budget per phase

Rough guideline, not a rule:

- Phases that open a single submodule (embedding, LM head, RMSNorm, block stack): 100–200 lines.
- Phases that open multi-operation structures (block opening, attention math, multi-head, RoPE, FFN): 300–500 lines.
- Phases that stitch and consolidate: 50–100 lines.

If a phase exceeds 600 lines, it is doing too much — split.

## 6.13 Patterns for attention/FFN-class phases (back-half phases)

The early phases (1–7) open one submodule at a time, each with simple input/output. The back half (Phases 8–13) opens multi-operation structures: attention math, multi-head reshape, GQA grouping, RoPE, SwiGLU. These have recurring patterns that should be written consistently across the file.

### 6.13.1 Reshape is a first-class operation

Any tensor reshape (`view`, `reshape`, `transpose`, `permute`) gets its own step. Do not combine "reshape and apply operation" into one step.

- Bad: "reshape Q to per-head and compute scores."
- Good: step N "reshape Q to `[1, n_h, N_tok, d_h]`"; step N+1 "compute scores `Q @ K.T`."

Reasoning: shape changes are where the most subtle bugs hide. Combining a reshape with the next operation hides whether the reshape was correct.

### 6.13.2 Every reshape step has an equivalence check

After any reshape, add a verification that the reshape is information-preserving — usually by slicing out one head's data after reshape and confirming it equals the same head's data before reshape.

Example:
```python
Q_4d = Q.view(1, N_tok, n_h, d_h).transpose(1, 2)   # [1, n_h, N_tok, d_h]
# equivalence check: head 0 of the 4d form equals the flat slice from the 2d form
assert torch.equal(Q_4d[0, 0], Q[0, :, 0:d_h])
```

### 6.13.3 GQA repeat is conceptually distinct from per-head reshape

These are two separate operations, two separate concepts, two separate steps:

1. Reshape K from `[1, N_tok, n_kv * d_h]` to `[1, n_kv, N_tok, d_h]`. (Per-head reshape.)
2. Repeat K along the head axis from `n_kv` to `n_h`. (GQA grouping.)

The first is a no-op semantically; the second introduces the asymmetry that GQA is built on. Combining them hides the GQA concept.

### 6.13.4 RoPE applies to Q and K, not V — state this explicitly

In Phase 11 (and anywhere RoPE is mentioned), the asymmetry (Q and K rotated; V not rotated) must be named explicitly in a Reflect line. The architectural reason: positional information should affect *which* tokens attend to which (the Q-K matmul) but not *what* they read (V is content, position-independent).

### 6.13.5 Multi-operation phases need an explicit gap-measurement step before the final stitch

Phases 9, 10, 11 each have a partial implementation that does not yet match the live model exactly (Phase 9 lacks RoPE; Phase 10 lacks RoPE; only Phase 11 closes the gap). At each phase boundary, add a step that *measures the gap to the live model* and names what is missing.

Example: Phase 9 Step 9.3.2 measures the gap and names it the "RoPE gap." Without this step, the reader is left with an unexplained mismatch.

### 6.13.6 Cross-block verification at the end of a multi-operation phase

After the final stitch of a phase that opens attention or FFN internals, add a step that runs the new helper across all 16 blocks and confirms zero mismatches. This catches block-specific bugs (e.g., one block having a slightly different head count) that single-block tests miss.

### 6.13.7 The `o_proj` boundary is the right hook point for attention verification

For attention phases, the reference target is almost always the input to `o_proj` (captured by hook). This is the only tensor in the live model that exposes the full pre-projection attention output. Hook it once per phase and reuse.

### 6.13.8 When the live model and the manual implementation use different conventions

Some operations (RoPE rotation, mask format, head ordering) have multiple valid conventions. HuggingFace uses one set of conventions; a textbook derivation may use another. When this happens:

1. Inspect the live model's convention by reading its returned tensor shapes and values, not by reading its source code.
2. Adopt the live model's convention for the manual implementation.
3. Add a Reflect line that names the alternative conventions and why they would also be correct.

### 6.13.9 Length budget for back-half phases is higher

Back-half phases run longer than front-half phases (200–500 lines vs 100–200). Resist the urge to compress. The cost of a hidden concept in attention is higher than the cost of a hidden concept in embedding.

## 6.14 Instructions for me, future-Claude

These are notes to my future self when asked to extend or revise this file.

### 6.14.1 Read the existing phases before writing new ones

When asked to flesh out a new phase or revise an existing one, read Phases 1–13 in full first. The conventions encoded in this file (template wording, header structure, Reflect tone, end-of-phase milestone format) are dense and easy to drift from if I work from memory.

### 6.14.2 The Predict block is the most-violated rule

Even when I know the rule, my own Predict blocks tend to drift toward tautology ("`d == 2048`" when the prior line said `d = config.hidden_size`). Before finalizing each step, re-read its Predict block and ask: is this a hypothesis the reader could falsify? If no, rewrite.

### 6.14.3 Every step's Verify must encode the Predict

The Verify block is the mechanical encoding of the Predict block. If the Predict says "shape `[1, N_tok, d]`," the Verify must include `assert tuple(x.shape) == (1, N_tok, d)`. If the Predict says "values match the live model within `atol=1e-4`," the Verify must include `torch.allclose(...)`. Predicts without matching Verifies are bugs.

### 6.14.4 When introducing a new helper, the Reflect line says *why this helper exists*

The Reflect line in a stitch step should not restate what the helper does (the Code block shows that). It should answer: what is the architectural insight that justifies wrapping these N operations into one named function? Examples:

- `manual_rmsnorm`: "Eight pedagogical sub-steps compress into four computational expressions — the eight existed for understanding; the four are sufficient for computation."
- `manual_block_stack`: "The helper exposes the architectural insight in two lines of code — depth is iteration."

### 6.14.5 Token 0 is the canonical sanity check for any position-dependent operation

If the operation depends on position (RoPE, causal masking, anything involving `position_ids`), token 0 is the position where the position-dependence reduces to identity. Use token-0 match as the structural sanity check before measuring the full-sequence gap.

### 6.14.6 The user will read the file in order

Do not place a forward reference whose absence would block a step. If Step P.L.S needs concept C, either C is opened in or before Phase P, or the step says "treat C as opaque; opens in Phase Q" explicitly.

### 6.14.7 When asked to "concentrate on" certain phases

If the user asks me to concentrate on phases X–Y, then:

- Fully flesh out phases X–Y with the six-block template.
- For any phase before X, refer to its existence and trust the user has read it (or that an earlier version of this file or a sibling file contains it).
- For any phase after Y, produce a "detailed stub" with: goal, milestone, reference target, suggested level breakdown (one paragraph per level), and a "patterns to follow" note pointing to Section 6.13.
- Do not produce a one-line stub. "Detailed stub" means at least 30–50 lines per phase, enough that a future author (or me, in a future turn) can fill it in without re-deriving the structure.

### 6.14.8 Future revisions should preserve numbering

If a future revision splits a step into two, prefer letter suffixes (`9.2.4a`, `9.2.4b`) over renumbering. This preserves cross-references in conversations and notes that reference specific step numbers.

### 6.14.9 The user prefers technical density over hand-holding

Reflect lines, milestone descriptions, and "what would break" paragraphs should be technical and specific, not motivational. Match the user's stated preferences: "no praise; no motivational language; explain failure modes; compare alternatives." If a sentence could appear in a marketing post or a self-help book, remove it.

### 6.14.10 If asked for a "skeleton" or "stub" version, do not be tempted to flesh

The user has been explicit about wanting different granularities for different parts of the file. Stubs are stubs; if I notice I am writing the full step template inside what was supposed to be a stub, stop and recognize the scope creep.

### 6.14.11 If the user reports a bug in the file, the fix is twofold

Fix the bug *and* add a row to Section 6.10 (common failure modes) so the same bug does not recur in a different phase. The failure-modes table is the institutional memory of this project.

---

*End of v3.*
