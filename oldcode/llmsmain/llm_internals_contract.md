# Learning Transformer LLM Internals — Master Contract

> This file is the single source of truth for the LLM internals learning project. It consolidates the goal, approach, decisions, acceptance criteria, design principles, and the full 19-phase roadmap. Read this file end-to-end before starting any session.

---

## 1. Goal

Build a complete mental model of how a transformer LLM converts input text into a single predicted token, by inspecting a real loaded model (Llama 3.2 1B Instruct) on EC2 — not by reading reference code.

The 50-level document `LLM Inference — 50-level Incremental Build` is the curriculum. Each level peels back one more layer of the architecture. Level 0 is "opaque box." Level 49 is "complete reference." The end state: be able to point to every tensor, every static weight, and every operation by name and by location in the loaded model.

The constraint: CPU-only inference, one model, one input, one predicted token. No KV cache, no batching, no training. This is intentional — those features add complexity that distracts from the core architecture.

---

## 2. What I'm trying to do

Learn how a transformer LLM works internally by loading a real model on an EC2 instance and inspecting it step by step. The work follows a 19-phase roadmap that maps to the 50-level document. The deliverable is a Jupyter notebook that, by the end, reimplements one-token inference for Llama 3.2 1B from raw weight tensors and elementary PyTorch operations, with no opaque module calls.

---

## 3. Approach — incremental construction with the document as a contract

Each level is a small, testable hypothesis: "if I add this one new step, this one new tensor shape should appear, with this specific value." Write the code, run it, check the shape. If it matches, the level is understood — because the result was produced manually. If it does not match, the bug is isolated to one concept, not 50.

The transformer core is opened progressively. Black boxes are allowed and explicit — a step can introduce a piece without immediately opening it. The piece stays opaque until a later phase opens it.

Stitching happens *as you go*, not at the end. When a piece is fully understood, it is wrapped into a small helper function. Later phases call the helper without re-deriving. This keeps cognitive load bounded as the work progresses.

### Why this approach (not alternatives)

- **Not "dump complete code, run it, study it":** When 50 concepts are present from the start, you cannot tell which concept caused which behavior. Errors cannot be localized.
- **Not "read reference code line by line":** Passive learning. Real reference code (HuggingFace `modeling_llama.py`) is full of optimizations and abstractions that obscure the architecture.
- **Not "build from scratch in pure PyTorch (no HuggingFace)":** Doubles the work. You would be debugging both your understanding and your implementation choices. Using HuggingFace lets you observe a known-correct model, so any mismatch is in your understanding, not the model.
- **Not "skip directly to attention/RoPE/GQA":** Most beginners do this. They never understand the boring outer pipeline (tokenize, embed, LM head, decode), and later, when something breaks at inference time, they have no idea how to debug.

---

## 4. Decisions already made

These decisions are settled. Do not re-debate them.

| # | Decision | Choice |
|---|---|---|
| 1 | Where to run | AWS EC2 (not local — laptop RAM constrained by Chrome + VS Code) |
| 2 | Instance type | t3.xlarge (16 GB RAM, 4 vCPU) |
| 3 | CPU or GPU | CPU-only (speed not the goal) |
| 4 | Region | ap-south-1 (Mumbai) |
| 5 | OS | Ubuntu 22.04 LTS, plain (not Deep Learning AMI) |
| 6 | Storage | EBS gp3, 30 GB |
| 7 | Pricing model | On-demand |
| 8 | Stop policy | Stop instance when not in use |
| 9 | Billing safety | Billing alarm at $20 |
| 10 | Access method | VS Code Remote-SSH (Jupyter runs in VS Code, no port forwarding needed) |
| 11 | Model to load | Llama 3.2 1B Instruct (HuggingFace, requires HF login) |
| 12 | Model precision | fp16 or bf16 |
| 13 | Learning approach | Inspect real model — NOT read from-scratch reference code |
| 14 | Inspection tool | PyTorch forward hooks + print + manual computation + assert |
| 15 | Library | HuggingFace transformers |
| 16 | Skip for now | KV cache, batching, sampling beyond argmax, training, MoE, RLHF, chain-of-thought |

### Hardware constraint (why local was rejected)

Laptop: i3-1215U, 16 GB RAM total but only ~6 GB free (Chrome + VS Code on company-managed laptop, no control). Intel UHD GPU — not usable for ML.

### Cost expectation

~$15–20/month if disciplined about stopping the instance. ~$130/month if left running 24/7.

### Accounts ready

- AWS account: yes
- HuggingFace account: yes (logged in on EC2)

---

## 5. Acceptance criteria

The goal is achieved when, for any level in the 50-level document, the following five things can be done without notes:

1. **Name the operation** — say what mathematical operation that step performs (matmul, RMSNorm, softmax, elementwise multiply, RoPE rotation, etc.).
2. **Name the tensors** — say what tensor goes in, what tensor comes out, and what the shapes are in terms of the size variables (`N_tok`, `d`, `d_h`, `n_h`, `n_kv`, `d_ff`, `vocab_size`).
3. **Name the static data** — say which learned weights or pre-built tables that step uses, and their shapes.
4. **Justify the shape change** (if any) — explain why the output shape differs from the input shape, in terms of what the operation is doing.
5. **Locate it in the real model** — point to the actual PyTorch module path (e.g., `model.model.layers[5].self_attn.q_proj`) inside the loaded Llama model that corresponds to that step.

### Per-phase milestones

- **End of Phase 6:** A full inference can be run manually — tokenize, embed, call core, final norm, LM head, last logits, softmax, argmax, decode — without using `model.generate()`.
- **End of Phase 7:** All 16 blocks listed; each preserves shape `[N_tok × d]`; `manual_core_loop` reproduces the model's output.
- **End of Phase 14:** `manual_attention(x_normed, block)` reproduces `block.self_attn(x_normed)` exactly within `1e-4` tolerance.
- **End of Phase 16:** `manual_ffn(y_normed, block)` reproduces `block.mlp(y_normed)` exactly.
- **End of Phase 18:** `manual_inference(text)` produces the same predicted token as `model.generate()`, with no opaque module calls anywhere in the manual function.
- **End of Phase 19:** The five acceptance criteria above can be done from memory.

### Out of scope

These are explicitly out of scope. If they appear in the model code, note and move on:

- KV cache (used for fast generation of multiple tokens — irrelevant to one-token inference)
- Batching (always batch size 1)
- Sampling strategies beyond argmax (temperature, top-k, top-p)
- Training, backward pass, gradients, optimizers
- MoE, RLHF, chain-of-thought reasoning

---

## 6. Design principles

These rules apply to every step. No exceptions.

1. **One concept per step.** Never two. If a step has two, it splits.

2. **Every step produces an observable result.** A printed shape, a printed value, a comparison between two tensors, or a successful assertion. Never just theory.

3. **Every step verifies against the previous step or the document.** Nothing is taken on trust. Either the new tensor shape matches what the document predicts, or the new code reproduces a result the model already produced.

4. **Black boxes are declared explicitly.** When a step introduces a piece without opening it, the step says: "we treat X as a black box; will open in Phase N." This makes deferred knowledge visible.

5. **Stitch as you go.** When a piece is fully understood, immediately wrap it into a helper function. Later phases call the helper without re-deriving it. Complexity is wrapped off the table the moment it is no longer needed for learning.

6. **Micro-checks are their own steps.** Inside each phase, there are explicit "stop and verify your understanding" steps with predict-then-confirm exercises.

7. **Mandatory end-of-phase self-check.** Without scrolling up, write down the input/output tensor shapes for that phase. If you cannot, re-read the phase before moving on. This rule has no exceptions.

8. **The notebook is the deliverable.** At the end, the notebook reads top-to-bottom as a complete record of understanding, with no external help needed.

---

## 7. Roadmap — 19 phases

### Phase 1 — Foundation: load the model

**Goal:** Get the model loaded. Read its config. Map the document's abstract sizes (`d`, `N`, `n_h`, `n_kv`, `d_h`, `vocab_size`, `d_ff`, `max_pos`) to concrete numbers.

| Step | What you do | New concept | Black boxes |
|---|---|---|---|
| 1.1 | Load tokenizer and model from HuggingFace. | Loading a model is one Python call; the result is a nested PyTorch module. | Everything inside the model. |
| 1.2 | Print `model.config`. Map `hidden_size → d`, `num_hidden_layers → N`, `num_attention_heads → n_h`, `num_key_value_heads → n_kv`, `vocab_size`, `intermediate_size → d_ff`, `max_position_embeddings → max_pos`. Compute `d_h = d / n_h`. Write each value down as a markdown cell. | The document's symbols are concrete numbers. | The role of each — only names mapped. |
| 1.3 **(micro-check)** | Without looking at config: predict on paper what `d` × `n_h` should equal, what `d_h` × `n_kv` represents, and whether `d_ff > d`. Then run code to verify. | Tests internalization of size relationships. | — |
| 1.4 | Print `model` (the whole module tree). Identify the embedding module, the layer list, the final norm, the LM head. | The outer architecture matches the document's outer pipeline. | Layer internals. |
| 1.5 | Tokenize an input string. Print `input_ids` and its shape. | Text becomes integer IDs. Shape is `[1, N_tok]`. | BPE algorithm. |
| 1.6 | Run `model.generate()` to produce one new token. Decode it. Print the predicted text. | The opaque pipeline works end-to-end. | Everything. |

**End-of-phase self-check:** Without scrolling, write down — what is `d`, `N`, `n_h`, `n_kv`, `d_h`, `vocab_size`, `d_ff`, `max_pos` for this model? What is the shape of `input_ids` for input `"The capital of France is"`?

---

### Phase 2 — Replace `generate()` with explicit five-step pipeline

**Goal:** Stop using `model.generate()`. Drive each outer step yourself: tokenize → call model → take last logits → softmax → argmax → decode.

| Step | What you do | New concept | Black boxes |
|---|---|---|---|
| 2.1 | Call `model(input_ids)` directly. Inspect the returned object's keys. Find `logits`. Print its shape. Confirm `[1, N_tok, vocab_size]`. | The model's raw output is `logits`. One row per input token. | What computed those logits. |
| 2.2 **(micro-check)** | Predict on paper: shape of `logits` for 5 tokens, 1 token, 100 tokens. Run with three different inputs to confirm. | Confirms `logits` shape depends on input length. | — |
| 2.3 | Slice last position: `last_logits = logits[0, -1, :]`. Print shape. Confirm `[vocab_size]`. | Only the last position predicts the next token. | Why earlier positions still produce logits (clear in Phase 5). |
| 2.4 | Apply `torch.softmax(last_logits, dim=-1)`. Assert sum = `1.0` within tolerance. Print top 5 (probability, token_id) pairs and decode each. | Softmax converts logits to probability distribution. | Numerical math of softmax. |
| 2.5 | Apply `torch.argmax(probs)`. Get token ID. Decode. Confirm matches Phase 1.6. | The full explicit pipeline reproduces `generate()`. | The transformer core. |
| 2.6 **(stitch)** | Wrap the five steps into helper `predict_one_token(text, model, tokenizer)`. Test. Confirm output equals 2.5. | The outer pipeline is now a one-line call. | — |

**End-of-phase self-check:** Without scrolling, list the 5 steps the helper performs, with input/output shape of each. Predict what `predict_one_token("Hello", model, tokenizer)` returns. Run to confirm.

---

### Phase 3 — Open the embedding step

**Goal:** Open the first opaque step in the pipeline. The embedding module is a row-lookup table `{E}`.

| Step | What you do | New concept | Black boxes |
|---|---|---|---|
| 3.1 | Find `model.model.embed_tokens`. Print type (`nn.Embedding`). Print weight shape. Confirm `[vocab_size, d]`. | Embedding is a learned matrix `{E}`, one row per token. | Why values are what they are. |
| 3.2 **(micro-check)** | Predict on paper: shape of embedded output for 4 tokens; for 10 tokens. Confirm by calling `model.model.embed_tokens(input_ids)` for two inputs. | Confirms embedding output shape understanding. | — |
| 3.3 | Manually do the lookup: `embedded = model.model.embed_tokens.weight[input_ids[0]]`. Compare against `model.model.embed_tokens(input_ids[0])`. Assert equal. | Embedding is just row indexing. | — |
| 3.4 | Take any token ID (e.g., 1234). Look up its embedding row. Print first 5 values. Note small magnitudes. | Embedding rows are real numbers, not abstractions. | What semantics these numbers represent. |

**End-of-phase self-check:** Without scrolling — shape of `{E}`? What does row `i` represent? Input/output shape of the embed step?

---

### Phase 4 — Open the LM head step

**Goal:** Open the last opaque step before `pick`. The LM head is a linear projection from `d` to `vocab_size`.

| Step | What you do | New concept | Black boxes |
|---|---|---|---|
| 4.1 | Find `model.lm_head`. Print type (`nn.Linear`). Print weight shape. Note: PyTorch stores `nn.Linear` weights as `[out, in]` = `[vocab_size, d]`. Document's `{H}` is `[d × vocab_size]` — same matrix, transposed convention. | Different conventions, same matrix. | — |
| 4.2 | Check if `model.lm_head.weight` is the same tensor as `model.model.embed_tokens.weight`. Use `id(...)` or `.data_ptr()` for identity. Llama 3.2 1B uses **weight tying** — same tensor, used twice. | Weight tying: `{E}` and `{H}` are sometimes the same matrix. | The mathematical reason this works. |
| 4.3 **(micro-check)** | Predict: `hidden_states` shape `[1, N_tok, d]` → `lm_head` → ? Confirm. | Tests matmul shape composition. | — |
| 4.4 | Manually compute logits: take a `[1, N_tok, d]` hidden tensor, apply `lm_head(hidden)`. Compare against actual `logits` from Phase 2.1. Assert equal. | LM head under your control. | — |

**End-of-phase self-check:** Without scrolling — shape of `{H}` in document convention? Shape of `lm_head.weight` in PyTorch convention? What is weight tying?

---

### Phase 5 — Open the final norm

**Goal:** Open the RMSNorm between the core and the LM head.

| Step | What you do | New concept | Black boxes |
|---|---|---|---|
| 5.1 | Find `model.model.norm`. Print type (`LlamaRMSNorm`). Print weight shape. Confirm `[d]`. | Final norm has one learned scale vector. | The math. |
| 5.2 | Read RMSNorm formula: for each row `x` of length `d`, `rms = sqrt(mean(x^2) + eps)`, then `(x / rms) * weight`. No mean subtraction (unlike LayerNorm), no learned bias. | RMSNorm normalizes by RMS only. Cheaper than LayerNorm. | Why this works as well as LayerNorm. |
| 5.3 | Manually implement RMSNorm in 3 lines: square, mean over last axis, sqrt + eps, divide, multiply by weight. Apply to a test tensor. | RMSNorm from scratch. | — |
| 5.4 | Take a `[1, N_tok, d]` hidden tensor. Apply manual RMSNorm. Compare against `model.model.norm(hidden)`. Assert equal within `1e-5`. | Manual reproduces model. | — |
| 5.5 **(stitch)** | Wrap into helper `manual_rmsnorm(x, norm_module)`. Reused in many later phases. | One-line helper. | — |

**End-of-phase self-check:** Without scrolling — RMSNorm formula in plain words. Shape of `{final_norm}`? Input/output shape of final norm step?

---

### Phase 6 — Drive the outer pipeline manually, end-to-end

**Goal:** Use Phases 3–5 helpers together. Replace `model(input_ids)` with manual: embed → call core → final_norm → LM head → last row → softmax → argmax → decode. Core is still a single black box.

| Step | What you do | New concept | Black boxes |
|---|---|---|---|
| 6.1 | Build manual pipeline: tokenize → embed (manual) → run all layers via single call to layer stack (still opaque) → manual_rmsnorm → matmul with `lm_head.weight.T` → take last row → softmax → argmax → decode. | Outer pipeline fully under your control. Core is opaque but boundary is exact. | Each transformer block. |
| 6.2 | Compare manual pipeline's token against `predict_one_token` (Phase 2.6) and `generate()` (Phase 1.6). All three must match. Assert. | Three independent paths give same result. | — |
| 6.3 **(stitch)** | Wrap into `manual_outer_pipeline(text, model, tokenizer)`. | Outer pipeline complexity wrapped off. | — |

**End-of-phase self-check:** Without scrolling, list every tensor flowing through `manual_outer_pipeline`, with shape: `input_ids`, `embedded`, `core_output`, `normed`, `logits`, `last_logits`, `probs`, `predicted_id`.

---

### Phase 7 — The core is N blocks

**Goal:** Open the layer stack. Confirm `N` independent blocks. Each block is still a black box internally.

| Step | What you do | New concept | Black boxes |
|---|---|---|---|
| 7.1 | Print `len(model.model.layers)`. Confirm equals `N`. Print `type(layers[0])`. Confirm all 16 same class. | Core is `N` blocks of same type, independent weights. | What a block does. |
| 7.2 **(micro-check)** | Predict: do `layers[0]` and `layers[5]` share weights? Confirm by comparing `id(layers[0].self_attn.q_proj.weight)` vs `id(layers[5].self_attn.q_proj.weight)`. | Confirms weight independence. | — |
| 7.3 | Attach forward hooks on every block. Capture input/output shape. Run forward. Print all 16. Confirm each is `[1, N_tok, d]` → `[1, N_tok, d]`. | Every block preserves `[N_tok, d]`. | What changes inside. |
| 7.4 | Manually loop: `x = embed(input_ids); for block in layers: x = block(x)`. Apply final norm + LM head. Confirm matches `manual_outer_pipeline`. | Drive the block loop yourself. | Block internals. |
| 7.5 **(stitch)** | Wrap into `manual_core_loop(embedded, model)`. | Core loop wrapped. | — |

**End-of-phase self-check:** Without scrolling — how many blocks? Input shape to each? Output shape? Do blocks share weights?

---

### Phase 8 — Crack open one block: identify the four children

**Goal:** Look inside `layers[0]`. Identify the four named children. Both attention and FFN are still opaque internally. From here on, all work is on block 0.

| Step | What you do | New concept | Black boxes |
|---|---|---|---|
| 8.1 | Print `layers[0]`. Identify 4 children: `input_layernorm`, `self_attn`, `post_attention_layernorm`, `mlp`. Note: `post_attention_layernorm` is a misleading name — it is actually the *pre-FFN* norm. | Block has 4 children. The naming is historical. | What attention and FFN do. |
| 8.2 **(micro-check)** | Predict: `input_layernorm` and `post_attention_layernorm` are both RMSNorm. Shape of each `.weight`? Confirm. | Both norms have weight shape `[d]`. | — |
| 8.3 | Hook the four children separately. Capture input/output shape. Run forward. Confirm each is `[1, N_tok, d]` → `[1, N_tok, d]`. | All four children preserve shape `[N_tok, d]`. | Internal transformations. |

**End-of-phase self-check:** Without scrolling — name 4 children in execution order. Input/output shape of each. Note the historical mis-naming.

---

### Phase 9 — Residual + pre-norm pattern (around attention)

**Goal:** Manually reproduce: save x → pre-norm → attention → add residual.

| Step | What you do | New concept | Black boxes |
|---|---|---|---|
| 9.1 | Take input `x` to block 0 (capture via hook). Apply `manual_rmsnorm(x, layers[0].input_layernorm)`. Compare against `layers[0].input_layernorm(x)`. Assert equal. | Pre-attention norm = `manual_rmsnorm`. | — |
| 9.2 | Call `attn_out = layers[0].self_attn(x_normed)` (still black box). Print shape. Confirm `[1, N_tok, d]`. | `self_attn` takes normalized input. Output is the *delta* to add to x. | What `self_attn` does internally. |
| 9.3 | Compute `result = x + attn_out`. | Residual: attention is added to x, not used as a replacement. | — |
| 9.4 **(micro-check)** | Predict: which is bigger in magnitude — `x` or `attn_out`? (Usually `x`.) Confirm with `x.norm()` and `attn_out.norm()`. | Confirms attention is a small *delta*. | — |
| 9.5 | Hook block to capture actual intermediate after attention residual. Compare manual `result` against captured. Assert equal. | Manual residual reproduces the model. | — |
| 9.6 **(stitch)** | Wrap into `manual_attn_substage(x, block)`: norm, call `self_attn` (opaque), add residual. | Wrapping is a one-line call. | `self_attn` internals. |

**End-of-phase self-check:** Without scrolling — write 3 steps of the attention sub-stage. Input shape? Output shape? Role of residual?

---

### Phase 10 — Residual + pre-norm pattern (around FFN)

**Goal:** Same as Phase 9, but for the FFN sub-stage.

| Step | What you do | New concept | Black boxes |
|---|---|---|---|
| 10.1 | Take FFN input `y` (output of Phase 9.5). Apply `manual_rmsnorm(y, layers[0].post_attention_layernorm)`. Compare. Assert equal. | Pre-FFN norm uses same helper. | — |
| 10.2 | Call `ffn_out = layers[0].mlp(y_normed)` (still black box). Print shape. Confirm `[1, N_tok, d]`. | FFN takes normalized input, outputs delta. | What `mlp` does internally. |
| 10.3 | Compute `block_out = y + ffn_out`. Compare against `layers[0](x)` for the original `x`. Assert equal. | Full block output reproduced. | Attention and FFN internals. |
| 10.4 **(stitch)** | Wrap into `manual_ffn_substage(y, block)`. | One-line call. | `mlp` internals. |
| 10.5 **(stitch, big)** | Wrap into `manual_block(x, block)` calling `manual_attn_substage` then `manual_ffn_substage`. Test against `block(x)` for all 16 blocks. Assert equal. | Block wrapped at residual+norm level. | `self_attn` and `mlp` internals. |

**End-of-phase self-check:** Without scrolling — write full block structure: save x, pre-attn norm, attention, residual, save y, pre-FFN norm, FFN, residual. Input/output shape of each.

---

### Phase 11 — Open `self_attn`: identify Q, K, V, O projections

**Goal:** Look inside `self_attn`. Identify the four projection sub-modules.

| Step | What you do | New concept | Black boxes |
|---|---|---|---|
| 11.1 | Print `layers[0].self_attn`. Identify `q_proj`, `k_proj`, `v_proj`, `o_proj`. All `nn.Linear`. | Attention has 4 linear projections. | What sits between Q/K/V and `o_proj`. |
| 11.2 | Print weight shapes: `q_proj.weight` `[d, d]`; `k_proj.weight` `[n_kv·d_h, d]`; `v_proj.weight` `[n_kv·d_h, d]`; `o_proj.weight` `[d, d]`. K and V have *fewer* output features than Q. This is GQA. | Shape asymmetry encodes GQA. | The mechanism by which `n_h` Q heads share `n_kv` K/V heads. |
| 11.3 **(micro-check)** | Predict: for `n_h=32`, `n_kv=8`, `d_h=64`, what should `k_proj.weight` shape be? Compute `n_kv·d_h = 512`. So `[512, 2048]`. Confirm. | Derive shape from document symbols. | — |
| 11.4 | Manually compute Q, K, V from `x_normed`: `Q = q_proj(x_normed)`, `K = k_proj(x_normed)`, `V = v_proj(x_normed)`. Print shapes. Confirm Q `[1, N_tok, d]`, K and V `[1, N_tok, n_kv·d_h]`. | Three independent matmuls. | The attention op. |

**End-of-phase self-check:** Without scrolling — name 4 projections in `self_attn`. Each weight shape. Output shape of Q, K, V given input `[1, N_tok, d]`.

---

### Phase 12 — Single-head simplified attention

**Goal:** Understand the five attention sub-steps (scores, scale, mask, softmax, weighted sum) using the *wrong* simplification of treating Q/K/V as single-headed. This produces output that does NOT match the model. Intentional — fixed in Phases 13 and 14.

> **Important note:** Phases 12–14 are the only phases where intermediate code does not match the model. Build attention in 3 stages: simplified (12), multi-head + GQA (13), RoPE (14). After Phase 14 it matches.

| Step | What you do | New concept | Black boxes |
|---|---|---|---|
| 12.1 | Compute `scores = Q @ K.transpose(-2, -1)`. Print shape. Confirm `[1, N_tok, N_tok]`. (Wrong — no heads — but instructive.) | Attention starts with similarity matrix between every query and every key position. | Why heads are needed (Phase 13). |
| 12.2 **(micro-check)** | Predict: `scores[0, i, j]` represents what? (Similarity between query at i and key at j.) | Semantic understanding of score matrix. | — |
| 12.3 | Divide `scores` by `sqrt(d_h)`. Print before/after. Document reasoning: scale prevents softmax saturation. | Scale is numerical fix, not learned. | Exact failure mode of unscaled softmax. |
| 12.4 | Build causal mask: lower-triangular `[N_tok, N_tok]`, `0` on/below diagonal, `-inf` above. Use `torch.triu(torch.full((N_tok, N_tok), -inf), diagonal=1)`. Print for 4 tokens. | Mask shape `[N_tok, N_tok]` (no head dim). Added to scaled scores. | Why future positions get `-inf`. |
| 12.5 | Add mask to `scaled_scores` to get `masked_scores`. Print one row to see `-inf` in upper triangle. | Future positions are now `-inf`. | — |
| 12.6 | Apply softmax over last axis: `attn_weights = torch.softmax(masked_scores, dim=-1)`. Confirm rows sum to 1. Rows with `-inf` entries become 0 there. | Softmax → row-wise probability distributions. | — |
| 12.7 **(micro-check)** | Predict: position 0 attends to how many keys? Position 5? Confirm by printing weights for 6-token input. | Confirms causal masking. | — |
| 12.8 | Compute `inner_result = attn_weights @ V`. Print shape — `[1, N_tok, n_kv·d_h]` (wrong but math works). Apply `o_proj`. Compare against actual `self_attn` output. **Expect mismatch.** | Single-head simplified attention computable. Mismatch fixed in 13 and 14. | Why mismatch — addressed next. |

**End-of-phase self-check:** Without scrolling — 5 sub-steps of attention op in order. Input/output shape of each. Acknowledge: produced wrong answer on purpose; what fixes it?

---

### Phase 13 — Multi-head splitting and GQA

**Goal:** Add per-head reshape. Attention will still NOT match the model — RoPE is missing. After Phase 14 it matches.

| Step | What you do | New concept | Black boxes |
|---|---|---|---|
| 13.1 | Reshape Q from `[1, N_tok, d]` to `[1, N_tok, n_h, d_h]`, then transpose to `[1, n_h, N_tok, d_h]`. Order matters: reshape splits last dim, transpose moves head axis forward. | Q has `n_h` heads. `d` is sliced into `n_h` chunks of `d_h`. | Why splitting heads improves expressiveness. |
| 13.2 | Reshape K from `[1, N_tok, n_kv·d_h]` to `[1, N_tok, n_kv, d_h]` to `[1, n_kv, N_tok, d_h]`. Same for V. Note `n_kv` heads, not `n_h`. | K and V have fewer heads than Q. This is GQA. | The repeat mechanism. |
| 13.3 | Repeat each K head `n_h / n_kv` times along head axis: `[1, n_kv, N_tok, d_h]` → `[1, n_h, N_tok, d_h]`. Use `K.repeat_interleave(n_h // n_kv, dim=1)`. Same for V. | GQA: each K/V head serves a *group* of Q heads. Repeat implements the group sharing. | Memory efficiency reasoning. |
| 13.4 **(micro-check)** | Predict: for `n_h=32`, `n_kv=8`, how many Q heads per K/V head? (32/8 = 4.) Confirm with `n_h // n_kv`. | GQA group ratio. | — |
| 13.5 | Redo Phase 12 sub-steps with per-head tensors: scores `[1, n_h, N_tok, N_tok]`, scale, broadcast mask to `[1, 1, N_tok, N_tok]`, softmax, weighted sum `[1, n_h, N_tok, d_h]`. | All 5 sub-steps run independently per head. | RoPE — Phase 14. |
| 13.6 | Concatenate per-head results: transpose `[1, n_h, N_tok, d_h]` → `[1, N_tok, n_h, d_h]`, reshape to `[1, N_tok, d]`. Apply `o_proj`. Compare against model. **Still expect mismatch — RoPE missing.** | Concat = transpose + reshape. After concat, `o_proj` mixes across heads. | RoPE. |

**End-of-phase self-check:** Without scrolling — per-head shape of Q? K? Why does K need `repeat_interleave`? After this phase, why does manual attention still not match?

---

### Phase 14 — RoPE on Q and K

**Goal:** Add position-dependent rotation. After this phase, manual attention matches the model exactly.

| Step | What you do | New concept | Black boxes |
|---|---|---|---|
| 14.1 | Find RoPE module. In Llama 3.2: `model.model.rotary_emb`. Call with `(value_tensor, position_ids)` where `position_ids = torch.arange(N_tok).unsqueeze(0)`. Returns `(cos, sin)` tables. Print shapes — typically `[1, N_tok, d_h]` each. | RoPE is precomputed, not learned. Sinusoidal frequencies, no weights. | Why these specific frequencies. |
| 14.2 | Read rotation formula. For each position, RoPE rotates pairs of dimensions in Q/K by an angle that depends on position. Rotate-half trick: split last dim into two halves, swap and negate, combine with cos/sin. Code: `(x * cos) + (rotate_half(x) * sin)` where `rotate_half(x)` swaps and negates the two halves. | Elementwise rotation, no matmul. Magnitudes preserved. | Trigonometric derivation. |
| 14.3 | Apply RoPE to Q (per-head, after split): `Q_rot = (Q * cos) + (rotate_half(Q) * sin)`. Print Q before/after. Confirm per-row magnitudes preserved within tolerance. | RoPE rotates in place; magnitudes survive. | Why it encodes relative position. |
| 14.4 | Apply RoPE to K (per-head, after split). Note: V is *not* rotated. | K rotated; V not. Position info enters via Q·K^T scores. Weighted sum over V carries no position bias. | Theory of why this works. |
| 14.5 **(micro-check)** | Predict: rotate Q at position 5 by some angle, rotate K at position 5 by the same angle. Does Q·K change? (No — rotations cancel when both rotate by the same angle.) | Tests relative-position intuition. | — |
| 14.6 | Redo full attention computation with rotated Q, K. Compare against `self_attn` output. **Assert equal within `1e-4`.** Manual attention matches the model. | Attention fully reproduced. | Nothing — attention is open. |
| 14.7 **(stitch)** | Wrap into `manual_attention(x_normed, block)` that does everything: Q/K/V, split heads, RoPE, repeat K/V, scores, scale, mask, softmax, weighted sum, concat, o_proj. Test against `block.self_attn(x_normed)` for all 16 blocks. Assert equal. | Attention is a one-line call. | — |

**End-of-phase self-check:** Without scrolling — what does RoPE rotate? What does it not rotate? Why is V not rotated? At what step does RoPE happen?

---

### Phase 15 — Open `mlp`: identify gate, up, down

**Goal:** Look inside the FFN. Identify the three projections.

| Step | What you do | New concept | Black boxes |
|---|---|---|---|
| 15.1 | Print `layers[0].mlp`. Identify `gate_proj`, `up_proj`, `down_proj`. All `nn.Linear`. | FFN has 3 projections. Gate and up are parallel; down comes after. | Why three. |
| 15.2 | Print weights: `gate_proj.weight` `[d_ff, d]`, `up_proj.weight` `[d_ff, d]`, `down_proj.weight` `[d, d_ff]`. | Gate and up share input/output shape. Down inverts dimensions. | Role of each. |
| 15.3 **(micro-check)** | Predict: for `d=2048`, `d_ff=8192`, what are 3 weight shapes? Confirm. | Dimensional reasoning for FFN. | — |

**End-of-phase self-check:** Without scrolling — name 3 projections in `mlp`. Each weight shape. Which are parallel, which is sequential?

---

### Phase 16 — Build the SwiGLU FFN manually

**Goal:** Implement the FFN from scratch. After this phase, manual FFN matches the model exactly.

| Step | What you do | New concept | Black boxes |
|---|---|---|---|
| 16.1 | Compute `gate = gate_proj(y_normed)` and `up = up_proj(y_normed)`. Both `[1, N_tok, d_ff]`. | Two parallel projections lift `d` to `d_ff`. | Why parallel. |
| 16.2 | Apply SiLU to `gate`: `gate_act = gate * torch.sigmoid(gate)`. Note `up` unchanged. | Only gate is activated. SiLU = "Sigmoid Linear Unit" = `x * sigmoid(x)`. | Why SiLU specifically. |
| 16.3 **(micro-check)** | Predict: SiLU(0)? (0.) SiLU(2)? (~1.76, since `2 * sigmoid(2) = 2 * 0.88 = 1.76`.) Verify. | Confirms SiLU shape understanding. | — |
| 16.4 | Compute `hidden = gate_act * up` (elementwise). Print shape `[1, N_tok, d_ff]`. | Elementwise product is the gating: `gate_act` modulates each dim of `up`. SwiGLU = "Swish-Gated Linear Unit". | Theoretical motivation. |
| 16.5 | Apply `down_proj(hidden)` → `[1, N_tok, d]`. Compare against `layers[0].mlp(y_normed)`. Assert equal. | FFN reproduced. | — |
| 16.6 **(stitch)** | Wrap into `manual_ffn(y_normed, block)`. Test against `block.mlp(y_normed)` for all 16 blocks. Assert equal. | FFN is one-line call. | — |

**End-of-phase self-check:** Without scrolling — 4 internal steps of SwiGLU FFN. Shape of each intermediate tensor. Why is gate activated but up is not?

---

### Phase 17 — Stitch attention and FFN into a manual block

**Goal:** Replace opaque calls inside `manual_block` with `manual_attention` and `manual_ffn`. Block now built from raw tensors and elementary ops, no opaque calls.

| Step | What you do | New concept | Black boxes |
|---|---|---|---|
| 17.1 | Update Phase 10.5 helper: `manual_block(x, block)` now uses `manual_rmsnorm + manual_attention + residual + manual_rmsnorm + manual_ffn + residual`. No call to `block.self_attn` or `block.mlp`. | Block fully manual. | Nothing. |
| 17.2 | For block 0, run `manual_block(x, layers[0])`. Compare against `layers[0](x)`. Assert equal. | One block has zero black boxes. | — |
| 17.3 | Repeat 17.2 for all 16 blocks (loop). Assert equal for every block. | Implementation generalizes — same math, different weights. | — |

**End-of-phase self-check:** Without scrolling — steps inside `manual_block` from input `x` to output. Input/output shapes. Confirm: no opaque PyTorch module called inside the block.

---

### Phase 18 — Stitch into manual full inference

**Goal:** Combine `manual_block` × 16 with the outer pipeline. End-to-end one-token inference with no opaque module calls anywhere.

| Step | What you do | New concept | Black boxes |
|---|---|---|---|
| 18.1 | Write `manual_inference(text, model, tokenizer)`: tokenize → embed (manual row lookup) → loop `manual_block(x, layers[i])` for all 16 blocks → manual_rmsnorm with `model.model.norm` → matmul with `lm_head.weight.T` → take last row → softmax → argmax → decode. | Full pipeline from raw tensors. | None. |
| 18.2 | Compare `manual_inference(text)` against `model.generate()` and `predict_one_token(text)` for several inputs. Assert all three match. | Full pipeline verified end-to-end. | — |

**End-of-phase self-check:** Without scrolling — list every step of `manual_inference`, in order. Input/output shapes. Confirm: no call to `model(x)`, `model.model(x)`, `block(x)`, `block.self_attn(x)`, or `block.mlp(x)`. Only raw tensor operations.

---

### Phase 19 — Final consolidation

**Goal:** Verify durable understanding. This is the acceptance criteria.

| Step | What you do |
|---|---|
| 19.1 | Without looking at the notebook, write down all tensor shapes from input text to output token. Check against the notebook. Repeat until from memory. |
| 19.2 | Without looking at the notebook, write down every learned weight tensor name and shape: `{E}`, `{attn_i.norm}`, `{W_Q}`, `{W_K}`, `{W_V}`, `{W_O}`, `{ffn_i.norm}`, `{W_gate}`, `{W_up}`, `{W_down}`, `{final_norm}`, `{H}`. Check. |
| 19.3 | For three random levels in the 50-level document, explain: operation, input/output tensor shapes, static data used, location in loaded model. Pick three random levels per day for a week. |

**End-of-phase self-check:** This phase is the goal. If 19.1, 19.2, 19.3 can be done from memory, the project is complete.

---

## 8. Summary of phases

| # | Phase | Focus |
|---|---|---|
| 1 | Foundation | Load model, map sizes |
| 2 | Outer pipeline | Replace `generate()` |
| 3 | Embed | Open the embedding lookup |
| 4 | LM head | Open the final projection |
| 5 | Final norm | Open RMSNorm |
| 6 | Manual outer pipeline | Stitch 1–5 |
| 7 | Block stack | Open the layer loop |
| 8 | Block children | Identify the 4 children |
| 9 | Attention sub-stage | Residual + pre-norm pattern |
| 10 | FFN sub-stage | Residual + pre-norm pattern |
| 11 | Attention projections | Identify Q/K/V/O |
| 12 | Single-head attention | 5 sub-steps (intentionally wrong) |
| 13 | Multi-head + GQA | Per-head reshape (still wrong) |
| 14 | RoPE | Final fix — attention matches |
| 15 | FFN projections | Identify gate/up/down |
| 16 | SwiGLU FFN | Build FFN from scratch |
| 17 | Manual block | Stitch attention + FFN |
| 18 | Manual inference | Stitch full pipeline |
| 19 | Consolidation | Verify durable understanding |

---

## 9. Mapping to the 50-level document

| Roadmap phase | Document levels |
|---|---|
| Phase 1 | (foundation, not in document) |
| Phase 2 | 0–10 |
| Phase 3 | 4 |
| Phase 4 | 6–7 |
| Phase 5 | 11–12 |
| Phase 6 | (stitch, not in document) |
| Phase 7 | 13 |
| Phase 8 | 14–16 |
| Phase 9 | 17–19 |
| Phase 10 | 38–40 |
| Phase 11 | 20–25 |
| Phase 12 | 26–30 (single-head simplified) |
| Phase 13 | 33–37 |
| Phase 14 | 31–32 |
| Phase 15 | 41–46 |
| Phase 16 | 47–48 |
| Phase 17 | (stitch, not in document) |
| Phase 18 | 49 |
| Phase 19 | (consolidation, not in document) |

Note: Multi-head (13) is opened before RoPE (14) — opposite of document order. Reason: head-splitting reshape is easier to grasp than RoPE; once heads are clear, RoPE drops in cleanly. Both norms (Phases 9, 10) are addressed before opening attention internals (Phase 11), because the residual pattern frames why attention and FFN exist in the first place.

---

## 10. Notebook structure convention

Each step in the notebook follows this template:

```
# ============ Phase N.M — <name> ============
# What changes vs Step N.(M-1): <one sentence>
# New tensor introduced: <name>, expected shape <shape>
# New static data introduced (if any): <name>, expected shape <shape>

# --- code that adds the one new step ---
<minimal new code>

# --- observation block ---
print("input shape:", ...)
print("output shape:", ...)
print("first 3 values:", ...)

# --- assertion block ---
assert output.shape == expected_shape, f"got {output.shape}, expected {expected_shape}"
```

Three rules:
1. The header restates what changed. Forces articulation of the delta.
2. The observation block is mandatory. Print shapes and a few values every step.
3. The assertion block is mandatory. Each `assert` encodes one prediction from the document.

---

*End of contract. Updates to this file are explicit version bumps. Do not modify silently.*
