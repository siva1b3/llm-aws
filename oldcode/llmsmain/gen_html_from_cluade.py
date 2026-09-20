"""Generates llm_levels.html with 50 levels.
Every row: #, Step, Op, In, In shape, Static, Out, Out shape, Dim out.
New/changed rows are highlighted yellow.
"""

import html, re

NONE = "—"

# ---- Shape map: tensor-name -> shape ----
SHAPES = {
    "text": "—", "text piece": "—",
    "token IDs": "[N_tok]",
    "1 token ID": "scalar",
    "vectors": "[N_tok × d]",
    "x": "[N_tok × d]", "x (held)": "[N_tok × d]", "x (vectors)": "[N_tok × d]", "x_normed": "[N_tok × d]",
    "y": "[N_tok × d]", "y (held)": "[N_tok × d]", "y (vectors)": "[N_tok × d]", "y_normed": "[N_tok × d]",
    "Q": "[N_tok × d]", "Q flat": "[N_tok × d]", "Q per-head": "[n_h × N_tok × d_h]", "Q_rot": "[n_h × N_tok × d_h]",
    "K": "[N_tok × (n_kv·d_h)]", "K flat": "[N_tok × (n_kv·d_h)]", "K per-head": "[n_kv × N_tok × d_h]", "K_rot": "[n_kv × N_tok × d_h]",
    "V": "[N_tok × (n_kv·d_h)]", "V flat": "[N_tok × (n_kv·d_h)]", "V per-head": "[n_kv × N_tok × d_h]",
    "raw scores": "[n_h × N_tok × N_tok]", "scaled scores": "[n_h × N_tok × N_tok]",
    "masked scores": "[n_h × N_tok × N_tok]", "attn weights": "[n_h × N_tok × N_tok]",
    "inner result": "[N_tok × d]", "per-head results": "[n_h × N_tok × d_h]",
    "merged result": "[N_tok × d]", "attn delta": "[N_tok × d]", "ffn delta": "[N_tok × d]",
    "gate": "[N_tok × d_ff]", "gate_act": "[N_tok × d_ff]", "up": "[N_tok × d_ff]", "hidden": "[N_tok × d_ff]",
    "logits": "[N_tok × vocab_size]", "logits (all positions)": "[N_tok × vocab_size]",
    "logits (last position)": "[vocab_size]", "probabilities": "[vocab_size]",
}

def shape_of(name):
    n = name.strip()
    if n in SHAPES: return SHAPES[n]
    base = n.split(" (")[0].strip()
    if base in SHAPES: return SHAPES[base]
    return "?"

def shape_list(csv):
    if csv == "...": return "..."
    return ", ".join(shape_of(p.strip()) for p in csv.split(","))

def dim_of(shape):
    s = shape.strip()
    if s in ("scalar", "[]"): return "scalar"
    if s == "—": return "—"
    if s == "...": return "..."
    if s == "?": return "?"
    if "," in s:
        return ", ".join(dim_of(x.strip()) for x in s.split(","))
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1]
        n = inner.count("×") + 1 if inner else 0
        labels = {1: "1D vector", 2: "2D matrix", 3: "3D tensor", 4: "4D tensor"}
        return labels.get(n, f"{n}D")
    return "?"

# ---- Op map: step-prefix -> op string (longer prefixes listed first) ----
OPS = [
    ("3.i.a.op.scores", "batched matmul (Q · Kᵀ)"),
    ("3.i.a.op.scale", "scalar divide (/ √d_h)"),
    ("3.i.a.op.mask", "elementwise add (−∞ to future)"),
    ("3.i.a.op.smax", "softmax (over key axis)"),
    ("3.i.a.op.weighted", "batched matmul (weights · V)"),
    ("3.i.a.op.rest", "opaque"),
    ("3.i.a.op", "opaque"),
    ("3.i.a.save", "identity (hold copy of x)"),
    ("3.i.a.norm", "RMSNorm"),
    ("3.i.a.q_split", "reshape (split heads)"),
    ("3.i.a.k_split", "reshape (split heads)"),
    ("3.i.a.v_split", "reshape (split heads)"),
    ("3.i.a.rope_q", "elementwise rotate (RoPE)"),
    ("3.i.a.rope_k", "elementwise rotate (RoPE)"),
    ("3.i.a.q", "matmul (x_normed · W_Q)"),
    ("3.i.a.k", "matmul (x_normed · W_K)"),
    ("3.i.a.v", "matmul (x_normed · W_V)"),
    ("3.i.a.inner", "opaque"),
    ("3.i.a.core", "opaque"),
    ("3.i.a.concat", "reshape (concat heads)"),
    ("3.i.a.proj", "matmul (· W_O)"),
    ("3.i.a.add", "elementwise add (residual)"),
    ("3.i.a : attention", "opaque"),
    ("3.i.a : stage A", "opaque"),
    ("3.i.b.save", "identity (hold copy of y)"),
    ("3.i.b.norm", "RMSNorm"),
    ("3.i.b.up", "matmul (y_normed · W_up)"),
    ("3.i.b.gate", "matmul (y_normed · W_gate)"),
    ("3.i.b.silu", "elementwise SiLU"),
    ("3.i.b.combine : elementwise multiply", "elementwise multiply (⊙)"),
    ("3.i.b.combine", "opaque (combine gate & up)"),
    ("3.i.b.inner_rest", "opaque"),
    ("3.i.b.inner", "opaque"),
    ("3.i.b.core", "opaque"),
    ("3.i.b.down", "matmul (hidden · W_down)"),
    ("3.i.b.add", "elementwise add (residual)"),
    ("3.i.b : FFN", "opaque"),
    ("3.i.b : stage B", "opaque"),
    ("block 1", "opaque"),
    ("block 2", "opaque"),
    ("block N", "opaque"),
    ("...", "..."),
    ("tokenize", "BPE segment + ID map"),
    ("embed", "row lookup in {E}"),
    ("LLM system", "opaque"),
    ("LLM core", "opaque"),
    ("LLM", "opaque"),
    ("final norm", "RMSNorm"),
    ("LM head", "matmul (vectors · {H})"),
    ("take last row", "slice (last position)"),
    ("softmax", "softmax over vocab"),
    ("pick (argmax)", "argmax"),
    ("pick", "argmax / sample"),
    ("decode", "inverse vocab lookup"),
    ("to token", "opaque"),
]

def op_of(step):
    s = step.strip()
    for prefix, op in OPS:
        if s.startswith(prefix):
            return op
    return "?"

def R(step, rin, sd, rout):
    return (step, rin, sd, rout)

HEAD = [
    R("tokenize", "text", "{vocab}", "token IDs"),
    R("embed", "token IDs", "{E : vocab_size × d}", "vectors"),
]

def tail_fn(fn_sd):
    return [
        R("final norm", "vectors", fn_sd, "vectors"),
        R("LM head", "vectors", "{H : d × vocab_size}", "logits (all positions)"),
        R("take last row", "logits (all positions)", NONE, "logits (last position)"),
        R("softmax", "logits (last position)", NONE, "probabilities"),
        R("pick (argmax)", "probabilities", NONE, "1 token ID"),
        R("decode", "1 token ID", "{vocab}", "text piece"),
    ]

TAIL = tail_fn("{final_norm : d}")

levels = []

# ==== Phase 1 ====
levels.append({"title": "Level 0 — opaque box",
    "note": "<b>New:</b> Entire pipeline is one box. <code>{model}</code> bundles vocab plus all learned weights.",
    "rows": [R("LLM system", "text", "{model}", "text piece")], "new_ids": [0]})

levels.append({"title": "Level 1 — split model call from decode",
    "note": "<b>New:</b> The model's true output is a token ID. Converting that integer to readable text uses <code>{vocab}</code>.",
    "rows": [R("LLM", "text", "{model internals}", "1 token ID"),
             R("decode", "1 token ID", "{vocab}", "text piece")], "new_ids": [0, 1]})

levels.append({"title": "Level 2 — split tokenize from the model",
    "note": "<b>New:</b> Text can't enter the model as a raw string. It's first split into integer token IDs using the same <code>{vocab}</code>.",
    "rows": [R("tokenize", "text", "{vocab}", "token IDs"),
             R("LLM", "token IDs", "{model internals}", "1 token ID"),
             R("decode", "1 token ID", "{vocab}", "text piece")], "new_ids": [0, 1]})

levels.append({"title": "Level 3 — expose embed as a separate step (internals opaque)",
    "note": "<b>New:</b> The LLM core doesn't consume integer IDs. Before it runs, each token ID is converted to a vector by an embed step. Internals not yet exposed.",
    "rows": [R("tokenize", "text", "{vocab}", "token IDs"),
             R("embed", "token IDs", "{embed internals}", "vectors"),
             R("LLM core", "vectors", "{core internals}", "1 token ID"),
             R("decode", "1 token ID", "{vocab}", "text piece")], "new_ids": [1, 2]})

levels.append({"title": "Level 4 — expose <code>{E}</code> as the embedding matrix",
    "note": "<b>New:</b> Embed is a row-lookup in <code>{E}</code>. Row <code>i</code> is the vector for token ID <code>i</code>. Learned matrix, but the operation is indexing, not matmul.",
    "rows": [R("tokenize", "text", "{vocab}", "token IDs"),
             R("embed", "token IDs", "{E : vocab_size × d}", "vectors"),
             R("LLM core", "vectors", "{core internals}", "1 token ID"),
             R("decode", "1 token ID", "{vocab}", "text piece")], "new_ids": [1]})

levels.append({"title": "Level 5 — LLM core outputs vectors, not a token ID",
    "note": "<b>New:</b> The core's real output is vectors. A new downstream <code>to token</code> step converts those vectors into a token ID.",
    "rows": [R("tokenize", "text", "{vocab}", "token IDs"),
             R("embed", "token IDs", "{E : vocab_size × d}", "vectors"),
             R("LLM core", "vectors", "{core internals}", "vectors"),
             R("to token", "vectors", "{to-token internals}", "1 token ID"),
             R("decode", "1 token ID", "{vocab}", "text piece")], "new_ids": [2, 3]})

levels.append({"title": "Level 6 — split <code>to token</code> into LM head + pick",
    "note": "<b>New:</b> <code>to token</code>'s first sub-step is a linear projection — the LM head — producing raw scores (logits). The remaining <code>pick</code> step is stateless.",
    "rows": [R("tokenize", "text", "{vocab}", "token IDs"),
             R("embed", "token IDs", "{E : vocab_size × d}", "vectors"),
             R("LLM core", "vectors", "{core internals}", "vectors"),
             R("LM head", "vectors", "{head internals}", "logits"),
             R("pick", "logits", NONE, "1 token ID"),
             R("decode", "1 token ID", "{vocab}", "text piece")], "new_ids": [3, 4]})

levels.append({"title": "Level 7 — expose <code>{H}</code> as the LM head matrix",
    "note": "<b>New:</b> LM head is a matmul with learned <code>{H}</code> of shape <code>[d × vocab_size]</code>.",
    "rows": [R("tokenize", "text", "{vocab}", "token IDs"),
             R("embed", "token IDs", "{E : vocab_size × d}", "vectors"),
             R("LLM core", "vectors", "{core internals}", "vectors"),
             R("LM head", "vectors", "{H : d × vocab_size}", "logits"),
             R("pick", "logits", NONE, "1 token ID"),
             R("decode", "1 token ID", "{vocab}", "text piece")], "new_ids": [3]})

levels.append({"title": "Level 8 — take only the last position's logits",
    "note": "<b>New:</b> LM head produces one logits row per input token. For predicting one next token, only the last position's row is used.",
    "rows": [R("tokenize", "text", "{vocab}", "token IDs"),
             R("embed", "token IDs", "{E : vocab_size × d}", "vectors"),
             R("LLM core", "vectors", "{core internals}", "vectors"),
             R("LM head", "vectors", "{H : d × vocab_size}", "logits (all positions)"),
             R("take last row", "logits (all positions)", NONE, "logits (last position)"),
             R("pick", "logits (last position)", NONE, "1 token ID"),
             R("decode", "1 token ID", "{vocab}", "text piece")], "new_ids": [3, 4, 5]})

levels.append({"title": "Level 9 — softmax converts logits to probabilities",
    "note": "<b>New:</b> <code>pick</code> doesn't consume raw logits. Softmax normalizes them into a distribution that sums to 1.",
    "rows": [R("tokenize", "text", "{vocab}", "token IDs"),
             R("embed", "token IDs", "{E : vocab_size × d}", "vectors"),
             R("LLM core", "vectors", "{core internals}", "vectors"),
             R("LM head", "vectors", "{H : d × vocab_size}", "logits (all positions)"),
             R("take last row", "logits (all positions)", NONE, "logits (last position)"),
             R("softmax", "logits (last position)", NONE, "probabilities"),
             R("pick", "probabilities", NONE, "1 token ID"),
             R("decode", "1 token ID", "{vocab}", "text piece")], "new_ids": [5, 6]})

levels.append({"title": "Level 10 — pick = argmax (for one-inference target)",
    "note": "<b>New:</b> Pick is a sampling policy. Simplest is argmax. Alternatives: temperature, top-k, top-p. For one-token prediction, argmax is sufficient.",
    "rows": [R("tokenize", "text", "{vocab}", "token IDs"),
             R("embed", "token IDs", "{E : vocab_size × d}", "vectors"),
             R("LLM core", "vectors", "{core internals}", "vectors"),
             R("LM head", "vectors", "{H : d × vocab_size}", "logits (all positions)"),
             R("take last row", "logits (all positions)", NONE, "logits (last position)"),
             R("softmax", "logits (last position)", NONE, "probabilities"),
             R("pick (argmax)", "probabilities", NONE, "1 token ID"),
             R("decode", "1 token ID", "{vocab}", "text piece")], "new_ids": [6]})

# ==== Phase 2 ====
def l11_rows():
    return [R("tokenize", "text", "{vocab}", "token IDs"),
            R("embed", "token IDs", "{E : vocab_size × d}", "vectors"),
            R("LLM core", "vectors", "{core internals}", "vectors"),
            R("final norm", "vectors", "{final-norm internals}", "vectors"),
            R("LM head", "vectors", "{H : d × vocab_size}", "logits (all positions)"),
            R("take last row", "logits (all positions)", NONE, "logits (last position)"),
            R("softmax", "logits (last position)", NONE, "probabilities"),
            R("pick (argmax)", "probabilities", NONE, "1 token ID"),
            R("decode", "1 token ID", "{vocab}", "text piece")]

levels.append({"title": "Level 11 — add final norm before LM head (weights opaque)",
    "note": "<b>New:</b> Between core output and LM head, a normalization step stabilizes vector magnitudes. Parameters not yet exposed.",
    "rows": l11_rows(), "new_ids": [3]})

def l12_rows():
    r = l11_rows(); r[3] = R("final norm", "vectors", "{final_norm : d}", "vectors"); return r

levels.append({"title": "Level 12 — expose <code>{final_norm}</code>",
    "note": "<b>New:</b> Final-norm's learned scale weights are <code>{final_norm}</code> of shape <code>[d]</code>. Operation is RMSNorm.",
    "rows": l12_rows(), "new_ids": [3]})

# ==== Phase 3 ====
def with_blocks(block_fn):
    r = list(HEAD); r.extend(block_fn()); r.extend(TAIL); return r

def l13_blocks():
    return [R("block 1", "vectors", "{block 1 internals}", "vectors"),
            R("block 2", "vectors", "{block 2 internals}", "vectors"),
            R("...", "...", "...", "..."),
            R("block N", "vectors", "{block N internals}", "vectors")]

levels.append({"title": "Level 13 — core is N sequential blocks (each opaque)",
    "note": "<b>New:</b> The core is <code>N</code> blocks in series. Each has its own independent weights. Each preserves shape <code>[N_tok × d]</code>.",
    "rows": with_blocks(l13_blocks), "new_ids": [2, 3, 4, 5]})

def l14_blocks():
    return [R("3.i.a : stage A (block i)", "vectors", "{A internals of block i}", "vectors"),
            R("3.i.b : stage B (block i)", "vectors", "{B internals of block i}", "vectors")]

levels.append({"title": "Level 14 — each block has two sub-stages A then B (both opaque)",
    "note": "<b>New:</b> A block is not atomic. It contains two internal sub-stages in sequence with different roles. Roles not yet named.",
    "rows": with_blocks(l14_blocks), "new_ids": [2, 3]})

def l15_blocks():
    return [R("3.i.a : attention (block i)", "vectors", "{attn_i internals}", "vectors"),
            R("3.i.b : stage B (block i)", "vectors", "{B internals of block i}", "vectors")]

levels.append({"title": "Level 15 — name sub-stage A as attention",
    "note": "<b>New:</b> The first sub-stage is called attention. Still opaque internally.",
    "rows": with_blocks(l15_blocks), "new_ids": [2]})

def l16_blocks():
    return [R("3.i.a : attention (block i)", "vectors", "{attn_i internals}", "vectors"),
            R("3.i.b : FFN (block i)", "vectors", "{ffn_i internals}", "vectors")]

levels.append({"title": "Level 16 — name sub-stage B as FFN",
    "note": "<b>New:</b> The second sub-stage is called the feedforward network (FFN). Still opaque internally.",
    "rows": with_blocks(l16_blocks), "new_ids": [3]})

# ==== Phase 4 — inside attention ====
def ffn_simple():
    return [R("3.i.b : FFN (block i)", "vectors", "{ffn_i internals}", "vectors")]

def l17_blocks():
    return [R("3.i.a.save : save x", "x (vectors)", NONE, "x (held)"),
            R("3.i.a.core : attention core", "x", "{attn_i core internals}", "attn delta"),
            R("3.i.a.add : add residual", "x (held), attn delta", NONE, "vectors")] + ffn_simple()

levels.append({"title": "Level 17 — residual connection around attention",
    "note": "<b>New:</b> Attention doesn't replace x; it produces a correction added back to x. Save and add have no static data — pure plumbing.",
    "rows": with_blocks(l17_blocks), "new_ids": [2, 3, 4]})

def l18_blocks():
    return [R("3.i.a.save : save x", "x", NONE, "x (held)"),
            R("3.i.a.norm : pre-attn norm", "x", "{attn_i.norm internals}", "x_normed"),
            R("3.i.a.core : attention core", "x_normed", "{attn_i core internals}", "attn delta"),
            R("3.i.a.add : add residual", "x (held), attn delta", NONE, "vectors")] + ffn_simple()

levels.append({"title": "Level 18 — pre-attention norm (weights opaque)",
    "note": "<b>New:</b> Before the attention core, input is normalized. Residual is added to the <i>un-normed</i> x — pre-norm architecture.",
    "rows": with_blocks(l18_blocks), "new_ids": [3, 4]})

def l19_blocks():
    return [R("3.i.a.save : save x", "x", NONE, "x (held)"),
            R("3.i.a.norm : pre-attn norm", "x", "{attn_i.norm : d}", "x_normed"),
            R("3.i.a.core : attention core", "x_normed", "{attn_i core internals}", "attn delta"),
            R("3.i.a.add : add residual", "x (held), attn delta", NONE, "vectors")] + ffn_simple()

levels.append({"title": "Level 19 — expose <code>{attn_i.norm}</code>",
    "note": "<b>New:</b> Pre-attn norm weights are <code>{attn_i.norm}</code> of shape <code>[d]</code>. RMSNorm, per-block.",
    "rows": with_blocks(l19_blocks), "new_ids": [3]})

def l20_blocks():
    return [R("3.i.a.save : save x", "x", NONE, "x (held)"),
            R("3.i.a.norm : pre-attn norm", "x", "{attn_i.norm : d}", "x_normed"),
            R("3.i.a.inner : attention inner", "x_normed", "{attn_i inner internals}", "inner result"),
            R("3.i.a.proj : output projection", "inner result", "{W_O internals}", "attn delta"),
            R("3.i.a.add : add residual", "x (held), attn delta", NONE, "vectors")] + ffn_simple()

levels.append({"title": "Level 20 — attention core ends with output projection (opaque)",
    "note": "<b>New:</b> Last step of the attention core is a linear projection mapping an internal result back into the block's vector space.",
    "rows": with_blocks(l20_blocks), "new_ids": [4, 5]})

def l21_blocks():
    return [R("3.i.a.save : save x", "x", NONE, "x (held)"),
            R("3.i.a.norm : pre-attn norm", "x", "{attn_i.norm : d}", "x_normed"),
            R("3.i.a.inner : attention inner", "x_normed", "{attn_i inner internals}", "inner result"),
            R("3.i.a.proj : output projection", "inner result", "{attn_i.W_O : d × d}", "attn delta"),
            R("3.i.a.add : add residual", "x (held), attn delta", NONE, "vectors")] + ffn_simple()

levels.append({"title": "Level 21 — expose <code>{W_O}</code>",
    "note": "<b>New:</b> Output projection uses <code>{attn_i.W_O}</code> of shape <code>[d × d]</code>.",
    "rows": with_blocks(l21_blocks), "new_ids": [5]})

def l22_blocks():
    return [R("3.i.a.save : save x", "x", NONE, "x (held)"),
            R("3.i.a.norm : pre-attn norm", "x", "{attn_i.norm : d}", "x_normed"),
            R("3.i.a.q : produce Q", "x_normed", "{W_Q internals}", "Q"),
            R("3.i.a.k : produce K", "x_normed", "{W_K internals}", "K"),
            R("3.i.a.v : produce V", "x_normed", "{W_V internals}", "V"),
            R("3.i.a.op : attention op", "Q, K, V", NONE, "inner result"),
            R("3.i.a.proj : output projection", "inner result", "{attn_i.W_O : d × d}", "attn delta"),
            R("3.i.a.add : add residual", "x (held), attn delta", NONE, "vectors")] + ffn_simple()

levels.append({"title": "Level 22 — attention inner consumes three derived tensors Q, K, V",
    "note": "<b>New:</b> The attention core doesn't use x_normed directly. Three separate projections produce Q, K, V. The attention op combines them.",
    "rows": with_blocks(l22_blocks), "new_ids": [4, 5, 6, 7]})

def l23_blocks():
    r = l22_blocks()
    # block rows start at 0 inside local list; in with_blocks, they're offset by 2 (HEAD).
    # But here l22_blocks returns block rows only. So indices within this function are block-local.
    r[2] = R("3.i.a.q : produce Q", "x_normed", "{attn_i.W_Q : d × d}", "Q")
    return r

levels.append({"title": "Level 23 — expose <code>{W_Q}</code>",
    "note": "<b>New:</b> <code>Q = x_normed · W_Q</code>, with <code>W_Q</code> of shape <code>[d × d]</code>.",
    "rows": with_blocks(l23_blocks), "new_ids": [4]})

def l24_blocks():
    r = l23_blocks()
    r[3] = R("3.i.a.k : produce K", "x_normed", "{attn_i.W_K : d × (n_kv·d_h)}", "K")
    return r

levels.append({"title": "Level 24 — expose <code>{W_K}</code>",
    "note": "<b>New:</b> <code>K = x_normed · W_K</code>, with <code>W_K</code> of shape <code>[d × (n_kv · d_h)]</code>. Second dim differs from W_Q because of grouped-query attention (see Level 37).",
    "rows": with_blocks(l24_blocks), "new_ids": [5]})

def l25_blocks():
    r = l24_blocks()
    r[4] = R("3.i.a.v : produce V", "x_normed", "{attn_i.W_V : d × (n_kv·d_h)}", "V")
    return r

levels.append({"title": "Level 25 — expose <code>{W_V}</code>",
    "note": "<b>New:</b> <code>V = x_normed · W_V</code>, with <code>W_V</code> of shape <code>[d × (n_kv · d_h)]</code>.",
    "rows": with_blocks(l25_blocks), "new_ids": [6]})

# ==== Phase 5 — inside the attention op ====
def attn_pre_qkv():
    return [R("3.i.a.save : save x", "x", NONE, "x (held)"),
            R("3.i.a.norm : pre-attn norm", "x", "{attn_i.norm : d}", "x_normed"),
            R("3.i.a.q : produce Q", "x_normed", "{attn_i.W_Q : d × d}", "Q"),
            R("3.i.a.k : produce K", "x_normed", "{attn_i.W_K : d × (n_kv·d_h)}", "K"),
            R("3.i.a.v : produce V", "x_normed", "{attn_i.W_V : d × (n_kv·d_h)}", "V")]

def attn_proj_add(proj_in="inner result"):
    return [R("3.i.a.proj : output projection", proj_in, "{attn_i.W_O : d × d}", "attn delta"),
            R("3.i.a.add : add residual", "x (held), attn delta", NONE, "vectors")]

def l26_blocks():
    op = [R("3.i.a.op.scores : Q·Kᵀ", "Q, K", NONE, "raw scores"),
          R("3.i.a.op.rest : rest of op", "raw scores, V", NONE, "inner result")]
    return attn_pre_qkv() + op + attn_proj_add() + ffn_simple()

levels.append({"title": "Level 26 — op begins with Q·Kᵀ scores",
    "note": "<b>New:</b> The attention op's first step computes similarity scores via <code>Q · Kᵀ</code>.",
    "rows": with_blocks(l26_blocks), "new_ids": [7, 8]})

def l27_blocks():
    op = [R("3.i.a.op.scores : Q·Kᵀ", "Q, K", NONE, "raw scores"),
          R("3.i.a.op.scale : scale by 1/√d_h", "raw scores", NONE, "scaled scores"),
          R("3.i.a.op.rest : rest of op", "scaled scores, V", NONE, "inner result")]
    return attn_pre_qkv() + op + attn_proj_add() + ffn_simple()

levels.append({"title": "Level 27 — scale by 1/√d_h",
    "note": "<b>New:</b> Raw scores are divided by <code>√d_h</code>. Prevents softmax saturation as d_h grows.",
    "rows": with_blocks(l27_blocks), "new_ids": [8, 9]})

def l28_blocks():
    op = [R("3.i.a.op.scores : Q·Kᵀ", "Q, K", NONE, "raw scores"),
          R("3.i.a.op.scale : scale by 1/√d_h", "raw scores", NONE, "scaled scores"),
          R("3.i.a.op.mask : causal mask", "scaled scores", "{causal_mask : N_tok × N_tok}", "masked scores"),
          R("3.i.a.op.rest : rest of op", "masked scores, V", NONE, "inner result")]
    return attn_pre_qkv() + op + attn_proj_add() + ffn_simple()

levels.append({"title": "Level 28 — apply causal mask",
    "note": "<b>New:</b> Each query position may only attend to itself and earlier ones. Mask sets future-position scores to <code>-∞</code>.",
    "rows": with_blocks(l28_blocks), "new_ids": [9, 10]})

def l29_blocks():
    op = [R("3.i.a.op.scores : Q·Kᵀ", "Q, K", NONE, "raw scores"),
          R("3.i.a.op.scale : scale by 1/√d_h", "raw scores", NONE, "scaled scores"),
          R("3.i.a.op.mask : causal mask", "scaled scores", "{causal_mask : N_tok × N_tok}", "masked scores"),
          R("3.i.a.op.smax : softmax (per row)", "masked scores", NONE, "attn weights"),
          R("3.i.a.op.rest : rest of op", "attn weights, V", NONE, "inner result")]
    return attn_pre_qkv() + op + attn_proj_add() + ffn_simple()

levels.append({"title": "Level 29 — softmax over keys",
    "note": "<b>New:</b> Softmax runs along the key axis. For each query position, scores across keys become a distribution.",
    "rows": with_blocks(l29_blocks), "new_ids": [10, 11]})

def l30_blocks():
    op = [R("3.i.a.op.scores : Q·Kᵀ", "Q, K", NONE, "raw scores"),
          R("3.i.a.op.scale : scale by 1/√d_h", "raw scores", NONE, "scaled scores"),
          R("3.i.a.op.mask : causal mask", "scaled scores", "{causal_mask : N_tok × N_tok}", "masked scores"),
          R("3.i.a.op.smax : softmax (per row)", "masked scores", NONE, "attn weights"),
          R("3.i.a.op.weighted : weighted sum with V", "attn weights, V", NONE, "inner result")]
    return attn_pre_qkv() + op + attn_proj_add() + ffn_simple()

levels.append({"title": "Level 30 — weighted sum with V",
    "note": "<b>New:</b> For each query position, inner result is a weighted sum of V rows using that row's attention weights.",
    "rows": with_blocks(l30_blocks), "new_ids": [11]})

# ==== Phase 6 — RoPE & heads ====
def attn_full(rope_q=False, rope_k=False, split_q=False, split_k=False, split_v=False, multihead_tail=False):
    rows = [R("3.i.a.save : save x", "x", NONE, "x (held)"),
            R("3.i.a.norm : pre-attn norm", "x", "{attn_i.norm : d}", "x_normed")]
    if split_q:
        rows.append(R("3.i.a.q : produce Q", "x_normed", "{attn_i.W_Q : d × d}", "Q flat"))
        rows.append(R("3.i.a.q_split : split Q into heads", "Q flat", NONE, "Q per-head"))
        qn = "Q per-head"
    else:
        rows.append(R("3.i.a.q : produce Q", "x_normed", "{attn_i.W_Q : d × d}", "Q"))
        qn = "Q"
    if split_k:
        rows.append(R("3.i.a.k : produce K", "x_normed", "{attn_i.W_K : d × (n_kv·d_h)}", "K flat"))
        rows.append(R("3.i.a.k_split : split K into heads", "K flat", NONE, "K per-head"))
        kn = "K per-head"
    else:
        rows.append(R("3.i.a.k : produce K", "x_normed", "{attn_i.W_K : d × (n_kv·d_h)}", "K"))
        kn = "K"
    if split_v:
        rows.append(R("3.i.a.v : produce V", "x_normed", "{attn_i.W_V : d × (n_kv·d_h)}", "V flat"))
        rows.append(R("3.i.a.v_split : split V into heads", "V flat", NONE, "V per-head"))
        vn = "V per-head"
    else:
        rows.append(R("3.i.a.v : produce V", "x_normed", "{attn_i.W_V : d × (n_kv·d_h)}", "V"))
        vn = "V"
    if rope_q:
        rows.append(R("3.i.a.rope_q : rotate Q", qn, "{RoPE : max_pos × d_h}", "Q_rot"))
        qs = "Q_rot"
    else:
        qs = qn
    if rope_k:
        rows.append(R("3.i.a.rope_k : rotate K", kn, "{RoPE : max_pos × d_h}", "K_rot"))
        ks = "K_rot"
    else:
        ks = kn
    rows.append(R("3.i.a.op.scores : Q·Kᵀ", f"{qs}, {ks}", NONE, "raw scores"))
    rows.append(R("3.i.a.op.scale : scale by 1/√d_h", "raw scores", NONE, "scaled scores"))
    rows.append(R("3.i.a.op.mask : causal mask", "scaled scores", "{causal_mask : N_tok × N_tok}", "masked scores"))
    rows.append(R("3.i.a.op.smax : softmax (per row)", "masked scores", NONE, "attn weights"))
    if multihead_tail:
        rows.append(R("3.i.a.op.weighted : weighted sum with V", f"attn weights, {vn}", NONE, "per-head results"))
        rows.append(R("3.i.a.concat : concat heads", "per-head results", NONE, "merged result"))
        pin = "merged result"
    else:
        rows.append(R("3.i.a.op.weighted : weighted sum with V", f"attn weights, {vn}", NONE, "inner result"))
        pin = "inner result"
    rows.append(R("3.i.a.proj : output projection", pin, "{attn_i.W_O : d × d}", "attn delta"))
    rows.append(R("3.i.a.add : add residual", "x (held), attn delta", NONE, "vectors"))
    return rows

levels.append({"title": "Level 31 — RoPE on Q",
    "note": "<b>New:</b> Without positional info, scores are order-insensitive. RoPE rotates each Q vector by a position-dependent angle. <code>{RoPE}</code> is pre-built, not learned.",
    "rows": with_blocks(lambda: attn_full(rope_q=True) + ffn_simple()), "new_ids": [6, 9]})

levels.append({"title": "Level 32 — RoPE on K",
    "note": "<b>New:</b> K is also rotated by <code>{RoPE}</code>. V is never rotated. Result: <code>Q_rot · K_rotᵀ</code> depends only on relative position.",
    "rows": with_blocks(lambda: attn_full(rope_q=True, rope_k=True) + ffn_simple()), "new_ids": [7, 9]})

levels.append({"title": "Level 33 — split Q into heads",
    "note": "<b>New:</b> Q is reshaped so <code>n_h</code> heads each get a <code>d_h</code>-dim slice. Downstream ops run per-head.",
    "rows": with_blocks(lambda: attn_full(rope_q=True, rope_k=True, split_q=True) + ffn_simple()), "new_ids": [3, 4, 7]})

levels.append({"title": "Level 34 — split K into heads",
    "note": "<b>New:</b> K is reshaped into <code>n_kv</code> heads.",
    "rows": with_blocks(lambda: attn_full(rope_q=True, rope_k=True, split_q=True, split_k=True) + ffn_simple()), "new_ids": [5, 6, 9]})

levels.append({"title": "Level 35 — split V into heads",
    "note": "<b>New:</b> V is reshaped into <code>n_kv</code> heads. Weighted sum runs per-head.",
    "rows": with_blocks(lambda: attn_full(rope_q=True, rope_k=True, split_q=True, split_k=True, split_v=True) + ffn_simple()), "new_ids": [7, 8, 14]})

levels.append({"title": "Level 36 — concat heads before output projection",
    "note": "<b>New:</b> Per-head results are concatenated back into one <code>d</code>-dim tensor per token. <code>W_O</code> then mixes across heads.",
    "rows": with_blocks(lambda: attn_full(rope_q=True, rope_k=True, split_q=True, split_k=True, split_v=True, multihead_tail=True) + ffn_simple()), "new_ids": [14, 15, 16]})

levels.append({"title": "Level 37 — GQA: K and V have fewer heads than Q",
    "note": "<b>New:</b> Q has <code>n_h</code> heads; K/V have <code>n_kv &lt; n_h</code>. Each K/V head serves a group of <code>n_h / n_kv</code> Q heads (repeated before scoring). This is why <code>W_K, W_V</code> have second dim <code>n_kv · d_h</code>. <i>Table identical to Level 36.</i>",
    "rows": with_blocks(lambda: attn_full(rope_q=True, rope_k=True, split_q=True, split_k=True, split_v=True, multihead_tail=True) + ffn_simple()), "new_ids": []})

# ==== Phase 8 — FFN ====
def ffn_half(pre=None, core='simple'):
    rows = [R("3.i.b.save : save y", "y (vectors)", NONE, "y (held)")]
    if pre == 'opaque':
        rows.append(R("3.i.b.norm : pre-FFN norm", "y", "{ffn_i.norm internals}", "y_normed"))
        cin = "y_normed"
    elif pre == 'exposed':
        rows.append(R("3.i.b.norm : pre-FFN norm", "y", "{ffn_i.norm : d}", "y_normed"))
        cin = "y_normed"
    else:
        cin = "y"
    if core == 'simple':
        rows.append(R("3.i.b.core : FFN core", cin, "{ffn_i core internals}", "ffn delta"))
    elif core == 'inner_down_opaque':
        rows.append(R("3.i.b.inner : FFN inner", cin, "{ffn_i inner internals}", "hidden"))
        rows.append(R("3.i.b.down : down projection", "hidden", "{W_down internals}", "ffn delta"))
    elif core == 'inner_down_exposed':
        rows.append(R("3.i.b.inner : FFN inner", cin, "{ffn_i inner internals}", "hidden"))
        rows.append(R("3.i.b.down : down projection", "hidden", "{ffn_i.W_down : d_ff × d}", "ffn delta"))
    elif core == 'up_opaque':
        rows.append(R("3.i.b.up : up projection", cin, "{W_up internals}", "up"))
        rows.append(R("3.i.b.inner_rest : rest of inner", "up", "{ffn_i inner-rest internals}", "hidden"))
        rows.append(R("3.i.b.down : down projection", "hidden", "{ffn_i.W_down : d_ff × d}", "ffn delta"))
    elif core == 'up_exposed':
        rows.append(R("3.i.b.up : up projection", cin, "{ffn_i.W_up : d × d_ff}", "up"))
        rows.append(R("3.i.b.inner_rest : rest of inner", "up", "{ffn_i inner-rest internals}", "hidden"))
        rows.append(R("3.i.b.down : down projection", "hidden", "{ffn_i.W_down : d_ff × d}", "ffn delta"))
    elif core == 'gate_opaque':
        rows.append(R("3.i.b.gate : gate projection", cin, "{W_gate internals}", "gate"))
        rows.append(R("3.i.b.up : up projection", cin, "{ffn_i.W_up : d × d_ff}", "up"))
        rows.append(R("3.i.b.combine : combine gate, up", "gate, up", NONE, "hidden"))
        rows.append(R("3.i.b.down : down projection", "hidden", "{ffn_i.W_down : d_ff × d}", "ffn delta"))
    elif core == 'gate_exposed':
        rows.append(R("3.i.b.gate : gate projection", cin, "{ffn_i.W_gate : d × d_ff}", "gate"))
        rows.append(R("3.i.b.up : up projection", cin, "{ffn_i.W_up : d × d_ff}", "up"))
        rows.append(R("3.i.b.combine : combine gate, up", "gate, up", NONE, "hidden"))
        rows.append(R("3.i.b.down : down projection", "hidden", "{ffn_i.W_down : d_ff × d}", "ffn delta"))
    elif core == 'silu':
        rows.append(R("3.i.b.gate : gate projection", cin, "{ffn_i.W_gate : d × d_ff}", "gate"))
        rows.append(R("3.i.b.silu : SiLU", "gate", NONE, "gate_act"))
        rows.append(R("3.i.b.up : up projection", cin, "{ffn_i.W_up : d × d_ff}", "up"))
        rows.append(R("3.i.b.combine : combine gate_act, up", "gate_act, up", NONE, "hidden"))
        rows.append(R("3.i.b.down : down projection", "hidden", "{ffn_i.W_down : d_ff × d}", "ffn delta"))
    elif core == 'silu_mul':
        rows.append(R("3.i.b.gate : gate projection", cin, "{ffn_i.W_gate : d × d_ff}", "gate"))
        rows.append(R("3.i.b.silu : SiLU", "gate", NONE, "gate_act"))
        rows.append(R("3.i.b.up : up projection", cin, "{ffn_i.W_up : d × d_ff}", "up"))
        rows.append(R("3.i.b.combine : elementwise multiply", "gate_act, up", NONE, "hidden"))
        rows.append(R("3.i.b.down : down projection", "hidden", "{ffn_i.W_down : d_ff × d}", "ffn delta"))
    rows.append(R("3.i.b.add : add residual", "y (held), ffn delta", NONE, "vectors"))
    return rows

def full_attn():
    return attn_full(rope_q=True, rope_k=True, split_q=True, split_k=True, split_v=True, multihead_tail=True)

FULL_ATTN_ROWS_LEN = 2 + len(full_attn())

def block_with_ffn(pre, core):
    return with_blocks(lambda: full_attn() + ffn_half(pre=pre, core=core))

ffn_specs = [
    ("Level 38 — residual connection around FFN",
     "<b>New:</b> FFN produces a correction added to y, not a replacement.",
     None, 'simple'),
    ("Level 39 — pre-FFN norm (weights opaque)",
     "<b>New:</b> Before FFN core, input is normalized. Residual added to un-normed y.",
     'opaque', 'simple'),
    ("Level 40 — expose <code>{ffn_i.norm}</code>",
     "<b>New:</b> Pre-FFN norm weights are <code>{ffn_i.norm}</code> of shape <code>[d]</code>.",
     'exposed', 'simple'),
    ("Level 41 — FFN core ends with down projection (opaque)",
     "<b>New:</b> Last step of FFN core is a linear projection from <code>d_ff</code> back to <code>d</code>.",
     'exposed', 'inner_down_opaque'),
    ("Level 42 — expose <code>{W_down}</code>",
     "<b>New:</b> <code>ffn_i.W_down</code> has shape <code>[d_ff × d]</code>.",
     'exposed', 'inner_down_exposed'),
    ("Level 43 — FFN inner has an up projection (opaque)",
     "<b>New:</b> A projection lifts y_normed from <code>d</code> to <code>d_ff</code>. Rest of inner still opaque.",
     'exposed', 'up_opaque'),
    ("Level 44 — expose <code>{W_up}</code>",
     "<b>New:</b> <code>ffn_i.W_up</code> has shape <code>[d × d_ff]</code>.",
     'exposed', 'up_exposed'),
    ("Level 45 — parallel gate projection (opaque)",
     "<b>New:</b> A second projection of y_normed produces <code>gate</code>, parallel to up.",
     'exposed', 'gate_opaque'),
    ("Level 46 — expose <code>{W_gate}</code>",
     "<b>New:</b> <code>ffn_i.W_gate</code> has shape <code>[d × d_ff]</code>.",
     'exposed', 'gate_exposed'),
    ("Level 47 — SiLU on gate",
     "<b>New:</b> <code>SiLU(x) = x · sigmoid(x)</code>, applied elementwise to gate only. Up is not activated.",
     'exposed', 'silu'),
    ("Level 48 — combine = elementwise multiply",
     "<b>New:</b> <code>hidden = gate_act ⊙ up</code>. SwiGLU gating: gate modulates up dimensions.",
     'exposed', 'silu_mul'),
]

for title, note, pre, core in ffn_specs:
    levels.append({"title": title, "note": note,
                   "rows": block_with_ffn(pre, core), "new_ids": None})

levels.append({"title": "Level 49 — complete reference",
    "note": "<b>New:</b> Full pipeline visible. No further concepts to add for one-inference-one-token. Backbone shape is <code>[N_tok × d]</code>. Attention scores <code>[n_h × N_tok × N_tok]</code> are the quadratic-in-sequence-length intermediate. FFN widens to <code>[N_tok × d_ff]</code>. LM head is the only step that changes per-token dim.",
    "rows": block_with_ffn('exposed', 'silu_mul'), "new_ids": []})

# ---- compute new_ids for FFN levels via diff ----
def diff_new_ids(prev, curr, start):
    if prev is None: return list(range(len(curr)))
    prev_set = set(prev[start:])
    return [i for i in range(start, len(curr)) if curr[i] not in prev_set]

prev = None
for lv in levels:
    if lv.get("new_ids") is None:
        lv["new_ids"] = diff_new_ids(prev, lv["rows"], FULL_ATTN_ROWS_LEN)
    prev = lv["rows"]

# ---- Render ----
def render_table(level):
    rows = level["rows"]
    new_ids = set(level.get("new_ids", []))
    out = ['<table>',
           '<thead><tr>'
           '<th>#</th><th>Step</th><th>Op</th>'
           '<th>In</th><th>In shape</th>'
           '<th>Static</th>'
           '<th>Out</th><th>Out shape</th><th>Dim out</th>'
           '</tr></thead><tbody>']
    for i, (step, rin, sd, rout) in enumerate(rows):
        cls = ' class="new"' if i in new_ids else ''
        op = op_of(step)
        ins = shape_list(rin)
        outs = shape_of(rout)
        dim = dim_of(outs)
        out.append(f'<tr{cls}>')
        out.append(f'<td>{i+1}</td>')
        out.append(f'<td class="step">{html.escape(step)}</td>')
        out.append(f'<td class="op">{html.escape(op)}</td>')
        out.append(f'<td>{html.escape(rin)}</td>')
        out.append(f'<td><code>{html.escape(ins)}</code></td>')
        out.append(f'<td>{html.escape(sd)}</td>')
        out.append(f'<td>{html.escape(rout)}</td>')
        out.append(f'<td><code>{html.escape(outs)}</code></td>')
        out.append(f'<td>{html.escape(dim)}</td>')
        out.append('</tr>')
    out.append('</tbody></table>')
    return '\n'.join(out)

phases = [
    (0, "Phase 1 — Outer pipeline (around the core)"),
    (11, "Phase 2 — Around the transformer core"),
    (13, "Phase 3 — Inside the core"),
    (17, "Phase 4 — Inside attention (outer wrapping)"),
    (26, "Phase 5 — Inside the attention op"),
    (31, "Phase 6 — Positional information (RoPE)"),
    (33, "Phase 7 — Multi-head (and GQA)"),
    (38, "Phase 8 — Inside FFN"),
    (49, "Phase 9 — Complete reference"),
]

def phase_for(idx):
    cur = None
    for start, name in phases:
        if idx >= start: cur = (start, name)
    return cur

css = """
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 1600px; margin: 20px auto; padding: 0 20px; color: #1a1a1a; line-height: 1.42; }
h1 { border-bottom: 3px solid #222; padding-bottom: 8px; font-size: 22px; }
h2.phase { background: #2c3e50; color: white; padding: 10px 14px; margin-top: 44px; font-size: 17px; border-radius: 4px; }
h3 { margin-top: 28px; color: #1a1a1a; border-bottom: 1px solid #bbb; padding-bottom: 4px; font-size: 16px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 10px 0; font-size: 12px; }
th, td { border: 1px solid #aaa; padding: 4px 6px; text-align: left; vertical-align: top; }
th { background: #e8ecef; font-weight: 600; white-space: nowrap; font-size: 12px; }
tr.new td { background: #fff2a8; }
td.step { font-weight: 600; white-space: nowrap; font-family: 'Consolas', 'Monaco', monospace; font-size: 11px; }
td.op { font-size: 11px; font-style: italic; color: #444; white-space: nowrap; }
code { font-family: 'Consolas', 'Monaco', monospace; font-size: 11px; background: #f4f4f4; padding: 1px 3px; border-radius: 3px; white-space: nowrap; }
.note { font-size: 13px; color: #333; margin: 4px 0 8px 0; }
.catalog { font-family: 'Consolas', 'Monaco', monospace; font-size: 12px; background: #f7f7f7; padding: 10px; border: 1px solid #ccc; white-space: pre; overflow-x: auto; border-radius: 3px; }
.footnote { font-size: 11px; color: #666; font-style: italic; margin-top: -4px; margin-bottom: 8px; }
.toc { font-size: 12px; columns: 3; column-gap: 28px; }
.toc a { text-decoration: none; color: #2c3e50; }
.toc a:hover { text-decoration: underline; }
summary { cursor: pointer; font-weight: 600; padding: 4px 0; }
details { margin: 6px 0; }
.intro { background: #f7f9fa; border-left: 4px solid #2c3e50; padding: 10px 14px; margin: 14px 0; font-size: 13px; }
.legend { font-size: 12px; background: #f7f9fa; padding: 8px 12px; border: 1px solid #ddd; border-radius: 3px; margin: 10px 0; }
.legend ul { margin: 4px 0; padding-left: 22px; }
"""

out = ['<!DOCTYPE html>', '<html lang="en"><head><meta charset="utf-8">',
       '<title>LLM Inference — 50-level Incremental Build (Op + Shapes + Dim)</title>',
       f'<style>{css}</style>', '</head><body>',
       '<h1>LLM Inference — 50-level Incremental Build</h1>',
       '<p class="intro"><b>Target:</b> one inference, one token predicted. No KV cache, no batching. Each level adds exactly one new concept. Yellow-highlighted rows mark what changed vs. the previous level.</p>',
       """<div class="legend"><b>Column meaning</b>
<ul>
<li><b>Step</b> — unique label of the step (e.g., <code>3.i.a.q : produce Q</code>).</li>
<li><b>Op</b> — the actual operation: row lookup, matmul, elementwise add, softmax, reshape, RMSNorm, etc.</li>
<li><b>In</b> — runtime tensor name(s) entering the step.</li>
<li><b>In shape</b> — tensor shape(s) of the inputs.</li>
<li><b>Static</b> — fixed data used (learned weights or pre-built tables), with shape annotated inline as <code>{name : shape}</code>.</li>
<li><b>Out</b> — runtime tensor name leaving the step.</li>
<li><b>Out shape</b> — tensor shape of the output.</li>
<li><b>Dim out</b> — scalar / 1D vector / 2D matrix / 3D tensor / 4D tensor.</li>
</ul></div>""",
       '<details open><summary>Sizes (declared once)</summary>',
       '<div class="catalog">N_tok       = number of input tokens             (variable per inference)\nd           = per-token vector dim                (e.g., 4096)\nd_ff        = FFN hidden dim                      (e.g., 14336)\nn_h         = number of query heads               (e.g., 32)\nn_kv        = number of K/V heads (GQA)           (e.g., 8)\nd_h         = per-head dim = d / n_h              (e.g., 128)\nvocab_size  = vocabulary size                     (e.g., 128256)\nN           = number of transformer blocks        (e.g., 32)\nmax_pos     = max supported positions             (e.g., 8192)</div>',
       '</details>',
       '<details><summary>Static data catalog (revealed progressively)</summary>',
       """<div class="catalog">{vocab}            : [vocab_size]                    BPE table (pre-built)
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
{H}                : [d × vocab_size]                LM head (learned)</div>""",
       '</details>',
       '<details><summary>Table of contents (50 levels)</summary><div class="toc">']

for idx, lv in enumerate(levels):
    plain = re.sub(r'<[^>]+>', '', lv["title"])
    out.append(f'<div><a href="#L{idx}">{plain}</a></div>')
out.append('</div></details>')

rendered_phase = None
for idx, lv in enumerate(levels):
    ph = phase_for(idx)
    if ph and ph[1] != rendered_phase:
        out.append(f'<h2 class="phase">{ph[1]}</h2>')
        rendered_phase = ph[1]
    out.append(f'<h3 id="L{idx}">{lv["title"]}</h3>')
    out.append(f'<p class="note">{lv["note"]}</p>')
    out.append(render_table(lv))
    if idx >= 14:
        out.append('<p class="footnote">Block rows <code>3.i.*</code> repeat for each block <code>i = 1..N</code>.</p>')

out.append('</body></html>')
html_text = '\n'.join(out)
with open('/home/claude/llm_levels.html', 'w', encoding='utf-8') as f:
    f.write(html_text)
print(f"Wrote {len(html_text)} bytes, {len(levels)} levels")
