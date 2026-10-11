# 03 — Implementation, Inspection, and Verification Rules

**Document role:** Authoritative coding discipline, observability conventions, and completion gates  
**Update when:** Code/inspection conventions or the evidence required to verify a learning step changes.

## 1. Non-negotiable implementation discipline

1. **One small milestone per iteration.** Implement, run, inspect, and explain the current step before expanding the system.
2. **No large code dumps.** Supply only the smallest complete change needed now; avoid huge scripts and multi-file rewrites when a tiny edit works.
3. **No premature abstractions.** Prefer transparent, direct code. Extract helpers only after a repeated, demonstrated need.
4. **No unnecessary dependencies.** Begin with Python, PyTorch, Hugging Face Transformers, and the chosen model's real requirements. Add tools only when the present operation demands them.
5. **No jumping ahead.** Resist explaining all layers, writing the full inference engine, or adding multi-pass generation prematurely.
6. **No production-engineering detours.** No web application, microservices, Docker, Kubernetes, distributed serving, complex instrumentation platform, or performance pipeline for this learning phase.
7. **Keep the baseline intact.** Add small investigation scripts or targeted instrumentation without silently changing the reference behavior.
8. **Make behavior reproducible.** Use a fixed input, CPU execution, inference/eval mode, and deterministic one-token selection initially. Record actual dependency versions after validation.
9. **Be explicit about observational limits.** If a library function, hook, or fused operation hides a value, say so and choose the smallest suitable alternative.
10. **Fix errors where they occur.** Never treat code as verified until real execution/output supports it.

## 2. Phase-1 execution contract

The minimal baseline is one model forward call from tokenized input and one post-processing step that chooses/decodes **one** token:

```text
Text → token IDs → ONE model forward call → logits → one ID → decoded text
```

For the selected causal language model, compare next-token scores from the appropriate final input position (typically `logits[0, -1, :]` for a single unpadded sample). The shape and indexing must be verified against the actual runtime output. Use deterministic argmax initially, not a stochastic generation loop.

- Set `model.eval()` and use `torch.inference_mode()` (or equivalent no-grad inference context).
- Run on CPU; do not silently switch devices or introduce GPU code.
- Do not call `generate()` as the core baseline if it hides repeated forward calls. Use the model forward method directly.
- Do not assume all outputs are printable words: one decoded token may be a text fragment, leading space, or special token.
- If the model has cache-related options, choose settings consistent with one-pass inspection and record them in the technical configuration after real validation.

The actual model implementation and configuration are authoritative over illustrative architecture diagrams.

## 3. For every operation, inspect the data contract

| Check | What to capture |
| --- | --- |
| Input | Actual value or bounded sample, source, Python object type |
| Numerical representation | Tensor shape, dtype, device, meaningful indices |
| Computation | Actual function/module and core mathematical transformation |
| Dependencies | Relevant files, configuration fields, weights, and where they reside |
| Output | Actual value or bounded sample, type, shape, destination |
| Reason | Why this step is necessary in the forward pass |
| Evidence | Command/script run and observed result |

Never claim a value is "from the model" if it was invented for teaching. Tag evidence as **Observed**, **Expected (not yet run)**, or **Illustrative**.

## 4. One preferred human-readable display per data type

| Data | Default presentation | Avoid |
| --- | --- | --- |
| Input / decoded text | Plain labeled text | Mixing prompt and output without labels |
| Token sequence | Numbered token list | Unlabeled concatenated tokens |
| Token ↔ ID mapping | Two-column mapping table | Printing the entire vocabulary |
| Configuration JSON | Pretty-printed relevant keys | Entire unrelated configuration dump |
| Asset inventory | Directory tree + size + purpose | Guessing which files exist |
| Pipeline status | Tiny progressive flow diagram | Full architecture spoiler |
| Vector / embedding | Shape + dtype + indexed selected values | Full long vector by default |
| Matrix / weights | Dimensions + selected rows/columns; optional heatmap if it helps | Full matrix dump |
| General tensor | Shape + dtype + device + selected values | Unbounded output |
| Transformation | Before/after shape and selected values | Shape-only explanation with no mechanism |
| Mathematical step | Small worked numerical example, then real-value link | Pretending toy data is actual |
| Attention weights | Token-labeled heatmap when useful | Unlabeled color image |
| Next-token logits | Top-k token/ID/score table | Dumping all vocabulary scores |
| Probabilities, if calculated | Top-k table (or bar chart if useful) | Conflating logits with probabilities |
| Transformer internals | Small component flow diagram | Revealing every component prematurely |
| Runtime asset placement | Disk → RAM → operation diagram | Treating each tensor as an external file |

**Terminal-first.** Introduce a table, saved report, plot, or heatmap only when it improves the current explanation. Use the simplest available mechanism; do not install visualization packages merely to decorate the output.

## 5. Avoid overwhelming output

- Always show shapes and dtypes for relevant tensors; show at most a small, deliberately chosen subset of values by default.
- Indicate indices and units when relevant (token position, embedding dimension, layer number, score versus probability).
- Limit top-k scores to a small number (e.g., 5–10), unless a deeper analysis is requested.
- For huge arrays, show a bounded preview and the method for deeper inspection if needed.
- Keep runtime prints distinct from explanation text so observations can be copied into investigation notes.
- Keep filenames and exact commands stable and reproducible.

## 6. When to inspect model source code or reproduce an operation

Start with a functioning high-level library call. When a box is opened:

1. Capture the actual input and output first.
2. Identify the library/module/function responsible.
3. Explain the mathematical and programmatic mechanism at a meaningful level.
4. If necessary, write a **small local version of that one operation**.
5. Compare with the original for matching inputs and parameters where practical; state expected numerical tolerances or differences.

Do not rewrite a whole Transformer before explaining one operation. Do not trust conceptual similarity as numerical verification.

## 7. Verification gate: all conditions required

An operation is **technically verified** only when evidence supports its execution and claimed behavior. It is **learner-confirmed** only when the human explicitly confirms understanding.

- [ ] Identify actual input and originating operation.
- [ ] Identify input representation, type, and dimensions where relevant.
- [ ] Identify executing code and explain the core computation.
- [ ] Explain why the operation is needed.
- [ ] Identify relevant assets/parameters and where they are used.
- [ ] Inspect actual output and its representation.
- [ ] Trace the output to the next operation.
- [ ] Run a relevant test/inspection and document the observed result.
- [ ] Resolve relevant discrepancies and misunderstandings.
- [ ] Obtain the learner's explicit confirmation **before advancing**.

Do not conflate **implemented**, **ran successfully**, **technically verified**, and **learner-confirmed**.

## 8. File-change and test discipline

- Before editing, inspect the existing file and current milestone (if workspace access exists).
- Change only code/docs relevant to the active box or a directly blocking error.
- State what command was actually run; never invent successful test results.
- Save relevant investigation results in [`05_INFERENCE_INVESTIGATION.md`](05_INFERENCE_INVESTIGATION.md) and current status in [`06_PROJECT_PROGRESS.md`](06_PROJECT_PROGRESS.md).
- If environment execution is unavailable, provide exact commands for the learner and mark the execution status **not run**.
- At meaningful checkpoints, keep the baseline run instructions reproducible in the technical configuration; do not write speculative installed versions.

## 9. Error handling for teaching

When errors occur, diagnose the smallest failing boundary. Show the root cause, the minimal correction, and the specific rerun needed. Do not mask an unresolved error by introducing a second framework, replacing the model, or abandoning the current black box. If a genuinely incompatible configuration forces a technical change, update the owning file and log the decision.
