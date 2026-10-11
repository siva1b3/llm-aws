# 05 — Inference Investigation Notebook

**Document role:** Evidence-based technical anatomy, discovered assets, progressively opened operations, and verified numerical findings  
**Update when:** An operation has actually been inspected; relevant code, assets, measurements, or equations have been verified; or a previous technical finding is corrected.

> **This is not a prewritten architecture tutorial.** Grow it one black box at a time, with the learner. Future internals remain collapsed until investigated. "Expected" values are not findings.

## 1. Investigation scope

Reference system: `HuggingFaceTB/SmolLM2-135M-Instruct` on CPU; **one complete forward pass** through the actual model, followed by selection and decoding of one predicted token. Technical choices live in [`04_TECHNICAL_CONFIGURATION.md`](04_TECHNICAL_CONFIGURATION.md).

The experiment's explanatory unit is:

**Input → representation → computation → supporting assets → output → next operation.**

A component can have nested internal operations. Open and document these individually when appropriate, rather than treating a technical label as an explanation.

## 2. Current onion view — not yet run

```text
[Input text] → [ BLACK BOX: LLM single forward pass ] → [next-token scores]
                                                       ↓
                                          [one token selection + decoding]
                                                       ↓
                                              [decoded token text]
```

**Current investigation state:** Planned only. No runtime data, intermediate tensors, or code inspection have yet been supplied/verified in this project handoff. The above is a conceptual outer shell, not an observed trace.

**Known structural constraint:** The chosen model has a sequence of Transformer blocks (the recorded expected count is 30), all used during real forward execution. We will inspect one representative block's internals in detail only when the onion reaches that block.

## 3. Fixed experiment identifier and traceability

Use one consistent input for comparisons initially:

```text
The capital of France is
```

When a real run exists, record:

- Actual invocation command and script path.
- Python, PyTorch, Transformers versions and model revision.
- Raw input string and any chat-template/special-token preparation applied.
- Execution settings relevant to data interpretation (device, dtype, caching, inference mode).
- Actual output and bounded intermediate samples.
- The observed result of checks/tests.

Never insert fabricated sample tensor values into an **Observed** entry.

## 4. Component record template — copy ONLY when investigating that component

### [Operation name] — [not started / implemented / executed / technically verified / learner-confirmed]

- **Why this operation is opened now:**
- **Where it sits:** `[understood] → [current] → [hidden remainder]`
- **Input and origin:**
- **Input type/shape/dtype/device:**
- **Core computation and purpose:**
- **Executing code / function / relevant source location:**
- **Assets and parameter references:** file path or in-memory tensor, where loaded from, read/write behavior.
- **Output and destination:**
- **Actual observed sample:** (never invent; label expected/illustrative separately).
- **Inspection command or script:**
- **Verification criterion and result:**
- **Simplifications and limitations:**
- **Learner confirmation:** Pending until explicitly given.
- **Open questions:**

For a nested black box, create a child subsection and use the same data contract. Only deepen as necessary for comprehension.

## 5. Asset investigation table — add actual entries gradually

| Asset / parameter | Actual location | Type / format | Loaded/used by | Observed evidence | Status |
| --- | --- | --- | --- | --- | --- |
| *No assets inspected yet* | — | — | — | — | Not started |

Relevant distinctions:

- Persistent downloaded files on disk versus library source code.
- Loaded learned weight tensors in RAM versus static configuration metadata.
- Tokenizer mappings in their actual representation versus generated token IDs.
- Intermediate tensors generated per invocation versus persistent model weights.
- Cache/state as implemented during the current forward invocation versus any future cross-invocation optimization.

Not every conceptual asset corresponds to an individual file. Verify what exists.

## 6. Runtime observation register — no claims before execution

| ID | Operation | Observation | Evidence location/command | Confidence / state |
| --- | --- | --- | --- | --- |
| — | — | No real inference observations recorded | — | Not started |

Use readable, bounded presentations. The default display for each data category is defined in [`03_IMPLEMENTATION_RULES.md`](03_IMPLEMENTATION_RULES.md); record actual observed samples here, not a full dump of millions of tensor values.

## 7. Computation reproductions and comparisons

| Operation | Original output | Reproduction output | Comparison criterion | Result |
| --- | --- | --- | --- | --- |
| *Nothing reproduced yet* | — | — | — | Not started |

Do not claim equality for operations merely because they have similar formulas. Inspect compatible inputs and parameters, calculate meaningful error/tolerance, and record actual results when reproduction is attempted.

## 8. What this file should NOT become

- A fully expanded Transformer diagram before the learner reaches its internals.
- A speculative list of files supposedly downloaded.
- A raw log dump.
- A duplication of current milestone checkboxes from [`06_PROJECT_PROGRESS.md`](06_PROJECT_PROGRESS.md).
- A replacement for the learner's confirmation gate.

Use this file to store *technical findings*. Use the progress file to store *where the learner is now and what to do next*.

## 9. First intended investigation (not yet executed)

Establish the minimal, real black-box baseline that performs **one forward call**, selects one next-token ID, and decodes it. Once executed, record only what was observed: invocation, the input, model-output/logit shape, selected token ID, and decoded token. Then stop for learner confirmation before peeling the next layer. Do not mark this step complete based on its description.
