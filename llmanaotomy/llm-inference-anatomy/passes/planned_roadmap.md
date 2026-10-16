# Planned roadmap

Updated: 2026-10-16. Direction confirmed by the learner. This is the full workflow; individual pass plans contain only their immediate work. Execution approval remains separate.

## Starting point and destination

**Starting point:** the learner confirms that the agent is running in this repository on EC2. Python, dependencies, and model assets remain unverified. No baseline inference or internal operation has been verified or learner-confirmed.

**Destination:** explain, using runnable code and real evidence, how the fixed input text becomes next-token scores and one decoded token through one complete CPU model forward pass. Explain every major operation, including the important computations inside one representative block, while executing all required blocks. Completion is assessed against [G1–G6](../definition/goal.md).

## Ordered steps

| Step | Work and expected output | Separate check and stopping point |
| --- | --- | --- |
| 1 — Python environment (pass-001) | Inspect local prerequisites and existing files, then create an isolated project environment through separately approved actions. | Verify its Python and pip paths and isolation. Report before installing anything. |
| 2 — CPU dependencies | Agree on the minimal packages and installation limits, then install them in the environment. Retain exact versions and setup commands. | Check imports, versions, and CPU availability. Stop on missing or incompatible prerequisites; agree on recovery before changing the approach. |
| 3 — Model assets | Agree on download limits and obtain the selected pretrained model and compatible tokenizer. Record origin, revision, actual files, sizes, and relevant use constraints. | Verify the required assets and loading configuration. Report before executing inference. Inspect assets in detail only when needed for the current lesson. |
| 4 — Working black box | Create a small baseline using the fixed prompt, one complete CPU forward call, deterministic argmax, and decoding of one token. Keep the internal architecture hidden. | Run and inspect actual input/output, logits shape and indexing, device, inference settings, and one-call behavior. Record evidence for G1; explain the outer contract and wait for learner confirmation. |
| 5 — Input preparation | Open the transformation from the fixed text to the actual model inputs, one component at a time. Retain a bounded trace of representations and responsible code/assets. | Compare actual tokens, IDs, special-token handling, types, shapes, and input preparation with the baseline. Resolve questions and obtain explicit understanding before advancing. |
| 6 — Model computation | Follow the actual implementation progressively. Open one major operation at a time, including meaningful nested computations within one representative block. Continue through the remaining model computation while preserving execution of every required block. | For each operation, inspect real input/output, selected values, shapes, dtypes, code, and parameters. Reproduce important computations where practical using matched inputs and agreed tolerances. Stop after each operation for learner confirmation. Supports G2–G4 and G6. |
| 7 — Scores to one token | Open the remaining steps from model output scores to the selected ID and decoded text. Explain their position outside the numerical model forward pass. | Verify the final input-position indexing, argmax result, ID-to-text decoding, and output limitations against runtime evidence. Obtain learner confirmation. Supports G2, G4–G6. |
| 8 — Assembled learning trace | Connect the confirmed operations into one complete explanation. Retain runnable instructions, asset provenance, inspection code, numerical comparisons, and evidence links. | Have the learner explain the path and distinguish preparation, model computation, selection, and decoding. Revisit gaps one at a time and obtain explicit final understanding confirmation. Supports G1–G6. |
| 9 — Completion review | Assess every goal criterion and required review; populate the final report with delivered outputs, evidence, acceptance, and limitations. | Close only when all applicable criteria and human reviews are satisfied. Otherwise record what remains and the next small action. |

## How we move through it

- Work in order. Each step may require several small actions or passes; only pass-001 is assigned now.
- Agree on the immediate pass plan, then approve each action and its separate check. Report the result before the next action. Close the current pass before opening another.
- During learning: observe → explain → inspect deeper if needed → verify → learner confirms. Code success alone never permits opening the next operation.
- Discover the detailed operation sequence from the real model when step 6 is reached; this roadmap does not preteach its architecture.
- Preserve the working baseline. Keep observations distinct from examples and expectations. Record actual versions and evidence as work occurs.
- CPU execution, one full forward pass per baseline invocation, and one decoded token remain the scope. Repeated generation, training, and production serving are deferred.

Selected settings, display contracts, and teaching rules remain in [constraints](../definition/constraints.md). Live progress belongs only in [current state](../status/current-state.md); immediate work belongs in the [pass plan](pass-001/plan.md).
