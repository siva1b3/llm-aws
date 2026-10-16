# Project flow — from a working black box to understanding

This is your reference map for learning how real input text becomes one predicted token.
Read the overview whenever you need orientation. Work through only the current small step during a lesson.

The endpoint is **understanding backed by real execution and your explicit confirmation**.
The model is HuggingFaceTB/SmolLM2-135M-Instruct, using CPU execution and one complete forward pass per experiment.

## Starting point and destination

At this roadmap's creation, the project records exist, but no baseline, model download, or runtime lesson is independently verified.
The selected EC2 environment is unverified. The whole LLM remains the current learning black box.

```text
Agree on the goal
  → establish the working environment
  → make the smallest real baseline work
  → understand one operation at a time
  → connect the understood operations
  → demonstrate and confirm the complete flow
```

The final result is a reproducible demonstration and your explanation of the complete path from text to one decoded token.
All required model blocks execute; we investigate one representative block in detail.
Repeated experiments are allowed for checks. Feeding predicted tokens back to generate a response is outside this phase.

## Parties and responsibilities

| Code | Party | Contribution and boundary |
| --- | --- | --- |
| H | You, the human learner | Review goals, approve project actions, inspect results, ask questions, confirm understanding, and accept outcomes. You need not write implementation code. |
| C | Codex | Write small changes, teach the current operation, inspect evidence, perform authorized checks, and maintain records. Cannot confirm understanding on your behalf. |
| L | Local Windows workspace, C:\main\llm-aws\llmanaotomy | Stores records and local edits. Codex operates it through tools under your access control. This does not establish remote access. |
| E | Future verified execution environment | Runs the model and produces evidence. Selected target: EC2 t3.large, ap-south-1, Ubuntu 24.04, CPU-only. Actual identity and access remain unresolved. |

E becomes involved only after its identity, access, and operator are established.
For each remote action, name the operator: Codex with verified authorized access, or you running Codex's exact commands.
If you operate E, provide the actual output for Codex to inspect. Environments produce evidence; they cannot approve or accept work.

## How to use this map

- Follow [current state](../status/current-state.md) for your actual position and [next actions](../status/next-actions.md) for the immediate next item.
- The numbers below are roadmap steps, not pass numbers, completed milestones, or execution approvals.
- Each row has one outcome and a separate check. Stop after the row for the stated review.
- Agree on the current pass and action scope before execution, following [operating instructions](../AGENTS.md).
- Keep later internals closed until their lesson. The block section deliberately leaves actual operation names to runtime inspection.
- If a row is too large, split it into smaller actions before starting. There is no fixed lesson duration.
- Preserve this file as the overall route. Store actual results in the active pass and current position in status records.

You requested this saved roadmap. It is a reference document; no pass has been activated and no project execution has been approved.

## Stage A — know where we are working

| Step | Small step and expected outcome | Parties and work | Separate check and stopping point |
| --- | --- | --- | --- |
| 01 | Review the destination: an agreed one-pass learning goal and roles. | H confirms or corrects intent; C explains boundaries; L stores records. | H confirms the goal and responsibilities. Correct disagreements before planning execution. |
| 02 | Establish the execution context: known machine identity, access, operator, and existing files. | H identifies available access; C inspects within approved scope; L stores findings; E is involved only if accessible. | C checks actual observations against selected settings. Stop if machine identity, operator, or access is unknown. |
| 03 | Resolve a missing environment, only if needed: an available agreed CPU workspace. | H approves any resource creation and costs; C proposes the smallest setup; the named operator prepares E; L records the result. | C checks reachability and actual configuration. Skip creation when an adequate environment already exists. |
| 04 | Establish Python: a known interpreter and usable project location. | C supplies the smallest setup or inspection; the named operator runs it on E; H reviews; L records observed settings. | C checks actual executable/version and project path. Stop on errors or conflicting existing files. |
| 05 | Establish dependencies: usable PyTorch and Transformers versions. | C prepares minimal dependency work; the named operator runs it on E; H reviews; L records versions. | C checks imports, actual versions, and CPU availability. Installation alone does not prove the model runs. |

Steps 02–05 require their own approved action scopes. Checking an existing environment does not authorize creating or replacing one.

## Stage B — make the outer black box work

Keep the view small: **input text → LLM black box → one-token output**.

| Step | Small step and expected outcome | Parties and work | Separate check and stopping point |
| --- | --- | --- | --- |
| 06 | Obtain the selected model assets: identifiable local files and a recorded revision. | C specifies the selected repository; H approves download scope; the named operator obtains assets on E; L stores inventory. | C checks actual paths, sizes, revision, and relevant configuration. Expected filenames are not evidence. |
| 07 | Load model and tokenizer: usable objects with known device and dtype. | C writes minimal loading code; the named operator runs it on E; H reviews; L records results. | C checks successful loading, model identity, CPU placement, and actual dtype. Resolve discrepancies before inference. |
| 08 | Prepare the baseline artifact: a small script for the fixed prompt and exactly one forward call. | C writes code in the agreed location; H reviews its purpose; L stores records; E is the selected run target. | C inspects one-call behavior, eval/no-gradient mode, final-position logits, argmax, and one-token decoding. Execution remains not run. |
| 09 | Run the baseline: actual logits, one selected ID, and one decoded token. | The named operator runs on E; C captures and checks results; H observes; L records evidence. | C checks actual shapes, indexing, input, selected ID, and decoded output against the baseline contract. No invented expected word is required. |
| 10 | Understand the outer contract: you can explain what entered and what came out. | C explains only the outer interface; H questions or paraphrases; L records explicit confirmation. E is unnecessary unless a rerun is approved. | H confirms the difference between one predicted token and a complete response. Stay here until the baseline is clear. |

The fixed input is `The capital of France is`.
The selected token may decode to a fragment, whitespace, or special token.
The baseline may use library calls as black boxes. We inspect their mechanisms only in the following lessons.

## Stage C — open the path into the model

Each row below is a learning boundary. Use the small-operation cycle later in this file before advancing.

| Step | Small step and expected outcome | Parties and work | Separate check and stopping point |
| --- | --- | --- | --- |
| 11 | Open text preparation: understand the exact text passed to the tokenizer. | C traces raw input and any preparation on E; H explains the distinction; L records evidence. | C compares actual supplied text with the fixed prompt, including any special-token or template handling. H confirms understanding. |
| 12 | Open tokenization: connect text pieces to actual token IDs. | C traces the tokenizer on E, splitting its mechanism into smaller lessons as needed; H follows one mapping; L records findings. | C checks real pieces, IDs, types, and relevant tokenizer assets. H explains the current transformation before another is opened. |
| 13 | Open model input preparation: understand the tensors actually passed into the model. | C inspects one input field at a time on E; H interprets its purpose; L records findings. | C checks shape, dtype, device, indices, and any mask/position inputs actually used. H confirms each field's role. |
| 14 | Open the first learned representation: understand how token IDs become model vectors. | C traces the actual operation and relevant parameters on E; H follows one selected value; L records evidence. | C checks input/output correspondence and a bounded parameter/value sample. H explains the mechanism, not just the shape change. |

## Stage D — understand one representative block

This stage contains several small lessons. The actual implementation determines their names and order.
Do not treat “understand the block” as one large coding or teaching action.

| Step | Small step and expected outcome | Parties and work | Separate check and stopping point |
| --- | --- | --- | --- |
| 15 | Locate one representative block: identify its real input, output, and place in the full model. | C inspects the actual model on E; H reviews the boundary; L records the selected block and reason. | C verifies the block exists and its observed interface. H confirms what remains inside the black box. |
| 16 | Open just the next internal operation: understand one real transformation. | C traces that operation's code, parameters, and values on E; H questions and explains; L stores findings. | C checks the full input-to-output contract. H explicitly confirms before another operation is exposed. |
| 17 | Reproduce a useful computation when practical: connect readable code with the actual result. | C writes a small version of the current operation; the named operator runs it on E; H interprets the comparison; L records evidence. | C compares matching inputs and parameters using stated tolerances. Explain differences; record limits when reproduction is impractical. |
| 18 | Reconnect the understood operation: know where its result goes next. | C traces the immediate destination on E; H explains the local connection; L updates current findings. | C checks actual data continuity. H confirms the connection. Return to 16 for the next internal operation only after this gate. |
| 19 | Close the representative block: explain its complete internal path using accumulated evidence. | C assembles only the already-understood pieces; H walks through them; L stores the confirmed explanation. E is used only for approved targeted checks. | C checks that no major operation remains unexplained. H explicitly confirms the whole block before moving outward. |

Repeat 16 → 17 where useful → 18 as many times as the real block and your understanding require.
If the current operation contains another meaningful black box, apply the same cycle inside it before moving sideways.

## Stage E — connect the model to the predicted token

| Step | Small step and expected outcome | Parties and work | Separate check and stopping point |
| --- | --- | --- | --- |
| 20 | Follow the remaining blocks: understand why all required blocks execute in sequence. | C traces actual block interfaces and count on E; H explains shared structure and distinct learned parameters; L records evidence. | C checks execution order and tensor continuity without repeating every block lesson. H confirms the full-stack relationship. |
| 21 | Open the remaining output transformations: understand how the final representation becomes vocabulary scores. | C exposes one actual operation at a time on E; H explains each; L records findings. | Apply the small-operation cycle to each operation, including any final normalization/projection present. Do not assume the architecture. |
| 22 | Inspect next-token scores: understand which position predicts the following token. | C shows the actual output shape and a small top-k score table from E; H interprets it; L records evidence. | C checks final-valid-position indexing and token/ID/score mapping. H distinguishes logits from probabilities. |
| 23 | Open token selection: understand why argmax chooses this ID. | C traces deterministic selection on E; H relates the selected ID to scores; L records findings. | C checks the selected ID against the actual maximum score. H confirms selection occurs after the model forward call. |
| 24 | Open decoding: understand how the chosen ID becomes the displayed text. | C traces the selected ID through the actual tokenizer decoder on E; H explains the result; L records evidence. | C checks exact decoded output, including spaces/fragments/special tokens. H confirms no second forward call is needed. |

## Stage F — demonstrate the complete understanding

| Step | Small step and expected outcome | Parties and work | Separate check and stopping point |
| --- | --- | --- | --- |
| 25 | Assemble the final explanation: a complete flow made from already-understood operations. | H explains the path to another developer; C checks it against evidence; L stores the explanation. | Check input, representation, computation, code, assets, output, and destination for every major transformation. Return to the specific gap if needed. |
| 26 | Demonstrate reproducibility: a documented command produces a supported one-pass trace. | C prepares the recorded invocation; the named operator runs on E; H inspects the result; L records versions, settings, and evidence. | C compares against the preserved baseline under matching conditions. Investigate differences rather than assuming a particular decoded word. |
| 27 | Accept and close: a reviewed outcome against all five goal criteria. | C assembles criterion evidence and final report; H gives explicit final understanding confirmation and acceptance; L stores the result. The named E operator resolves any running work. | H and C review G1–G5, remaining limitations, and output ownership. Keep completion pending if a criterion or required review is unmet. |

## The small-operation cycle

Use this cycle for each operation in steps 11–24. A lesson may need several short exchanges.

| Part | Codex's contribution | Your contribution | Outcome and check |
| --- | --- | --- | --- |
| A. Locate | Show understood → current → hidden remainder. | Confirm the question you are trying to answer. | One clear operation and learning objective. |
| B. Inspect input | Show real values or a bounded sample, type, shape, dtype, and device where relevant. | Identify what the data represents. | Evidence tied to the actual incoming operation. |
| C. Trace mechanism | Open the executing function and relevant parameters/assets; explain plain-language intuition first. | Ask about the current mechanism. | A meaningful computational explanation rather than an architecture label. |
| D. Inspect output | Show actual output, changed representation, and its immediate destination. | Describe what changed and why. | Input-to-output continuity supported by evidence. |
| E. Verify | Perform the agreed focused check; compare a small reproduction where useful. | Inspect the comparison or demonstrate your interpretation. | Supported claim, stated tolerance/limits, and resolved discrepancies. |
| F. Confirm | Ask for understanding confirmation and record the actual response. | Explicitly confirm, or identify what remains unclear. | Permission to leave this learning boundary only when you are ready. |

The named operator uses E for actual runtime work; L stores records. No environment is required for a purely conversational explanation.
Code written, code executed, technically verified, and learner-confirmed are separate states.
If confused, stay with the current operation and try a smaller example. If execution fails, stop dependent work and resolve that failure.

## How the steps serve your goal

| Goal criterion | Supporting steps | What you should have at the end |
| --- | --- | --- |
| G1: real one-call baseline | 02–10, 26 | Actual reproducible one-pass input, logits, selected ID, and decoded output. |
| G2: explain every major operation | 11–24 | Evidence-backed explanations of each input, computation, parameter/asset, output, and destination. |
| G3: understand one block while all blocks execute | 15–20 | Detailed understanding of one block and verified placement within the full stack. |
| G4: reproduce useful computations | 17, repeated where practical | Readable comparisons with actual inputs, parameters, tolerances, and stated limitations. |
| G5: explain and confirm the complete flow | 25–27 | Your complete explanation, explicit understanding confirmation, and accepted final outcome. |

The detailed [goal](../definition/goal.md) and [constraints](../definition/constraints.md) remain authoritative for scope and methods.
Update this roadmap when the learning route changes by agreement. Keep its step IDs stable where possible.
At a checkpoint, record the current roadmap step in current state and next actions; record execution details in the active pass.
Never mark a roadmap step complete merely because its code exists or a later step is listed here.
