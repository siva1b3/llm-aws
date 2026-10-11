# Low-level design (LLD)

This document owns the agreed implementation contract for the current operation.
LLD means low-level design.
The baseline contract is planned. No baseline script or runtime result is verified in the supplied record.

Use [HLD](hld.md) for component responsibilities and overall boundaries.
Use [technical reference](technical-reference.md#initial-runtime-settings) for selected settings and the fixed prompt.
Use [implementation rules](implementation-rules.md) for coding, displays, and verification standards.

## Initial baseline contract

Start with the smallest complete Python script that demonstrates the outer single-pass interface.
Function names and file paths remain undecided until the actual workspace is inspected.
Logical steps do not require separate helper functions.

| Step | Input contract | Intended behavior | Output contract |
| --- | --- | --- | --- |
| Load assets | Selected repository and validated environment | Load the compatible tokenizer and pretrained model | Usable tokenizer/model objects |
| Prepare input | Fixed text | Apply and record the chosen preparation | Model-compatible numerical inputs |
| Invoke model | Prepared inputs and loaded parameters | Follow evaluation and inference settings, invoke forward once | Actual model output |
| Read scores | Output logits and valid input position | Check dimensions and select the correct position | Next-token score vector |
| Select ID | Score vector | Apply deterministic argmax | Exactly one integer token ID |
| Decode ID | Selected ID and compatible tokenizer | Decode only that ID | Labeled decoded text |
| Inspect result | Actual inputs, outputs, and settings | Print bounded, readable observations | Reproducible baseline evidence |

These are intended interfaces, not observed types, shapes, or values.
Confirm actual tokenizer keys, object types, tensor dimensions, dtype, and device during execution.
Record template and special-token preparation rather than assuming the raw prompt reaches the model unchanged.

## Indexing and assumptions

Read scores from the last valid input position.
For a single unpadded sample, `logits[0, -1, :]` is a candidate indexing expression.
The [single-pass rules](implementation-rules.md#single-pass-execution) require runtime checks before relying on it.
Do not assume that padding or another preparation method preserves that indexing.
Do not hardcode recorded architecture expectations without inspecting the actual configuration.

## Failure handling and verification

Stop at the smallest failing boundary if loading, preparation, invocation, indexing, or decoding fails.
Record the actual error and the correction needed for a rerun.
Do not report prediction success after an unresolved failure.
Follow [error and handoff rules](implementation-rules.md#file-changes-errors-and-handoffs).

Check one model forward invocation, correct score selection, one selected ID, and one decoded token.
Check the actual shape, dtype, device, and relevant settings.
Use the [data contract](implementation-rules.md#data-contract-for-each-operation) to capture the command and observed results.
Technical verification and learner confirmation remain separate gates.

## Evolution

Add detail only for the current operation after the baseline and learning gates permit it.
Link verified code locations and evidence when they become available.
Do not prewrite the internal design of every Transformer operation.
Keep unfinished details in [LLD status](../project-status/lld.md).
Keep actual code inspections, comparisons, tolerances, and corrections in [LLD evidence](../project-evidence/lld.md).
Use the project decision process for changes to agreed scope or technical choices.
