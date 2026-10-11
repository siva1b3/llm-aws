# High-level design (HLD)

This document owns the agreed component boundaries, responsibilities, and overall data path.
HLD means high-level design.
Implementation and runtime verification remain pending.

The [project guide](project-guide.md) owns scope and success criteria.
The [technical reference](technical-reference.md) owns model, environment, and runtime settings.

## Initial design

The application prepares text, invokes the full model once, and selects and decodes one predicted token.
This is the planned outer structure:

```text
Fixed input text
  → tokenizer and input preparation
  → [full model: ONE forward pass, internals hidden]
  → next-token logits
  → select ONE token ID
  → decode that ID to text
```

This diagram describes responsibilities. It does not reveal or verify the internal computation of any component.

| Component | Responsibility | Output boundary |
| --- | --- | --- |
| Input preparation | Accept the fixed text and document any template or special-token preparation | Token IDs and other required model inputs |
| Model invocation | Execute the selected model's required blocks once in their actual order | Model output containing logits |
| Score selection | Identify scores at the last valid input position | Scores for the next token |
| Token selection | Choose the highest-scoring ID through deterministic argmax | One token ID |
| Decoding | Convert the selected ID with the compatible tokenizer | Text, possibly a fragment, space, or special token |
| Inspection | Capture bounded observations without silently changing baseline behavior | Labeled terminal output and supporting records |

The boundaries are logical responsibilities. They do not require separate services, files, or helper functions.
Model loading precedes invocation and uses the selected configuration and assets.
The [planned resource lifecycle](technical-reference.md#planned-resource-lifecycle) describes disk and memory placement.

## Constraints and responsibility

Follow the [Phase 1 scope](project-guide.md#phase-1-one-complete-forward-pass).
Execute all required blocks, while teaching one representative block's internals when the learner reaches it.
Selection and decoding must not trigger another model invocation.
Keep future generation loops outside the current design.

The assistant implements and explains the current operation.
The learner confirms understanding through the [learning method](learning-method.md).
Follow the [verification gate](implementation-rules.md#verification-and-advancement-gate) before advancing.

## Evolution

Expand only the component currently reached through onion-peeling.
Update agreed boundaries when an authorized decision changes them.
Keep proposals and unresolved choices in [HLD status](../project-status/hld.md).
Keep observations and corrections in [HLD evidence](../project-evidence/hld.md).
Evidence can challenge the design without automatically changing scope.
Use [LLD](lld.md) for interfaces and the current implementation contract.
