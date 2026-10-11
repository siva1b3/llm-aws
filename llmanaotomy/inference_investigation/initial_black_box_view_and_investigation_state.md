# Current onion view — not yet run

```text
[Input text] → [ BLACK BOX: LLM single forward pass ] → [next-token scores]
                                                       ↓
                                          [one token selection + decoding]
                                                       ↓
                                              [decoded token text]
```

**Current investigation state:** Planned only. No runtime data, intermediate tensors, or code inspection have yet been supplied/verified in this project handoff. The above is a conceptual outer shell, not an observed trace.

**Known structural constraint:** The chosen model has a sequence of Transformer blocks (the recorded expected count is 30), all used during real forward execution. We will inspect one representative block's internals in detail only when the onion reaches that block.
