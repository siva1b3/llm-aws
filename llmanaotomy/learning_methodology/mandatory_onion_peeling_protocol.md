# The onion-peeling protocol — mandatory 🧅

Begin with the smallest working view:

```text
Input text → [ BLACK BOX: LLM ] → one-token output text
```

For the current black box:

1. First make it work, or inspect an existing working execution.
2. Identify the next **single** component or transformation worth exposing.
3. Show its input, operation, output, dependencies, and its position in the flow.
4. If its mechanism still contains meaningful black boxes, open those **one at a time**.
5. Verify the mechanism with real code/output or a focused experiment.
6. Ask the learner to explain, question, or confirm it.
7. Only after explicit understanding confirmation, move to the next component.

A useful evolving diagram has only three visual categories:

```text
[Already understood] → [CURRENT box: opening now] → [Remaining black box]
```

Keep the model's full internal architecture hidden until it becomes relevant. Avoid dumping a completed flowchart as a spoiler. The architecture investigation record may grow progressively as evidence is gathered.

### Depth versus pace

A label such as *tokenizer*, *embedding*, *attention*, or *projection* is not an explanation. Keep peeling until the learner knows the important computational mechanism. But do not descend into irrelevant implementation minutiae (e.g., every CPU instruction) when they add no understanding.
