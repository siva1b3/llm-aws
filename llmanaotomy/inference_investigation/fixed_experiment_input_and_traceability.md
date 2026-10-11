# Fixed experiment identifier and traceability

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
