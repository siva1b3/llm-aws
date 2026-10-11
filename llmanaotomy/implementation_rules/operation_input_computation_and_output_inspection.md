# For every operation, inspect the data contract

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
