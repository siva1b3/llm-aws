# Asset investigation table — add actual entries gradually

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
