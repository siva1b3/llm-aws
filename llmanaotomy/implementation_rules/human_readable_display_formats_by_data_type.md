# One preferred human-readable display per data type

| Data | Default presentation | Avoid |
| --- | --- | --- |
| Input / decoded text | Plain labeled text | Mixing prompt and output without labels |
| Token sequence | Numbered token list | Unlabeled concatenated tokens |
| Token ↔ ID mapping | Two-column mapping table | Printing the entire vocabulary |
| Configuration JSON | Pretty-printed relevant keys | Entire unrelated configuration dump |
| Asset inventory | Directory tree + size + purpose | Guessing which files exist |
| Pipeline status | Tiny progressive flow diagram | Full architecture spoiler |
| Vector / embedding | Shape + dtype + indexed selected values | Full long vector by default |
| Matrix / weights | Dimensions + selected rows/columns; optional heatmap if it helps | Full matrix dump |
| General tensor | Shape + dtype + device + selected values | Unbounded output |
| Transformation | Before/after shape and selected values | Shape-only explanation with no mechanism |
| Mathematical step | Small worked numerical example, then real-value link | Pretending toy data is actual |
| Attention weights | Token-labeled heatmap when useful | Unlabeled color image |
| Next-token logits | Top-k token/ID/score table | Dumping all vocabulary scores |
| Probabilities, if calculated | Top-k table (or bar chart if useful) | Conflating logits with probabilities |
| Transformer internals | Small component flow diagram | Revealing every component prematurely |
| Runtime asset placement | Disk → RAM → operation diagram | Treating each tensor as an external file |

**Terminal-first.** Introduce a table, saved report, plot, or heatmap only when it improves the current explanation. Use the simplest available mechanism; do not install visualization packages merely to decorate the output.
