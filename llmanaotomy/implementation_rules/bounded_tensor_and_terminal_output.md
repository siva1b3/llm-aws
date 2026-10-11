# Avoid overwhelming output

- Always show shapes and dtypes for relevant tensors; show at most a small, deliberately chosen subset of values by default.
- Indicate indices and units when relevant (token position, embedding dimension, layer number, score versus probability).
- Limit top-k scores to a small number (e.g., 5–10), unless a deeper analysis is requested.
- For huge arrays, show a bounded preview and the method for deeper inspection if needed.
- Keep runtime prints distinct from explanation text so observations can be copied into investigation notes.
- Keep filenames and exact commands stable and reproducible.
