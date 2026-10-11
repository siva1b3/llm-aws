# Asset inventory: expected categories, not confirmed filenames

Do **not** pretend the repository has been downloaded. When downloaded, inspect real local file listings and record exact paths, sizes, and purposes in [`../inference_investigation/README.md`](../inference_investigation/README.md) when each asset is used.

Potential assets include:

- Model architecture/configuration JSON.
- Tokenizer JSON, configuration, special-token mappings, or vocabulary-related files.
- Model parameter files (commonly safetensors) and any index, if present.
- Optional generation configuration or repository documentation.
- Cached files and executable implementations provided by installed Python packages.

The exact set depends on the model revision and download method. Not every named category must be a separate file. Learned embeddings typically live inside model weight tensors, not a standalone token-to-vector text table.

Record the repository revision/commit when actually downloaded for reproducibility.
