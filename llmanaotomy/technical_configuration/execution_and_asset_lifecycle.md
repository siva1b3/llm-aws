# Execution and asset lifecycle we will inspect

```text
Downloaded/cached files on disk
               ↓ loading
Python code + tokenizer/model objects + parameters in host RAM
               ↓ a real single forward invocation
Temporary tensors and next-token logits in host RAM
               ↓ one-token selection and decoding
Human-readable decoded token text
```

This is an **overview of resource categories**, not a claim to know every internal numerical transformation yet. Open and examine those internal steps only under the onion-peeling protocol.
