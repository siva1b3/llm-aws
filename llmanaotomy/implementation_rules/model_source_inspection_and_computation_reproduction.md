# When to inspect model source code or reproduce an operation

Start with a functioning high-level library call. When a box is opened:

1. Capture the actual input and output first.
2. Identify the library/module/function responsible.
3. Explain the mathematical and programmatic mechanism at a meaningful level.
4. If necessary, write a **small local version of that one operation**.
5. Compare with the original for matching inputs and parameters where practical; state expected numerical tolerances or differences.

Do not rewrite a whole Transformer before explaining one operation. Do not trust conceptual similarity as numerical verification.
