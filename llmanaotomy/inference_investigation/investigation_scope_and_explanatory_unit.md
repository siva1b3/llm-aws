# Investigation scope

Reference system: `HuggingFaceTB/SmolLM2-135M-Instruct` on CPU; **one complete forward pass** through the actual model, followed by selection and decoding of one predicted token. Technical choices live in [`../technical_configuration/README.md`](../technical_configuration/README.md).

The experiment's explanatory unit is:

**Input → representation → computation → supporting assets → output → next operation.**

A component can have nested internal operations. Open and document these individually when appropriate, rather than treating a technical label as an explanation.
