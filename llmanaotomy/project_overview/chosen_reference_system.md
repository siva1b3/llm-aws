# Chosen reference system

We use a **small, dense, decoder-only Transformer** running on CPU, with pretrained weights and model files from Hugging Face. Specific current choices live only in [`../technical_configuration/README.md`](../technical_configuration/README.md). These are **chosen settings**, not evidence that the machine is provisioned or that inference has executed.

We may use a high-level inference library to get a working baseline, but subsequently open its black boxes and examine actual computations. Real inference must remain faithful to the selected model; simplified educational examples must be clearly labeled as such.
