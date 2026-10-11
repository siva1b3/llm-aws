# Goal

Understand, through real code and real runtime evidence, how input text travels through a small pretrained decoder-only Transformer and becomes a prediction for the next token. The learner should understand the representations, computations, software, parameters, and supporting assets at every **major operation** of a single forward pass.

This is an **investigation and teaching project**, not a production inference service. Memorizing architecture labels, calling a high-level API, or making a model produce convincing text is not sufficient.

The desired explanatory contract for every major operation is:

> **Input → representation → computation → assets/parameters → output → next operation.**

The code is a means of discovering and verifying that mechanism.
