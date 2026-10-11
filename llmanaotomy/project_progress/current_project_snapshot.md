# Snapshot

| Field | Current state |
| --- | --- |
| Project | LLM inference anatomy through real code and onion-peeling |
| Phase | Phase 1: exactly one forward pass |
| Learner goal | Understand every major transformation from text input to a single next-token prediction and decoded token |
| Model selected | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| Instance selected | AWS EC2 `t3.large`, Mumbai, Ubuntu 24.04, 30 GiB gp3, CPU-only |
| Implementation milestone | **Not started / no verified baseline** |
| Learning milestone | Initial conceptual contract agreed; no runtime black box yet learner-confirmed |
| Active black box | The entire single-forward-pass LLM interface |
| Runtime execution | **Not performed/verified here** |
| Model download / assets | **Not performed/verified here** |
| Blocking issues | None known; environment existence needs confirmation before running |
