# Non-negotiable implementation discipline

1. **One small milestone per iteration.** Implement, run, inspect, and explain the current step before expanding the system.
2. **No large code dumps.** Supply only the smallest complete change needed now; avoid huge scripts and multi-file rewrites when a tiny edit works.
3. **No premature abstractions.** Prefer transparent, direct code. Extract helpers only after a repeated, demonstrated need.
4. **No unnecessary dependencies.** Begin with Python, PyTorch, Hugging Face Transformers, and the chosen model's real requirements. Add tools only when the present operation demands them.
5. **No jumping ahead.** Resist explaining all layers, writing the full inference engine, or adding multi-pass generation prematurely.
6. **No production-engineering detours.** No web application, microservices, Docker, Kubernetes, distributed serving, complex instrumentation platform, or performance pipeline for this learning phase.
7. **Keep the baseline intact.** Add small investigation scripts or targeted instrumentation without silently changing the reference behavior.
8. **Make behavior reproducible.** Use a fixed input, CPU execution, inference/eval mode, and deterministic one-token selection initially. Record actual dependency versions after validation.
9. **Be explicit about observational limits.** If a library function, hook, or fused operation hides a value, say so and choose the smallest suitable alternative.
10. **Fix errors where they occur.** Never treat code as verified until real execution/output supports it.
