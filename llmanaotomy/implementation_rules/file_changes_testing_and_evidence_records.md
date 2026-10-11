# File-change and test discipline

- Before editing, inspect the existing file and current milestone (if workspace access exists).
- Change only code/docs relevant to the active box or a directly blocking error.
- State what command was actually run; never invent successful test results.
- Save relevant investigation results in [`../inference_investigation/README.md`](../inference_investigation/README.md) and current status in [`../project_progress/README.md`](../project_progress/README.md).
- If environment execution is unavailable, provide exact commands for the learner and mark the execution status **not run**.
- At meaningful checkpoints, keep the baseline run instructions reproducible in the technical configuration; do not write speculative installed versions.
