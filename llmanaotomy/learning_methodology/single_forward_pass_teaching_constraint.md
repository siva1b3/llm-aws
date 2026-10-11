# Phase-specific teaching constraint

Only **one forward pass** is in scope now. Do not use the need for realistic text generation as a reason to introduce repeated inference or an autoregressive loop. Select and decode a single predicted token after one full model invocation. Future generation behavior stays a closed box until the scope changes in [`../project_overview/README.md`](../project_overview/README.md).
