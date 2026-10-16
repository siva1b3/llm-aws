# Pass-003 — What do we need before the model can run?

Updated: 2026-10-16 (Asia/Calcutta).
Disposition: proposed, reserved and inactive. Recreated at the learner's explicit request. Plan approval, execution approval, and learner confirmations are pending.

## Learning purpose

You should be able to identify the selected model's real files, explain their broad purposes, and distinguish a model saved on disk from a model loaded into memory and a model actually executing.

I am your teacher, technical guide, and implementer. You are the learner. I handle the supporting technical work, explain choices when needed, and use actual observations to teach. You ask questions, inspect the observations, and explain or explicitly confirm understanding. You are not expected to write setup code or manage a technical task list.

This is step 3 of the [roadmap](../planned_roadmap.md). It prepares our first prediction demonstration. The model stays a closed box:

```text
Input text → [model: closed box] → one predicted token
```

That diagram shows our intended demonstration, not an observed run. This pass stops before inference. It supports prerequisites for [G1–G6](../../definition/goal.md); completing it does not satisfy those final criteria.

## Lessons and their outcomes

The table uses the same columns as the roadmap. Each row is one small action with a separate check. We work in order. After each row, we discuss the result and stay with that question until it is clear. Later rows are planned outcomes, not lessons already delivered.

| Step and position | What we do | Your role — learner | My role — teacher and technical guide | Outcome of the step | Separate check / evidence |
| --- | --- | --- | --- | --- | --- |
| **A1. Where will the model live? — not started** | Establish where the files can live and distinguish disk storage from working memory. Inspect the actual target, existing files, free disk space, and RAM. | Look at the observed locations and resources. Ask questions and explain or confirm what disk and RAM each hold. | Explain the distinction with a simple example. Recommend a location, perform the approved inspection, and show only the relevant observations. | **You can explain why downloading needs disk space and loading needs RAM.** We know the actual local starting conditions. | **V1:** separately check the target, existing files, and resource observations. Do not declare capacity sufficient until file requirements are known. Ask you to distinguish disk from RAM, then stop. |
| **A2. What are we getting when we download a pretrained model? — not started** | Review the selected source and connect its required assets to their broad purposes: learned weights, model configuration, and tokenizer assets. Agree on exactly what to obtain. | Explain or confirm the purpose of each category. Ask about unfamiliar terms; review the proposed download and limits with my guidance. | Inspect authorized source metadata, identify the exact revision and compatible file set, and explain only the broad asset roles. Propose sizes, destination, and resource limits with reasons. | **You can explain why installing a library is different from obtaining a pretrained model and why its supporting assets matter.** | **V2:** compare the proposal with source metadata, model identity, use terms, available resources, and trust requirements. Check your explanation of the asset purposes. Resolve missing decisions and stop before download. |
| **A3. Bring those model files onto our machine — not started** | Download only the agreed files from the agreed revision and observe what downloading actually produced. | Inspect the result and ask questions about what has changed on the machine. | Perform the authorized download within agreed limits. Explain the actual outcome and disclose partial effects or errors. | **You can identify what “downloaded” means in our real example.** The agreed files are present on disk if the check passes. | **V3:** separately confirm the download outcome, revision, and expected file presence; inspect partial effects before any retry. Report and stop before the detailed file walkthrough. |
| **A4. Connect the real files to their purposes — not started** | Walk through a bounded inventory of the downloaded files, their sizes, and their roles. Keep internal computations closed. | Point to the actual assets and connect them to the purposes discussed earlier. Question anything that remains unclear. | Show a short directory tree and explain which files the later loader uses. Identify provenance and file identifiers without dumping large file contents. | **You can explain what we have on disk using actual files rather than labels alone.** | **V4:** compare the inventory with the approved file set and reliable source metadata; retain identifiers and integrity limits. Check your file-to-purpose explanation, then stop. |
| **A5. What changes when we load the model? — not started** | Load the verified local tokenizer and model on CPU, offline. Observe loaded objects, parameter device, and numerical precision without running a prediction. | Explain or confirm what remains on disk, what is now in RAM, and why loading has not yet produced a prediction. | Provide minimal loading code, explain it in small pieces, and show bounded actual observations. Introduce numerical precision only as needed for this loading choice. | **You can distinguish downloaded files, loaded objects and weights, and execution.** A successful loading check prepares the later demonstration. | **V5:** separately verify local loading, CPU placement, agreed dtype, and absence of forward/generation calls. Check your distinction between downloaded, loaded, and executed; stop before inference. |
| **A6. Explain what we now have — not started** | Connect the confirmed lessons into a short account of what the model needs and what loading accomplished. Review this pass together. | Explain the asset roles and disk-to-RAM transition in your own words, or identify questions. Decide whether you understand and accept the pass for closure. | Review your explanation against actual evidence, revisit gaps within this boundary, and retain reproducible instructions and the learning checkpoint. | **You can explain our readiness for a first prediction and what has not yet happened.** | **V6:** review technical evidence, provenance, instructions, limitations, and explicit learning confirmations. Record acceptance and closure only with your agreement. The next roadmap step needs its own plan. |

## Supporting technical boundaries

- Work type: teaching with bounded model-asset preparation in the existing repository. Target: the learner-confirmed EC2 environment described in [current state](../../status/current-state.md). This chat now uses a worktree; earlier environment checks do not prove its virtual environment or model paths exist here. Confirm the actual execution target within A1 before dependent work.
- Inputs: the selected `HuggingFaceTB/SmolLM2-135M-Instruct`, CPU-only choice, and software decisions in [constraints](../../definition/constraints.md); recorded setup evidence in [pass-002](../pass-002/evidence.md). No package reinstallation or model compatibility is inferred.
- Design: verified files on disk → local tokenizer/model objects and weights in RAM. No prompt tokenization lesson, internal architecture dissection, forward call, or generation. Download and loading remain separate actions and procedures.
- Dependencies: A1/V1 and its learner confirmation precede A2; A2/V2 and resolved download choices precede A3; A3/V3 precede A4; A4/V4 and learner confirmation precede A5; A5/V5 and learner confirmation precede A6. Stop after every action/check pair.
- Before applicable execution, I will specify exact commands, target, affected paths, expected observations, and stopping point. I will inspect existing user changes and preserve unrelated files. Scripts, when useful, will keep action and verification procedures separate.
- Destination, exact revision/file set, download bytes, disk/RAM allowances, time and retry limits, and partial-download cleanup remain unresolved. I propose these choices; we agree before affected execution. No remote executable model code, package upgrades, system changes, resource creation, publication, or release is included.
- No migration or rollout is involved. Files persist after downloading. On failure, stop, retain partial effects, and check actual state. No automatic retry, deletion, replacement, or dependency repair outside approved scope.

## Evidence, completion, and authority

All actions are **not started**. V1–V6 are **not run**. All learner confirmations and execution approvals are **pending**. Observed, expected, and illustrative values must remain distinct.

I own code, technical checking, explanations, and checkpoint records. You own understanding confirmation, plan and execution decisions, acceptance, and closure. No additional contributors are assigned. Technical success does not establish understanding or permission to advance.

Completion requires verified assets and provenance, successful offline CPU loading, reproducible commands, explicit understanding of the asset roles and downloaded/loaded/executed distinction, and your acceptance and closure decision. G1–G6 remain outstanding. We do not cross into the prediction demonstration in this pass.

Material risks: insufficient resources can interrupt download/loading; I check requirements before those actions. A changing source can break reproducibility; I retain an exact revision. Incompatibility can block loading; I preserve the failure and propose a focused diagnosis. Too much explanation can obscure the lesson; I teach only the current question and revisit it until you confirm understanding. Resource decisions remain unresolved; no recovery work is authorized by this proposal.

## Current status

Plan approval and execution approval are pending. All lessons are not started, checks are not run, and learner confirmations are pending.

Next action: review this plan together. After agreement, explain A1/V1 and obtain execution approval before starting.
