# Constraints and applicable guidance

Status: extracted, awaiting human review.
The selected configuration below is source-reported, not runtime-verified.
Sources: [methodology](../../sources/02_LEARNING_METHODOLOGY.md),
[implementation](../../sources/03_IMPLEMENTATION_RULES.md), and [configuration](../../sources/04_TECHNICAL_CONFIGURATION.md).

## People and authority

| Responsibility | State |
| --- | --- |
| Understanding confirmation | Human learner, explicitly required by the source contract |
| Teaching and minimal implementation | Assistant; learner runs or reviews when direct access is unavailable |
| Engineering planning/execution approval owner | Not named; confirm before opening a pass |
| Required engineering reviewers | Not specified; clarify when relevant |
| Final learning confirmation | Learner |
| Broader acceptance, release, AWS operation, cleanup ownership | Not specified; resolve before dependent work |
| Branch, commit, push, merge policy | Not supplied; preparation does not authorize remote changes |

The user's current instruction authorizes preparation in this destination.
It does not establish a reviewed plan or authorize infrastructure expenditure.

## Selected environment and model

| Setting | Source-selected value |
| --- | --- |
| Provider and region | AWS EC2, ap-south-1 (Mumbai) |
| Instance | t3.large; source lists 2 vCPUs and 8 GiB RAM |
| Platform | x86-64, Ubuntu Server 24.04 LTS |
| Storage and payment | 30 GiB gp3 EBS, On-Demand |
| Accelerator | None; CPU-only by design |
| Model | HuggingFaceTB/SmolLM2-135M-Instruct |
| Model family | Small dense pretrained decoder-only causal Transformer |
| Expected architecture | Approximately 135M parameters; 30 blocks; hidden size 576; vocabulary 49,152 |
| Precision | Prefer ordinary FP32; inspect actual parameter dtype |
| Software | Python 3, PyTorch, Hugging Face Transformers, compatible repository tokenizer |
| Presentation | Simple terminal output first; charts optional when demonstrably useful |

Expected architecture values must be checked against actual downloaded configuration, tokenizer, and implementation.
No machine, downloaded file, installed version, model revision, cache path, dtype, or device has been observed here.
Record actual versions and model revision after validation.
Use a reproducible dependency list or lockfile once the baseline works.
Do not replace selected major choices without explaining the reason and obtaining learner approval.
Record approved changes here, preserve rationale in decisions, and reassess affected evidence.

## Initial runtime contract

- Fixed input: `The capital of France is`.
- Start with batch size 1 unless a documented reason emerges.
- Use CPU, `model.eval()`, and `torch.inference_mode()` or equivalent no-gradient context.
- Execute one explicit causal-LM forward call through the full model.
- Do not hide repeated calls behind a `generate()` loop.
- Use logits from the last valid input position; verify dimensions and tokenizer behavior before indexing.
- Choose the highest-scoring next-token ID with deterministic argmax.
- Decode only that selected ID; report fragments, whitespace, and special tokens honestly.
- Keep optional caching out of the learning path when it obscures computation; record actual flags and behavior.
- Record raw input and any actual chat-template or special-token preparation.
- Preserve the working baseline while adding focused investigation scripts or instrumentation.

## Teaching requirements

Teach plain-language intuition before necessary mathematics.
Build first and introduce prerequisites only when the current operation needs them.
Open one black box at a time; decompose an opaque operation before moving to the next.
Use a small progressive diagram: understood → current → hidden remainder.
Avoid full architecture spoilers, long lectures, large code dumps, premature abstractions, and unnecessary dependencies.
Do not expect the learner to write implementation code manually.
Provide small complete edits or exact minimal commands when direct access is unavailable.
Use a small illustrative numerical example when real tensors overwhelm understanding.
Label observations as Observed, Expected (not yet run), or Illustrative.

Leave space for questions after explaining the current operation.
Correct misconceptions using evidence from the same operation.
Stop for missing execution output, unresolved discrepancies, unclear concepts, or absent explicit learner confirmation.
Do not infer understanding from silence, successful code, or earlier general agreement.

## Inspection and verification contract

For each current operation, capture:

1. Actual input and originating operation.
2. Object type, relevant shape, dtype, device, and meaningful indices.
3. Executing function/module, core mathematical computation, and purpose.
4. Relevant files, configuration, learned parameters, location, loading, and use.
5. Actual output or bounded sample, representation, and next destination.
6. Command/script and real test or inspection result.
7. Resolved discrepancies and explicit learner confirmation before advancing.

Use the source [display conventions](../../sources/03_IMPLEMENTATION_RULES.md) as the detailed reference.
Use labeled text, numbered tokens, token-ID tables, relevant JSON keys, and real asset trees with sizes and purposes.
Show indexed vector/matrix/tensor samples with shapes and dtypes; avoid full vocabulary or parameter dumps.
Show before/after values and shapes rather than shapes alone.
For logits, use a small top-k token/ID/score table; the source's 5–10 entries are examples, not mandatory targets.
Distinguish scores from probabilities.
Use token-labeled attention heatmaps only when useful.
Explain disk, RAM, weights, configuration, generated IDs, intermediate tensors, and actual cache state distinctly.
Not every conceptual asset corresponds to its own file.
State when hooks, library calls, or fused operations hide values.
Reproduce only the current operation when useful; compare matching inputs and parameters with meaningful error/tolerance.
Do not mistake similar formulas for numerical equality.

## Resource, security, and recovery requirements

Keep secrets out of files, prompts, and transcripts.
Do not enable `trust_remote_code=True` without documented necessity and review.
No public web-serving ports are required.
Do not add CUDA, GPU drivers, quantization, distributed execution, or deployment machinery for this phase.
The source cautions about T3 credits, possible surplus-credit charges, EBS charges, and public IPv4 costs.
These are preserved cautions, not a current pricing assessment.
Stop the instance when unused; stopping does not imply storage charges cease.
Budget, runtime/spending limits, account access, and resource cleanup owner are unresolved before paid-resource work.
No deadline, numerical performance target, or service-level commitment was supplied.
Diagnose errors at the smallest failing boundary.
Preserve failed observations and baseline behavior.
Do not mask errors by changing models, frameworks, or the current learning operation.
If a major choice becomes incompatible, explain the impact and obtain the required change authority.

## Guide selection

| Guide | Applicability and reason |
| --- | --- |
| [Core workflow](../../software_files/core/software-ai-development-workflow.md) | Applicable: record ownership, lifecycle, approvals, evidence, and closure |
| [Architecture](../../software_files/core/architecture-and-design-guide.md) | Applicable proportionately: current operation's boundaries, data flow, and design rationale |
| [Quality](../../software_files/core/engineering-quality-guide.md) | Applicable: small readable code, real numerical checks, dependencies, security, and reproducibility |
| [LLM engineering](../../software_files/individual/llm-engineering-guide.md) | Applicable sections 3, 4, 7–9 only as relevant to task limits, model/config identity, inference evidence, and changes |
| [Infrastructure](../../software_files/individual/infrastructure-deployment-operations-guide.md) | Applicable sections 3–4 and 9–11 as relevant to EC2 environment, access, configuration, costs, shutdown, and handoff |
| ML and deep learning | Not selected: no training, fine-tuning, dataset work, or comparative model-quality evaluation; core numerical checks cover this scope |
| GPU infrastructure | Not applicable: CPU-only |
| Frontend; backend/API; database; distributed systems | Not applicable: no UI, service, application database, or distributed system |

LLM guide examples do not mandate RAG, tools, agents, evaluation datasets, model judges, or production benchmarks.
Infrastructure examples do not mandate Docker, Kubernetes, deployment pipelines, alerts, or production release procedures.
Revisit guide selection only when approved scope changes.

## Adoption reconciliation and pending decisions

The core governs engineering passes; the source methodology governs teaching pace.
Both gates apply: permission to execute is distinct from confirmation of understanding.
The source's “next action” describes future environment inspection; preparation does not execute that action.
Original progress and investigation files remain historical input.
Use current-state and active-pass records for new status and findings in this project.
No material technical conflict was found during extraction.

Before the next stage: review scope, derived criteria, guide selection, record ownership, and the engineering approval owner.
Before runtime work: establish actual EC2 availability, access, existing code, resource limits, and cleanup ownership.
Exact versions, model revision, asset inventory, dtype, cache flags, and numerical tolerances await evidence during authorized work.
These unknowns do not block preparation.
