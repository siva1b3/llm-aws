# Constraints

Status: extracted, awaiting human review. Sources: [methodology](../../sources/02_LEARNING_METHODOLOGY.md), [implementation](../../sources/03_IMPLEMENTATION_RULES.md), and [configuration](../../sources/04_TECHNICAL_CONFIGURATION.md).

## Supplied configuration

These are source-recorded selections, not verified infrastructure, installations, or fresh approvals.

| Area | Supplied selection |
| --- | --- |
| Environment | AWS EC2, ap-south-1 Mumbai, t3.large, On-Demand |
| Intended capacity | 2 vCPUs, 8 GiB RAM, 30 GiB gp3 EBS |
| OS and device | Ubuntu Server 24.04 LTS, x86-64, CPU only; no GPU or CUDA |
| Model | HuggingFaceTB/SmolLM2-135M-Instruct, small dense decoder-only Transformer |
| Stack | Python 3, PyTorch, Hugging Face Transformers, model-compatible tokenizer |
| Input | The capital of France is |
| Execution | Batch size 1 initially; model.eval() and torch.inference_mode() or equivalent no-gradient context |
| Output | One explicit forward call; last valid input position's logits; deterministic argmax; decode only one ID |
| Precision | Ordinary FP32 preferred; actual dtype requires inspection |
| Presentation | Terminal first; optional plots only when the current explanation needs them |

The source expects approximately 135M parameters, 30 blocks, hidden size 576, and vocabulary size 49,152.
These are unverified expectations. Actual downloaded configuration and executable implementation take precedence.
Record installed versions, model revision, cache paths, actual dtype/device, and actual cache settings only after observation.
Validate tensor indexing and tokenizer behavior; do not assume logits dimensions from examples.
Document raw input and any chat-template or special-token preparation.
Do not use an opaque multi-token generate() loop as the baseline.
One token may be whitespace, a fragment, or a special token.
Keep optional caching out of the learning path when it obscures the one-pass computation.

## Teaching and implementation contract

Build first, then execute, observe, explain, inspect, verify, and obtain explicit learner confirmation.
Begin with text -> black-box LLM -> one-token text. Reveal one transformation at a time.
Open nested mechanisms only when needed; do not reveal a completed internal architecture prematurely.
Teach plain-language intuition before necessary mathematics. Use small chunks and just-in-time prerequisites.
When the learner is confused, stay with the same operation and change the explanation.
The assistant writes code directly when accessible; otherwise provide exact minimal changes for the learner to execute.
Do not make the learner write implementation code manually or repeat information available through authorized inspection.

Keep changes small, runnable, transparent, and scoped to the current operation or a directly blocking error.
Avoid large code dumps, premature abstractions, unnecessary dependencies, and production detours.
Preserve the reference baseline. Use targeted instrumentation or small investigation scripts.
Begin with a functioning library call before reproducing individual operations.
Do not rewrite a whole Transformer before explaining one operation.
Use matching inputs and parameters for comparisons; record justified tolerances or differences when checks are planned.
Fix the smallest failing boundary; do not mask failures by changing frameworks or models without reason and authority.

## Inspection and verification gate

For every opened operation, capture actual input and origin, type, shape, dtype, device, computation, purpose, dependencies, output, and destination.
Identify executing code and relevant asset or parameter locations, including disk versus RAM.
Record the command, script, observed result, and limits of hooks or fused operations.
Resolve relevant runtime discrepancies and misunderstandings before advancement.
Technical verification requires execution evidence. Advancement additionally requires explicit learner confirmation.
Keep implemented, executed, technically verified, and learner-confirmed statuses separate.
Label observed, expected-not-run, and illustrative values distinctly.

Use labeled plain text, numbered tokens, token/ID tables, relevant configuration keys, and observed asset trees with sizes and purposes.
For vectors, matrices, and tensors, show shape, dtype, device, meaningful indices, and selected bounded values.
Show before/after representations and computation rather than shapes alone.
Use a small top-k token/ID/score table for logits; distinguish probabilities when calculated.
The source suggests 5-10 scores as an example, not a fixed acceptance target.
Use token-labeled attention heatmaps or small numerical examples only when they improve the current explanation.
Keep toy values separate from actual model evidence; never print an entire vocabulary or huge tensor by default.
Keep runtime output separate from explanations. Preserve reproducible commands and filenames.
Inventory actual assets gradually; expected categories do not prove particular files exist.
Learned embeddings usually reside in parameter tensors; do not assume separate token-to-vector files.

## Access, safety, resources, and ownership

EC2 existence, reachability, model download, installed dependencies, and runtime behavior remain unverified.
No credentials or sensitive raw data belong in records, code, prompts, or transcripts.
Do not enable trust_remote_code=True without documented necessity and review.
No public web-serving ports are required. No quantization or deployment machinery is selected.
The source requires stopping the instance when unused; storage charges may continue.
It flags CPU-credit, storage, and public IPv4 costs. Those cautions are supplied context, not a current pricing assessment.
Budget, spending limits, access method, resource owner, shutdown owner, and retention choices remain unspecified.
No paid-resource creation, model execution, or remote access was authorized by preparation.
The learner owns conceptual confirmation. Named planning/execution approval owner and required reviewers remain unresolved.
Formal acceptance owner and any release or operating owner also remain unresolved; no deployment is in current scope.
No deadline, performance target, numerical comparison tolerance, or new technology has been invented.

## Applicable guides

| Guide | Status and reason |
| --- | --- |
| Core workflow | Applicable: selected lifecycle and project records |
| Architecture and quality | Applicable: proportionate design, correctness, provenance, and inspection checks |
| LLM engineering | Applicable: model/tokenizer/configuration identity, input/output interpretation, and bounded inference evidence |
| Infrastructure and operations | Applicable: selected EC2 environment, access, configuration, costs, and cleanup |
| ML/deep learning | Not applicable as a separate lifecycle: training, adaptation, datasets, and comparative model-quality evaluation are excluded |
| GPU infrastructure | Not applicable: CPU-only scope |
| Frontend/backend/database/distributed systems | Not applicable: no service, UI, persistent application database, or distributed execution |

Guide locations are linked in the [overview](../README.md). Apply relevant sections only; service evaluation suites are not automatic requirements.
Revisit applicability if the human changes scope.

## Review questions

Before planning, confirm the extracted goal and criteria, guide applicability, workflow adoption, and approval/review ownership.
Before environment work, resolve actual environment access, resource authority, spending boundaries, and shutdown responsibility.
Exact runtime versions, downloaded asset details, and comparison tolerances can wait for their authorized investigation stage.
