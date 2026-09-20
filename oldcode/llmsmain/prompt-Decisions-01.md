# Context: Learning LLM Internals on AWS EC2

## My background
- Name: Ranjit
- Location: Andhra Pradesh, India
- New to LLM internals — this is my first time peeling apart a transformer
- Speed is NOT my goal. Understanding the architecture is.

## What I'm trying to do
Learn how a transformer LLM works internally by loading a real model on an EC2 instance and inspecting it step by step. I'm following a 50-level incremental document that peels the model from an opaque box (Level 0) down to full reference architecture (Level 49), covering: tokenize, embed, transformer blocks (attention + FFN with residuals and pre-norm), final norm, LM head, softmax, argmax, and decode. The document uses Llama-3-style architecture as the reference (RMSNorm, RoPE, GQA, SwiGLU FFN).

## Decisions already made (do not re-debate these)

| # | Decision | Choice |
|---|---|---|
| 1 | Where to run | AWS EC2 (not local — laptop RAM constrained by Chrome + VS Code) |
| 2 | Instance type | t3.xlarge (16 GB RAM, 4 vCPU) |
| 3 | CPU or GPU | CPU-only (speed not the goal) |
| 4 | Region | ap-south-1 (Mumbai) |
| 5 | OS | Ubuntu 22.04 LTS, plain (not Deep Learning AMI) |
| 6 | Storage | EBS gp3, 30 GB |
| 7 | Pricing model | On-demand |
| 8 | Stop policy | Stop instance when not in use |
| 9 | Billing safety | Billing alarm at $20 |
| 10 | Access method | SSH + Jupyter port-forward (Jupyter on EC2, browser on laptop) |
| 11 | Model to load | Llama 3.2 1B Instruct (HuggingFace, requires HF login) |
| 12 | Model precision | fp16 or bf16 |
| 13 | Learning approach | Inspect real model — NOT read from-scratch reference code |
| 14 | Inspection tool | PyTorch forward hooks + print |
| 15 | Library | HuggingFace transformers |
| 16 | Skip for now | MoE, chain-of-thought, RLHF (revisit only after level 49) |

## Accounts ready
- AWS account: yes
- HuggingFace account: yes

## Hardware constraint (why local was rejected)
Laptop: i3-1215U, 16 GB RAM total but only ~6 GB free (Chrome + VS Code on company-managed laptop, no control). Intel UHD GPU — not usable for ML.

## Cost expectation
~$15–20/month if disciplined about stopping the instance. ~$130/month if left running 24/7.

## My response style preferences
- Direct, technically rigorous, no filler or praise
- Challenge my assumptions if I'm wrong
- Start with concise answer, expand with reasoning
- I'm a beginner to LLM internals — explain clearly, avoid jargon dumps
- English is my second language — use precise, simple wording
- Ask clarifying questions if the request is ambiguous

## Where I am in the plan
[UPDATE THIS LINE EACH SESSION — examples:]
- "Haven't launched EC2 yet. Need launch steps."
- "EC2 launched, need setup commands."
- "Setup done, model loaded. Ready for levels 0–10 inspection."
- "Working on levels 17–37 (attention internals)."

## What I want from you in this chat
[STATE YOUR CURRENT REQUEST HERE]