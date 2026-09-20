# LLM Infrastructure: End-to-End Flow

**Scope**: data acquisition → pretraining → post-training → inference → feedback loop, with monitoring at every stage.

**Audience**: engineers learning the system end-to-end; architects evaluating tradeoffs, failure modes, and design rationale.

**Reading guide**:
- For sequential learning, read top to bottom.
- For component lookup, jump to **Section 6** (cross-cutting reference tables and glossary).
- For monitoring/SRE concerns, every stage has a dedicated `.3 Monitoring` subsection.
- For design rationale, every stage has a dedicated `.4 Tradeoffs & Failure Modes` subsection.

**Number conventions**: every concrete number is labeled by source:
- **public** — from a vendor spec, paper, or published reference
- **derivable** — arithmetic from public numbers (math shown)
- **typical** — common industry range, not a single source
- **illustrative** — example only; real values vary by org and time

If a number lacks a basis label, it should be inferable from the surrounding context.

---

# Stage 1: Data Acquisition & Preparation

**Purpose**: turn raw internet content into a deduplicated, quality-filtered, tokenized corpus that pretraining streams from at GPU-saturating throughput.

**Output**: a sharded, tokenized dataset on a parallel filesystem, sized in the low petabytes, capable of being read at 100s of GB/s by a training cluster.

## 1.1 Flow

### Step 1 — Crawl

  - **1.1** Seed list construction
      - 1.1.1 CommonCrawl dumps ingested wholesale (CommonCrawl publishes ~3-4 billion pages/month, ~400 TB compressed WARC per monthly dump — *public, commoncrawl.org*)
      - 1.1.2 Targeted crawlers for high-value domains (arxiv, github, stackexchange, wikipedia) on independent schedules
      - 1.1.3 Licensed dataset ingestion (books, papers, code) — bypasses crawl entirely
  - **1.2** Per-domain politeness controls
      - 1.2.1 robots.txt compliance
      - 1.2.2 QPS cap per domain (typical: 1–10 req/sec/domain)
      - 1.2.3 User-agent identification
  - **1.3** Fetch and store
      - 1.3.1 HTTP fetch with retry/backoff
      - 1.3.2 Write raw bytes to **RawLake** (original encoding preserved, no transformation)
      - 1.3.3 Metadata sidecar (URL, fetch timestamp, HTTP headers, content-type)

  **Output of Step 1**: raw bytes in RawLake. Volume scales linearly with crawler fleet size; CommonCrawl-class operation = 100s of TB/month.

### Step 2 — Extract

  - **2.1** Format detection (MIME sniffing, magic bytes)
  - **2.2** Per-format extraction
      - 2.2.1 HTML → text via boilerplate removal (Trafilatura, Resiliparse — newspaper3k deprecated)
      - 2.2.2 PDF → text
          - 2.2.2.1 Born-digital PDFs: PyMuPDF or pdfplumber (fast, ~10–100 ms/page typical)
          - 2.2.2.2 Scanned PDFs: OCR via Tesseract or commercial OCR (slow, ~1–5 sec/page typical)
      - 2.2.3 Code repos → file-level extraction with language detection (linguist, enry)
      - 2.2.4 EPUB/MOBI → text via ebooklib or pandoc
  - **2.3** Language identification (fastText `lid.176` model, ~10K docs/sec/CPU-core typical)
  - **2.4** UTF-8 normalization, NFC unicode form
  - **2.5** Write to **TextLake** as compressed plain text (zstd level 3–9 typical; 3–5× compression on natural text)

  **Output of Step 2**: cleaned UTF-8 text in TextLake. Typical compression ratio from RawLake → TextLake is 5–20× depending on source mix (HTML has high boilerplate-to-content ratio).

### Step 3 — Deduplicate

  - **3.1** Exact dedup
      - 3.1.1 SHA-256 hash per document
      - 3.1.2 Drop duplicates, keep first occurrence by fetch timestamp
      - 3.1.3 Typical removal: 10–30% of documents (mirrors, reposts)
  - **3.2** Near-dedup via MinHash + LSH
      - 3.2.1 Compute MinHash signature per document (typical: 128–256 hash functions)
      - 3.2.2 LSH bucketing (typical: 9 bands × 13 rows, or similar, tuned for ~0.8 Jaccard threshold)
      - 3.2.3 Within-bucket pairwise Jaccard check
      - 3.2.4 Drop matches, keep canonical (longest, or earliest)
      - 3.2.5 Typical removal: additional 30–60% of documents
  - **3.3** Cross-source dedup (web vs. books vs. papers — same content often appears in multiple)

  **Compute profile of Step 3**: CPU-bound and IO-bound. Spark or Ray on commodity CPU nodes. **No GPUs.** (The original document put this on A100s; that's wrong.)

  **Output of Step 3**: deduplicated TextLake. After exact + near-dedup, total surviving documents typically 20–40% of input.

### Step 4 — Quality filter

  Sequential filters; documents must pass all.

  - **4.1** Heuristic filters (cheap, reject 30–50% of post-dedup)
      - 4.1.1 Length bounds (typical: 50 ≤ word count ≤ 100,000)
      - 4.1.2 Mean word length (typical: 3 ≤ μ ≤ 10 chars)
      - 4.1.3 Repetition ratio (fraction of duplicated lines; reject > 0.3 typical)
      - 4.1.4 Symbol-to-word ratio (reject > 0.1 typical — catches code-in-prose artifacts)
      - 4.1.5 Stopword presence (reject if too few stopwords — catches keyword spam)
      - 4.1.6 (See Gopher / RefinedWeb / FineWeb papers for exact thresholds — these are public)
  - **4.2** Model-based quality classifier
      - 4.2.1 Small encoder (DeBERTa-v3-base ~140M params, or smaller fastText)
      - 4.2.2 Trained on (high-quality reference, low-quality random web) pairs
      - 4.2.3 Inference on L4/A10/T4 GPUs or CPU with batching — **not A100s**
      - 4.2.4 Threshold tuned to retain top 10–30% by score
  - **4.3** PII filter
      - 4.3.1 Regex pass (emails, phones, credit cards, SSNs, API keys)
      - 4.3.2 NER pass (spaCy or fine-tuned BERT for names/addresses)
      - 4.3.3 Action: redact (replace with token) OR drop document (configurable per filter)
  - **4.4** Benchmark decontamination
      - 4.4.1 N-gram hash table of known eval set examples (MMLU, HumanEval, GSM8K, MATH, etc.)
      - 4.4.2 Reject documents containing 13-gram (typical) overlap with eval data
      - 4.4.3 **Critical**: without this, benchmarks become invalid

  **Output of Step 4**: high-quality, decontaminated, PII-cleaned text. Typical retention from post-dedup: 10–30%.

### Step 5 — Mix and shard

  - **5.1** Per-source bucketing (web, code, books, papers, math, multilingual, etc.)
  - **5.2** Mix ratio sampling
      - 5.2.1 Ratios are a tuned hyperparameter (see published mixes: GPT-3, Llama, RefinedWeb)
      - 5.2.2 Typical: web 40–60%, code 10–20%, books 5–15%, papers 5–10%, multilingual 5–20%
      - 5.2.3 Upsampling: small high-quality sources (math, papers) often sampled with replacement
  - **5.3** Shard writing
      - 5.3.1 Format: MosaicML StreamingDataset, WebDataset, or Megatron `.bin`+`.idx`
      - 5.3.2 Shard size: typically 100 MB – 2 GB per shard (tuned for parallel FS read patterns)
      - 5.3.3 Random shuffle within shard; shards consumed in random order at training time
  - **5.4** Write to **GoldDataset** (NVMe parallel FS)

### Step 6 — Tokenize

  - **6.1** Tokenizer training (one-time per model family)
      - 6.1.1 Algorithm: byte-level BPE (GPT-2/3 style) or SentencePiece (LLaMA style)
      - 6.1.2 Vocab size: 32K (LLaMA), 50K (GPT-2), 100K (GPT-4), up to 256K (Gemini, multilingual)
      - 6.1.3 Training data: sample of GoldDataset, typically 10s of GB
      - 6.1.4 Hardware: single high-memory CPU box, hours of runtime
      - 6.1.5 Output: tokenizer model file (~MB scale), stored in **ModelArtifacts**
  - **6.2** Corpus tokenization
      - 6.2.1 Apply tokenizer across full GoldDataset
      - 6.2.2 Embarrassingly parallel; CPU fleet, hundreds to thousands of cores
      - 6.2.3 Pack into fixed-length sequences (typical: 2048, 4096, or 8192 tokens per sequence)
      - 6.2.4 Output format: int32 or uint32 token arrays, optionally int16 if vocab ≤ 65535
      - 6.2.5 Write to GoldDataset (alongside text, **not replacing it** — text remains source of truth)

  **Storage math for tokenized output** (derivable):
  - 1 trillion tokens × 2 bytes/token (uint16 packed) = 2 TB
  - Modern frontier datasets: 10–15 trillion tokens → 20–30 TB packed

## 1.2 Components

### Storage spec sheet

| Component | Storage type | Media | Capacity (typical) | Read throughput | Cost per GB-month (illustrative, AWS list)¹ | Why this tier |
|---|---|---|---|---|---|---|
| **RawLake** | Object (S3-class) | HDD / archive | 10s of PB | Low (cold) | $0.001–$0.004 (Glacier–Standard-IA) | Write-once, read-rare. Archive economics. |
| **TextLake** | Object (S3-class) | HDD / standard | Single-digit PB | Medium | $0.021–$0.023 (Standard) | Re-read on filter/tokenizer changes. |
| **GoldDataset** | Parallel FS (Lustre/Weka/VAST/DAOS) | NVMe SSD | Low PB | 100s of GB/s aggregate | $0.10–$0.30 (self-hosted, amortized; cloud parallel FS higher) | Pretraining streams 24/7; GPU idle = catastrophic cost. |
| **ModelArtifacts** | Object, versioned | Standard SSD | GBs | Low | $0.023 (S3 Standard) | Tokenizer files, configs. Small, versioned. |

¹ AWS list prices Nov 2025, illustrative only. Real costs depend on contracts, region, volume.

### Compute spec sheet

| Cluster | Hardware | Scale (typical) | Workload profile | Why this hardware |
|---|---|---|---|---|
| **Crawler fleet** | CPU servers (low-spec, lots of NICs) | 100s–1000s of nodes | IO-bound (network) | No CPU/GPU needed; bandwidth + storage IO dominate |
| **Extract fleet** | CPU servers, high RAM | 100s–1000s of nodes | CPU-bound (parsing) | PDFs and HTML parsing are CPU work |
| **OCR fleet** (subset of extract) | CPU or small GPU (T4/L4) | 10s–100s of nodes | Mixed | OCR has GPU-accelerated paths (PaddleOCR, TrOCR) for accuracy-critical sources |
| **Dedup cluster** | CPU servers, high RAM, fast local SSD | 100s of nodes (Spark/Ray) | Shuffle-heavy | MinHash/LSH is hash-and-shuffle; no GPU benefit |
| **Quality classifier fleet** | Small GPUs (L4 24GB, A10 24GB, T4 16GB) | 10s–100s of nodes | Inference of small models (sub-500M params) | Big GPUs are overkill; cost-per-classification matters |
| **Tokenizer training** | Single high-memory CPU box | 1 node, 512GB–1TB RAM | Single-threaded-ish | Training corpus must fit in memory for fast iteration |
| **Tokenization fleet** | CPU servers | 100s of nodes | Embarrassingly parallel | Pure CPU work; trivially scalable |

### Throughput math (derivable)

For a 10T-token training run, GoldDataset must sustain enough read throughput to keep a training cluster fed.

- A 10K-H100 cluster processes roughly 5–15M tokens/sec (varies with model size, sequence length, parallelism — see Megatron-LM paper for derivations)
- At 2 bytes/token packed, that's 10–30 MB/sec of *unique* data flow per training step
- But: each token in the dataset is read multiple times across epochs (typical: 1–4 passes); training also re-reads via shuffle buffers
- Practical requirement: **100s of GB/s aggregate read throughput** from GoldDataset, sustained for months
- This is why parallel FS (Lustre/Weka/VAST) — single-node NFS cannot sustain this

## 1.3 Monitoring

### Throughput metrics (per stage)

| Metric | What it measures | Why it matters | Typical alert threshold |
|---|---|---|---|
| Documents/sec ingested | Crawl rate | Detects crawler stalls | < 50% of 7-day baseline → page |
| Bytes/sec written to RawLake | Crawl bandwidth | Storage IO health | < 50% of baseline → page |
| Documents/sec extracted | Extract throughput | Detects extractor crashes | < 50% of baseline → page |
| Queue depth between stages | Backlog | Detects bottlenecks | > 7 days of work backed up → page |

### Yield metrics (the critical ones)

| Metric | What it measures | Why it matters | Typical alert threshold |
|---|---|---|---|
| Retention rate per filter | Fraction passing each filter | Detects filter bugs and input distribution shifts | Δ > 10% from 30-day baseline → page |
| Per-source retention rate | Same, broken down by source | Catches source-specific issues (e.g. arxiv extraction broken) | Δ > 25% per source → page |
| Exact dedup rate | % dropped by SHA dedup | Spikes mean crawler is re-fetching | > 50% (typical baseline 10–30%) → investigate |
| Near-dedup rate | % dropped by MinHash | Same | > 80% (typical baseline 30–60%) → investigate |
| Output token count | Final tokens in GoldDataset | Sanity check on full pipeline | Δ > 5% from expected → page |

### Quality metrics

| Metric | What it measures | Method | Cadence |
|---|---|---|---|
| Quality score distribution per source | Classifier output histogram | Aggregate over all docs/source/day | Daily review |
| Manual sample review | Human eyeballs on random shards | Pull N=100 docs/shard, reviewer rates | Weekly |
| Eval contamination check | Held-out eval n-grams found in GoldDataset | Hash table scan | Per release |
| PII leakage audit | Found PII in nominally-clean output | Sample + regex + manual | Monthly |

### Cost metrics

| Metric | Why |
|---|---|
| $/TB processed per stage | Spot regressions (a Python rewrite of a Rust filter, etc.) |
| Cluster utilization % | Idle clusters = wasted money |
| Storage tier hit rate (Glacier retrieval rate) | Catches accidentally-hot cold storage |

### What is NOT monitored in real-time

- **PII leakage**: by definition, you don't know what you missed. Caught via periodic audits and red-team holdouts, not metrics.
- **Subtle quality regressions**: surface only as downstream pretraining loss curves, weeks later.
- **Data poisoning**: sophisticated adversarial content slips through. No metric catches it; only post-hoc investigation does.

## 1.4 Tradeoffs & Failure Modes

### Decision: batch pipeline, not streaming

| | Batch (chosen) | Streaming (rejected) |
|---|---|---|
| Latency from crawl to trainable | Days–weeks | Hours |
| Cost | Low (commodity CPU, HDD) | High (always-on infra) |
| Reprocessability | Trivial — rerun the batch | Hard — replay log + side effects |
| Debuggability | Idempotent, deterministic | Stateful, harder |
| When it breaks | If model needs fresh data (last week's news, library releases) | Operational complexity at scale |

**Mitigation for batch's freshness problem**: a parallel "fresh" pipeline at daily cadence, blended at low weight into the training mix.

### Decision: dedup before quality, not after

- **Pro**: quality filter runs on 30–50% less data → significant compute savings (quality filter is the most expensive stage after extraction)
- **Con**: dedup decisions are effectively permanent. The *choice* of which copy to keep matters and can't be revisited without re-running dedup.
- **Failure mode**: if dedup keeps the SEO-spam copy over the original article (because the spam copy was fetched first), quality filter can't recover the original.

### Decision: per-source mix ratios as tuned hyperparameter

- **Pro**: upweight high-value sources (code +10%, papers +5%) to materially boost capability per training token
- **Con**: each mix-ratio experiment requires a full (or scaled) training run to evaluate. Weeks per data point.
- **Mitigation**: scaling laws on small models for relative ordering; the caveat is that mix effects don't always transfer to large scale.
- **Real-world failure**: a mix ratio that looks good at 1B-param scale can produce a worse 100B-param model. This has happened publicly (see various model post-mortems).

### Decision: tokenize once and pack, store binary

- **Pro**: pretraining reads sequential bytes at max throughput; zero per-step tokenization overhead
- **Con**: a tokenizer change requires re-tokenizing the full corpus (10s of TB of CPU work)
- **Failure mode**: tokenizer bugs discovered mid-training. Choices: live with it, or restart. Real example: GPT-2 tokenizer's quirks with whitespace and code persisted for years.

### Failure mode: benchmark contamination

- **Cause**: eval set examples (MMLU, HumanEval, etc.) are on the open web; crawler picks them up
- **Consequence**: model trains on test set → benchmark scores invalid → silent overclaiming
- **Mitigation**: explicit decontamination filter with hash table of known eval n-grams (13-gram typical)
- **Residual risk**: paraphrased eval examples slip through n-gram filter. No clean solution.

### Failure mode: PII leakage

- **Cause**: regex misses obfuscated PII; NER misses non-Western names; new PII formats appear (e.g. new ID schemes)
- **Mitigation**: layered defense — regex + NER + output-side filtering at inference time
- **Residual risk**: treat all PII filtering as best-effort, not guaranteed. Legal exposure is real.

### Failure mode: legal/licensing exposure

- **Cause**: crawled content is copyrighted, license-restricted, or jurisdiction-sensitive (GDPR right-to-erasure)
- **Mitigation**: source allowlists/denylists, per-source license tracking, takedown workflow
- **Operational reality**: this is now a dedicated team's job at major labs, not an afterthought.

### Failure mode: data poisoning

- **Cause**: adversary publishes targeted content (e.g. "if user says X, respond Y") at scale across many domains
- **Mitigation**: dedup + quality filter catches crude versions
- **Residual risk**: sophisticated attacks are largely unsolved. Active research.

### Failure mode: scaling cliff in dedup

- **Cause**: pairwise Jaccard checks in LSH grow with bucket density; one hot bucket can dominate runtime
- **Symptom**: dedup job runs 10× longer than expected; one Spark stage stuck on a few tasks
- **Mitigation**: salt buckets, cap per-bucket pairwise comparisons, sample within hot buckets

---

# Stage 2: Pretraining

**Purpose**: turn the tokenized GoldDataset into a base model — a neural network with learned weights that can predict the next token given a context, with no task-specific behavior yet.

**Output**: a base model checkpoint (weights + optimizer state) stored in **ModelVault**, ready to be fed into Stage 3 (post-training).

**Duration**: weeks to months of continuous training on a dedicated supercomputer.

## 2.1 Flow

### Step 1 — Cluster preparation

  - **1.1** Reserve training cluster
      - 1.1.1 Typical frontier-class cluster: 10K–100K H100 or H200 GPUs (*typical*; xAI Colossus: ~100K H100s, public; Meta's Llama-3 training clusters: 24K H100s, public)
      - 1.1.2 GPUs organized in **nodes** (8 GPUs/node, NVLink-connected within node — *public, NVIDIA HGX H100 spec*)
      - 1.1.3 Nodes organized into **rails** or **pods** (typical: 256–1024 GPUs per pod, connected via InfiniBand non-blocking fabric)
      - 1.1.4 Pods connected via fat-tree InfiniBand topology (typical: 400 Gb/s per port, NDR generation — *public, NVIDIA Quantum-2*)
  - **1.2** Verify cluster health before run start
      - 1.2.1 Per-GPU diagnostics (DCGM, nvidia-smi): memory ECC errors, throttling, link state
      - 1.2.2 InfiniBand fabric health: link error counters, congestion, route validation
      - 1.2.3 Burn-in tests: synthetic NCCL all-reduce benchmark, expected ≥ 90% of theoretical bandwidth
      - 1.2.4 At cluster scale, ~1–5% of GPUs are typically dead-on-arrival or marginal — must be identified and excluded
  - **1.3** Load model code, configs, dataset path
      - 1.3.1 Framework: Megatron-LM, NVIDIA NeMo, MosaicML Composer, or PyTorch FSDP / DeepSpeed
      - 1.3.2 Config includes: model architecture, parallelism strategy, optimizer hyperparameters, learning rate schedule

### Step 2 — Choose parallelism strategy

  This is the most important design decision in pretraining. A frontier model does not fit in one GPU's HBM; it must be sharded across GPUs along multiple axes simultaneously.

  - **2.1** Tensor Parallelism (TP)
      - 2.1.1 Splits individual layers across GPUs (each GPU holds a slice of weight matrices)
      - 2.1.2 Requires high-bandwidth all-reduce on every layer's forward and backward pass
      - 2.1.3 **Bound to within a single node** (typically TP=8) because only NVLink (900 GB/s per GPU, *public, NVIDIA*) can sustain the bandwidth; InfiniBand cannot
  - **2.2** Pipeline Parallelism (PP)
      - 2.2.1 Splits the model by layer groups across nodes (node A holds layers 1–10, node B holds 11–20, etc.)
      - 2.2.2 Communication is point-to-point sends between adjacent stages, much lower bandwidth than TP
      - 2.2.3 Introduces "pipeline bubbles" — idle time at start/end of each batch; mitigated by microbatching (1F1B schedule, interleaved schedules)
      - 2.2.4 Typical: PP = 8–32 across nodes
  - **2.3** Data Parallelism (DP)
      - 2.3.1 Each replica processes a different microbatch in parallel
      - 2.3.2 Gradients all-reduced across DP replicas at end of each step
      - 2.3.3 This is the **outermost** parallelism axis — it's how you scale to more GPUs once TP and PP are saturated
  - **2.4** Sequence / Context Parallelism (SP / CP)
      - 2.4.1 Splits the sequence dimension across GPUs (needed for long-context training)
      - 2.4.2 Critical at 100K+ token contexts, where activation memory dominates
  - **2.5** Expert Parallelism (EP) — Mixture of Experts (MoE) only
      - 2.5.1 Splits experts across GPUs; tokens route to a subset of experts via a gating network
      - 2.5.2 Adds all-to-all communication (each token goes to its assigned expert, then results return)
      - 2.5.3 Frontier MoE models: Mixtral-8x7B (public), DeepSeek-V3 (public, 671B total / 37B active), GPT-4 (rumored MoE)

  **Worked example: 1024-GPU job for a dense 70B-param model** (*illustrative*)
  - TP = 8 (within node)
  - PP = 8 (across 8 nodes per pipeline replica)
  - DP = 16 (16 pipeline replicas in parallel)
  - Total: 8 × 8 × 16 = 1024 GPUs

  **Correction to the original document**: the original said "each forward pass uses all GPUs simultaneously via tensor parallelism." That's wrong. TP is bounded to one node. A forward pass on one microbatch traverses one pipeline stage at a time. Different DP replicas process different microbatches in parallel; that's not the same as "all GPUs working on one forward pass."

### Step 3 — Initialize and load weights

  - **3.1** Random initialization
      - 3.1.1 Standard schemes: scaled normal (μ-Parametrization / muP increasingly common), Xavier/Glorot, He initialization
      - 3.1.2 Critical: initialization scale interacts with learning rate; getting this wrong = NaN losses or slow convergence
  - **3.2** Shard weights across parallelism dimensions per the chosen strategy
  - **3.3** Materialize on GPU HBM (each GPU now holds its slice of weights, optimizer state, gradients)

  **HBM math** (derivable, dense model)
  - Params: N (e.g. 70B)
  - Bytes per param: weights (2 bytes bf16) + gradients (2 bytes bf16) + optimizer state (8 bytes fp32 Adam moments) + master weights (4 bytes fp32) = **16 bytes/param**
  - For 70B: 70 × 16 = **1120 GB** total state across the cluster
  - Per GPU on a 1024-GPU job: 1120 / 1024 ≈ **1.1 GB/GPU** (theoretical minimum)
  - Plus activation memory, communication buffers, fragmentation: typically 30–60 GB/GPU in practice
  - This is why H100 80GB HBM (*public*) matters — anything smaller forces aggressive recomputation

### Step 4 — Training loop

  Repeat for trillions of tokens.

  - **4.1** Data load
      - 4.1.1 Streaming dataset reader pulls next batch from GoldDataset
      - 4.1.2 Prefetch + double-buffer to overlap with compute
      - 4.1.3 Sustained read: 100s of GB/s aggregate (see 1.2 throughput math)
  - **4.2** Forward pass
      - 4.2.1 Input tokens flow through TP-sharded layers within a node
      - 4.2.2 Activations passed point-to-point to next pipeline stage
      - 4.2.3 Output: logits over vocabulary for each position
      - 4.2.4 Loss computed: cross-entropy between predicted distribution and actual next token
  - **4.3** Backward pass
      - 4.3.1 Gradients computed in reverse layer order
      - 4.3.2 Activation memory reclaimed (or recomputed if checkpointing enabled)
      - 4.3.3 Gradients passed back through pipeline stages
  - **4.4** Gradient communication
      - 4.4.1 **TP dimension**: all-gather + reduce-scatter within node (NVLink, 900 GB/s per GPU)
      - 4.4.2 **PP dimension**: no all-reduce needed; gradients are local per stage
      - 4.4.3 **DP dimension**: all-reduce across DP replicas (InfiniBand, 400 Gb/s per link)
      - 4.4.4 **FSDP variant**: gradients reduce-scattered, then params all-gathered for next step
  - **4.5** Optimizer step
      - 4.5.1 Adam/AdamW updates moments, applies learning rate
      - 4.5.2 Modern alternatives gaining ground: Lion (Google, 2023), Distributed Shampoo (used at scale by various labs), Sophia
      - 4.5.3 Learning rate schedule: typically linear warmup (~1% of steps) then cosine decay or WSD (warmup-stable-decay)
  - **4.6** Logging + telemetry
      - 4.6.1 Loss, gradient norm, learning rate, throughput written to metrics store
      - 4.6.2 Per-rank metrics aggregated for SRE visibility

  **Token throughput math** (derivable, *illustrative*)
  - H100 dense throughput at bf16: ~700–1000 TFLOPS practical (~50–70% of theoretical peak — *typical*)
  - Per-token compute for forward+backward, dense model: ≈ 6 × N FLOPs (Kaplan scaling law approximation, *public*)
  - 70B model, 10K H100s at 800 TFLOPS practical: (10000 × 800e12) / (6 × 70e9) ≈ **1.9M tokens/sec**
  - For 10T-token training run: 10e12 / 1.9e6 ≈ **5.3M seconds = 61 days** of pure compute

### Step 5 — Checkpointing

  - **5.1** Cadence: every N steps (typical: every 500–2000 steps, or every few hours of wall clock)
  - **5.2** Contents
      - 5.2.1 Model weights (sharded per parallelism strategy)
      - 5.2.2 Optimizer state (Adam moments, master fp32 weights)
      - 5.2.3 RNG state (for deterministic resume)
      - 5.2.4 Dataloader state (position in dataset shards)
      - 5.2.5 Step count, learning rate scheduler state
  - **5.3** Write
      - 5.3.1 Each rank writes its shard in parallel
      - 5.3.2 Target: **CheckpointStore** (NVMe parallel FS, separate from GoldDataset to avoid IO contention)
      - 5.3.3 Async checkpointing: GPU computes while checkpoint streams to disk (overlaps the IO cost)
  - **5.4** Retention policy
      - 5.4.1 Keep last K checkpoints on hot storage (typical K = 3–10)
      - 5.4.2 Tier older checkpoints to object storage
      - 5.4.3 Mark milestone checkpoints (e.g. every 100B tokens) as permanent

  **Checkpoint size math** (derivable)
  - 70B model, full optimizer state: 70B × 16 bytes/param = **1.12 TB per checkpoint**
  - A 1T-param MoE checkpoint: ~16 TB
  - The original document's "5 TB for frontier model" is for *weights only* — checkpoint with optimizer state is 4× larger

### Step 6 — Continuous evaluation

  - **6.1** Reserved eval pool (separate from training GPUs)
      - 6.1.1 Typical: ~1–5% of cluster capacity, e.g. 100–500 H100s reserved
      - 6.1.2 Runs the most recent checkpoint against a battery of benchmarks
  - **6.2** Eval suite
      - 6.2.1 Perplexity on held-out data (cheapest, runs every N steps)
      - 6.2.2 Few-shot benchmarks (MMLU, HellaSwag, ARC, GSM8K, HumanEval — *public*)
      - 6.2.3 Internal eval sets (more frequent, faster, less leak-prone)
      - 6.2.4 Loss-on-held-out-domains (code, math, multilingual) — catches regressions hidden in aggregate loss
  - **6.3** Anomaly detection
      - 6.3.1 Loss spikes (>3σ above rolling baseline) trigger inspection
      - 6.3.2 Gradient norm spikes precede loss spikes by 10–100 steps typically
      - 6.3.3 Slow eval regressions (e.g. code benchmark drifting down over 1000 steps) flagged for review

### Step 7 — Recovery from failure

  Hardware failure is **inevitable** at scale. A 10K-GPU cluster has tens of GPUs failing per week (*typical*).

  - **7.1** Per-step failure detection
      - 7.1.1 NCCL timeout (typical 10–30 min) indicates a hung rank
      - 7.1.2 GPU fault counters (XID errors, ECC) flag dying hardware
      - 7.1.3 Loss = NaN/Inf: numerical instability
  - **7.2** Recovery procedure
      - 7.2.1 Stop the world (all ranks)
      - 7.2.2 Diagnose failed node(s) — usually a GPU or InfiniBand link
      - 7.2.3 Drain the dead node, hot-spare a replacement
      - 7.2.4 Reload last good checkpoint across the cluster
      - 7.2.5 Resume training from saved step count
  - **7.3** Restart wall-clock cost
      - 7.3.1 Checkpoint load: 5–30 minutes for frontier-class model (deserialize, re-shard, place on correct ranks — not just raw read throughput)
      - 7.3.2 Cluster re-initialization: 5–15 minutes (NCCL handshake, dataloader resume)
      - 7.3.3 Total recovery: typically **15–60 minutes**, not 30 seconds as the original document claimed

### Step 8 — End of run

  - **8.1** Final checkpoint written to ModelVault (replicated + air-gapped backup)
  - **8.2** Model is now the **base model** — knows how to predict next tokens, but has no instruction-following, no safety training, no chat formatting
  - **8.3** Hand off to Stage 3 (post-training)

## 2.2 Components

### Storage spec sheet

| Component | Storage type | Media | Capacity (typical) | Throughput | Why this tier |
|---|---|---|---|---|---|
| **GoldDataset** | Parallel FS (Lustre/Weka/VAST/DAOS) | NVMe SSD | Low PB | 100s GB/s read | Streamed continuously to GPU cluster |
| **CheckpointStore** | Parallel FS (separate cluster from GoldDataset) | NVMe SSD | 100s of TB | 100s GB/s write burst, low average | Checkpoint writes are bursty; separate cluster avoids IO contention with training data reads |
| **ModelVault** | Replicated object storage + air-gapped backup | Standard SSD + tape/cold | 10s–100s of TB | Low (cold archive) | Final weights are the most valuable IP. Production replicas exist on inference clusters. |
| **MetricsStore** | Time-series DB (Prometheus, VictoriaMetrics, InfluxDB) | SSD | GBs–TBs | High write rate, moderate read | Per-rank per-step metrics; queried by dashboards |

### Compute spec sheet

| Cluster | Hardware | Scale (typical) | Workload profile | Notes |
|---|---|---|---|---|
| **Training cluster** | H100 80GB or H200 141GB (or B200 192GB in 2025+) | 10K–100K GPUs | bf16/fp8 compute, NVLink+IB heavy | Liquid-cooled at H200/B200 generation; air-cooled with rear-door HX at H100 |
| **Eval cluster** | H100 (smaller pool) | 100–1000 GPUs | Inference of mid-training checkpoint | Separate to avoid stealing capacity from training |
| **Coordination plane** | CPU servers | 10s of nodes | Scheduler, monitoring, log aggregation | Kubernetes or Slurm typical |

### Network spec sheet

| Layer | Hardware | Bandwidth | Role |
|---|---|---|---|
| Intra-node (GPU↔GPU) | NVLink 4 / NVSwitch | 900 GB/s per GPU (*public, NVIDIA H100 spec*) | TP traffic |
| Inter-node (within rail/pod) | InfiniBand NDR (Quantum-2) | 400 Gb/s per port, non-blocking fat-tree | PP + DP all-reduce |
| Inter-pod | InfiniBand or Ethernet RoCE | Oversubscribed (typical 2:1 or 4:1) | DP all-reduce across pods |
| Storage network | InfiniBand or dedicated Ethernet | Sized for 100s GB/s aggregate to parallel FS | GoldDataset + CheckpointStore IO |

### Power and cooling

| Metric | Typical scale |
|---|---|
| Per-GPU power draw (H100 SXM) | 700 W (*public, NVIDIA*) |
| Per-node power (8 GPUs + CPU + NIC) | 8–10 kW |
| Per-rack power (4–8 nodes) | 30–80 kW (liquid-cooled racks) |
| 100K H100 cluster total IT power | ~70–90 MW (*derivable*: 100K × 700W ≈ 70 MW, plus 20–30% overhead for CPU/storage/network) |
| Datacenter total with PUE 1.2–1.3 | ~85–120 MW |
| Cooling | Air with rear-door HX at H100 generation; liquid required at H200/B200 |

## 2.3 Monitoring

### Compute health (per GPU, per step)

| Metric | What it measures | Why it matters | Alert threshold |
|---|---|---|---|
| GPU utilization % | SM occupancy | Detects idle GPUs (data stall, comm bottleneck) | < 60% sustained → investigate |
| HBM utilization % | Memory used / capacity | OOM risk, recomputation pressure | > 90% → reduce batch or increase recomp |
| HBM bandwidth GB/s | Memory throughput | Detects compute-bound vs memory-bound phases | Below kernel-specific expected → kernel bug |
| GPU temperature °C | Thermal | Throttling risk | > 85°C → throttle or cool harder |
| ECC error count | Memory errors | Hardware decay | Any uncorrectable → drain GPU |
| XID errors | NVIDIA driver fault codes | Hardware failures | Any fatal XID → drain node |
| Power draw W | Per-GPU power | Misconfiguration or hardware issue | < 70% expected → underutilized |

### Network health

| Metric | What it measures | Why it matters | Alert threshold |
|---|---|---|---|
| NCCL all-reduce time | Comm latency | Network or stragglers | > 2× rolling baseline → page |
| IB link error counters | Bit errors | Cable / switch faults | Any growing counter → replace |
| Congestion notifications | ECN marks | Topology bottlenecks | Sustained → re-route or scale fabric |
| Per-rank straggler ratio | Slowest / fastest rank time | One slow GPU stalls the whole step | > 1.2 → investigate slow rank |

### Training health (per step / per N steps)

| Metric | What it measures | Why it matters | Alert threshold |
|---|---|---|---|
| Loss | Cross-entropy on training batches | The thing being optimized | Spike > 3σ above rolling → page; sustained increase → rollback |
| Gradient norm | L2 norm of gradients | Stability indicator | Spike often precedes loss spike by 10–100 steps |
| Gradient clipping rate | Fraction of steps where grad was clipped | Optimizer health | > 5% → reduce LR or investigate |
| Weight norm | L2 norm of weights | Drift / instability | Diverging → instability |
| Activation stats | Mean/var per layer | Layer health | Layers drifting to zero or exploding → architectural bug |
| Tokens/sec throughput | Cluster throughput | Cost efficiency | < 80% of expected MFU → investigate |
| MFU (Model FLOPs Utilization) | Practical FLOPs / theoretical peak | Efficiency benchmark | 40–55% is typical at frontier scale (*public, various papers*); below = inefficient |

### Eval health (per checkpoint)

| Metric | What it measures | Cadence |
|---|---|---|
| Held-out perplexity | Generalization | Every 500–2000 steps |
| Per-domain held-out loss | Code, math, multilingual loss separately | Every 500–2000 steps |
| Few-shot benchmarks | MMLU, HumanEval, etc. | Every 5K–20K steps (expensive) |
| Internal evals | Custom benchmarks not on the open web | Every checkpoint |

### Operational metrics

| Metric | What it measures | Why |
|---|---|---|
| Mean time between failures (MTBF) | Hours between cluster-stopping faults | Capacity planning, reliability |
| Mean time to recovery (MTTR) | Wall clock from failure to resume | Throughput tax |
| Effective training time % | Time spent on forward/backward / total wall clock | Real cluster efficiency |
| Checkpoint write time | Wall clock to persist checkpoint | If > async budget, checkpoint stalls training |

### What is NOT monitored in real-time

- **Final model capability**: only observable after training completes and full eval suite runs.
- **Subtle distribution shifts in data**: caught via downstream eval drift, not training-time metrics.
- **Reward hacking or memorization**: surfaces only in post-training or deployment.

## 2.4 Tradeoffs & Failure Modes

### Decision: parallelism strategy (TP × PP × DP)

| Axis | When to increase | When to decrease |
|---|---|---|
| TP | Activation memory tight, want lower per-step latency | Beyond 8 — TP requires NVLink; cross-node TP is impractical |
| PP | Model too large to fit with just TP+DP | Many stages → big pipeline bubbles → wasted compute |
| DP | Want to scale to more GPUs once TP and PP are set | Beyond ~1000 DP replicas, all-reduce becomes the bottleneck |

**Failure mode**: choosing TP=16 on a node with NVLink only on TP=8 boundaries. Cross-NVLink-domain TP traffic falls back to InfiniBand → 10–50× slowdown on every layer.

### Decision: FSDP vs Megatron-style 3D parallelism

| | FSDP (PyTorch native) | Megatron-LM 3D |
|---|---|---|
| Ease of use | Higher (built into PyTorch) | Lower (custom config, more code) |
| Memory efficiency | Excellent (shards everything, all-gathers on demand) | Excellent if 3D config is correct |
| Communication cost | All-gather + reduce-scatter on every step | Configurable; can amortize better at high TP |
| Best for | Mid-size training, research, dynamic shapes | Frontier-scale fixed configurations |
| Failure mode | Communication-bound at very large DP if not tuned | Brittle config, expert-only |

### Decision: bf16 vs fp8 training

| | bf16 (mature) | fp8 (newer, H100+) |
|---|---|---|
| Hardware support | All modern GPUs | H100/H200/B200 only (*public*) |
| Throughput | 1× baseline | 1.5–2× baseline (*typical, varies by op*) |
| Stability | Well-understood | Requires careful scaling, hybrid precision still needed |
| Failure mode | None major | Loss spikes from fp8 over/underflow; some layers must stay bf16 |

### Decision: activation checkpointing (recomputation)

- **Pro**: cuts activation memory by ~10× → enables larger models on fixed HBM
- **Con**: ~30% extra compute (recompute activations during backward)
- **Failure mode**: forgetting to checkpoint a specific layer → OOM at random step. Selective checkpointing tools help.

### Decision: dense vs MoE

| | Dense | MoE |
|---|---|---|
| Training cost | 6N FLOPs/token (Kaplan approx) | ~6N_active FLOPs/token (much less for same total params) |
| Inference cost | Same as training cost per param | Cheaper per token (only active experts compute) |
| Communication | Standard TP/PP/DP | Adds all-to-all for expert routing |
| Memory | Whole model active | Total params much larger, but per-step compute cheaper |
| Failure mode | None inherent | Load imbalance across experts (some experts overused, others starve) — solved with auxiliary load-balancing loss |

### Decision: continuous eval during training

- **Pro**: catch regressions early, decide whether to rollback before wasting weeks of compute
- **Con**: eval cluster is dedicated capacity (1–5% of fleet) that could be training
- **Failure mode**: eval frequency too low → discover problem too late; eval frequency too high → eval becomes a bottleneck

### Failure mode: loss spike + bad rollback

- **Cause**: numerical instability (fp8 underflow, learning rate too high, bad batch)
- **Symptom**: loss jumps from ~2.5 to 10+ in one step; gradient norm spikes
- **Recovery**: rollback to last good checkpoint, possibly with reduced LR or skipped batch
- **Time cost**: 15–60 minutes wall clock per rollback (see Step 7.3)
- **Worst case**: spike is reproducible from a corrupted dataset shard. Without diagnosing root cause, you rollback, rollback, rollback.

### Failure mode: silent stragglers

- **Cause**: one GPU running 5–10% slower (thermal throttle, marginal hardware, kernel scheduling)
- **Symptom**: NCCL all-reduce time creeping up; tokens/sec dropping; loss curve healthy but training slow
- **Detection**: per-rank step time, slowest/fastest ratio
- **Mitigation**: drain the slow rank, hot-spare in a replacement

### Failure mode: dataset shard corruption

- **Cause**: bit rot, bad write, tokenizer bug in one shard
- **Symptom**: loss spike at predictable step (when shard is read)
- **Mitigation**: checksum shards; skip-and-log on bad batches rather than crash

### Failure mode: gradient explosion

- **Cause**: pathological batch, LR too high, missing warmup
- **Symptom**: gradient norm → ∞, loss → NaN
- **Mitigation**: gradient clipping (typical: clip to L2 norm 1.0); LR warmup over ~1% of steps

### Failure mode: power event

- **Cause**: datacenter power blip, breaker trip on a rack
- **Symptom**: subset of cluster goes dark mid-step
- **Recovery**: full cluster restart from last checkpoint
- **Mitigation**: UPS for short blips; geo-redundancy doesn't help for a single training run (can't span sites due to InfiniBand latency)

---

# Stage 3: Post-training

**Purpose**: turn the base model (a raw next-token predictor) into a deployable assistant — one that follows instructions, refuses harmful requests, behaves consistently across formats, and is good at the capabilities the product needs (code, math, reasoning, tool use).

**Output**: a final aligned model checkpoint in **ModelVault**, ready for inference deployment.

**Duration**: weeks. Much shorter than pretraining, much smaller compute (typically 512–2048 GPUs vs. 10K–100K).

**Key shift**: pretraining is data-bound and compute-heavy; post-training is **data quality bound** and **iteration speed bound**. The bottleneck moves from FLOPs to high-quality human and synthetic data.

## 3.1 Flow

Post-training is a sequence of four sub-stages, each refining the model further. They are roughly ordered but in practice are interleaved and iterated.

### Step 1 — Supervised Fine-Tuning (SFT)

  - **1.1** Curate instruction dataset
      - 1.1.1 Human-written prompt-response pairs (typical: 10K–1M examples — *typical*; Anthropic, OpenAI, Meta all in this range publicly)
      - 1.1.2 Sources: contractor annotation platforms (Scale AI, Surge AI, Invisible), in-house writers, synthetic data
      - 1.1.3 Format: standardized chat template with roles (system / user / assistant), special tokens for turn boundaries
      - 1.1.4 Quality control: per-example reviewers, sample audits, calibration sessions
      - 1.1.5 Storage: **HumanData** (encrypted object storage, strict access controls — PII and content risk)
  - **1.2** Mix
      - 1.2.1 Balance task types: general chat, code, math, reasoning, refusals, multilingual
      - 1.2.2 Upsample rare-but-important examples (e.g. safety refusals)
  - **1.3** Train
      - 1.3.1 Initialize from base model checkpoint
      - 1.3.2 Standard supervised loss: cross-entropy on assistant tokens only (user/system tokens masked out)
      - 1.3.3 Lower learning rate than pretraining (typical: 10× lower)
      - 1.3.4 Short run: 1–3 epochs over SFT data (overfitting on small SFT data is a real risk)
      - 1.3.5 Hardware: 512–2048 H100s typical, days to ~2 weeks wall clock
  - **1.4** Evaluate
      - 1.4.1 Instruction-following benchmarks (IFEval, MT-Bench — *public*)
      - 1.4.2 Internal eval sets for behaviors of interest
      - 1.4.3 Compare against base model and prior SFT versions

  **Output of Step 1**: SFT model. Knows how to format responses, follow instructions, do basic refusals. Not yet preference-aligned, not yet fully safe.

### Step 2 — Preference Learning (RLHF / RLAIF / DPO)

  This is the largest, most compute-heavy post-training stage. The model is shaped toward producing outputs that humans (or an AI judge) prefer.

  - **2.1** Generate response pairs
      - 2.1.1 For each prompt in a curated set, the SFT model generates 2–8 candidate responses
      - 2.1.2 Sampling: high temperature (0.7–1.0) for diversity, varying seeds
      - 2.1.3 Hardware: inference on the SFT model, typically 100s of GPUs
  - **2.2** Rank responses
      - 2.2.1 **RLHF (human feedback)**: human raters pick preferred response or rank N
      - 2.2.2 **RLAIF (AI feedback / Constitutional AI)**: a separate critic model evaluates against a constitution / rubric and selects preferred
          - Constitutional AI is the specific Anthropic technique: critic + revise loop, then train on the revisions (*public, Anthropic CAI paper*)
          - "RLAIF" is the generic term; CAI is one implementation
      - 2.2.3 Output: dataset of (prompt, chosen, rejected) tuples
      - 2.2.4 Scale: 10K–1M comparisons (*typical*)
  - **2.3** Choose preference algorithm
      - 2.3.1 **PPO (Proximal Policy Optimization)** — classical RLHF
          - Requires: policy model + reference model + reward model + value model, all loaded simultaneously
          - Train a reward model first (separate step) on the preference data
          - Then run RL loop: policy generates, reward model scores, PPO updates policy with KL penalty toward reference
          - Most compute-heavy option; multiple model instances in memory
      - 2.3.2 **DPO (Direct Preference Optimization)** — newer, simpler
          - Skips the reward model entirely; trains directly on preference pairs
          - Requires only: policy model + reference model
          - Much cheaper; many labs have shifted toward DPO and variants (IPO, KTO, ORPO)
      - 2.3.3 **REINFORCE-style** methods (RLOO, GRPO) — middle ground; used in some recent frontier models
  - **2.4** Train
      - 2.4.1 Hardware: 512–1024 H100s typical, 1–4 weeks wall clock
      - 2.4.2 Heavy use of inference (generating responses for RL loop) — inference and training colocated on same cluster
      - 2.4.3 KL divergence from reference model tracked carefully — too much divergence = reward hacking
  - **2.5** Evaluate
      - 2.5.1 Win rate vs reference (AlpacaEval, Arena-style head-to-head)
      - 2.5.2 Per-category eval: helpfulness, harmlessness, honesty
      - 2.5.3 Watch for **alignment tax**: capability regression on reasoning/math from over-aligning

  **Output of Step 2**: preference-aligned model. Substantially more helpful, more consistent in style, better refusals.

  **Correction to the original document**: the original quoted PPO-era hardware costs ("policy + reference + reward simultaneously") as if they applied to DPO. DPO eliminates the reward model. The actual cost depends entirely on which algorithm is chosen.

### Step 3 — Safety / Red-team training

  - **3.1** Generate adversarial inputs
      - 3.1.1 Internal red team (humans actively trying to jailbreak the model)
      - 3.1.2 Automated red-teaming: another LLM generates jailbreaks at scale
      - 3.1.3 Categorized attack patterns: prompt injection, persona switching, persuasion, technical bypass
  - **3.2** Run model on adversarial inputs
      - 3.2.1 If model complies with harmful request → record as negative example
      - 3.2.2 If model refuses appropriately → record as positive example
      - 3.2.3 If model over-refuses (refuses benign requests) → record as negative (over-refusal is also a failure)
  - **3.3** Curate safety dataset
      - 3.3.1 Stored in **HumanData** under stricter encryption / access controls
      - 3.3.2 Includes both jailbreak attempts (with correct refusals) and benign-looking but harmful requests
  - **3.4** Fine-tune on safety data
      - 3.4.1 Often SFT or DPO on this dataset
      - 3.4.2 Watch carefully for over-refusal regression on standard prompts
      - 3.4.3 Hardware: 512 H100s typical, ~1 week
  - **3.5** Iterate
      - 3.5.1 Red team finds new bypass → add to dataset → retrain → red team again
      - 3.5.2 Ongoing process; never "done"

  **Output of Step 3**: safety-hardened model. Refuses harmful requests, resistant to common jailbreaks, but not invincible.

### Step 4 — Capability training (synthetic data, verifier loops)

  - **4.1** Synthetic data generation for verifiable domains
      - 4.1.1 **Math**: model generates problems and solutions; verifier (Python execution, theorem prover) checks correctness; keep only verified-correct examples
      - 4.1.2 **Code**: model generates code + test cases; sandbox executes; keep passing examples
      - 4.1.3 **Reasoning**: chain-of-thought traces, verified by step-by-step checking
      - 4.1.4 **Tool use**: synthetic dialogs where model uses tools, with success verified
  - **4.2** Verification infrastructure
      - 4.2.1 Sandbox execution farm: CPU servers running isolated containers (gVisor, Firecracker)
      - 4.2.2 Resource limits per execution: CPU time, memory, network blocked
      - 4.2.3 Verifier results determine whether example enters training set
  - **4.3** Fine-tune on verified synthetic data
      - 4.3.1 Often mixed with original SFT/preference data
      - 4.3.2 Multiple rounds; each round generates higher-quality data using the improved model
      - 4.3.3 Hardware: 512 H100s, multiple rounds over 2–4 weeks
  - **4.4** Verify no capability regressions on non-targeted domains

  **Output of Step 4**: capability-enhanced model. Better at math, code, reasoning, tool use.

  **Note**: increasingly this verified synthetic data is also mixed back into **pretraining** (not just post-training), which blurs the stage boundaries.

### Step 5 — Final assembly and quantization

  - **5.1** Final model written to ModelVault
      - 5.1.1 Format: safetensors (industry standard) or framework-specific
      - 5.1.2 Replicated; air-gapped backup
  - **5.2** Quantization for inference
      - 5.2.1 Training format: bf16 or fp32 master weights → 2 bytes/param
      - 5.2.2 Inference format: fp8, int8, or int4 → 0.5–1 byte/param
      - 5.2.3 Quantization methods: GPTQ, AWQ, SmoothQuant, fp8 native (H100+)
      - 5.2.4 Calibration: run on representative data, measure activation distributions, set scales
      - 5.2.5 Evaluate post-quantization model — typically <1% quality regression if done well
  - **5.3** Inference-ready artifact written separately
      - 5.3.1 Smaller (2–4× than training-format)
      - 5.3.2 Stored alongside full-precision version
      - 5.3.3 Shipped to inference fleet

  **Important**: the training-format checkpoint and the inference-format artifact are **different files**. The original document implied a single 5 TB safetensors file went directly from ModelVault to inference. In reality, there's a quantization step.

## 3.2 Components

### Storage spec sheet

| Component | Storage type | Capacity | Why |
|---|---|---|---|
| **HumanData** | Encrypted object storage, strict ACLs | 100s of GB – low TB | SFT, preference, safety datasets; PII risk + content risk |
| **SyntheticData** | Standard object storage | 1–10 TB | Generated math/code/reasoning datasets |
| **ModelVault** | Replicated object + air-gapped backup | 10s–100s of TB | All training-format checkpoints |
| **InferenceArtifacts** | Replicated to regional inference clusters | TBs (per model, post-quantization) | Quantized weights ready to serve |

### Compute spec sheet

| Cluster | Hardware | Scale (typical) | Workload |
|---|---|---|---|
| **Post-train cluster** | H100 80GB or H200 | 512–2048 GPUs | SFT, DPO/PPO, safety fine-tune, capability fine-tune |
| **Inference for data gen** | H100, lower-end OK | 100s of GPUs | Generate responses for preference data, synthetic data |
| **Critic / judge cluster** | H100 | 64–256 GPUs | RLAIF judge inference |
| **Sandbox farm** | CPU servers, isolated (gVisor/Firecracker) | 100–500 nodes | Code execution, math verification |
| **Eval cluster** | H100 | 100s of GPUs | Continuous eval across post-training stages |

### Human infrastructure (this stage has significant non-GPU costs)

| Role | Function | Typical scale |
|---|---|---|
| Annotators / data labelers | Write prompts, write responses, rank pairs | 500–5000 contractors per lab |
| QA reviewers | Audit annotator output, calibrate | 50–200 |
| Red team | Generate adversarial examples, jailbreak | 50–200 internal + bug bounty |
| Policy / safety team | Define refusal policies, rubrics | 10–50 internal |
| ML engineers (post-train) | Run experiments, iterate | 20–100 per lab |

## 3.3 Monitoring

### Per-experiment metrics

| Metric | What it measures | Why |
|---|---|---|
| Training loss | Standard | Health of fine-tuning |
| KL divergence from reference | How far policy has drifted from SFT/base | Too high = reward hacking risk |
| Reward score (RLHF) | Reward model's verdict on policy outputs | Improving = on track; plateau or decline = problem |
| Win rate vs reference | Head-to-head win rate against prior model | The metric that matters most |
| Refusal rate on benign prompts | Over-refusal rate | Catches alignment tax / over-correction |

### Eval suites (per checkpoint)

| Eval | What it measures | Cadence |
|---|---|---|
| **MT-Bench / Arena-style** | General chat quality | Every meaningful checkpoint |
| **MMLU / BBH / MATH / HumanEval** | Capability benchmarks | Per stage end |
| **HHH (Helpful / Harmless / Honest)** | Behavioral evals | Per stage end |
| **Red-team evals** | Jailbreak resistance | Every safety iteration |
| **Internal evals** | Custom benchmarks, harder to game | Continuous |
| **Capability regressions** | Whether targeted training hurt other capabilities | Every stage |

### Data quality monitoring

| Metric | Why |
|---|---|
| Inter-annotator agreement | Disagreement signals unclear rubric or hard examples |
| Annotator throughput | Bottleneck on data flow |
| Per-annotator quality | Spot drifting or low-quality contributors |
| Audit pass rate | What % of annotations pass QA review |

### Safety-specific monitoring

| Metric | Why |
|---|---|
| Jailbreak success rate (held-out red team set) | Resistance trend over time |
| Categorized refusal accuracy | Right things refused, wrong things not over-refused |
| Dual-use response audit | Sample outputs in sensitive categories, manual review |

### What is NOT monitored well

- **Long-horizon failure modes** (sycophancy, deception, manipulation): hard to measure with current evals; active research.
- **Capability emergence**: model gaining new abilities mid-training is real and largely unpredictable.
- **Distributional misalignment**: model behaves well on eval distribution, fails on deployment distribution. Only deployment surfaces this.

## 3.4 Tradeoffs & Failure Modes

### Decision: RLHF (PPO) vs DPO vs others

| | PPO | DPO | REINFORCE-style (GRPO, RLOO) |
|---|---|---|---|
| Models loaded simultaneously | 4 (policy, ref, reward, value) | 2 (policy, ref) | 2 (policy, ref) |
| Compute cost | Highest | Lowest | Middle |
| Reward model needed | Yes | No | Optional |
| Online sampling needed | Yes | No (preference data is offline) | Yes |
| Stability | Tricky (KL coeff, value loss) | Mostly stable | Tunable |
| Adoption (2024–2025) | Declining at frontier labs | Common | Rising for verifiable rewards |
| When it shines | When you can iterate the reward model | When preference data is fixed | When rewards are verifiable (math, code) |

### Decision: human feedback vs AI feedback (Constitutional AI)

| | Human (RLHF) | AI critic (RLAIF / CAI) |
|---|---|---|
| Cost per preference label | $0.50–$5 (*illustrative*) | ~$0.001 (just compute) |
| Throughput | 100s of labels/hour | 100Ks of labels/hour |
| Quality | High for what humans agree on; noisy on edge cases | Consistent; encodes the rubric well |
| Failure mode | Human bias, fatigue, disagreement | Critic blind spots; model can game its own critic |
| Where used | High-stakes rubric items, calibration | Bulk preference data, scalable oversight |

### Decision: SFT data scale

- **Too little SFT data (< 10K)**: model doesn't learn instruction format well, defaults to base-model behavior
- **Too much SFT data (> 1M)**: marginal returns diminish; high-quality 10K-100K often beats noisy 1M
- **Failure mode**: contractor-written data that mimics what contractors *think* good responses look like, not what users actually want

### Decision: alignment tax vs capability

- **Pro of aligning aggressively**: safer, more useful in practice
- **Con**: every alignment pass can degrade math, code, or reasoning by a few %
- **Mitigation**: mix capability training into alignment passes; eval per-domain continuously
- **Failure mode**: over-refusal — model refuses benign requests because they pattern-match to refused categories

### Failure mode: reward hacking

- **Cause**: policy finds a way to maximize reward without actually being better (e.g. very long, verbose responses that the reward model rates highly)
- **Symptom**: reward going up, human ratings going down
- **Mitigation**: KL penalty toward reference, periodic human spot-check of policy outputs, adversarial reward model training

### Failure mode: distribution shift between SFT data and deployment

- **Cause**: SFT prompts written by contractors; real users have different vocabulary, formality, requests
- **Symptom**: model performs well on internal eval, regresses in production
- **Mitigation**: include production-traffic-derived prompts in SFT mix (with strict PII filtering)

### Failure mode: jailbreak arms race

- **Cause**: new jailbreak techniques constantly emerge (DAN, role-play, encoded requests, multilingual bypass)
- **Symptom**: model patched against last week's jailbreak fails to this week's
- **Mitigation**: continuous red-teaming, fast iteration loop, automated jailbreak detection in production

### Failure mode: synthetic data feedback loop

- **Cause**: synthetic data generated by Model_N is used to train Model_N+1, which generates Model_N+2's data, etc.
- **Symptom**: gradual collapse to model's own preferred patterns; loss of diversity
- **Mitigation**: anchor with substantial real-data inclusion; verify synthetic data with non-model verifiers (Python, theorem provers) wherever possible

### Failure mode: quantization regression

- **Cause**: bf16 training-format model degrades when quantized to fp8/int4
- **Symptom**: deployed model performs worse than checkpoint evals suggested
- **Mitigation**: evaluate post-quantization model on the full eval suite before shipping; quantization-aware training in extreme cases

---

# Stage 4: Inference (serving)

**Purpose**: serve the final model to users at scale — accept a user message, run the model, stream back tokens — under latency, throughput, cost, and safety constraints.

**Output**: a streaming response back to the user, plus logs and feedback events for the downstream feedback loop.

**Scale shift**: inference is the largest GPU footprint in a mature LLM company. Training a frontier model is a one-time cost of months; inference is billions of requests per day, indefinitely. Inference GPU fleet typically **3–10× larger than the training cluster**, distributed globally.

**Key constraints**:
- **Latency**: time-to-first-token (TTFT) must be < a few hundred ms; time-per-output-token (TPOT) must keep up with human reading speed (~30–60 tokens/sec sustained feels fast).
- **Throughput**: tokens/sec per GPU is the cost-of-goods metric.
- **Concurrency**: thousands of users per GPU node, batched together.

## 4.1 Flow

### Step 1 — User sends request

  - **1.1** User device
      - 1.1.1 Phone, laptop, or other client; running Claude.ai web app or native iOS/Android app
      - 1.1.2 Message typed; on send, packaged as HTTPS POST with auth token
  - **1.2** Network path
      - 1.2.1 Device → ISP → public internet → Anthropic edge PoP
      - 1.2.2 TLS 1.3 termination at edge
      - 1.2.3 Typical edge latency: 10–50ms RTT depending on geography

### Step 2 — Edge gateway

  - **2.1** TLS termination, HTTP/2 or HTTP/3
  - **2.2** Authentication
      - 2.2.1 Validate session token / API key
      - 2.2.2 Lookup user identity, tier (free / pro / team / enterprise / API)
  - **2.3** Rate limiting
      - 2.3.1 Per-user request rate, per-user token rate, per-organization rate
      - 2.3.2 Check against **RateLimitDB** (in-memory KV store, typically Redis cluster with sharding)
      - 2.3.3 If over limit: reject 429 with retry-after
  - **2.4** Bot / abuse detection
      - 2.4.1 Heuristics + ML model to flag suspicious patterns
      - 2.4.2 Hard blocks for known-bad signatures
  - **2.5** Route to regional **AppCluster**
      - 2.5.1 Latency-based routing (user → nearest healthy region)
      - 2.5.2 Failover to next-nearest region if primary is degraded

  **Hardware**: edge servers at 30–100 global PoPs, CPU-only, no GPUs. Typical: Cloudflare-class edge or comparable.

### Step 3 — Application cluster (orchestration layer)

  This is the "non-AI" backend that prepares the model request.

  - **3.1** Fetch conversation history
      - 3.1.1 Query **ConvoStore** (distributed database) for past messages in this thread
      - 3.1.2 Hot recent conversations in low-latency tier; cold conversations may be tiered to object storage
  - **3.2** Fetch user memories / context
      - 3.2.1 Query **MemoryStore** for user-specific facts and preferences
      - 3.2.2 Vector search for semantically-relevant past content; KV lookup for explicit facts
  - **3.3** Fetch system prompt and tool definitions
      - 3.3.1 Query **ConfigStore** (in-memory, replicated, very fast)
      - 3.3.2 Tool definitions injected based on entitlements (e.g. web search, code execution availability)
  - **3.4** Input-side safety classifiers
      - 3.4.1 Fast classifiers run on the user message (CSAM, illegal content categories)
      - 3.4.2 If flagged: block before sending to model, return policy response
  - **3.5** Assemble model request
      - 3.5.1 System prompt + conversation history + user message + tool defs + sampling params
      - 3.5.2 Token count estimated; if over context window, truncate or summarize history
      - 3.5.3 Routing decision: which model size, which region's inference cluster
  - **3.6** Send to inference cluster
      - 3.6.1 Internal network (data center fabric, 100Gb+)
      - 3.6.2 gRPC or HTTP/2 streaming

  **Hardware**: Kubernetes-managed CPU servers, hundreds per region. Stateless; horizontally scales.

### Step 4 — Inference cluster (the GPU work)

  This is where the model actually runs. Modern inference is **disaggregated**: prefill (process the prompt) and decode (generate tokens) often run on separate GPU pools because their compute profiles differ.

  - **4.1** Request lands on inference server
      - 4.1.1 Server runs vLLM, TensorRT-LLM, SGLang, or custom inference stack
      - 4.1.2 Multiple concurrent requests batched together (continuous batching)
  - **4.2** Tokenize input
      - 4.2.1 Apply tokenizer to prompt
      - 4.2.2 Output: sequence of token IDs
  - **4.3** Check prefix cache (KV cache reuse)
      - 4.3.1 Hash prompt prefix; lookup in **KVCacheStore** (HBM-resident, with CPU-RAM spillover)
      - 4.3.2 If system prompt or conversation prefix matches a cached entry: reuse its KV states, skip recomputing those tokens
      - 4.3.3 Prefix cache hit rate at scale: 30–70% (*typical*; depends on prompt structure)
      - 4.3.4 Cache eviction: LRU or similar policy under HBM pressure
  - **4.4** Prefill phase (process the prompt)
      - 4.4.1 Compute KV states for all prompt tokens not in cache
      - 4.4.2 Highly parallel — all tokens computed in one pass
      - 4.4.3 Compute-bound (uses GPU FLOPs heavily, low HBM bandwidth pressure)
      - 4.4.4 Output: first token logits, KV cache populated
  - **4.5** Decode phase (generate tokens one at a time)
      - 4.5.1 Loop:
          - 4.5.1.1 Forward pass using cached KVs from all previous tokens
          - 4.5.1.2 Compute logits for the single next position
          - 4.5.1.3 Apply sampling (temperature, top-p, top-k, repetition penalty)
          - 4.5.1.4 Emit chosen token
          - 4.5.1.5 Detokenize incrementally (token → text fragment, handling sub-word boundaries)
          - 4.5.1.6 Stream emitted token back upstream immediately (don't wait for end)
          - 4.5.1.7 Append new token to context; update KV cache
      - 4.5.2 Termination: stop token emitted, max length reached, or stop-sequence matched
      - 4.5.3 Memory-bandwidth bound (each decode step reads all KVs from HBM)
  - **4.6** Optional: speculative decoding
      - 4.6.1 A small "draft" model proposes K future tokens
      - 4.6.2 Main model verifies all K in one forward pass (cheap because it's a single forward)
      - 4.6.3 Accepted prefix is emitted; rejected tokens trigger fallback
      - 4.6.4 Speedup: 1.5–3× typical for chat workloads (*typical*; methods: Medusa, EAGLE, draft-target speculation)
  - **4.7** Output-side safety classifiers
      - 4.7.1 Streaming classifiers run on emitted tokens in parallel
      - 4.7.2 If flagged: terminate generation, return safety error
      - 4.7.3 Mild flags logged but not blocked

  **Correction to original**: original showed detokenization as a terminal step after the whole response is generated. In reality, detokenization is **interleaved with generation** — vLLM and TRT-LLM detokenize incrementally so streaming can start at the first token.

### Step 5 — Continuous batching (the central efficiency trick)

  - **5.1** Without batching
      - 5.1.1 Each request occupies the GPU alone
      - 5.1.2 GPU utilization: typically < 20% on chat workloads
      - 5.1.3 Throughput per dollar: poor
  - **5.2** With static batching
      - 5.2.1 Fixed batch of N requests start together, finish together
      - 5.2.2 Short responses wait for long ones
      - 5.2.3 GPU underutilized late in batch
  - **5.3** Continuous batching (used in vLLM, TRT-LLM, SGLang)
      - 5.3.1 Requests join/leave the batch at each decode step
      - 5.3.2 Short requests finish; new requests join immediately
      - 5.3.3 GPU stays at high utilization (70%+)
      - 5.3.4 **This is what makes inference economics work** at chat scale
  - **5.4** PagedAttention (vLLM innovation, *public*)
      - 5.4.1 KV cache split into fixed-size blocks (like OS pages)
      - 5.4.2 Logical-to-physical block mapping per request
      - 5.4.3 Eliminates KV cache fragmentation
      - 5.4.4 Enables higher batch sizes for the same HBM

### Step 6 — Response streaming back

  - **6.1** Token stream flows: inference → AppCluster → EdgeGateway → user device
  - **6.2** Protocol: Server-Sent Events (SSE) over HTTPS, or WebSocket
  - **6.3** Each token forwarded immediately on receipt (no buffering)
  - **6.4** Parallel: AppCluster writes conversation to **ConvoStore** (typically async, after stream completes)
  - **6.5** Logs written to **SafetyLog** and observability pipelines

### Step 7 — User feedback

  - **7.1** User actions captured
      - 7.1.1 Thumbs up / thumbs down rating
      - 7.1.2 Regenerate request (implicit negative)
      - 7.1.3 Edit / continue (implicit signals)
      - 7.1.4 Time spent reading / next message (engagement signals)
  - **7.2** Events written to **FeedbackStore** (event log, e.g. Kafka)
  - **7.3** Feeds into Stage 5 (Feedback Loop)

## 4.2 Components

### Storage spec sheet

| Component | Storage type | Capacity (typical) | Latency requirement | Why |
|---|---|---|---|---|
| **ConvoStore** | OLTP DB (Spanner/CockroachDB/Postgres) for hot; object storage for cold | Hot: 100s of TB. Cold: PBs | < 50 ms read | Recent conversations served quickly; cold tier for archive |
| **MemoryStore** | KV store + vector DB (pgvector, custom) | 10s–100s of TB | < 100 ms | Per-user state |
| **ConfigStore** | In-memory replicated KV (etcd-like) | GBs | < 10 ms | Hot path; cannot be slow |
| **RateLimitDB** | Redis cluster (with persistence) | TBs | < 5 ms | Every request hits this |
| **KVCacheStore** | GPU HBM (primary) → CPU RAM (spillover) → optional NVMe | Per-node: 10s–100s of GB | μs from HBM, ms from CPU RAM | Read on every decode step |
| **SafetyLog** | Log aggregation (Elasticsearch/OpenSearch/ClickHouse) | PBs | Eventual consistency OK | Audit + analysis |
| **FeedbackStore** | Kafka cluster + warehouse downstream | 10+ PB | Eventual OK | Feeds training loop |

### Compute spec sheet

| Cluster | Hardware | Scale (typical, per region) | Workload |
|---|---|---|---|
| **EdgeGateway** | CPU servers, fast NICs | 100s of nodes across 30–100 PoPs globally | TLS, routing, rate limit |
| **AppCluster** | CPU servers, mid-spec | 500–5000 nodes | Orchestration, no GPU |
| **InferenceCluster (prefill)** | H100/H200/B200, can be lower-end | 1000s of GPUs | Compute-bound prefill |
| **InferenceCluster (decode)** | H100/H200/B200, memory-bandwidth optimized | 1000s of GPUs | Memory-bound decode |
| **Classifier pool** | A10/L4/T4 GPUs | 100s of nodes | Input/output safety classifiers, smaller models |
| **Observability cluster** | CPU, big storage | 100s of nodes | Logs, metrics, traces |

**Disaggregated prefill/decode**: increasingly, frontier inference stacks separate prefill GPUs from decode GPUs, because:
- Prefill is compute-bound → benefits from raw FLOPs
- Decode is memory-bandwidth-bound → benefits from HBM bandwidth, not FLOPs
- Separating them lets each pool be sized independently

### Network spec sheet

| Layer | Bandwidth requirement | Why |
|---|---|---|
| Edge ↔ user | Internet-class; 10–50 ms RTT | Streaming response, low-latency UX |
| Edge ↔ AppCluster | Internal DC, 25–100 Gb/s | Internal traffic |
| AppCluster ↔ InferenceCluster | Internal DC, 100 Gb/s | Streaming tokens back |
| Intra-inference-node (GPU↔GPU) | NVLink 900 GB/s | Tensor parallelism on the model |
| Inter-inference-node (within instance) | InfiniBand NDR | Pipeline parallelism, if used |

## 4.3 Monitoring

This is the largest monitoring surface in the entire system. Inference is a live service with strict SLOs.

### Latency SLOs

| Metric | What it measures | Typical target |
|---|---|---|
| **TTFT (Time To First Token)** | User send → first token visible | < 300 ms p50, < 1s p99 |
| **TPOT (Time Per Output Token)** | Avg time between subsequent tokens | < 50 ms p50, < 200 ms p99 |
| **End-to-end latency** | User send → response complete | Varies with output length |
| **TPS (Tokens Per Second per stream)** | Sustained generation rate | > 30 tokens/sec is "fast" |

### Throughput / efficiency metrics

| Metric | What it measures | Why |
|---|---|---|
| **Tokens/sec per GPU** | Throughput per unit cost | The cost-of-goods metric |
| **Active batch size** | How many requests in flight | Driver of efficiency |
| **GPU compute utilization** | SM occupancy on inference nodes | Should be 60–80%+ with continuous batching |
| **HBM bandwidth utilization** | Memory throughput on decode | Should be near peak during decode |
| **MFU for inference** | Realized FLOPs / theoretical | Frontier inference: 30–50% (*typical*); lower than training |
| **Prefix cache hit rate** | KV cache reuse % | Higher = lower compute cost |
| **Speculative decoding accept rate** | Fraction of draft tokens accepted | Indicates speculation quality |

### Capacity / queuing metrics

| Metric | Why |
|---|---|
| Requests/sec per region | Capacity utilization |
| Queue depth at AppCluster → Inference | Backpressure signal |
| Time-in-queue p99 | If high: inference fleet undersized |
| Available KV cache blocks | Low = upcoming OOM on inference node |
| Concurrent active streams per node | Saturation indicator |

### Quality / safety metrics

| Metric | Why |
|---|---|
| Refusal rate (overall) | Drift detection in model behavior |
| Refusal rate by category | Catches over-refusal in specific topics |
| Safety classifier trigger rate | Flagged outputs |
| Severe classifier hit rate | Should be tiny; spike = attack or regression |
| Time to detect attack pattern | Operational metric for safety response |

### Reliability metrics

| Metric | Why |
|---|---|
| Per-request error rate (5xx) | Service health |
| GPU failure rate | Inference fleet decay |
| Cluster availability % | SLA compliance |
| Geographic failover counts | Regional health |
| Cold start time (new pod) | Scale-up responsiveness |

### Cost / economics metrics

| Metric | Why |
|---|---|
| Cost per million input tokens | Economics |
| Cost per million output tokens | Economics (output is more expensive than input) |
| GPU-hours per active user per day | Capacity planning |
| Cache hit rate $ savings | KV cache value |

### What is hard to monitor

- **User satisfaction**: thumbs-up/down is sparse and biased. Real satisfaction requires deeper signals.
- **Subtle hallucinations**: model confidently wrong, no metric catches it in real time.
- **Long-conversation degradation**: behavior changes over a long context, only visible in retrospective analysis.

## 4.4 Tradeoffs & Failure Modes

### Decision: continuous batching vs static batching

| | Continuous (chosen) | Static |
|---|---|---|
| GPU utilization | 70%+ | Often < 30% |
| Implementation complexity | High (state machine per request) | Simple |
| Latency for first request | Slightly higher (joins next step) | Optimal for that one request |
| Throughput | Much higher | Lower |
| When static wins | Workloads with uniform sequence lengths | Rare in chat |

### Decision: disaggregated prefill/decode vs unified

| | Unified | Disaggregated |
|---|---|---|
| GPU pool management | Simpler (one fleet) | More complex (two fleets) |
| Hardware optimization | Compromise | Each pool optimized for its phase |
| Network overhead | None | Must transfer KV cache from prefill → decode node |
| Best for | Lower-traffic deployments | High-traffic, latency-sensitive frontier serving |

### Decision: speculative decoding

- **Pro**: 1.5–3× speedup on chat workloads, no quality loss (verified by target model)
- **Con**: extra GPU memory for draft model; complexity in inference stack
- **Failure mode**: draft model trained on different distribution → low accept rate → no speedup, plus overhead
- **Mitigation**: train draft model on production traffic distribution

### Decision: KV cache spillover hierarchy

| Tier | Latency | Capacity | Reality |
|---|---|---|---|
| GPU HBM | μs | 10s of GB per GPU | Primary; eviction policy critical |
| CPU RAM | ~10s of μs (PCIe transfer) | 100s of GB per node | Common spillover |
| NVMe | ms | TBs | **Rare in production** — latency blows decode budget |
| Network-attached cache | ms+ | Large | Emerging (LMCache etc.); still experimental |

**Correction to original**: original listed NVMe spillover casually. In practice, NVMe-tier KV cache is research-stage. Most production stacks: HBM with CPU-RAM as eviction destination, prefix caching as primary reuse mechanism.

### Decision: quantization precision

| | bf16 | fp8 | int8 | int4 |
|---|---|---|---|---|
| Quality regression | Baseline | < 1% | 1–3% | 3–10% |
| Throughput uplift | 1× | 1.5–2× | 2× | 3–4× |
| Hardware support | Universal | H100+ | Universal | Universal |
| When used | Reasoning-critical | Default for frontier | Resource-constrained | Edge / mobile |

### Decision: regional distribution

- **Pro**: low latency for global users; resilience to regional failures
- **Con**: weight replication across regions (10s of TB per model per region), capacity planning per region
- **Failure mode**: regional traffic spike → cross-region failover → latency degradation; mitigated by overprovisioning

### Failure mode: KV cache OOM under load

- **Cause**: traffic spike with long contexts; KV memory budget exceeded
- **Symptom**: new requests rejected with "out of capacity"; existing requests degraded
- **Mitigation**: admission control (reject early if can't fit), preemption (kill lowest-priority), more aggressive eviction

### Failure mode: head-of-line blocking with long requests

- **Cause**: a few requests with 100K+ token outputs hog batch slots
- **Symptom**: short requests queue up; TTFT p99 spikes
- **Mitigation**: separate pools for short vs long requests; priority lanes; max output length caps

### Failure mode: classifier false positives

- **Cause**: output safety classifier flags benign content
- **Symptom**: user gets generic refusal on innocuous query; bad UX
- **Mitigation**: classifier calibration; tiered response (log mild, block severe); fast retraining loop on flagged samples

### Failure mode: model rollout regression

- **Cause**: new model version ships with regression on some niche
- **Symptom**: user complaints; metric shift on specific eval categories
- **Mitigation**: canary deploys (1% → 10% → 100%); per-version traffic splitting; instant rollback capability

### Failure mode: dependency cascade

- **Cause**: a downstream service (ConvoStore, MemoryStore) slows or fails
- **Symptom**: AppCluster requests time out; inference idle but user-facing errors
- **Mitigation**: circuit breakers, graceful degradation (serve without memory if memory is down), aggressive timeouts

### Failure mode: prompt injection / jailbreak in production

- **Cause**: adversarial user input bypasses safety training
- **Symptom**: model emits harmful content
- **Mitigation**: output classifier (defense in depth), monitoring for attack patterns, fast incident response to add to next safety training round

### Failure mode: regional power / network event

- **Cause**: data center power blip, fiber cut, BGP misconfiguration
- **Symptom**: region drops out
- **Mitigation**: cross-region failover at edge; per-region health checks; overprovisioned capacity

---

# Stage 5: Feedback Loop

**Purpose**: take signals from production — explicit ratings, implicit behavior, safety events, support tickets — and feed them back into the next training cycle as curated training data.

**Output**: curated subsets added to **HumanData** (for post-training) and occasionally back to **GoldDataset** (for pretraining), plus operational dashboards.

**Why this stage matters**: pretraining + post-training give you Model_N. The feedback loop is how you decide what's wrong with Model_N and shape Model_N+1. Without it, models stop improving.

## 5.1 Flow

### Step 1 — Signal capture (in production)

  - **1.1** Explicit signals
      - 1.1.1 Thumbs up / thumbs down on responses
      - 1.1.2 Written feedback when user explains the rating
      - 1.1.3 Report buttons (safety, accuracy, other)
  - **1.2** Implicit signals
      - 1.2.1 Regenerate request (user wanted something different)
      - 1.2.2 Stop-and-retry (user cancelled then asked something similar)
      - 1.2.3 Edit (user copied and modified — model didn't quite have it)
      - 1.2.4 Continue conversation (positive signal that response was useful)
      - 1.2.5 Time-to-next-message (engagement)
      - 1.2.6 Session length, return rate (broader engagement)
  - **1.3** Operational signals
      - 1.3.1 Safety classifier hits (output blocked or flagged)
      - 1.3.2 Errors, timeouts, retries
      - 1.3.3 Customer support tickets referencing specific outputs

  All written to **FeedbackStore** as immutable events with conversation context (subject to retention policy).

### Step 2 — Ingestion and routing

  - **2.1** **FeedbackStore** is an append-only event log
      - 2.1.1 Typical stack: Kafka cluster, dozens to low hundreds of brokers
      - 2.1.2 Retention: 7–30 days hot in Kafka, indefinite in downstream warehouse
  - **2.2** Streamed to **Warehouse** for analytics
      - 2.2.1 Typical: Snowflake, BigQuery, Databricks, or in-house columnar store
      - 2.2.2 Schema: user_id (hashed), conversation_id, message_id, model_version, signal_type, signal_value, timestamp, content_hash
      - 2.2.3 PII scrubbing at ingestion; raw content stored under stricter access controls
  - **2.3** Streamed to **MetricsStore** for real-time monitoring (aggregate counts, not raw events)

### Step 3 — Aggregation and analysis

  - **3.1** Batch jobs (Spark/Ray on CPU clusters, ~1000 nodes typical)
  - **3.2** Aggregations
      - 3.2.1 Thumbs-down rate by topic / category / user segment / model version
      - 3.2.2 Regenerate rate as proxy for dissatisfaction
      - 3.2.3 Per-prompt-cluster failure rate (cluster prompts by semantic similarity, find clusters with high failure rate)
      - 3.2.4 Comparative analysis across model versions (Model_N vs Model_N-1)
  - **3.3** Pattern discovery
      - 3.3.1 Topic modeling on user prompts with high failure rates
      - 3.3.2 Embedding clustering to find similar failure cases
      - 3.3.3 Causal analysis: did Model_N regress on a specific behavior?
  - **3.4** Outputs
      - 3.4.1 Internal dashboards for product / research teams
      - 3.4.2 Candidate examples for human review (next step)

### Step 4 — Human review and curation

  - **4.1** Reviewers triage flagged examples
      - 4.1.1 Was the model actually wrong? (sometimes thumbs-down is user error or disagreement)
      - 4.1.2 What's the correct response?
      - 4.1.3 Is this a one-off or a pattern?
  - **4.2** Categorize
      - 4.2.1 Pretraining gap (model lacks knowledge) → flag for data team
      - 4.2.2 Post-training gap (model has knowledge but behaves wrong) → flag for post-train team
      - 4.2.3 Safety gap → flag for safety team
      - 4.2.4 Product / UX issue (not a model issue) → flag for product
  - **4.3** Convert to training data
      - 4.3.1 Write corrected responses (becomes new SFT examples)
      - 4.3.2 Rank pairs (becomes new preference data)
      - 4.3.3 Add to safety dataset for refusal training
      - 4.3.4 Store in **HumanData** with provenance tags

### Step 5 — Feed back into training

  - **5.1** Curated data flows into next post-training cycle
      - 5.1.1 Mixed with existing post-training data
      - 5.1.2 Weight set per category importance
  - **5.2** For pretraining gaps, source content added to next pretraining mix
      - 5.2.1 E.g. if model is weak on a coding library, add more high-quality examples of that library to pretraining
  - **5.3** A/B testing the next model
      - 5.3.1 Canary release to small % of traffic
      - 5.3.2 Compare metrics: did the targeted improvements land? Did anything regress?
      - 5.3.3 Decide whether to roll out fully, iterate, or rollback

## 5.2 Components

### Storage spec sheet

| Component | Storage type | Capacity (typical) | Why |
|---|---|---|---|
| **FeedbackStore** | Kafka cluster | 10s of TB hot, retention 7–30 days | High write throughput, event log semantics |
| **Warehouse** | Columnar analytics DB (Snowflake/BigQuery/in-house) | 10+ PB | Long-horizon analytics, joins across user/model/session |
| **HumanData (feedback subset)** | Encrypted object storage | TBs added per cycle | Reviewed and curated training data |
| **MetricsStore** | Time-series DB | TBs | Real-time dashboards |
| **EmbeddingStore** | Vector DB | 100s of TB | Semantic search and clustering of failure patterns |

### Compute spec sheet

| Cluster | Hardware | Scale | Workload |
|---|---|---|---|
| **Analytics cluster** | CPU (Spark/Ray) | 1000+ nodes | Batch aggregations |
| **Embedding cluster** | Small/mid GPUs (L4/A10) | 100s of nodes | Embed prompts for clustering |
| **Review tooling backend** | CPU | 10s of nodes | Annotation UI, queue management |

### Human infrastructure

| Role | Function | Typical scale |
|---|---|---|
| Triage reviewers | First-pass categorization of flagged examples | 100s |
| Domain experts | Deep review of technical / sensitive cases | 10s |
| Product analysts | Read dashboards, identify trends | 10s |

## 5.3 Monitoring

### Signal volume metrics

| Metric | Why |
|---|---|
| Events/sec into FeedbackStore | Capacity health |
| Per-signal-type rate (thumbs up/down/regenerate/etc.) | Trend detection |
| Per-model-version signal rate | Comparative quality across versions |

### Quality metrics

| Metric | Why |
|---|---|
| Overall thumbs-down rate | Aggregate satisfaction proxy |
| Per-category thumbs-down rate | Where model struggles |
| Per-segment satisfaction (free vs pro, by region) | Equitable performance |
| Regenerate rate | Implicit dissatisfaction |
| Conversation length / return rate | Engagement proxy |

### Pipeline health

| Metric | Why |
|---|---|
| Ingestion lag (event → warehouse) | Stale dashboards = stale decisions |
| Aggregation job runtime | Batch SLA |
| Review queue depth | Reviewer capacity |
| Time-to-feedback-incorporated (signal → training data) | Loop speed |

### Sampling bias monitoring

| Metric | Why |
|---|---|
| Demographics of feedback givers | Loud users skew signal |
| Topics represented in feedback | Niche topics may be underrepresented |
| Geographic / language distribution | Avoid optimizing only for one segment |

### What is hard to monitor

- **True quality**: thumbs-up/down is a noisy proxy. Real quality requires deeper studies (longitudinal user surveys, expert review).
- **Long-term effects**: did Model_N+1 actually retain users better than Model_N? Months of data needed.
- **Counterfactuals**: what would have happened if we hadn't shipped the change?

## 5.4 Tradeoffs & Failure Modes

### Decision: implicit signals vs explicit signals

| | Implicit (regen, edits, engagement) | Explicit (thumbs, ratings) |
|---|---|---|
| Volume | Very high | Sparse (< 1% of conversations) |
| Noise | High (many reasons users regen) | Lower (intentional) |
| Bias | Engagement bias (loud users) | Self-selection bias |
| Best for | Aggregate trends, A/B tests | Specific failure cases |

### Decision: scale of human review

- **Pro of more reviewers**: more training data, faster iteration
- **Con**: expensive, quality variance
- **Failure mode**: reviewer fatigue / drift; calibration sessions needed regularly

### Decision: feedback retention policy

- **Pro of long retention**: more data for analysis, longer-horizon studies
- **Con**: privacy risk, storage cost, regulatory exposure (GDPR right-to-erasure)
- **Mitigation**: time-bounded retention; hashed identifiers; user controls

### Failure mode: feedback loop overfits to vocal minority

- **Cause**: thumbs-down is given disproportionately by certain user segments
- **Symptom**: model optimizes for those segments, regresses for silent majority
- **Mitigation**: segment-aware analysis; A/B testing across segments; explicit balancing

### Failure mode: Goodhart's law on feedback metrics

- **Cause**: optimizing for thumbs-up rate creates incentive for sycophancy, hedging, padding
- **Symptom**: model gets higher ratings but is less useful (longer responses, more agreement)
- **Mitigation**: multiple competing metrics, periodic human audit, hold-out evals not used for training

### Failure mode: stale feedback informing new model

- **Cause**: feedback collected on Model_N used to train Model_N+1, which has different capabilities
- **Symptom**: training data targets gaps that no longer exist; misses new gaps
- **Mitigation**: rapid iteration; canary releases generating fresh feedback; freshness weighting

### Failure mode: PII / sensitive content in feedback

- **Cause**: user writes sensitive info in their messages; gets captured in feedback pipelines
- **Symptom**: privacy incident
- **Mitigation**: scrubbing at ingestion; strict access controls on warehouse; encryption at rest; audit trails

### Failure mode: feedback poisoning

- **Cause**: coordinated thumbs-down campaigns to shape model behavior
- **Symptom**: anomalous signal patterns; targeted topic regressions
- **Mitigation**: anomaly detection on feedback patterns; per-user weighting; sample audits

---

# Section 6: Cross-cutting reference

This section consolidates the storage, compute, and terminology references that appear scattered through the stages. Use it for lookup, not learning.

## 6.1 Consolidated storage tier reference

| System | Stage(s) | Storage tier | Media | Capacity (typical) | Throughput characteristics | Cost tier¹ | Notes |
|---|---|---|---|---|---|---|---|
| **RawLake** | 1 | Object (archive) | HDD | 10s of PB | Low read | Lowest | Write-once raw scraped bytes |
| **TextLake** | 1 | Object (standard) | HDD | Single-digit PB | Medium read | Low | Cleaned text, re-readable |
| **GoldDataset** | 1, 2 | Parallel FS | NVMe SSD | Low PB | 100s GB/s aggregate read | High | Pretraining data; GPU-feeding |
| **ModelArtifacts** | 1, 2, 3 | Object, versioned | SSD | GBs | Low | Low | Tokenizers, configs |
| **CheckpointStore** | 2 | Parallel FS (separate cluster) | NVMe SSD | 100s of TB | 100s GB/s burst write | High | Training checkpoints |
| **ModelVault** | 2, 3 | Replicated object + air-gap | SSD + tape | 10s–100s of TB | Low (cold) | Medium | Canonical weights, valuable IP |
| **HumanData** | 3, 5 | Encrypted object, strict ACLs | SSD | 100s of GB – low TB | Medium | Medium | SFT, preference, safety data |
| **SyntheticData** | 3 | Object | SSD | 1–10 TB | Medium | Low | Generated math/code/reasoning |
| **InferenceArtifacts** | 3, 4 | Replicated to inference regions | SSD | TBs per model | Read on model load | Medium | Quantized weights |
| **ConvoStore** | 4 | OLTP DB + cold tier | SSD (hot), HDD (cold) | Hot 100s TB, cold PB | < 50 ms read | Medium | User conversations |
| **MemoryStore** | 4 | KV + vector DB | SSD | 10s–100s of TB | < 100 ms | Medium | Per-user memories |
| **ConfigStore** | 4 | In-memory replicated KV | RAM | GBs | < 10 ms | High (RAM) | System prompts, configs |
| **RateLimitDB** | 4 | Redis | RAM + SSD | TBs | < 5 ms | High | Rate limit counters |
| **KVCacheStore** | 4 | GPU HBM → CPU RAM | HBM + RAM | 10s–100s GB per node | μs (HBM) – ms (RAM) | Highest (HBM) | Per-decode-step reads |
| **SafetyLog** | 4 | Log aggregation (ES/OS/ClickHouse) | SSD | PBs | Eventual consistency OK | Medium | Audit + analysis |
| **FeedbackStore** | 5 | Kafka cluster | SSD | 10s of TB hot | High write rate | Medium | Event log |
| **Warehouse** | 5 | Columnar analytics DB | SSD (compressed) | 10+ PB | Analytical queries | Medium | Long-horizon analytics |
| **MetricsStore** | 2, 4, 5 | Time-series DB | SSD | TBs | High write, moderate read | Low-medium | Metrics, traces, logs |
| **EmbeddingStore** | 5 | Vector DB | SSD | 100s of TB | Similarity search | Medium | Failure pattern clustering |

¹ Relative cost tier, not absolute dollars. Exact $ figures depend on org, region, contracts, and change frequently.

## 6.2 Consolidated compute cluster reference

| Cluster | Stage(s) | Hardware | Scale (typical) | Workload | Notes |
|---|---|---|---|---|---|
| **Crawler fleet** | 1 | CPU, fast NICs | 100s–1000s nodes | IO-bound | No CPU/GPU heavy work |
| **Extract fleet** | 1 | CPU, high RAM | 100s–1000s nodes | CPU-bound parsing | Includes OCR subset |
| **Dedup cluster** | 1 | CPU, high RAM, local SSD | 100s nodes | Spark/Ray shuffle | No GPU |
| **Quality classifier fleet** | 1 | Small GPU (L4/A10/T4) | 10s–100s nodes | Small-model inference | Not A100s |
| **Tokenization fleet** | 1 | CPU | 100s nodes | Embarrassingly parallel | |
| **Training cluster** | 2 | H100 80GB / H200 141GB / B200 192GB | 10K–100K GPUs | bf16/fp8 dense + comm | Frontier-class supercomputer |
| **Eval cluster (training-time)** | 2 | H100 | 100s–1000s GPUs | Mid-training checkpoint eval | Separate to avoid stealing capacity |
| **Coordination plane (training)** | 2 | CPU | 10s nodes | Scheduler, monitoring | Kubernetes/Slurm |
| **Post-train cluster** | 3 | H100/H200 | 512–2048 GPUs | SFT, DPO/PPO, safety, capability | Smaller than pretraining |
| **Inference for data gen** | 3 | H100 | 100s GPUs | Synthetic data, response sampling | |
| **Critic / judge cluster** | 3 | H100 | 64–256 GPUs | RLAIF judge inference | |
| **Sandbox farm** | 3 | CPU, isolated (gVisor/Firecracker) | 100–500 nodes | Code execution, math verification | |
| **Post-train eval cluster** | 3 | H100 | 100s GPUs | Continuous eval | |
| **EdgeGateway** | 4 | CPU, fast NICs | 100s nodes across 30–100 PoPs | TLS, routing, rate limit | Global |
| **AppCluster** | 4 | CPU, mid-spec | 500–5000 nodes per region | Orchestration | Stateless |
| **Inference cluster (prefill)** | 4 | H100/H200/B200 | 1000s GPUs per region | Compute-bound prefill | Disaggregated from decode |
| **Inference cluster (decode)** | 4 | H100/H200/B200 (HBM-bandwidth optimized) | 1000s GPUs per region | Memory-bound decode | Disaggregated from prefill |
| **Classifier pool (inference)** | 4 | A10/L4/T4 | 100s nodes | Input/output safety classifiers | Small models |
| **Observability cluster** | 4, 5 | CPU, big storage | 100s nodes | Logs, metrics, traces | |
| **Analytics cluster** | 5 | CPU (Spark/Ray) | 1000+ nodes | Batch aggregations | |
| **Embedding cluster** | 5 | Small/mid GPU (L4/A10) | 100s nodes | Embed prompts for clustering | |
| **Review tooling backend** | 5 | CPU | 10s nodes | Annotation UI, queue | |

## 6.3 Where the money goes (rough breakdown)

This is the cost shape of a frontier LLM operation, in order of magnitude. Absolute dollars omitted because they change quickly; relative proportions are more stable.

| Category | Approximate share of total infra spend (*typical, order of magnitude*) | Notes |
|---|---|---|
| Inference GPU fleet | ~50–70% | Largest by far in mature companies; grows with usage |
| Training GPU clusters (amortized) | ~15–25% | Concentrated bursts during model training |
| GPU power and cooling | ~5–10% | Liquid cooling at H200/B200 generation |
| Storage (all tiers combined) | ~3–8% | GoldDataset's parallel FS is the most expensive per GB |
| Networking (InfiniBand, edge, transit) | ~2–5% | Mostly the InfiniBand fabric on training clusters |
| CPU clusters (data prep, app, analytics) | ~2–5% | Cheap relative to GPUs |
| Human infrastructure (annotators, reviewers) | ~5–15% | Significant but smaller than GPUs |

**Headline**: GPUs dominate. ~70–90% of infrastructure cost is GPU hardware + power. Everything else is small relative to that.

## 6.4 Critical performance numbers (publicly sourced where possible)

| Number | Value | Source |
|---|---|---|
| H100 SXM HBM capacity | 80 GB | Public (NVIDIA H100 datasheet) |
| H100 SXM HBM bandwidth | 3.35 TB/s | Public (NVIDIA H100 datasheet) |
| H100 SXM FP16 / BF16 peak | 989 TFLOPS (with sparsity, half without) | Public (NVIDIA H100 datasheet) |
| H100 SXM FP8 peak | 1979 TFLOPS (with sparsity, half without) | Public (NVIDIA H100 datasheet) |
| H100 SXM power | 700 W | Public (NVIDIA H100 datasheet) |
| H200 SXM HBM capacity | 141 GB | Public (NVIDIA H200 datasheet) |
| H200 SXM HBM bandwidth | 4.8 TB/s | Public (NVIDIA H200 datasheet) |
| NVLink per GPU (H100) | 900 GB/s | Public (NVIDIA) |
| InfiniBand NDR per port | 400 Gb/s | Public (NVIDIA Quantum-2) |
| Practical MFU for frontier pretraining | 40–55% | Public (various papers: Megatron-LM, Llama, etc.) |
| Practical MFU for inference | 30–50% | Typical |
| Kaplan scaling law per-token FLOPs (dense) | ≈ 6N | Public (Kaplan 2020) |
| CommonCrawl monthly volume | ~3–4B pages, ~400 TB WARC | Public (commoncrawl.org) |

## 6.5 Glossary

Sorted alphabetically. Terms used in this document.

- **All-reduce**: collective communication operation where each rank's local value is combined (typically summed) and the result distributed back to all ranks. Used to combine gradients across data-parallel replicas.
- **AppCluster**: orchestration layer between edge and inference; prepares the model request.
- **B200**: NVIDIA Blackwell-generation datacenter GPU (2024+); 192 GB HBM3e.
- **Batching (continuous)**: technique where requests join and leave the in-flight batch at each decode step, keeping GPU utilization high. Used in vLLM, TRT-LLM, SGLang.
- **bf16**: brain float 16 — 16-bit floating point format with 8-bit exponent, 7-bit mantissa. Standard training precision.
- **BPE**: Byte Pair Encoding — tokenizer training algorithm.
- **CAI**: Constitutional AI — Anthropic's technique for using an AI critic + revision loop, then training on the revisions. Specific implementation of RLAIF.
- **CheckpointStore**: parallel filesystem for training checkpoints, separated from GoldDataset to avoid IO contention.
- **ConfigStore**: in-memory replicated KV store for system prompts and configs.
- **ConvoStore**: distributed database for user conversation history.
- **CP (Context Parallelism)**: parallelism over the sequence dimension; needed for long-context training.
- **DP (Data Parallelism)**: parallelism where each replica processes different microbatches; gradients all-reduced.
- **DPO**: Direct Preference Optimization — preference learning without an explicit reward model. Trains policy directly on (chosen, rejected) pairs.
- **EP (Expert Parallelism)**: parallelism over MoE experts; tokens routed to a subset of experts.
- **FSDP**: Fully Sharded Data Parallel — PyTorch's native sharded parallelism. Shards weights, gradients, optimizer state across data-parallel ranks.
- **FeedbackStore**: append-only event log for user feedback signals.
- **fp8**: 8-bit floating point format. H100+ hardware support; ~2× throughput vs bf16.
- **GoldDataset**: tokenized, sharded, training-ready corpus on parallel filesystem.
- **H100 / H200**: NVIDIA Hopper-generation datacenter GPUs (2022 / 2023).
- **HBM**: High Bandwidth Memory — stacked DRAM on the GPU package. Where weights and activations live.
- **HumanData**: encrypted storage for human-curated training data (SFT, preferences, safety).
- **InfiniBand**: high-bandwidth, low-latency interconnect used for GPU-to-GPU communication across nodes.
- **KL divergence**: measure of distance between two probability distributions. Used as a penalty term in RLHF to keep policy close to reference.
- **KV cache**: stored key and value tensors from past tokens, reused at each decode step to avoid recomputation.
- **KVCacheStore**: in-HBM cache of KV tensors, with CPU RAM as spillover.
- **LSH**: Locality-Sensitive Hashing — used with MinHash for near-deduplication.
- **MFU**: Model FLOPs Utilization — practical FLOPs achieved divided by hardware theoretical peak. Frontier training: 40–55%.
- **MinHash**: signature-based document similarity estimation; combined with LSH for scalable dedup.
- **MoE**: Mixture of Experts — architecture where each token is routed to a subset of "experts" (sub-networks). Cheaper inference per active parameter.
- **ModelVault**: canonical storage for trained model weights.
- **NCCL**: NVIDIA Collective Communications Library — implements all-reduce, all-gather, etc. on GPU.
- **NVLink**: NVIDIA's high-bandwidth intra-node GPU-to-GPU interconnect (900 GB/s per GPU on H100).
- **OCR**: Optical Character Recognition — extracting text from scanned images / PDFs.
- **PagedAttention**: vLLM's KV cache management technique; splits cache into fixed blocks like OS pages.
- **PII**: Personally Identifiable Information.
- **PP (Pipeline Parallelism)**: parallelism that splits the model by layer groups across nodes; point-to-point sends between stages.
- **PPO**: Proximal Policy Optimization — RL algorithm used for RLHF. Requires policy + reference + reward + value models.
- **Prefill**: inference phase that processes the prompt and produces the first token + initial KV cache.
- **Decode**: inference phase that generates tokens one at a time using cached KV from previous tokens.
- **RawLake**: archive storage for raw scraped bytes.
- **RLAIF**: Reinforcement Learning from AI Feedback — uses an AI critic instead of (or in addition to) human raters. CAI is one implementation.
- **RLHF**: Reinforcement Learning from Human Feedback — preference learning using human-ranked response pairs.
- **safetensors**: safe serialization format for tensors; industry-standard for distributing model weights.
- **SFT**: Supervised Fine-Tuning — training on instruction-following examples; first stage of post-training.
- **SP (Sequence Parallelism)**: parallelism over the sequence dimension; sometimes used interchangeably with CP, sometimes distinct.
- **Speculative decoding**: technique where a small draft model proposes tokens that the main model verifies in a single forward pass. 1.5–3× speedup typical.
- **SSE**: Server-Sent Events — HTTP-based streaming protocol used for response streaming.
- **TextLake**: cleaned, extracted text from RawLake.
- **TP (Tensor Parallelism)**: parallelism that splits individual layers across GPUs within a node; requires NVLink-class bandwidth.
- **TPOT**: Time Per Output Token — average time between subsequent tokens during generation.
- **TTFT**: Time To First Token — latency from request received to first token visible to user.
- **Warehouse**: columnar analytics database for long-horizon feedback analysis.
- **WSD**: Warmup-Stable-Decay learning rate schedule.
- **XID**: NVIDIA driver fault code; presence often indicates hardware failure.
