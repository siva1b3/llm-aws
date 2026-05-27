# Prior Art for Multi-Agent LLM Conversation Systems with Compartmentalized Memory, Theory of Mind, and Common-Ground Promotion

## Executive Summary — The "Lay of the Land"

The architecture you describe — **two LLM agents conversing turn-by-turn, with separated self / theory-of-mind / common-ground memory compartments, an immutable identity anchor, an append-only first-person turn log, a per-turn generate→comprehend→update cognitive cycle, explicit promotion logic for shared beliefs, and asymmetric (receive-only) memory updates** — sits at the intersection of four distinct research streams that have never been fully unified in a single published artifact:

1. **Generative-agent / cognitive-architecture work** (Park et al., DeepMind Concordia, CoALA) — provides observation/reflection/planning loops and modular memory but typically uses *one* memory stream, not separated self/ToM/common-ground compartments.
2. **Theory-of-mind dialogue research** (MindDial, ToM-agent, SymbolicToM, Hypothetical Minds, MindCraft) — explicitly tracks first- and second-order beliefs and even has "common ground" alignment, but is usually evaluated on benchmark Q&A or a single-task game, not on continuous open-domain conversation between two persistent agents.
3. **Stateful agent-memory infrastructure** (MemGPT/Letta, Zep/Graphiti, Mem0, A-MEM) — provides immutable persona blocks, episodic-vs-semantic tiers, and append-only event stores, but is built for a single agent talking to a user, not for two agents talking to each other with mutually private ToM compartments.
4. **Multi-agent orchestration frameworks** (AutoGen, CAMEL, AgentVerse, CrewAI, MetaGPT, ChatDev, AI Town, MiroFish/OASIS) — produce multi-LLM dialogue and even social simulations at scale, but they almost universally share a single chat history and lack a formal compartmentalized belief model.

**The most direct prior art is `MindDial` (Qiu et al., SIGDIAL 2024) and `ToM-agent` (Yang et al., 2025), both of which already implement a "self belief / belief-about-listener / belief-gap → common-ground" three-tier mind module.** The most architecturally adjacent open-source agent stack is **Letta (formerly MemGPT)**, whose core/persona/human "memory blocks" map cleanly onto your self / ToM / common-ground compartmentalization. The closest commercial product in spirit is **MiroFish (and its OASIS engine from CAMEL-AI)**, which is a *swarm* of generative agents with persistent per-agent memory built on Zep's temporal knowledge graph — but MiroFish is about thousands of agents posting to a simulated social platform, not two agents conducting a deep first-person dialogue with formal mutual-belief tracking.

**The unique gap your POC could fill** is the integration: nobody has shipped a clean, two-agent, dialogue-first system that combines (a) an immutable identity anchor, (b) three explicitly separated memory compartments per agent (private self, private ToM-of-other, shared common ground), (c) a strict per-turn generate→comprehend→update cycle, (d) asymmetric updates (only on receive), and (e) explicit promotion rules from private belief to common ground. Each of these features exists in isolation in the literature; their combination, especially in an open-domain conversation rather than a task-game, is open territory.

---

## 1. Closest Matches — Systems Implementing Most of the Features

### 1.1 MindDial (Qiu, Liu, Li, Zhu, Zheng — SIGDIAL 2024)
- **Built by:** UCLA / BIGAI (Song-Chun Zhu's group), with Zilong Zheng. Published at SIGDIAL 2024 ([ACL Anthology](https://aclanthology.org/2024.sigdial-1.63/)).
- **Year / Status:** 2023 preprint, SIGDIAL 2024 final; research only ([arXiv 2306.15253](https://arxiv.org/abs/2306.15253)).
- **What it does:** A conversational framework with an explicit "mind module" that tracks **three levels of belief**: (1) the speaker's own belief, (2) the speaker's prediction of the listener's belief, and (3) **the common belief based on the gap between the first two**. Responses are generated explicitly to "resolve the belief difference" — i.e., to *promote* private beliefs to common ground ([ACL Anthology](https://aclanthology.org/2024.sigdial-1.63/)).
- **Match to your architecture:** Very high on (2) separated self/ToM/common-ground compartments and (6) promotion logic. Lower on (3) immutable anchor, (4) append-only turn-tagged log, and (5) the explicit generate/comprehend/update split — these are not centerpieces of the paper. Tested on MutualFriend (alignment) and CaSiNo (negotiation), so it is task-grounded rather than open-ended. Two agents conversing turn-by-turn is the canonical setup.
- **Strengths:** First clean paper that operationalizes Clark-and-Brennan common ground inside an LLM dialogue agent with mind-module ToM. Ablations confirm the three-level design improves task outcome.
- **Limitations:** Limited to scenarios with well-defined knowledge bases/goals; the authors explicitly say generalization to "more casual conversation scenarios" is open ([ACL Anthology](https://aclanthology.org/2024.sigdial-1.63/)).

### 1.2 ToM-agent (Yang et al., 2025)
- **Built by:** Bo Yang and colleagues. ([arXiv 2501.15355](https://arxiv.org/abs/2501.15355), OpenReview ICLR submission).
- **Year / Status:** January 2025 preprint; research only.
- **What it does:** Two LLM agents (Alice, Bob) hold a multi-turn dialogue. Each has a **Self-BDI module** (own Beliefs/Desires/Intentions) plus a **Counterpart BDI Tracking module** (private inferred BDIs about the other agent, with confidence values). It includes **first-order ToM and second-order ToM** ("Alice ponders whether Bob has understood her") and a **counterfactual-reflection loop** that compares predicted vs. actual responses to update beliefs about the partner ([arXiv html](https://arxiv.org/html/2501.15355v1)).
- **Match to your architecture:** Probably the single closest match overall. Hits (1) two-agent conversation, (2) separated self vs. theory-of-mind compartments, (5) per-turn generate/comprehend/update with an explicit reflection step, and (7) asymmetric updates of belief about the other — the BDI tracker updates after observing the counterpart's utterance.
- **Gap vs. your design:** Does **not** have an explicit shared "common ground" compartment with promotion rules — it is bilateral private modeling rather than tri-partite. Also no explicit immutable-anchor concept; the persona is part of self-BDI which is mutable.
- **Datasets:** Empathetic and persuasion dialogue corpora.
- **Strengths/Limitations:** Disentangles confidence from mental states; counterfactual reflection is novel. Limited evaluation rigor; small-scale.

### 1.3 Hypothetical Minds (Cross, Xiang, Bhatia, Yamins, Haber — Stanford, 2024)
- **Built by:** Stanford (Daniel Yamins, Nick Haber labs). [arXiv 2407.07086](https://arxiv.org/abs/2407.07086), [Project page](https://locross93.github.io/HM-Website/), [GitHub](https://github.com/locross93/Hypothetical-Minds).
- **Year / Status:** July 2024; active research.
- **What it does:** A modular cognitive architecture with **perception, memory, hierarchical planning, and a Theory-of-Mind module** that *generates, evaluates, and refines hypotheses* in natural language about other agents' strategies. Hypotheses get reinforced when they predict the other's behavior correctly. Evaluated on Melting Pot (DeepMind's multi-agent benchmark).
- **Match to your architecture:** Strong on (2)-ToM compartment, (5) cognitive cycle, (6) refinement = essentially promotion logic, and (7) asymmetric updates (the agent updates only when it sees the opponent's actions). Unique value: each hypothesis carries a score that gets reinforced — closest thing to your "confidence" / promotion threshold.
- **Gap:** Designed for embodied, action-oriented multi-agent games, not turn-by-turn linguistic dialogue. No shared common-ground compartment. The "self" persona is implicit.
- **Strengths:** Cognitively motivated, ablations are clean, code is open-source.
- **Limitations:** RL-style game environments rather than open conversation.

### 1.4 SymbolicToM (Sclar, Kumar, West, Suhr, Choi, Tsvetkov — ACL 2023)
- **Built by:** UW/CMU. ACL 2023 Outstanding Paper. [arXiv 2306.00924](https://arxiv.org/abs/2306.00924), [GitHub](https://github.com/msclar/symbolictom).
- **What it does:** A plug-and-play, decoding-time algorithm that maintains a graph **per character** representing each character's beliefs **and** their estimation of other characters' beliefs ($B_{Bob}$ and $B_{Bob,Alice}$), and higher-order recursion ([arXiv abs](https://arxiv.org/abs/2306.00924)).
- **Match:** Excellent on (2)-ToM compartmentalization at multiple orders of recursion; explicit symbolic structure; not a conversation system but a *reasoning-over-narrative* system. Doesn't model self/common-ground/promotion explicitly.
- **Use as building block:** The graph representation it uses for belief tracking is a viable template for the ToM compartment in your POC.

### 1.5 Theory of Mind for Multi-Agent Collaboration (Li et al., EMNLP 2023)
- **Built by:** Huao Li and collaborators. [arXiv 2310.10701](https://arxiv.org/abs/2310.10701).
- **What it does:** LLM agents in a cooperative text game maintain **explicit textual belief states** that are updated each turn from observations and communications, and used for ToM inference about teammates.
- **Match:** Demonstrates the value of an explicit belief-update prompt step (your "comprehension/memory-update") and surfaces the same problem you're addressing — that LLMs in multi-agent dialogue rapidly disseminate misinformation when belief tracking is ad-hoc. Confirms higher-order ToM is hard for LLMs unless explicitly scaffolded.

---

## 2. Partial Matches — Academic Papers with Some of the Features

### 2.1 Generative Agents (Park, O'Brien, Cai, Morris, Liang, Bernstein — Stanford/Google, UIST 2023)
- The foundational paper ([arXiv 2304.03442](https://arxiv.org/abs/2304.03442); 25-agent Smallville). Architecture: a **memory stream** (append-only natural-language record of experience, time-tagged), **reflection** (periodic synthesis of higher-level beliefs), and **planning** ([Stanford HAI](https://hai.stanford.edu/news/computational-agents-exhibit-believable-humanlike-behavior)).
- **Match:** Hits (4) append-only log with timestamps and (6) higher-level belief synthesis (reflection ≈ promotion), and the agents do converse turn-by-turn. The architecture has become "the standard for sandboxed agent simulation" ([ResearchGate review](https://www.researchgate.net/publication/375063078_Generative_Agents_Interactive_Simulacra_of_Human_Behavior)).
- **Gap:** Single unified memory stream — **no separation between self / ToM-of-others / common ground.** Each agent has private memories but does not maintain a structured model of "what the other agent privately believes about me" or "what we both know we know." This is exactly the gap your design targets.
- **Follow-up:** Park et al. 2024 ["Generative Agent Simulations of 1,000 People"](https://arxiv.org/abs/2411.10109) extends this to digital twins of real interview participants, replicating GSS responses at 85% of human test-retest consistency. Code at [StanfordHCI/genagents](https://github.com/joonspk-research/genagents).

### 2.2 MindCraft (Bara, CH-Wang, Chai — EMNLP 2021 Outstanding Paper)
- [arXiv 2109.06275](https://arxiv.org/abs/2109.06275); [GitHub](https://github.com/sled-group/MindCraft).
- A Minecraft dataset of human pairs collaborating, with **explicit annotations of each partner's beliefs about the world and about each other as the interaction unfolds** — exactly the data needed to train/evaluate the kind of architecture you describe. The paper argues that "theory of mind plays an important role in maintaining common ground during human collaboration."
- **Use:** Not an architecture but the canonical *dataset* and *task formulation* you'd want to test against.

### 2.3 FANToM (Kim, Sclar, Zhou, Le Bras, G. Kim, Choi, Sap — EMNLP 2023)
- [arXiv 2310.15421](https://arxiv.org/abs/2310.15421); [project page](https://hyunw.kim/fantom/).
- **Information-asymmetric conversation** benchmark: characters leave/rejoin a conversation, creating natural common-ground/private-knowledge splits. Tests whether models track *who knows what*.
- All SOTA LLMs perform "significantly worse than humans" even with chain-of-thought ([Allen AI blog](https://allenai.org/blog/does-gpt-4-have-theory-of-mind-capabilities-cd2d766e51a7)). **This is the most natural evaluation suite for your POC.**

### 2.4 Hi-ToM (Wu, He, Jia, Mihalcea, Chen, Deng — EMNLP Findings 2023)
- [arXiv 2310.16755](https://arxiv.org/abs/2310.16755). Higher-order (up to 4th-order) ToM benchmark; LLM performance degrades rapidly with order. Useful for evaluating whether your separated ToM compartment actually buys you higher-order reasoning.

### 2.5 NegotiationToM, OpenToM, ToMi, BigToM, ToMBench
- [NegotiationToM](https://arxiv.org/html/2404.13627) tests BDI tracking in real CaSiNo negotiations. ToMi (Le et al. 2019) is the original Sally-Anne LLM benchmark used in SymbolicToM. Together these define the evaluation surface.

### 2.6 CoALA — Cognitive Architectures for Language Agents (Sumers, Yao, Narasimhan, Griffiths — Princeton, 2023)
- [arXiv 2309.02427](https://arxiv.org/abs/2309.02427).
- Not an implementation but the **conceptual framework** that organizes LLM agents into modular memory (working / episodic / semantic / procedural), structured action space, and a decision cycle — explicitly inspired by Soar and ACT-R ([Emergent Mind summary](https://www.emergentmind.com/papers/2309.02427)).
- **Use:** The reference framework you should cite when describing your separated compartments and per-turn cycle.

### 2.7 SOTOPIA (Zhou, Zhu, et al. — CMU/Allen, ICLR 2024)
- [arXiv 2310.11667](https://arxiv.org/abs/2310.11667); [Sotopia.world](https://sotopia.world/).
- Open-ended environment for two LLM agents to role-play social interactions with **private goals** (a critical feature that maps to your "self" compartment with hidden information) and a holistic evaluation framework. Sotopia-π (training extension) gets Mistral-7B to GPT-4-level social ability. Very useful both as evaluation environment and for the "agent-vs-agent with private goals" pattern.

### 2.8 SimToM (Wilf, Lee, Liang, Morency — CMU, ACL 2024)
- [arXiv 2311.10227](https://arxiv.org/abs/2311.10227). Two-stage prompting — perspective-take, then answer — to enforce ToM. A lightweight version of your generate/comprehend split. Decompose-ToM (Sarangi et al. 2025, [arXiv 2501.09056](https://arxiv.org/html/2501.09056)) generalizes this with recursive perspective simulation.

### 2.9 Reflexion (Shinn, Cassano, Berman, Gopinath, Narasimhan, Yao — NeurIPS 2023)
- [arXiv 2303.11366](https://arxiv.org/abs/2303.11366); [GitHub](https://github.com/noahshinn/reflexion). Verbal self-reflection feedback stored in episodic memory between trials. Inspires the "memory update" step but is single-agent and trial-based, not turn-based.

### 2.10 Multi-Agent Debate (Du, Li, Torralba, Tenenbaum, Mordatch — ICML 2024)
- [Project page](https://composable-models.github.io/llm_debate/). Multiple LLM instances debating to consensus. Foundational for "multiple LLMs in conversation," but each instance shares the full debate history; no compartmentalized private memory. Liang et al. ([EMNLP 2024](https://aclanthology.org/2024.emnlp-main.992/)) extend with a "tit-for-tat" judge.

### 2.11 EM-LLM and Human-Like Memory Architectures
- [EM-LLM (arXiv 2407.09450)](https://arxiv.org/html/2407.09450v1) brings episodic-memory event segmentation (Bayesian surprise + graph refinement) to LLMs.
- [A-MEM (Xu et al. 2025)](https://arxiv.org/abs/2502.12110) builds a Zettelkasten-style note network with auto-linking. Both are infrastructure pieces for a "human-like" memory module.

### 2.12 Adaptive Theory of Mind (A-ToM, 2026)
- [arXiv 2603.16264](https://arxiv.org/html/2603.16264). Hypothetical agents at multiple ToM orders run in parallel; performance is selected via online expert-advice algorithms. Useful for the "asymmetric update" + confidence weighting story.

### 2.13 Belief-Aligned Conversational Agents (Feb 2025)
- [ResearchGate](https://www.researchgate.net/publication/389176615_Enhancing_Conversational_Agents_with_Theory_of_Mind_Aligning_Beliefs_Desires_and_Intentions_for_Human-Like_Interaction). Explicit BDI alignment in LLaMA-3 yields ~67% win-rate over baseline. Confirms the practical value of explicitly maintained ToM components in dialogue.

### 2.14 DToM-Track / Dynamic ToM as Memory
- [arXiv 2603.14646](https://arxiv.org/pdf/2603.14646). Frames ToM as a *temporally extended representational memory* problem; finds LLMs reliably infer current belief but fail to retrieve prior belief states once updated. Directly motivates your "append-only" log instead of overwriting.

---

## 3. Commercial Products and Startups in This Space

### 3.1 MiroFish (Guo Hangjiang, 2026) — and its engine OASIS / CAMEL-AI
- **What it actually is:** Confirmed via [GitHub 666ghj/MiroFish](https://github.com/666ghj/MiroFish), [Medium write-up by Agent Native](https://agentnativedev.medium.com/mirofish-swarm-intelligence-with-1m-agents-that-can-predict-everything-114296323663), and [DEV community](https://dev.to/arshtechpro/mirofish-the-open-source-ai-engine-that-builds-digital-worlds-to-predict-the-future-ki8). MiroFish is a **swarm-prediction engine**: it ingests a seed document (news article, policy draft, novel), extracts a knowledge graph (Neo4j), spawns hundreds-to-thousands of generative agents with personas, and simulates them posting/replying on a synthetic Twitter+Reddit to forecast public-opinion or financial outcomes. Built by a Chinese undergraduate; topped GitHub trending in March 2026; received a $4.1M seed from Chen Tianqiao. An English/offline fork [nikmcfly/MiroFish-Offline](https://github.com/nikmcfly/MiroFish-Offline) and CLI fork [amadad/mirofish-cli](https://github.com/amadad/mirofish-cli) exist.
- **Engine:** Powered by **OASIS (Open Agent Social Interaction Simulations)** from CAMEL-AI, scaling to ~1M agents.
- **Memory architecture:** Per-agent persistent memory via **Zep Cloud** knowledge graphs (the original) or local Neo4j (offline fork) — short-term recent rounds + long-term auto-summarized facts/relationships ([Medium architecture article](https://medium.com/@balajibal/mirofish-multi-agent-swarm-intelligence-for-predictive-simulation-09771e60b188)).
- **Match to your architecture:** Far on the *swarm/social-simulation* side; not built around a single deep two-agent dialogue. Has persistent identity, memory, and emergent common-ground via shared platform, but no explicit "private ToM about specific other agents" compartment or formal promotion rules. Importantly, MiroFish's posts are *broadcast*, not bilateral — it lacks the dyadic, turn-by-turn, first-person framing your design specifies.
- **What you can take from it:** Knowledge-graph-backed personas, Zep's temporal memory pattern, OASIS's scale primitives.

### 3.2 Inworld AI (NPCs)
- **Character Brain / Contextual Mesh** architecture for game NPCs, with personality models, emotions engine, **autonomous goals**, **long-term memory** that synthesizes/de-duplicates information across sessions ([Inworld blog](https://inworld.ai/blog/introducing-long-term-memory)), and a **fourth-wall** distinction between **personal knowledge and common knowledge** in the Contextual Mesh ([Inworld blog](https://inworld.ai/blog/ai-npcs-and-the-future-of-video-games), [Lightspeed write-up](https://lsvp.com/stories/inworld-ai-npcs-character-engine/)). The "personal vs common knowledge" split is a near-direct commercial cousin of your "self" vs "common ground" compartments — though framed for player-NPC interaction, not NPC-NPC.
- **Status:** Active commercial product, used in shipped games. Memory is real but proprietary; no public API for the specific data structures.

### 3.3 Character.AI
- Strong on persona library, weaker on architectural sophistication. Relies on a single proprietary model with limited memory; "Chat Memories" lets users pin messages as a workaround ([Skywork.ai 2026 guide](https://skywork.ai/skypage/en/character-ai-guide-persona-roleplay/2029467207405101056)). No published evidence of compartmentalized self/ToM/common-ground memory.

### 3.4 Replika
- Single ongoing companion per user; "Memory bank" + "Diary" features ([eesel.ai overview](https://www.eesel.ai/blog/replika-ai)). Architecturally simpler — there is no second AI agent and no formal common-ground tracking, but persistent identity over years is a working precedent.

### 3.5 Project December (Jason Rohrer, 2020–)
- GPT-3-based bespoke chatbots, famous for the "Jessica simulation" ([San Francisco Chronicle / Nieman Storyboard](https://niemanstoryboard.org/2021/08/31/jason-fagone-follows-the-creation-life-and-different-life-of-a-chatbot-romance/)). Architecturally simple (prompt + back-and-forth), but pioneered the *single-agent persistent persona* paradigm.

### 3.6 Fable Studio — "Virtual Beings" (Lucy from *Wolves in the Walls*)
- Emmy-winning interactive characters that combine human-written context with GPT-3 completions ([Fable blog](https://www.fable-studio.com/behind-the-scenes/ai-generation), [Voicebot.ai](https://voicebot.ai/2020/12/28/fable-studio-introduces-virtual-beings-ready-to-converse-with-you/)). Lucy "remembers your interactions and caters next week's content" via Instagram. Fable has since pivoted to Showrunner (the "Netflix of AI") for AI-generated TV episodes. Demonstrates the *persistent character with memory across user contacts* pattern at production quality.

### 3.7 Letta (formerly MemGPT) — most architecturally relevant commercial stack
- [Letta.com](https://www.letta.com), [docs](https://docs.letta.com/concepts/memgpt/), [GitHub](https://github.com/letta-ai/letta). The MemGPT paper (Packer et al.) introduced the OS-style memory hierarchy: **core memory blocks** (`persona`, `human`) that are *always in context* and treated as immutable identity, **archival memory** for long-term storage, and tools for the agent to self-edit memory ([Letta blog: Agent Memory](https://www.letta.com/blog/agent-memory)).
- **Match to your architecture:** Highest of any commercial product on (3) **immutable anchor** (core blocks function exactly as your "anchor that never gets summarized away"), (4) **structured persistent state**, and the natural extension to multiple labeled memory blocks per agent maps directly to your self / ToM / common-ground tri-partition. Letta also supports multi-agent shared-memory patterns.
- **Limitation:** Built for an agent talking to a user, not two agents talking to each other with separated mutual-belief compartments. Your design is essentially "Letta with a third memory block per peer + promotion logic + bilateral dialogue loop."

### 3.8 Zep / Graphiti (memory infrastructure)
- [getzep.com](https://www.getzep.com), [arXiv 2501.13956](https://arxiv.org/abs/2501.13956). Temporally-aware knowledge graph for agent memory; bi-temporal edges with `valid_from / valid_to / invalid_at`. Outperforms MemGPT on Deep Memory Retrieval benchmark (94.8% vs 93.4%). Used as the storage backend in MiroFish.

### 3.9 Mem0 (mem0.ai)
- [GitHub](https://github.com/mem0ai/mem0). Vector-first agent memory with optional graph layer; the dominant lightweight alternative to Zep. Mem0's add-only extraction (no UPDATE/DELETE) maps onto your "append-only" rule.

### 3.10 Other digital-twin / multi-agent simulation startups
- **Concordia (Google DeepMind, open-source library)** — [GitHub](https://github.com/google-deepmind/concordia), [arXiv 2312.03664](https://arxiv.org/abs/2312.03664). Tabletop-RPG-style framework where a Game-Master entity arbitrates physically-grounded interactions among agents. Strong "components" architecture (memory, reasoning chains as separate modules) that could be configured into your tri-partite memory split. v2.0 (2025) emphasizes a unified Entity-Component pattern.
- **Smallville reference implementation** — [joonspk-research/generative_agents](https://github.com/joonspk-research/generative_agents) — original Stanford code.
- Multi-agent policy/business prediction tools beyond MiroFish (e.g., Chopra et al.'s 8.6M-agent COVID simulation cited in [arXiv 2503.09639](https://arxiv.org/html/2503.09639v2)) exist but treat agents as decision nodes, not conversational personalities.

---

## 4. Open-Source Frameworks That Could Be Assembled Into This

| Framework | Where it helps | Where it falls short |
|---|---|---|
| **AI Town** ([a16z-infra/ai-town](https://github.com/a16z-infra/ai-town)) | TypeScript/Convex implementation of generative agents with shared global state, transactions, and a simulation engine; usable as a substrate for two persistent agents with persistent memory. | Single memory stream per agent; no compartmentalization; no formal ToM. |
| **AgentVerse** ([OpenBMB/AgentVerse](https://github.com/OpenBMB/AgentVerse), [arXiv 2308.10848](https://arxiv.org/abs/2308.10848)) | ICLR 2024; supports both task-solving and simulation modes; modular for adding belief modules. | No built-in private/shared memory compartments. |
| **AutoGen / Microsoft Agent Framework** ([microsoft/autogen](https://github.com/microsoft/autogen)) | Mature multi-agent conversation primitives; AssistantAgent has a `memory` slot; group-chat moderator. AutoGen is now in maintenance mode in favor of Microsoft Agent Framework. | All agents typically share the same chat history; no first-class ToM compartment. |
| **CAMEL** ([camel-ai/camel](https://github.com/camel-ai/camel), [arXiv 2303.17760](https://arxiv.org/abs/2303.17760)) | Inception-prompted role-playing between *exactly two* agents (your turn-by-turn pattern); explicit "AI Society" / "Mind exploration" framing. | Symmetric prompts only; no ToM module, no common-ground tracking. |
| **CrewAI** ([crewaiinc/crewai](https://github.com/crewaiinc/crewai)) | Role-based teams; high adoption; integrates Mem0. | Task-pipeline orientation; not optimized for two-agent deep dialogue. |
| **LangGraph** | Stateful graph-based control flow with checkpointers — good substrate for the per-turn cycle and append-only state ([LangChain memory walkthrough](https://dev.to/sreeni5018/five-agent-memory-types-in-langgraph-a-deep-code-walkthrough-part-2-17kb)). | You build the compartmentalization yourself. |
| **Letta** ([letta-ai/letta](https://github.com/letta-ai/letta)) | As above — closest production-grade fit for the immutable-anchor + structured-blocks pattern. | Single-agent default; multi-agent shared memory is newer. |
| **MetaGPT / ChatDev** | Structured multi-agent SOPs. | Software-engineering orientation; fixed roles. |
| **Concordia** | Component-based; clean separation of perception/memory/reasoning; v2.0 makes this serializable. | Game-master intermediation rather than direct dialogue. |
| **Hypothetical Minds code** ([locross93/Hypothetical-Minds](https://github.com/locross93/Hypothetical-Minds)) | Reference implementation of a hypothesis-tracking ToM module you could lift directly. | Coupled to Melting Pot environments. |
| **SymbolicToM** ([msclar/symbolictom](https://github.com/msclar/symbolictom)) | Reference implementation of symbolic per-character belief graphs. | Reading-comprehension oriented. |
| **MindCraft** ([sled-group/MindCraft](https://github.com/sled-group/MindCraft)) | Dataset + code for ToM-annotated dialogue. | Minecraft-bound. |
| **OASIS** (CAMEL-AI) | Up to 1M agents, persistent memory, scalable LLM call distribution. | Social-network simulation, not deep dyadic dialogue. |

A practical assembly recipe: **Letta** (for immutable persona + persistent compartments) + **LangGraph** (for the per-turn generate→comprehend→update state machine) + a **SymbolicToM-style belief graph** (for the ToM compartment) + a custom **promotion/grounding module** (your novel piece) + **FANToM / SOTOPIA / MindCraft** as evaluation harnesses.

---

## 5. Foundational Research This Builds On

### 5.1 Linguistics / Pragmatics / Cognitive Science
- **Clark & Brennan (1991), "Grounding in communication"** — the canonical theory of common ground as a mutual-knowledge structure built collaboratively turn-by-turn ([Wikipedia](https://en.wikipedia.org/wiki/Grounding_in_communication), [Semantic Scholar](https://www.semanticscholar.org/paper/Grounding-in-communication-Clark-Brennan/5a9cac54de14e58697d0315fe3c01f3dbe69c186)). Stalnaker (1978/2002) on context as a shared set of presuppositions. **Read these before building**: they give you precise vocabulary for "presentation/acceptance" cycles, evidence of grounding, and the principle of least collaborative effort.
- **Belief-Desire-Intention (BDI)** model — Bratman (1987); Rao & Georgeff. The classical symbolic precursor to your self-compartment design ([MDPI agentic-AI survey](https://www.mdpi.com/1999-5903/17/9/404)).
- **SOAR (Laird, Newell, Rosenbloom)** and **ACT-R (Anderson)** — cognitive architectures with separated working/episodic/semantic/procedural memory and explicit decision cycles. CoALA explicitly translates this into LLM-agent terms ([arXiv 2309.02427](https://arxiv.org/abs/2309.02427)).
- **Society of Mind (Minsky)** — invoked as inspiration in CAMEL ([alphaXiv](https://www.alphaxiv.org/overview/2303.17760)) and in Multi-Agent Debate ([arXiv 2305.14325](https://arxiv.org/abs/2305.14325)).
- **Theory of Mind classics in AI**: Premack & Woodruff (1978); Baron-Cohen Sally-Anne false-belief test (1985); Rabinowitz et al. "Machine Theory of Mind" (ICML 2018).

### 5.2 NLP/LLM Foundations Directly Underpinning Your POC
- **Generative Agents** ([Park et al. 2023](https://arxiv.org/abs/2304.03442)) — observation/reflection/planning architecture.
- **MemGPT** (Packer et al. 2023) — OS-style memory hierarchy and self-editing tools, now maintained as Letta.
- **ReAct** (Yao et al. 2023) — reason-act loops; baseline for the per-turn cycle.
- **Reflexion** ([Shinn et al. 2023](https://arxiv.org/abs/2303.11366)) — verbal-feedback memory updates between trials.
- **Multi-Agent Debate** ([Du et al. 2024](https://composable-models.github.io/llm_debate/)) — multi-LLM dialogue benefits.
- **CAMEL** ([Li et al. 2023](https://arxiv.org/abs/2303.17760)) — two-agent role-playing as a generative method.
- **Symmetric Machine ToM** (Sclar, Neubig, Bisk 2022) — predecessor work.
- **Conversational alignment frameworks** — Gabriel/Kasirzadeh et al.'s CONTEXT-ALIGN ([arXiv 2505.22907](https://arxiv.org/html/2505.22907v1)) provides desiderata for "human-like" conversational AI that map onto your goals.
- **Memory in LLM-Multi-Agent Systems survey** ([TechRxiv 2025](https://www.techrxiv.org/users/1007269/articles/1367390/master/file/data/LLM_MAS_Memory_Survey_preprint_/LLM_MAS_Memory_Survey_preprint_.pdf?inline=true)) — explicitly notes that hybrid topologies "separate global context from local context" and that "shared memory stores this mutual knowledge explicitly" — directly speaks to your tri-partite design.

### 5.3 Benchmarks You Will Want
- **FANToM** (information-asymmetric conversation) — [arXiv 2310.15421](https://arxiv.org/abs/2310.15421).
- **Hi-ToM** (higher-order belief) — [arXiv 2310.16755](https://arxiv.org/abs/2310.16755).
- **NegotiationToM** — [arXiv 2404.13627](https://arxiv.org/html/2404.13627).
- **MindCraft / Collaborative Plan Acquisition** — [arXiv 2109.06275](https://arxiv.org/abs/2109.06275).
- **SOTOPIA** — [arXiv 2310.11667](https://arxiv.org/abs/2310.11667).
- **OneCommon / Dynamic-OneCommon** (Udagawa & Aizawa) — common-ground-creation dialogue.
- **LongMemEval** — long-term memory in dialogue (used by Zep and Mem0 evaluations).
- **DToM-Track** ([arXiv 2603.14646](https://arxiv.org/pdf/2603.14646)) — dynamic ToM as memory.

---

## 6. Where Your Architecture Is Genuinely Novel — and Where to Be Careful

### Novel contributions, given current literature
1. **Three explicitly named memory compartments per agent (self / private ToM-of-other / shared common ground) operationalized at the prompt level**. MindDial has the three *belief levels*, but they're computed per-turn; your design persists them as separate stores. ToM-agent has self-BDI and counterpart-BDI but no common-ground store. Letta has labeled blocks but no ToM block.
2. **Explicit promotion rules** (private belief → common ground) backed by an evidence/acceptance protocol — this is what Clark & Brennan call grounding, and what is conspicuously absent from every implementation surveyed except MindDial's belief-difference resolution.
3. **Asymmetric updates (only on receive)**. This is implicit in Hypothetical Minds and ToM-agent but rarely stated as a hard architectural rule. It avoids a known failure mode where agents over-fit to their own outputs.
4. **First-person framing with explicit author/turn tags as an append-only log.** Most generative-agent systems use a flat memory stream without author tags; speaker confusion is a documented failure mode (see "Speaker Verification in Agent-generated Conversations" — [arXiv 2405.10150](https://arxiv.org/pdf/2405.10150)). DToM-Track shows LLMs have recency bias and fail to retrieve prior belief states once updated, motivating your append-only design ([arXiv 2603.14646](https://arxiv.org/pdf/2603.14646)).
5. **Immutable anchor as a first-class architectural primitive for both agents in a dyadic conversation.** Letta's persona blocks do this for one agent; the architectural commitment to enforce it on *both* sides during dialogue is unusual.

### Cautions
- **Higher-order ToM is hard.** Hi-ToM and FANToM both show LLMs degrading rapidly past 2nd-order beliefs ([Hi-ToM](https://arxiv.org/abs/2310.16755), [FANToM Allen AI blog](https://allenai.org/blog/does-gpt-4-have-theory-of-mind-capabilities-cd2d766e51a7)). Your architecture should explicitly cap recursion or use Hypothetical-Minds-style reinforcement of best-scoring hypotheses rather than try to reason all the way down.
- **Promotion logic is non-trivial.** Clark & Schaefer's "presentation + acceptance" model implies grounding requires *evidence* of understanding (back-channels, demonstrations, repetition). Your promotion rule should specify what counts as evidence — e.g., explicit acknowledgment, paraphrase, action consistent with the belief.
- **LLMs presume common ground rather than build it.** The "Grounding Gaps" paper ([arXiv 2311.09144](https://arxiv.org/html/2311.09144.pdf)) shows RLHF-trained models skip clarification/acknowledgment acts that humans use to build common ground. You may need to explicitly prompt for or score grounding acts.
- **MiroFish-style emergent simulation can polarize faster than reality.** The OASIS paper documents herd-behavior bias in LLM agents — your two-agent setup avoids this but inherits it if scaled.
- **"First machine with a soul" claims warrant skepticism.** The Project December / Replika / Character.AI ecosystem demonstrates how rapidly users anthropomorphize even simple persistent-persona systems — relevant for your POC's evaluation framing.

---

## 7. Concrete Recommendations for the POC

1. **Adopt MindDial's three-belief mind-module formulation** ([Qiu et al. 2024](https://aclanthology.org/2024.sigdial-1.63/)) as your starting prompt template; it is the closest published artifact to what you want and the paper provides exact prompt structures.
2. **Build the storage layer on Letta** for the immutable-anchor + persistent-blocks property; add a third labeled block per peer for ToM and a shared block for common ground. Letta's Conversations API already supports cross-agent shared memory.
3. **Use a SymbolicToM- or Hypothetical-Minds-style hypothesis graph** inside the ToM compartment so beliefs about the partner are scored and refineable rather than free-form text.
4. **Tag every entry in the append-only log with `{turn_id, author_id, addressee_id, modality}`** — there's empirical evidence that LLMs confuse speakers in agent-generated conversations ([arXiv 2405.10150](https://arxiv.org/pdf/2405.10150)).
5. **Define promotion explicitly with a dual-acknowledgment rule** (private belief → shared common ground only after both agents have produced acknowledging utterances or congruent actions). This is closer to Clark-Schaefer's contribution model than to current LLM systems.
6. **Evaluate on FANToM (information asymmetry) and SOTOPIA (private goals)**, plus MindCraft for situated grounding. These three give you ToM accuracy, social-goal completion, and common-ground-construction metrics respectively.
7. **Frame the contribution carefully**: not "first multi-agent LLM dialogue" (CAMEL/AutoGen own that) and not "first ToM module" (MindDial/ToM-agent/Hypothetical Minds own that), but rather **"first system to operationalize Clark-and-Brennan grounding as separate persistent memory compartments with explicit promotion rules in two-agent open-domain dialogue."**

---

## Key Source Index

- Park et al. 2023 Generative Agents — https://arxiv.org/abs/2304.03442
- Park et al. 2024 1,000 People — https://arxiv.org/abs/2411.10109
- MindDial (Qiu et al. SIGDIAL 2024) — https://aclanthology.org/2024.sigdial-1.63/
- ToM-agent (Yang et al. 2025) — https://arxiv.org/abs/2501.15355
- Hypothetical Minds (Cross et al. 2024) — https://arxiv.org/abs/2407.07086
- SymbolicToM (Sclar et al. ACL 2023) — https://arxiv.org/abs/2306.00924
- ToM for Multi-Agent Collaboration (Li et al. EMNLP 2023) — https://arxiv.org/abs/2310.10701
- MindCraft (Bara et al. EMNLP 2021) — https://arxiv.org/abs/2109.06275
- FANToM (Kim et al. EMNLP 2023) — https://arxiv.org/abs/2310.15421
- Hi-ToM (Wu et al. EMNLP-Findings 2023) — https://arxiv.org/abs/2310.16755
- NegotiationToM — https://arxiv.org/html/2404.13627
- SOTOPIA (Zhou et al. ICLR 2024) — https://arxiv.org/abs/2310.11667
- SimToM (Wilf et al. ACL 2024) — https://arxiv.org/abs/2311.10227
- CoALA (Sumers et al. 2023) — https://arxiv.org/abs/2309.02427
- Reflexion (Shinn et al. NeurIPS 2023) — https://arxiv.org/abs/2303.11366
- Multi-Agent Debate (Du et al. ICML 2024) — https://arxiv.org/abs/2305.14325
- CAMEL (Li et al. NeurIPS 2023) — https://arxiv.org/abs/2303.17760
- AgentVerse (Chen et al. ICLR 2024) — https://arxiv.org/abs/2308.10848
- Concordia (Vezhnevets et al., DeepMind) — https://arxiv.org/abs/2312.03664; https://github.com/google-deepmind/concordia
- AI Town — https://github.com/a16z-infra/ai-town
- AutoGen — https://github.com/microsoft/autogen
- Letta / MemGPT — https://www.letta.com ; https://github.com/letta-ai/letta
- Zep / Graphiti — https://www.getzep.com ; https://arxiv.org/abs/2501.13956
- Mem0 — https://github.com/mem0ai/mem0
- A-MEM — https://arxiv.org/abs/2502.12110
- MiroFish — https://github.com/666ghj/MiroFish ; https://github.com/nikmcfly/MiroFish-Offline
- OASIS / CAMEL-AI — referenced via MiroFish docs and https://medium.com/@balajibal/mirofish-multi-agent-swarm-intelligence-for-predictive-simulation-09771e60b188
- Inworld AI Long-Term Memory — https://inworld.ai/blog/introducing-long-term-memory
- Fable Studio Lucy — https://www.fable-studio.com/behind-the-scenes/ai-generation
- Project December — https://projectdecember.net/ ; https://niemanstoryboard.org/2021/08/31/jason-fagone-follows-the-creation-life-and-death-of-a-chatbot-romance/
- Clark & Brennan grounding — https://en.wikipedia.org/wiki/Grounding_in_communication
- Grounding Gaps in LLM Generations — https://arxiv.org/html/2311.09144.pdf
- Speaker Verification in Agent-generated Conversations — https://arxiv.org/pdf/2405.10150
- DToM-Track (Dynamic ToM as Memory) — https://arxiv.org/pdf/2603.14646
- Conversational Alignment with AI in Context (Gabriel/Kasirzadeh et al.) — https://arxiv.org/html/2505.22907v1
- Memory in LLM-MAS survey — https://www.techrxiv.org/users/1007269/articles/1367390/master/file/data/LLM_MAS_Memory_Survey_preprint_/LLM_MAS_Memory_Survey_preprint_.pdf?inline=true
- Adaptive ToM for LLM-MA Coordination — https://arxiv.org/html/2603.16264

*Note on dates and unverified claims:* Several sources cited above are 2025–2026 preprints; some figures (e.g., MiroFish's GitHub-trending and $4.1M investment claims) come from Medium write-ups that should be independently verified before being repeated in a publication. The MiroFish phenomenon is described differently across sources — the original repository in Chinese frames it as a "swarm intelligence engine," while English forks emphasize the prediction-engine framing; the underlying OASIS paper from CAMEL-AI is the authoritative technical reference. Replika and Character.AI feature claims (e.g., specific subscription tiers, message limits) cited in 2026 industry blogs should be verified against the vendors' current sites, as these change frequently.