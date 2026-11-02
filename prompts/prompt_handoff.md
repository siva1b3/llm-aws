<summary>
Fill this template using the content of this chat. Decide the section structure based on what the chat was actually about. Do not force content into sections that do not fit; do not invent content to fill a section.

WORKFLOW:
Step 1 — Before producing any handoff, output a proposal:
- 1–2 lines describing what the chat covered
- A markdown table comparing the sizes you offer (up to three: Small, Medium, Large) with these columns:
  | Size | Approx. length | Includes | Drops vs next size up |
  - "Approx. length" calibrated to THIS chat's content density, not a fixed scale
  - "Includes" and "Drops vs next size up" must name CATEGORIES of content, not just describe verbosity differences
- If two sizes would only differ by verbosity rather than by category, offer fewer sizes. One or two sizes is acceptable when the chat does not warrant three.
- After the table, one line: "Which size do you want?"
- Do not produce any handoff until the user answers.

Step 2 — Produce only the size the user requested.

SIZE PRINCIPLE:
Sizes are defined by what CATEGORIES of content they include, not by verbosity. Each step up adds a new kind of content, not more words about the same content. Examples of categories: environment facts, decisions, reasoning behind decisions, rejected alternatives, conventions, constraints, edge cases raised, open questions, supporting context. Different chats will draw on different category sets.

FIDELITY RULE (applies to all sizes, especially Large):
Every claim must trace to something actually said or established in the chat. No inferred detail, no plausible-sounding additions, no "this would probably also include..." If a size would require invented content to reach its length, produce it shorter and say so. Large is "more fidelity to what happened," not "more content."

FIXED ELEMENTS (do not change):
1. The handoff starts with a header in this exact form, with [project/topic] replaced by a short descriptive name:
   "# [project/topic] — context handoff"
   The header alone orients the next chat. Do not add a framing sentence beneath it. The next line after the header must be a real content section.
2. The handoff ends with a section that tells the next chat what to do next. Name it whatever fits the content (Open, Next steps, Pending questions, Where I left off, Unresolved). It must be the last section.

FLEXIBLE ELEMENTS:
Everything between the header and the final section. You decide:
- Which sections to include based on what the chat covered
- Section names that match the actual content
- Order — foundational context first, recent/active work last
- Format within sections (prose, lists, tables, code blocks) based on what carries the information best
- Depth, scaled to the requested size and bounded by the fidelity rule

HARD RULES — DO NOT INCLUDE:
- Framing sentences under the header ("Continuing prior work", "Below is the current state", "This document captures...")
- A "how to use this document" section or any meta-framing about how to read the document
- Restated interaction preferences or communication style instructions (live in userPreferences, persist automatically)
- Numbered cross-reference systems ("see section 3.2")
- A "Goals" section that restates aspirations rather than concrete decisions
- Conversational filler, exploratory tangents that led nowhere, things discussed but not decided
- Praise, framing language, summary-of-the-summary paragraphs
- Echoing these instructions or any bracketed guidance in your output

HARD RULES — MUST INCLUDE WHERE APPLICABLE:
- Concrete facts over abstractions (versions, names, paths, exact values)
- Decisions with one-line reasons (what + why), not deliberation history
- Distinction between what exists and what is planned (past tense for built, present/future for pending)
- Constraints that shaped decisions
- Anything established in the chat that would surprise someone who did not see it established

VOICE:
Direct, factual, dense. Each line carries information. Write for someone who needs to act on this tomorrow.
</summary>