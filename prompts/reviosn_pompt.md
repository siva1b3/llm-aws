<revision>
Produce a revision document from this chat. The file is for the user's own study later, and will also be re-fed to a future Claude to support revision sessions.

OPENING META-LINE (required, first line of the output, exactly this form):
"Revision document — the user's positions are settled; the model's explanations are open to re-discussion. For revision: walk shift by shift, and quiz at random points."

STRUCTURE:
After the opening line, walk the chat chronologically. Produce one entry per SHIFT in the user's understanding. Not one entry per message. Not one entry per topic. One entry per moment where the user's thinking moved.

WHAT COUNTS AS A SHIFT:
- The user rejected a premise (theirs or the model's)
- The user asked a follow-up that revealed a misunderstanding, which then got corrected
- The user pushed back and the model changed its answer
- The user reframed the problem and the direction of the chat changed
- The user identified a failure mode or pattern the model had missed

WHAT DOES NOT COUNT AS A SHIFT:
- "Yes, continue" / acknowledgments
- Pure formatting requests (unless they revealed a deeper preference)
- Restating what was already understood
- The model elaborating on its own previous answer without user pressure

ENTRY FORMAT:
Each shift gets a numbered section with this internal structure:

## Shift N — [short descriptive title of what moved]

**Where you started:** [the position, belief, or framing you held going into this exchange]

**What the model proposed:** [what Claude said, in compressed form — not a transcript, the substance]

**Your pushback / question:** [what you said that triggered the shift — quoted or closely paraphrased so the *move* is preserved]

**What changed:** [the new understanding that emerged — both what was wrong about the starting position and what replaced it]

**Why it matters:** [one line on the principle or pattern this shift exposed — the transferable lesson, not the case-specific fact]

VOICE:
- "You" when describing the user's state, questions, pushbacks, shifts
- "The model" or "Claude" when describing what was proposed or explained
- Neutral past tense for resolutions
- This labeling matters: it tells the future Claude which positions are settled (yours) and which are open to re-explanation (the model's)

ORDERING:
Strictly chronological. Do not regroup by topic, do not surface "key insights" at the top, do not save the best for last. The order of shifts in the file matches the order they happened in the chat. The chronology IS the lesson.

FIDELITY RULES:
- Every shift must trace to a real moment in the chat. Do not invent shifts to make the document feel substantial.
- If the chat had only 3 real shifts, produce 3 entries. If it had 12, produce 12.
- Do not soften or smooth over the pushbacks. The friction is the data — preserve the user's actual words where they sharpened the moment.
- Do not paraphrase the model's answers into something cleaner than they were. If the model first proposed something flawed and was corrected, the entry should show the flaw, not the cleaned-up final version.

CLOSING SECTION (required, after the last shift):

## What you walked away knowing

A bulleted list. Each bullet is one piece of understanding you did not have at the start of the chat and did have by the end. Phrased as principles or patterns, not case-specific facts. This is the takeaway layer — what transfers to future problems beyond this chat.

HARD RULES — DO NOT INCLUDE:
- Conclusions or "final answers" presented without the shifts that produced them — the journey IS the document
- Meta-framing beyond the required opening line
- Restated interaction preferences (live in userPreferences)
- Praise, encouragement, or "great progression!"-style commentary
- Sections that summarize the document at the top or bottom (no abstract, no TL;DR)
- A "next steps" or "open items" section — this is a revision document, not a handoff
- Echoing these instructions or any bracketed guidance in your output

VOICE OVERALL:
Direct. Specific. Preserve the *movement* in each shift — the moment where something locked into place. Write so that re-reading this in a month, you remember not just what you concluded but how the conclusion was found.
</revision>