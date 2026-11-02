# A prompt refiner block — make the model rewrite your request before it answers

Companion to the read-back gate post. Different tool, opposite job.

The gate confirms your request unchanged, so a wrong deliverable doesn't get built. This one changes your request — it rewrites what you asked into a sharper version of itself before anything gets answered. Use it when the problem is your wording, not the cost of the output.

Storing it here with the reasoning and the failure modes, because most of the wording exists to stop specific ways this goes wrong.

## The block

```
Rewrite my request as a clearer, more precise version of itself — the prompt
I should have written.

If we have been talking already, treat the whole conversation as the request.
Fold in what we have settled. Do not re-ask what I have already answered.

Keep my intent exactly. Do not add requirements, scope, tone, or a persona I
did not ask for. Precision, not padding — do not make it longer than it needs
to be.

If I am asking for information, make the question sharper.
If I am asking for a task, make the task, the constraints, and the expected
output format explicit.
If I do not yet know what I am asking, do not force it into a tight
specification — sharpen the wording but keep it open, and name the directions
it could go.

Then, only if there is something real to say:
- anything you guessed in order to make it precise — one line each
- anything only I can answer — as a question

Skip either of those when there is nothing to put in them. Do not pad.

Do not answer the request itself. Wait for me to reply "confirmed."
```

One-turn variant, when a second round trip isn't worth it:

```
First rewrite my request as a clearer version of itself — folding in anything
we have already settled in this conversation, and adding nothing I did not ask
for. Then answer that version. Note any real guesses at the end.
```

## What this is actually doing

Worth being clear about the mechanism, because the intuition here is usually wrong.

There is no special machine dialect that models understand better. No phrasing trick, no magic structure, no benefit to writing like a robot. Prompts that look machine-generated often perform worse, because they're padded with ceremony — "You are a world-class expert in..." — that carries no information at all.

What actually improves output is information density. The task stated explicitly. The constraints stated explicitly. The output format stated explicitly. The ambiguity removed. That's all a refined prompt is. It reads clearer to a human too — that's the test. If the rewritten version would confuse another person, it isn't a better prompt, it's a worse one wearing a costume.

The second benefit, and for me the bigger one: English is my second language. This block converts imprecise wording into precise wording before it reaches the model, in a step I can inspect. That's a real gain and it has nothing to do with prompting theory.

## Why each part is worded the way it is

**"the prompt I should have written"** — sets the target. Not a summary, not an expansion, not a plan. The same request, said better.

**"treat the whole conversation as the request"** — the block used to read as if my message stood alone. Used on turn five, that's simply wrong: the request is my last message plus everything already agreed. Without this line the refiner re-asks questions I already answered and strips constraints we settled three turns ago.

**"Do not add requirements, scope, tone, or a persona I did not ask for"** — the single most important line. A refiner wants to be helpful, and being helpful looks like adding. See the failure modes.

**"Precision, not padding"** — left alone, refiners inflate. Length feels like effort. A three-line question becomes a twenty-line specification with headers, and none of the added lines carry information I supplied.

**Three request types, named explicitly** — info, task, and don't-know-yet need different treatment. The earlier version silently assumed everything was a task, which is why it felt heavy on plain questions. A question about a fact doesn't have an "expected output format" worth specifying, and pretending it does produces noise.

**The "if I do not yet know what I am asking" branch** — this is a safety net, not a recommendation. More on that below.

**"only if there is something real to say" / "Skip either of those when there is nothing"** — the guessed-at list and the open-questions list used to be mandatory numbered sections. Mandatory sections generate filler when there's nothing real to put in them, and filler trains you to stop reading the output — which defeats the point, because those two lists are the part that actually matters. Demoting them from headings to one-liners helps too: a heading invites the model to fill it.

**"Do not answer the request itself. Wait for me to reply 'confirmed.'"** — same hard stop as the gate. Without an explicit stop, models rewrite and then immediately answer, which removes your chance to catch a bad rewrite.

## Failure modes

**Silent addition.** The dominant one. The refiner adds constraints I never stated — a tone, a length, an audience, an expert persona, a format. Those additions then look like my own requirements when I read the rewrite back, and I approve them without noticing. This is exactly why guesses are forced into a separate line instead of being allowed into the rewrite body. That short list is the part to read carefully; the rewrite itself is the part that reads smoothly and therefore gets skimmed.

**Inflation.** If the rewrite is noticeably longer than what I wrote, the question to ask is what was added — not whether it looks thorough. Usually the extra length is scaffolding, not specification.

**Over-constraining.** The non-obvious failure. Sometimes a vague question produces a better answer than a precise one, because the vagueness leaves room for the model to find a framing I didn't know to ask for. Refining "what should I know about X" into a tight three-part specification gets me exactly those three parts and loses everything outside them. Precision is not free — it trades away discovery.

**Meaning drift.** Refinement can improve the grammar while shifting the meaning. The rewrite reads well, so it gets accepted, and then a good answer arrives to a slightly different question. Check the rewrite for meaning, not for clarity. Clarity is what makes this failure hard to catch.

**Mid-conversation contamination** — the one worth actually worrying about. When this is invoked on turn five, the refiner has to decide what "we settled" means, and it cannot reliably distinguish a decision I made from something the model proposed that I didn't object to. Over several turns of back-and-forth, unchallenged suggestions get promoted to requirements. The refined request then specifies things I never chose, in wording I'll recognise, because I read it once when it went past.

The defence: read the mid-conversation rewrite specifically for constraints I don't remember deciding. If a line surprises me, it's probably the model's, not mine.

**Ritual creep.** Same as the gate. Every failure makes you want to add a line, and past a certain length the whole block becomes background texture the model skims rather than follows. This version is at the practical limit.

## On the "I don't know what I'm asking" case

The block handles it, but a refiner is a weak tool there and it's worth being honest about that.

Refinement works by removing ambiguity. When you're exploring, the ambiguity is doing useful work — it's what lets an answer arrive from a direction you didn't think to specify. Tighten too early and you get exactly what you asked for and nothing better.

The stronger move is to ask the messy question directly and let two or three turns of conversation find the shape. Then use the refiner afterwards, once you know what you actually want, to lock it down. The branch in the block is there for when you use it anyway — not a reason to use it.

## Do not stack this with the read-back gate

They look similar and do opposite jobs. This one changes the request. The gate confirms the request unchanged. Running both means: refine → confirm → restate → confirm → build. Four round trips before a single line of output. That is ritual, not process.

One per task:

**Refiner** — when the problem is my wording. Imprecise ask, clear cost.

**Gate** — when the problem is the cost of being wrong. Clear ask, expensive deliverable.

They rarely apply to the same request. If it seems like both apply, the honest answer is usually that the request is expensive and vague, and the fix is to think for another thirty seconds before sending it — not to run two protocols.

## Stop point

Between the two blocks this covers the ground: a gate for expensive output, a refiner for imprecise wording, and a rule that they don't stack. Any further patching should be a note about when to use them, not more lines inside them.
