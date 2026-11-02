# A read-back gate for LLM prompts — force the model to restate your request before it builds anything

This is a prompt block I paste at the end of a request when the output is going to be expensive — a document, a phased plan, a chunk of code, anything where a misunderstanding costs a full regeneration cycle. It stops the model from producing anything until it has restated my request back to me and I have approved the restatement.

Posting it here as a reference for myself, with the reasoning and the failure modes, because the wording is not arbitrary. Every line in it is fixing a specific way this fails.

---

## The block

```
Before doing anything, restate my request back in your own words so I can
confirm you understood it. Keep the restatement shorter than my request —
I should be able to check it in a few seconds.

Cover only what applies:
- if there is a goal behind the task, state it
- if I gave requirements, constraints, or a format, list them
- if the scope could be larger or smaller than I intended, say what you
  will and will not do

Always name the assumptions you are making — anything you filled in that I
did not state. Do not say "no assumptions"; if you are ready to start, you
have made some.

If the request is simple, just tell me plainly what you understood me to be
asking — do not invent sections that do not apply.

If anything is ambiguous or missing, list it as a numbered question instead
of guessing. If you think my approach is flawed or there is a better way,
say so now.

Do not do the task or produce any part of the output yet. Wait for me to
reply "confirmed." If I change anything after your restatement, restate
once more and wait again.
```

Short version, for smaller tasks:

```
Restate my request in your own words, including any assumptions you are
making, and list anything unclear as questions. Don't start until I reply
"confirmed."
```

---

## What it actually is

This is a read-back handshake. Aviation uses it — the controller issues a clearance, the pilot reads it back, the controller confirms. Surgical teams use it before an incision. Network change management uses it: you state the intended config change, someone confirms the intended end state, and only then does it get committed to the device.

The principle is the same in all three: put a cheap verification step between *understanding* and *execution*, so that a misunderstanding gets caught before effort is spent instead of after. The read-back costs seconds. The wrong output costs a full cycle — and worse, sometimes you don't notice it was wrong until you're deep into using it.

The structure it creates is a four-state loop:

1. You state the requirement.
2. The model restates it.
3. You approve, or you correct — and if you correct, it goes back to state 2.
4. The model executes.

Nothing large gets built until the spec is locked.

---

## Why each part is worded the way it is

**"in your own words"** — an echo of your own phrasing proves nothing. A model can repeat your sentence back verbatim while having misread it completely. Forcing genuine rewording is what turns a *claim* of understanding into *evidence* of it. If the model understood, the paraphrase will be right. If it didn't, the paraphrase is where you see it.

**"Keep the restatement shorter than my request"** — this is about economics, not correctness. The gate only pays for itself because checking is cheaper than rebuilding. If the read-back is longer than what you asked for, checking becomes its own reading task, and you will start skimming. Once you skim, the gate is open and you haven't noticed. Capping the length keeps verification cheap, which is the whole reason the protocol works.

**"Cover only what applies"** — early versions of this had four mandatory sections: goal, requirements, assumptions, scope. That's fine for a build task and useless for "summarise this paragraph." Forcing all four every time produces filler, and filler trains you to stop reading the restatement. Making them conditional lets the read-back scale with the request.

**"Always name the assumptions"** — this one is deliberately *not* conditional. It is the highest-value line in the block, because silent assumptions are where almost all wrong output originates. But if you phrase it as "*if* you are making assumptions, name them," you have handed the model an exit, and "none" is the cheap answer. Hence the follow-up sentence closing that exit: if it's ready to start, it has filled something in.

**"say what you will and will not do"** — catches scope drift in both directions. Creep, where the model adds a reasonable-sounding requirement you never asked for. And shrink, where it quietly drops part of what you asked. Shrink is the more dangerous one because it's invisible in a confident-sounding restatement.

**"list it as a numbered question instead of guessing"** — separates what the model knows from what it's filling in. Numbered so you can answer 1, 3, 5 without rewriting the whole thing.

**"If you think my approach is flawed... say so now"** — the entire protocol depends on an honest counterpart. A model that rephrases your words flatteringly and says "got it" defeats the gate completely. This line explicitly licenses pushback *before* effort is spent rather than after, when neither side wants to throw the work away.

**"Do not do the task or produce any part of the output yet"** — earlier wording was "do not produce the solution yet," which has a gap: many requests have no solution. Write a summary. Draft an email. Translate this. A model can reason that no solution is involved and just proceed. "The task or any part of the output" closes that. "Any part" also blocks the common half-compliance move where it restates and then adds "here's a quick start to get you going."

**"Wait for me to reply 'confirmed'"** — pick one uncommon token and keep it consistent, so it functions as a command rather than as conversation. "Yes" is a weak trigger: it appears constantly in ordinary back-and-forth and can fire by accident.

**"If I change anything, restate once more and wait again"** — makes the gate a loop instead of a one-shot. Without it, the model restates, you tweak one requirement, and it runs on the stale understanding. Specs change across turns; the gate has to re-arm.

---

## Failure modes

These are the ways it breaks. Most of them are not the model's fault.

**Rubber-stamping.** The big one. Once the block is in place, the weak point moves from the model to you. The realistic failure is not a bad restatement — it's a restatement that *looks* thorough, produces a feeling of reassurance, and gets a reflexive "confirmed." That converts the whole thing into ceremony. The block cannot protect against this; only the habit of reading adversarially can. Ask specifically: did it shrink the job, did it add something I never asked for, is that assumption actually mine.

**Self-exemption.** Do not add an escape clause like "you may skip this if the request is simple." The moment the model can decide whether the gate applies, it will decide to skip too often — usually on exactly the requests where it was needed. A gate that can be self-exempted is not a gate. Control it from outside, by choosing whether to paste it.

**Ritual creep.** The block wants to grow. Every time it fails you'll want to add a line. Past a certain length, long instruction blocks get followed *less* carefully, not more — they become background texture the model skims. The version above is at the practical limit. Freeze it.

**Overuse.** Pasting this on a quick factual question turns one round trip into two for no benefit. Keep the two versions and match the gate to the cost of being wrong. If a wrong answer costs nothing, don't gate.

**Drift in long conversations.** In a long chat, an instruction from twenty messages ago loses force. If you're still working the same thread hours later, re-paste the block rather than assuming it's still governing.

**Confident-but-wrong restatements.** The failure that actually gets through is a restatement that is fluent, well-organised, and subtly wrong about one requirement. Thoroughness in the read-back is not evidence of accuracy. Check content, not polish.

---

## When it's worth it

Worth it: documents, multi-part plans, code you'll actually run, anything with a format spec, anything where you've iterated the requirements across several messages, and anything where you'd be annoyed to throw away the output.

Not worth it: factual questions, quick lookups, small edits, anything you can verify at a glance.

The split that matters is this — the *restatement* should flex to fit the request, plain for simple things and structured for complex ones. The *gate* should not flex at all. One adapts, the other is a hard rule. Keeping those two separate is most of what makes this work.
