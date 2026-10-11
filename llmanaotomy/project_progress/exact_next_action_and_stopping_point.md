# Exact next small action

**First:** Establish the actual working environment context (whether the chosen EC2 instance is provisioned/reachable, and if files already exist). If direct workspace or shell access is available, inspect rather than ask the learner to repeat information already available. If it is unavailable, request the single minimum missing fact/action needed to proceed.

**Then:** Create or run a minimal Python script that executes **exactly one** model forward pass for the fixed input, reads next-token logits, selects a single argmax ID, decodes that ID, and prints bounded, human-readable results. The assistant writes the code; the learner executes if necessary. Do not use a multi-token generation loop.

**Stop after:** The actual baseline output is examined and explained. Request the learner's confirmation before opening the tokenizer or the next internal box.
