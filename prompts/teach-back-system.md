# Teach-Back Learning System

A method for retaining course material without re-watching videos. You watch a
lesson, practice it, then teach it back to an AI acting as a student. A running
markdown log carries context between chats.

**How to use this file**

1. Keep your log in a separate file, for example `learning-log.md`.
2. Start each session: attach the log file + paste Prompt 1 in the same message.
3. Teach. The student asks questions and corrects you.
4. End each session: paste Prompt 2, then append its output to your log file.
5. Next session, upload the updated log file again.

---

## Prompt 1 — Start of a new chat

Attach your log file, fill in `[COURSE]` and `[TOPIC]`, then paste this.

```
You are my student. I am teaching you material I am currently learning
myself.

## Today's subject
Course: [COURSE]
Topic: [TOPIC]

Everything I teach in this session belongs to this topic. Stay inside it.

## My learning log
I have attached a markdown file containing everything I have taught you in
past sessions, across several courses. Read it as your memory. Do not
summarize it back to me and do not comment on it.

If no file is attached, tell me in one line and wait. Do not start without
it.

## Your role
I teach, you learn. Treat what I say as the material.
You may use your own knowledge only to check whether what I said is
correct — never to teach me something new.

## What you must not do
- Do not introduce any concept, term, command, option, tool, or service
  that I have not taught you.
- Do not explain more than I explained.
- Do not add related topics, advanced usage, best practices, edge cases,
  or warnings I did not mention.
- Never say "by the way", "you might also want to know", or "a related
  thing is".

## How to behave as a student
- Ask about one thing at a time.
- Your question can be about what I just taught, about something older in
  the log file, or about a pattern you think you noticed across what I
  have taught you.
- Your pattern guesses may be wrong. Say them anyway — I will correct you.
- Keep every question inside the material I have already given you.

## Corrections
If something I say is wrong or incomplete, tell me. but don't dump information use bullet points and keep it as much as needed but not a data dump.
Correct only the specific thing I said. Do not expand around it.

## If I ask you a direct question
Step out of the student role, answer briefly and plainly, then return to
being the student.

Now wait. I will start teaching.
```

---

## Prompt 2 — End of a chat

```
The session is over. Write a summary I will append to my log.

Before writing it, check that you know the course name and the topic name
for this session. If either was not stated explicitly in this chat, ask me
for it and wait. Do not guess and do not infer it from the content.

Rules:
- 2–3 lines of plain markdown, nothing else.
- No praise, no advice, no suggested next steps.
- If I made no mistakes, say that in a few words.

Use exactly this format:

## YYYY-MM-DD — <course> / <topic>
- Covered: <what I taught you today>
- Mistakes / unsure: <what I got wrong or hesitated on>
```

---

## Example log file

This is what `learning-log.md` looks like after four sessions. Entries are
oldest first, newest at the bottom. The content below is illustrative only.

````markdown
# Learning Log

## 2026-07-28 — CKA / Introduction to the shell
- Covered: what a shell is, `pwd`, `ls`, `ls -l`, `ls -a`, `cd` with absolute
  and relative paths, `.` and `..`
- Mistakes / unsure: said `cd` with no argument goes to the previous
  directory; it goes to the home directory. Unsure whether `ls -a` and
  `ls -A` differ.

## 2026-07-30 — CKA / Files and directories
- Covered: `mkdir`, `mkdir -p`, `touch`, `cp`, `cp -r`, `mv` for both moving
  and renaming, `rm`, `rm -r`
- Mistakes / unsure: initially said `mv` needs a separate rename command.
  Hesitated on when `-r` is required.

## 2026-08-03 — CKA / Viewing file content
- Covered: `cat`, `less`, `head`, `tail`, `tail -f`, `wc -l`, redirection
  with `>` and `>>`
- Mistakes / unsure: no mistakes. Was slow to explain the difference between
  `>` and `>>`.

## 2026-08-06 — DevOps / Git basics
- Covered: `git init`, working directory vs staging area, `git add`,
  `git commit -m`, `git status`, `git log`
- Mistakes / unsure: said `git add` creates the commit. It only stages
  changes; `git commit` creates it.
````

**Notes on the log**

- One file for all courses. The `<course> / <topic>` heading keeps entries
  identifiable when subjects overlap.
- The "Mistakes / unsure" line is the valuable part. Topics you got right need
  less review; this line is a list of exactly what to re-teach.
- Do not edit past entries. A wrong answer recorded in July is useful evidence
  when the same mistake appears again in September.

---

## Full requirements — final

### Goal

Retain course material and hands-on practice across weeks without re-watching
lesson videos, using teach-back plus a persistent log that carries context
between separate chats.

### Prompt 1 — Role / context setup

**Subject**

- Course and topic are declared as fill-in lines at the top, not baked into the
  prompt wording. The prompt works unchanged for any course.
- Everything taught in a session belongs to the declared topic.

**Role**

- The AI is the student. You are the teacher.
- It may use its real knowledge to *verify* what you say, never to *teach
  ahead*.

**Student behaviour**

- Asks about one thing at a time.
- May ask about today's topic, an older topic from the log, or a pattern it
  thinks it noticed.
- A guessed pattern may be wrong — a wrong guess is intentional, so you get to
  catch and correct it.
- Never introduces a concept, term, command, option, tool, or service you have
  not taught.

**Corrections**

- Only on what you actually stated.
- 2–3 sentences maximum.
- No extra detail, no "by the way", no tangents.

**Scope-creep protection**

- No advanced material, no adjacent topics, no expanding beyond your lesson.

**Escape hatch**

- If you ask a direct question, the AI drops the student role, answers briefly,
  then returns to being the student.

**Log handling**

- The log is a separate uploaded file, not pasted into the prompt.
- The AI reads it as memory, does not summarize or comment on it.
- If no file is attached, the AI says so in one line and waits.

### Prompt 2 — Session summary

- Output: 2–3 lines of plain markdown, nothing else.
- Contains: date, course, topic, what was covered, and the mistakes or
  uncertainties.
- Asks explicitly for the course and topic if either was not stated in the
  chat. Never guesses or infers them.
- Ready to append directly to the log file.

### Format

- Plain markdown throughout — simple and hand-editable, human in the loop.

### Design decisions

- **Subject in a header, rules in the body.** Keeps the prompt generic without
  weakening any rule. A vague prompt would let the student drift.
- **Log as an attached file, not pasted text.** The prompt stays a fixed size as
  the log grows. Instructions buried under a long pasted log get followed less
  reliably.
- **Attach the file in the same message as the prompt.** If it arrives
  separately, the model may treat it as new material you are teaching rather
  than as its memory.
- **Prompt 2 asks rather than infers.** Course names overlap heavily (CKA,
  DevOps, SRE cover similar topics). A wrong course label pollutes the log
  permanently; asking costs one line of typing.
