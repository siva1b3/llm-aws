# Project context guide

Use this guide to organize project information. It applies to learning, research, software development, writing, and other projects.

Organize information by its purpose:

1. What guides the work?
2. Where does the work stand now?
3. What does the evidence show?

These purposes define three categories. The categories do not require one file per topic or a fixed number of folders.

## 1. The three categories

| Category | Contains | How it changes | Suggested folder |
| --- | --- | --- | --- |
| **Stable contract** | Purpose, goals, scope, constraints, responsibilities, working agreements, and success criteria | Update when an agreement or governing decision changes | `project-rules/` |
| **Current state** | Active work, confirmed progress, unresolved questions, blockers, and the next action | Replace outdated status as work progresses | `project-status/` |
| **Accumulated evidence** | Experiments, observations, results, explanations, and the support for conclusions | Add records as work produces evidence. Correct findings when needed. | `project-evidence/` |

The category names describe the purpose of the information. The folder names help readers find that information. Use these folders only if they help the project.

### Stable contract

Describe the project's agreements:

- What must the project achieve?
- What is outside the scope?
- Who is responsible for the work?
- How must the work proceed?
- What counts as success?

Include agreed technical choices when they govern the work.

Stable does not mean permanent. These agreements usually continue across sessions. Use the project's decision process to change them. Do not change the goal or an agreed constraint through a progress note alone.

### Current state

Record enough information for the reader to resume work:

- The active task.
- The work that is complete.
- The results that are verified.
- The questions and blockers that remain.
- The next concrete action.

Keep this record current. Replace outdated status. Link to earlier evidence when needed. Readers must not need to reconstruct the current status from a history of updates.

Distinguish the states that matter to the project. Planned, implemented, executed, verified, and understood can have different meanings. A successful run does not prove understanding. Silence does not prove agreement.

### Accumulated evidence

Keep the records that support conclusions. A useful record contains:

- The question that the work addresses.
- The inputs or conditions.
- The method.
- The observations.
- The interpretation of those observations.
- The limitations of the findings.

Include explanations that help readers understand actual work. Label hypotheses, expectations, and illustrative examples. Do not present them as observed results.

Retain useful findings after the active task changes. If a finding changes, explain the correction. Identify the earlier conclusion as superseded when relevant. Add enough explanation for readers to understand raw logs.

## 2. Where questions belong

An unanswered question is not evidence. Keep it beside an evidence record when readers need that context.

| Question | Where to keep it |
| --- | --- |
| A question that affects active work or the next decision | Current state |
| A question raised by an experiment or investigation | With that evidence record, marked unresolved |
| A proposed change to the goal or an agreement | Current state until the decision. Update the stable contract after agreement. |

If an investigation question blocks current work, summarize it in the current state. Link to the full record. Keep the answer connected to supporting observations when an answer becomes available.

Questions do not need a fourth category.

## 3. How the categories connect

The stable contract guides the work. The current state identifies the next action. Work produces evidence. Use that evidence to update the current state. Evidence can also support a deliberate change to the stable contract.

For example, a research project uses the categories as follows:

| Information | Category |
| --- | --- |
| Agreed research question and acceptance criteria | Stable contract |
| Active analysis task | Current state |
| Records of the method and observed results | Accumulated evidence |

Evidence can challenge an agreement. Evidence does not automatically authorize a new objective.

## 4. Folder and file names

Use short, descriptive names. Prefer lowercase words separated by hyphens. Avoid spaces and unexplained abbreviations. Do not use a long list of the file's contents as its name.

| Purpose | Example filename |
| --- | --- |
| Connected goals, scope, and working agreements | `project-guide.md` |
| Latest checkpoint and next action | `current-progress.md` |
| Agreed technical choices and relevant settings | `technical-reference.md` |
| A real experiment and its findings | `tokenization-observations.md` |
| Explanation of a folder's purpose and reading route | `README.md` |

These names are examples. They do not require you to create each file. Name evidence records after the actual investigation. Add a date or sequence number only when it helps distinguish records or show their order.

Keep `AGENTS.md` at the root when an agent uses it as the entry point. Use it to explain the reading order. Link to the documents that contain the details.

Each folder README should explain:

- What belongs in the folder.
- What belongs elsewhere.
- Which file to read first.
- When to update the records.

Use the README for navigation. Do not copy the documents into it.

## 5. Keep context together

- Keep sections together when readers need them together to understand or act.
- Use headings and a table of contents before you create another file.
- Split only when a part has its own reading purpose and enough context to stand alone.
- Choose one document as the source for each rule or fact. Link to that document elsewhere.
- Keep the current state concise. Retain detailed findings in evidence records.
- Add evidence records when real work produces observations. Do not create empty records for future investigations.

A reference can contain more than one kind of information. Technical choices may be stable, while observed environment details change. Label that distinction within the reference. A different update frequency alone does not require another category or a separate file.

## 6. Clear writing and ASD-STE100

This revision uses [asd-ste100.md](asd-ste100.md) as its writing reference. That reference applies STE-inspired guidance to explanatory documents. It does not provide the official dictionary of approved words.

ASD-STE100 Simplified Technical English includes writing rules and an approved dictionary. See the [official explanation](https://www.asd-ste100.org/STE_faq.html).

Use the writing guidance when it fits the document and audience:

- Use direct instructions.
- Limit instruction sentences to 20 words and description sentences to 25 words.
- Keep one topic per paragraph.
- Use a consistent term for each concept.
- Explain necessary technical terms.
- Separate observations from assumptions and instructions.
- Remove wording that adds no meaning.
- Preserve conditions, uncertainty, and technical meaning when you rewrite.

These practices alone do not establish ASD-STE100 compliance. This guide does not claim compliance. Check the applicable rules and vocabulary against the standard before you claim compliance. Use appropriate review. The [official guidance on AI and STE](https://www.asd-ste100.org/STE_downloads.html) cautions against treating plausible AI output as verified compliance.

## 7. Maintain the context

1. Before work, read the current state.
2. Read the applicable agreements.
3. During work, distinguish plans, assumptions, and observations.
4. After meaningful work, retain useful evidence.
5. Update the current state with actual status, unresolved issues, and one next action.
6. When an agreement changes, update the document that contains it.
7. Update any status or evidence records that the change affects.

Check the structure with three questions:

- Can a new participant understand the purpose?
- Can that participant resume work?
- Can that participant find the evidence for important conclusions without searching through many fragments?
