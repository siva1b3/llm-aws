# Four-Pass Walkthrough — Synthetic Example

Every record here is illustrative. No command shown here was run on EC2. The timestamps, paths, version labels, authorization, outputs, and statuses are invented solely to demonstrate the format. They must never be used as evidence of real project progress.

## Sequence

| Pass | Action | Outcome | What happens next |
| --- | --- | --- | --- |
| pass_001 | step_01: check the environment | Passed | Begin the CLI action |
| pass_002 | step_02: implement the CLI | Failed: whitespace-only title accepted | Preserve failure; inspect effects; prepare an in-scope repair |
| pass_003 | step_02: repair the CLI | Passed | Begin the storage action |
| pass_004 | step_03: storage, add, and list | Passed | step_04 and step_05 remain pending |

Four passes complete three planned actions because step_02 needs two attempts. This example does not complete the overall task-tracker goal. Completion behavior and final end-to-end verification still remain.

## Read One Pass

1. Read `input.json` to see its objective, participants, authority, expected outcomes, and checks.
2. Read `evidence/checks.txt` to see the illustrative observations.
3. Read `output.json` to see how each check and party outcome maps to evidence.
4. For the failed pass, inspect `state_changes`, `unresolved_items`, and `next_action` before opening the retry.

All records use the real schemas, but set `record_kind` to `example`. Source versions beginning `example-` are scenario labels, not real commits. The authorization is a simulated instruction stated here: “For this example only, imagine the human authorized steps 01–03, isolated checks, and up to three in-scope attempts per revision.” This sentence grants no real execution permission.

Paths in evidence files refer to an imagined EC2 environment. Relative evidence paths in output files resolve within the corresponding example pass. Source `protected/` paths refer to the real package's governing documents; example records do not mutate them.

## Decisions Demonstrated

- The failed CLI pass leaves a code file behind. A failed status does not mean no changes occurred.
- The retry stays on action revision 1 because the required behavior and scope are unchanged; only the faulty implementation changes.
- The retry references the prior failed pass and reruns all CLI checks.
- Each successful check has an evidence reference; a success summary alone is insufficient.
- The separate real `passes/` directory remains empty and the real plan remains proposed.
