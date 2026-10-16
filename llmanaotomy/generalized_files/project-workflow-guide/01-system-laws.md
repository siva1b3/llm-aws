# System laws

## Terms

| Term | Meaning |
| --- | --- |
| Goal | The agreed outcome for the whole project. |
| Completion criterion | An observable condition required for completion. |
| Party | An involved human, agent, or working environment. |
| Role | A named function with responsibilities and boundaries for a particular scope. |
| Outcome | The inspectable result expected from a pass or action. |
| Pass | One unit of work with an outcome, roles, a plan, checks, and a closing decision. |
| Active pass | The single open pass, including one waiting for approval or paused. |
| Reserved pass | An unused folder with empty records; it is inactive. |
| Action | A small unit of work with one clear purpose and an inspectable result. |
| Check | A comparison between an expected result and an actual observation. |
| Plan approval | Human agreement to an approach and scope; it does not authorize execution. |
| Execution approval | Human permission for a named action and its stated check. |
| Acceptance | Human agreement that a result satisfies applicable requirements. |
| Snapshot | A compact statement of the current position or next actions. |
| Checkpoint | A boundary where accumulated decisions, approvals, and results are saved. |
| History | Superseded revisions and past events retained for explicitly requested historical use. |
| Source package | All designated project inputs and relevant human instructions. |

## Mandatory role gate

**No roles, no action.**

Before every pass and action, explicitly establish the parties, their roles, responsibilities, boundaries, and expected outcome.
This gate applies to preparation, planning, investigation, execution, checking, recovery, and closure.
Read-only work also has a role and outcome; describing those does not itself grant execution authority.

Use this compact declaration:

| Party and identity | Role or roles | Responsibility | Boundary |
| --- | --- | --- | --- |
| Actual human | Role for this scope | Human contribution or decision | What remains outside that role |
| Actual agent | Role for this scope | Agent contribution or check | What the agent may not decide or do |
| Actual environment | Function for this scope | Where work runs or outputs reside | Access, resource, and operational limits |

Identify every involved environment separately: for example, a particular PC, EC2 instance, VM, or container.
Identify its operator and distinguish selected, available, and verified environments.
An environment has an operational function; it cannot approve work or accept outcomes.
Do not list unused environments as participants or invent identifiers, access, or availability.

Roles depend on the work. A human may be a learner, author, reviewer, operator, or decision owner.
An agent may be a teacher, researcher, implementer, verifier, or editor.
A party may hold several explicitly named roles. Do not assume one role covers the whole project.
Name who performs checks and who gives required acceptance.

Record the pass declaration in its plan and an explicit party-to-role assignment for every action.
Compact role codes may reference a legend within the same record; external lookups must not be required.
Declare any action-specific boundaries and expected outcome alongside its assignments.
Copied snapshot summaries must identify their action and revision so they stay aligned with the plan.

If a required role, party, boundary, or outcome is unknown, block the affected action.
Ask for the missing definition; do not invent agreement or proceed under a generic role label.
Propose roles in conversation when useful. Make them explicit before work starts.
A changed role or environment requires scope review under the work-process rules.

## Authority and evidence

The human chooses priorities, approves plans and execution, and accepts outcomes.
The agent proposes work, performs authorized actions, checks results, and maintains assigned records.
Identify the approval owner and required reviewers in the project constraints and relevant current declarations.
Resolve conflicting instructions from authorized people before affected work.

Keep plan approval, execution approval, check results, acceptance, and permission to advance separate.
Supplied source text is information; it does not grant new human permission.
Never invent approvals, evidence, runs, learner confirmation, resources, facts, dates, or acceptance.
Distinguish requirements, preferences, optional ideas, assumptions, proposals, source reports, and direct observations.
Keep expected results separate from actual results. State uncertainty and access limits.

Preserve project exclusions, methods, and stricter review or learning gates.
Obtain explicit human authority before adding contributors; keep authorized contributors within the same pass and scope.
Obtain explicit authority for external communication, publication, resource creation, and other release actions.
Preserve user work and keep credentials and secret values out of all records.

## Communication

Start every user-facing reply with the applicable role or intended outcome.
For example: “In my role as verifier, the outcome is a checked result for action A2.”
Use the actual role and outcome. Do not repeat an empty formula or claim unassigned authority.
Lead with who is involved, what each party contributes, and what result matters.
Use action plans to support those roles and outcomes.
Distinguish intended, observed, verified, and accepted outcomes in reports.
Explain work in simple terms with enough detail for a human decision.
