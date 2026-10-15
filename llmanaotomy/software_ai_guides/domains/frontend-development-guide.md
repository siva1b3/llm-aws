# Frontend development guide

## 1. Purpose and authority

Use this guide for browser applications, websites, and user-facing web interfaces.
It owns frontend behavior, state, accessibility, browser integration, and rendering considerations.

Follow the [core workflow](../core/software-ai-development-workflow.md) for scope, approvals, records, and completion.
Use [architecture and design](../core/architecture-and-design-guide.md) for HLD and LLD.
Use [engineering quality](../core/engineering-quality-guide.md) for shared coding, security, testing, and review requirements.
This guide adds technical detail. It does not authorize work or create additional mandatory project files.

### Apply the version 3 workflow

Use the core's conversation, approval, action, check, and checkpoint sequence for engineering work.
Draft and revise plans in conversation without changing Markdown during discussion.
Obtain plan approval, then explain the next action, its separate check, limits, and stopping point.
Obtain final execution approval for that action and check before acting.
Save the agreed plan and first execution approval at the core's plan checkpoint before execution.
Keep reserved pass records empty and inactive until that checkpoint.

Divide journey investigation, design, component implementation, integration, verification, and release into small actions with inspectable results.
Keep action and verification procedures distinct, including separate executable artifacts when applicable.
Report each result and obtain execution approval before dependent work or the next action.
Do not create an unattended procedure that crosses action approval or required human review boundaries.
Retain valid approvals for unchanged scope and preserve stricter project-specific learning or review gates.

Save management records at the core's defined checkpoints, including failures, blockers, withdrawal, and substantive session handovers.
Keep routine results and subsequent approvals in conversation between checkpoints.
Preserve partial effects, running operations, unknown outcomes, accumulated approvals, and one next action at checkpoints.
Check unknown outcomes within applicable authority before retries that could duplicate effects.
Obtain new approval for recovery unless the approved scope explicitly covers that recovery branch and its check.

Interface state, autosave, event handling, and product confirmation flows follow product requirements independently of project Markdown checkpoint timing.

### Transfer guidance during record preparation

Read all designated sources, including design references, content requirements, browser-support notes, review reports, instruction files, and status records.
Transfer applicable requirements into generated definitions and operating instructions under the core's ownership rules.
Preserve supplied user journeys, branding, selected components, API contracts, supported browsers, accessibility requirements, and review obligations.
Keep selected configuration distinct from observed configuration, and source-reported results distinct from newly verified facts.
Record unresolved choices as focused questions; do not invent frameworks, design systems, devices, performance targets, or reviewers.

Preparation does not authorize browser investigation, new mockups or design plans, application implementation, builds, or publication.
Keep preparation checks in current state, reserved pass records at zero bytes, and the final report empty until closure.
Transfer necessary guidance into internal owners so future operation requires neither this guide nor other preparation inputs.
Use internal relative links; keep preparation provenance in the overview's historical note.

## 2. Related guides

| Guide | Use when |
| --- | --- |
| [Backend and APIs](backend-api-development-guide.md) | The interface communicates with a backend. |
| [LLM engineering](llm-engineering-guide.md) | The interface presents generated content or model-driven actions. |
| [Infrastructure and operations](infrastructure-deployment-operations-guide.md) | The frontend is built, hosted, released, or monitored. |

Do not require a backend, framework, or build system solely because this guide mentions one.
Keep API contract ownership explicit. Do not maintain conflicting client and server specifications.

## 3. Define the user experience

Identify users, tasks, supported devices, browsers, and input methods.
Record supplied branding, content, localization, accessibility, and performance requirements.
Separate mandatory behavior from visual preferences and optional enhancements.
Do not invent a design system or replace an established one without an agreed reason.

For each important journey, define:

- Entry point, user intent, actions, and successful outcome.
- Loading, empty, partial, error, permission-denied, and completed states.
- Validation, cancellation, retry, and recovery behavior.
- Navigation, deep links, refresh, and back-button behavior.
- Responsive behavior and keyboard interaction.

Use observable acceptance criteria. Avoid criteria such as "looks modern" without explaining the expected result.

## 4. HLD considerations

Describe routes, major views, shared components, and external integrations.
Identify which behavior runs in the browser and which runs on the server.
Explain rendering and caching choices when they affect discoverability, performance, security, or operation.
Identify state owners and the lifetime of local, shared, URL, and server-provided state.
Define authentication integration without treating the browser as the authorization authority.

Choose technologies from project requirements and existing constraints.
Do not introduce global state or a new component library for a local problem without a reason.

## 5. LLD considerations

Specify component inputs, outputs, events, and side effects.
Define forms, field validation, error placement, and submission behavior.
Keep display formatting separate from authoritative data values where necessary.
Specify asynchronous request cancellation and stale-response handling.
Prevent repeated submissions where they can cause duplicate effects.
Coordinate server-side idempotency with the backend when disabling a button is insufficient.

Define cache invalidation, optimistic updates, and recovery from rejected updates.
Make URL state and browser storage behavior explicit when used.
Avoid storing sensitive information in browser storage without an assessed need and appropriate controls.
Define cleanup for subscriptions, timers, listeners, and long-running streams.

## 6. Accessibility and content

Prefer semantic controls and meaningful document structure.
Provide accessible names, labels, and useful error descriptions.
Preserve visible focus and logical keyboard navigation.
Manage focus after navigation, dialogs, and significant state changes.
Do not communicate essential information through color alone.
Account for zoom, text resizing, reduced motion, and assistive technology where applicable.

Identify the project's accessibility target and required human review.
Use applicable current standards when assessing a named compliance requirement.
Automated checks alone do not establish complete accessibility.

Keep content accurate, concise, and consistent with user actions.
Explain failures with a useful next step. Do not expose internal stack traces or secrets.

## 7. Security and browser boundaries

Treat API responses, URL parameters, uploaded content, and generated text as untrusted input.
Use destination-appropriate output handling. Avoid unsafe HTML insertion.
Define session expiration and access-denied behavior with the backend.
Keep private credentials out of browser bundles and public configuration.
Coordinate cookie, request-forgery, and cross-origin controls with the actual backend architecture.
Validate destinations used for navigation or external links when they originate from untrusted content.

For LLM interfaces, separate generated suggestions from confirmed actions.
Render generated content safely and make uncertainty visible where the user needs it.

## 8. Performance and reliability

Agree on representative pages, devices, networks, and user journeys before measuring.
Assess bundle loading, image delivery, rendering, request waterfalls, and repeated work where relevant.
Use loading indicators that reflect actual progress rather than invented percentages.
Handle slow or failed dependencies without trapping the user.
Bound large lists, uploads, and streaming output according to the agreed experience.
Avoid optimization that makes the code harder to maintain without measured benefit.

## 9. Verification

Select applicable checks from the quality guide and discuss expected results in conversation during planning.
Save agreed expectations as `not run` at the core's plan checkpoint before execution.
Explain the affected journey, separate check, browser conditions, limits, and stopping point when requesting execution approval.
Keep implementation and checking procedures distinct, including executable artifacts when applicable.

| Area | Evidence to consider |
| --- | --- |
| Behavior | Critical journeys, forms, navigation, validation, and recovery. |
| Integration | Actual API contracts, authentication states, cancellation, and stale responses. |
| Visual result | Relevant viewport sizes, overflow, readable content, and agreed design comparisons. |
| Accessibility | Automated checks plus required keyboard and assistive-technology review. |
| Compatibility | Supported browser and device conditions. |
| Performance | Representative loading and interaction measurements. |
| Security | Unsafe content handling, secret exposure, and permission boundaries. |

Screenshots support appearance claims. They do not establish correct application behavior.
Record untested browser or accessibility conditions as limitations.
Save tested output versions, browser conditions, observations, evidence, and limitations at the applicable core checkpoint.
Keep routine results in conversation between checkpoints and save failures promptly.
Passing technical checks does not establish design acceptance or completion of required human accessibility or learner review.

## 10. Release and maintenance

Use the infrastructure guide for builds, hosting, configuration, rollout, and recovery.
Check deployed routes, asset paths, cache behavior, and relevant backend connectivity.
Verify that old cached clients remain compatible where releases can overlap.
Provide source maps and diagnostics only through the project's agreed exposure policy.
Update affected component documentation, contracts, tests, and design notes within their approved action scope.
Report each action and check result, then obtain execution approval before the next action.
Resolve or explicitly transfer control of running builds and publishing operations before pass closure.
Close work under the core workflow after applicable checks and required reviews.
