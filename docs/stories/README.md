# Stories

This directory holds **stories** — human-authored narratives of how milkie is
used to accomplish something at its public API surface. Each story is the
authoritative source of intent for a coherent usage scenario; the E2E test
that bears the same id is its executable verification.

This README is the convention for the stories system itself. Treat it as
the rulebook: if a story file disagrees with this README, the README wins
until updated.

## What a story is

- A **narrative** of one coherent end-to-end usage scenario, written in
  prose with enumerable acceptance criteria
- A **human-led** artifact — product, design, or engineering writes it
  before tests are written; tests follow the story, not the other way
- **One scenario per file** — each story is self-contained

## What a story is not

- Not an API reference — see `docs/en/guide.md` / `docs/zh/guide.md`
- Not a test — see the matching file under `tests/e2e/`
- Not an architecture decision — see `ARCHITECTURE.md`
- Not a subsystem invariant — see "Invariants vs Scenarios" below
- Not a release note or changelog

## Directory layout

```
docs/stories/
  README.md                                 ← this file
  INDEX.md                                  ← projected views
  s-001-react-with-tools.md
  s-002-multi-turn-resume.md
  ...
tests/e2e/
  s-001-react-with-tools.e2e.test.ts        ← same id and slug
  s-002-multi-turn-resume.e2e.test.ts
  ...
```

The story file and its E2E test share **the same id and the same slug**.
This is the linking mechanism; no tool is required to follow it.

## Naming

- Format: `s-NNN-kebab-slug.md`
- `NNN` is zero-padded, monotonically increasing, never reused
- The slug is a short, lowercase, hyphen-separated phrase that names the
  scenario; it should still make sense a year from now
- Do not rename a story's id or slug after `status: active` — downstream
  references (test files, INDEX, links from design docs) depend on them

## Frontmatter schema

```yaml
---
id: s-001                       # required, matches filename prefix
title: ReAct agent with intra-agent parallel tools
                                # required, one-line human title
status: active                  # required: draft | active | deprecated
kind: scenario                  # required: scenario | invariant
                                # (invariants are exceptional — see below)
capability: plan-and-act        # required, coarse capability tag
                                # one tag per story; primary axis of classification
subsystems:                     # optional, one or more milkie-internal subsystems
  - agent-runtime
  - agent-trace
requires:                       # required, target capabilities the story exercises
  - FSM Core                    # entries should map to ARCHITECTURE.md's
  - working context             # Implementation Status vocabulary
  - Direct LLM/tool execution
  - Trajectory observability
owner: "@xupeng"                # required, GitHub handle or team
created: 2026-05-23             # required, ISO date
tests:                          # required if status != draft
  - tests/e2e/s-001-react-with-intra-agent-parallel-tools.e2e.test.ts
related:                        # optional, links to other artifacts
  - ARCHITECTURE.md#agent-runtime
  - docs/superpowers/specs/2026-05-16-agent-system-design.md
deprecated_reason: ""           # required iff status == deprecated
---
```

**`requires` lists the target capabilities a story exercises.** Each entry
should map to an entry in `ARCHITECTURE.md`'s `## Implementation Status`
section (use that vocabulary verbatim). A story is **ready** to move from
`draft` to `active` only when every entry in its `requires` list appears in
the "Implemented today" section of `ARCHITECTURE.md`. Until then it is
**partial** (some implemented, some target) or **blocked** (none yet).

This is enforced by convention, not by tooling — the `requires` field is
the source of truth for readiness, and `INDEX.md` derives the readiness
view from it.

**`capability` is the primary classification axis.** One tag per story.
Use short kebab-case slugs that describe the user-visible capability
(`plan-and-act`, `observability`, `interrupt-resume`, `multi-state-fsm`,
etc.). Two stories should share a `capability` only if they really
are two facets of the same capability.

**`subsystems` is optional** and lists only **milkie-internal subsystems**:
`agent-runtime`, `agent-trace`, `evolution`. Most stories touch
`agent-runtime` and `agent-trace` so the field rarely discriminates;
fill it when it adds signal (e.g. an `agent-trace`-only story, or an
`evolution`-touching story), skip it when it would just be noise.

Infrastructure layers (Context Layer, Data, Execution, Foundation) are
outside milkie's scope per `ARCHITECTURE.md` and **never appear here**.
A story may *exercise* an infrastructure layer (e.g. checkpoint to a
Redis-backed state store), but the story is still about an internal
subsystem's user-visible behavior; the infrastructure layer is a
dependency, not the subject.

## Status lifecycle

```
draft ────► active ────► deprecated
```

- `draft` — story written, may still change shape, E2E test absent or
  red. Free to edit liberally.
- `active` — E2E test exists and is green; the system supports this
  scenario. Edits must keep id/slug stable; substantive narrative
  changes require updating the test in the same PR.
- `deprecated` — the capability has been removed or replaced. The story
  file is kept for history with `deprecated_reason` filled in; the
  matching E2E test should be removed or skipped in the same PR.

A story never moves backwards. A bug in `active` is fixed in code; a
narrative error is corrected in place. If a scenario is genuinely
superseded, deprecate the old story and open a new one with a new id.

## Writing style

- **Free narrative + enumerable acceptance criteria.** Not Gherkin.
  Acceptance criteria must be a checklist of testable assertions; the
  matching E2E test asserts each item.
- Suggested section order:
  1. **场景叙事 / Narrative** — what the user is trying to do and why
  2. **关键交互流 / Interaction flow** — ordered steps, including FSM
     turns, tool calls, parallel branches, sub-agent spawns
  3. **验收准则 / Acceptance criteria** — bullet list of testable claims
  4. **不在此 story 范围 / Out of scope** — explicit exclusions with
     pointers to related stories
- Prefer the codebase's existing language (Chinese is acceptable; mix
  English terms where they match identifiers).
- Avoid duplicating test code in the story body. Stories describe
  intent, not implementation.

## Granularity

- **One story = one coherent user-facing capability = one E2E test
  file.**
- If a scenario splits naturally into two independently testable user
  outcomes, write two stories.
- If a scenario only makes sense as a chain (e.g. "interrupt then
  resume"), keep it in one story; the E2E test may have multiple
  `it(...)` blocks but lives in one file.

## Invariants vs Scenarios

Most stories are **scenarios** (user-visible behaviors). Some assertions
are **invariants** — subsystem-level hard contracts that have no
user-visible scenario but must always hold (e.g. "Agent Trace replay
produces identical state for the same log"). These do **not** belong in
`docs/stories/` as stories.

Place invariants in the corresponding subsystem's design doc under
`docs/design/{subsystem}.md` (to be created) or, for now, in
`ARCHITECTURE.md`'s invariant list. Only use `kind: invariant` in a
story file when an invariant is verifiable through a narrated scenario;
otherwise it is not a story.

## INDEX.md rules

`INDEX.md` is a projection over the stories in this directory. It
lists every non-deleted story with id, title, status, subsystems, and
matching test file(s). Multiple views (by status, by subsystem, by
capability) may be rendered in the same file.

The index is allowed to be regenerated from frontmatter. Hand-edits
between regenerations are tolerable but not durable. A small script
(future work) should be able to rebuild it from the directory.

## Adding a new story

1. Pick the next free `s-NNN`.
2. Create `docs/stories/s-NNN-<slug>.md` with `status: draft`.
3. Write narrative + acceptance criteria. No test yet required.
4. Open a PR to gather feedback on the story shape.
5. When the story stabilizes, write the matching E2E test under
   `tests/e2e/s-NNN-<slug>.e2e.test.ts`.
6. When the test goes green in CI, flip `status: active` and update
   `INDEX.md`.

## Deprecating a story

1. Set `status: deprecated` and fill in `deprecated_reason`.
2. Remove or `it.skip` the matching E2E test.
3. Update `INDEX.md`.
4. Do not delete the file — history matters.

## Relationship to existing spec docs

`docs/superpowers/specs/2026-05-16-agent-e2e-scenarios.md` predates this
system. Its scenarios will be migrated into individual story files over
time; the spec doc may then be retired or kept as a historical
testing-strategy note. Until migration completes, both formats coexist;
new scenarios should use the story format from the start.
