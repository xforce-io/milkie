# Stories Index

Projected view over every story in this directory. Conventions live in
`README.md`. A regenerator script may overwrite the tables below; the
**Notes** section is hand-maintained.

Last updated: 2026-05-24 (Phase 2: basic Agent Trace event log implemented; cache/replay/fork still target)

## By id

| ID | Title | Status | Subsystems | Test |
|----|-------|--------|------------|------|
| [s-001](./s-001-react-with-intra-agent-parallel-tools.md) | ReAct agent with intra-agent parallel tools (plan-and-act) | draft | agent-runtime · agent-trace | `tests/e2e/s-001-react-with-intra-agent-parallel-tools.e2e.test.ts` |
| [s-002](./s-002-inspect-a-completed-run.md) | Inspect a completed agent run | draft | agent-trace | `tests/e2e/s-002-inspect-a-completed-run.e2e.test.ts` |
| [s-003](./s-003-explain-a-decision-with-context.md) | Explain an agent decision with its full context | draft | agent-trace | `tests/e2e/s-003-explain-a-decision-with-context.e2e.test.ts` |
| [s-004](./s-004-lineage-from-artifact-to-source.md) | Trace lineage from an artifact back to its source | draft | agent-trace | `tests/e2e/s-004-lineage-from-artifact-to-source.e2e.test.ts` |
| [s-005](./s-005-deterministic-replay.md) | Deterministically replay a recorded agent run | draft | agent-trace · agent-runtime | `tests/e2e/s-005-deterministic-replay.e2e.test.ts` |
| [s-006](./s-006-fork-at-event-for-what-if.md) | Fork a run at an event to explore a counterfactual | draft | agent-trace · agent-runtime | `tests/e2e/s-006-fork-at-event-for-what-if.e2e.test.ts` |
| [s-007](./s-007-inter-agent-parallel-code-review.md) | Inter-agent parallel via named sub-agent tools | draft | agent-runtime · agent-trace | `tests/e2e/s-007-inter-agent-parallel-code-review.e2e.test.ts` |
| [s-008](./s-008-long-task-interrupt-and-resume.md) | Interrupt a long-running agent and resume from checkpoint | draft | agent-runtime · agent-trace | `tests/e2e/s-008-long-task-interrupt-and-resume.e2e.test.ts` |
| [s-009](./s-009-multi-turn-with-tool-error-recovery.md) | Multi-turn conversation with tool error recovery | draft | agent-runtime · agent-trace | `tests/e2e/s-009-multi-turn-with-tool-error-recovery.e2e.test.ts` |
| [s-010](./s-010-skill-versioned-load-and-ab-experiment.md) | Skill loaded at epoch boundary, A/B experiment on skill version | draft | agent-runtime · agent-trace · evolution | `tests/e2e/s-010-skill-versioned-load-and-ab-experiment.e2e.test.ts` |
| [s-011](./s-011-multi-state-fsm-intent-routing-and-slot-filling.md) | Multi-state FSM with intent routing, slot filling, and escalation | draft | agent-runtime · agent-trace | `tests/e2e/s-011-multi-state-fsm-intent-routing-and-slot-filling.e2e.test.ts` |

## By implementation readiness

Derived from each story's `requires:` field cross-referenced with
`ARCHITECTURE.md`'s `## Implementation Status` section. A story is **ready**
when every entry in its `requires:` appears under "Implemented today",
**partial** when some are implemented and some are target only, and
**blocked** when none are implemented yet.

### Ready (all requires implemented today)

These can move from `draft` to `active` as soon as someone writes the E2E
test. Implementation gates are clear.

- **s-001** ReAct + intra-agent parallel — needs only FSM Core, working context, LLM/tool execution, Trajectory observability
- **s-007** Inter-agent parallel — adds Sub-agent as named tool (assumed implemented; verify in code)
- **s-009** Multi-turn + error recovery — adds Error handling FSM transition (verify)
- **s-011** Multi-state FSM intent routing — adds Action state with ctx.emit (verify)

### Partial (some requires implemented, some target only)

These work today in degraded / basic form; full story validation requires
target capabilities to land.

- **s-002** Inspect a run — basic timeline via Trajectory observability ✓; rich view needs Event-sourced Agent Trace event log
- **s-003** Explain a decision — span data exists ✓; full working context snapshot needs Event-sourced log
- **s-008** Interrupt + resume — FSM Core ✓, State stores ✓; Yield point + interrupt signal and Supervisor tree need verification
- **s-010** Skill load + A/B — FSM Core ✓, working context ✓; Skill epoch loading needs verification; Evolution components are target only

### Blocked (all or most requires are target only)

Cannot be validated until target infrastructure lands. Story content stands
as design specification.

- **s-004** Lineage — Event-sourced log ✓ basic recording exists; still needs Lineage query API (causedBy graph traversal + lineage projection)
- **s-005** Deterministic replay — IOPort ✓, Event-sourced log ✓ (basic); still needs Response cache, Non-determinism log, Replay engine
- **s-006** Fork at event — IOPort ✓, Event-sourced log ✓ (basic); still needs Response cache, Fork primitive

## By status

### draft

- s-001 through s-011 (all 11 stories)

### active

_(none yet — moving to active requires E2E test for Ready stories)_

### deprecated

_(none yet)_

## By subsystem

Subsystems are **milkie-internal only**: `agent-runtime`, `agent-trace`,
`evolution`. Infrastructure layers (Context Layer, Data, Execution,
Foundation) are outside milkie's scope per `ARCHITECTURE.md` and never
appear here.

### agent-runtime

- s-001, s-005, s-006, s-007, s-008, s-009, s-010, s-011

### agent-trace

- s-001 through s-011 (every story)

### evolution

- s-010

## By capability

| capability | stories |
|---|---|
| plan-and-act | s-001 |
| observability | s-002 |
| explainability | s-003 |
| lineage | s-004 |
| replay | s-005 |
| fork | s-006 |
| inter-agent-parallel | s-007 |
| interrupt-resume | s-008 |
| multi-turn-with-error-handling | s-009 |
| skill-load-and-ab | s-010 |
| multi-state-fsm | s-011 |

## Notes

- All 11 stories are `draft`. Readiness varies — see the "By implementation readiness" view above.
- The first wave of likely E2E test writing should focus on the four **Ready** stories (s-001, s-007, s-009, s-011), then **Partial** stories at the level they're implementable today (s-008 most likely next).
- Several migrated stories (s-009, s-010, s-011) carry **internal sub-scenarios** that may be split per the README's granularity rule after discussion. Flagged inside each story.
- Migration source: `docs/superpowers/specs/2026-05-16-agent-e2e-scenarios.md` (Cases 2–6 ported as s-007 to s-011).
- The "Test" path column reserves filenames per the convention; matching E2E test files may not yet exist while stories are in `draft`.
- **Verify-in-code TODOs**: The Ready / Partial classifications above are based on documentation and earlier conversation; some require source verification:
  - Sub-agent as named tool (s-007, s-011)
  - Error handling FSM transition (s-009)
  - Action state with ctx.emit (s-011)
  - Yield point + interrupt signal (s-008)
  - Skill epoch loading (s-010)
- When code lands closing a target capability, update `ARCHITECTURE.md`'s Implementation Status first, then this index's readiness view will need to be regenerated.
