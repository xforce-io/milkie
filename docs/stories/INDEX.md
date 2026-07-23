# Stories Index

Projected view over every story in this directory. Conventions live in
`README.md`. A regenerator script may overwrite the tables below; the
**Notes** section is hand-maintained.

Last updated: 2026-07-23 (#214 minimal learning loop: s-016 task-outcome draft;
s-006 title/narrative → current-task repair; pointer Out/related on
s-002/003/005/009/010)

## By id

| ID | Title | Status | Subsystems | Test |
|----|-------|--------|------------|------|
| [s-001](./s-001-react-with-intra-agent-parallel-tools.md) | ReAct agent with intra-agent parallel tools (plan-and-act) | active | agent-runtime · agent-trace | `tests/e2e/s-001-react-with-intra-agent-parallel-tools.e2e.test.ts` |
| [s-002](./s-002-inspect-a-completed-run.md) | Inspect a completed agent run | active | agent-trace | `tests/e2e/s-002-inspect-a-completed-run.e2e.test.ts` |
| [s-003](./s-003-explain-a-decision-with-context.md) | Explain an agent decision with its full context | active | agent-trace | `tests/e2e/s-003-explain-a-decision-with-context.e2e.test.ts` |
| [s-004](./s-004-lineage-from-artifact-to-source.md) | Trace lineage from an artifact back to its source | draft | agent-trace | `tests/e2e/s-004-lineage-from-artifact-to-source.e2e.test.ts` |
| [s-005](./s-005-deterministic-replay.md) | Deterministically replay a recorded agent run | active | agent-trace · agent-runtime | `tests/e2e/s-005-deterministic-replay.e2e.test.ts` |
| [s-006](./s-006-fork-at-event-for-what-if.md) | Fork a failed run at an event to repair the current task | draft | agent-trace · agent-runtime | `tests/e2e/s-006-fork-at-event-for-what-if.e2e.test.ts` |
| [s-007](./s-007-inter-agent-parallel-code-review.md) | Inter-agent parallel via named sub-agent tools | active | agent-runtime · agent-trace | `tests/e2e/s-007-inter-agent-parallel-code-review.e2e.test.ts` |
| [s-008](./s-008-long-task-interrupt-and-resume.md) | Interrupt a long-running agent and resume from checkpoint | active | agent-runtime · agent-trace | `tests/e2e/s-008-long-task-interrupt-and-resume.e2e.test.ts` |
| [s-009](./s-009-multi-turn-with-tool-error-recovery.md) | Multi-turn conversation with tool error recovery | active | agent-runtime · agent-trace | `tests/e2e/s-009-multi-turn-with-tool-error-recovery.e2e.test.ts` |
| [s-010](./s-010-skill-versioned-load-and-ab-experiment.md) | Skill loaded at epoch boundary, A/B experiment on skill version | draft | agent-runtime · agent-trace · evolution | `tests/e2e/s-010-skill-versioned-load-and-ab-experiment.e2e.test.ts` |
| [s-011](./s-011-multi-state-fsm-intent-routing-and-slot-filling.md) | Multi-state FSM with intent routing, slot filling, and escalation | active | agent-runtime · agent-trace | `tests/e2e/s-011-multi-state-fsm-intent-routing-and-slot-filling.e2e.test.ts` |
| [s-012](./s-012-batch-replay-suite-and-classify-divergences.md) | Batch replay a saved suite and classify divergences | draft | agent-trace | `tests/e2e/s-012-batch-replay-suite-and-classify-divergences.e2e.test.ts` |
| [s-013](./s-013-variant-search-with-bounded-cost.md) | Variant search with bounded amortized cost | draft | agent-trace · evolution | `tests/e2e/s-013-variant-search-with-bounded-cost.e2e.test.ts` |
| [s-014](./s-014-reverse-reference-lineage-query.md) | Reverse-reference lineage query | draft | agent-trace | `tests/e2e/s-014-reverse-reference-lineage-query.e2e.test.ts` |
| [s-015](./s-015-subagent-reads-parent-trace-runtime.md) | Sub-agent reads parent's in-flight trace at runtime | draft | agent-runtime · agent-trace | `tests/e2e/s-015-subagent-reads-parent-trace-runtime.e2e.test.ts` |
| [s-016](./s-016-record-and-query-task-outcome.md) | Record and query task outcome after a run | draft | agent-trace | `tests/e2e/s-016-record-and-query-task-outcome.e2e.test.ts` |

## By implementation readiness

Derived from each story's `requires:` field cross-referenced with
`ARCHITECTURE.md`'s `## Implementation Status` section. A story is **ready**
when every entry in its `requires:` appears under "Implemented today",
**partial** when some are implemented and some are target only, and
**blocked** when none are implemented yet.

### Ready (all requires implemented today)

These can move from `draft` to `active` as soon as the matching E2E test
exists and is green. Implementation gates are clear.

- **s-001** ReAct + intra-agent parallel — FSM Core ✓, working context ✓, LLM/tool execution ✓, Trajectory observability ✓ (active)
- **s-002** Inspect a completed run — Trajectory observability ✓, Agent Trace event log ✓ (active)
- **s-003** Explain a decision with context — Trajectory observability ✓, Agent Trace event log ✓ (active)
- **s-007** Inter-agent parallel — Sub-agent as named tool ✓ confirmed in `AgentRuntime.ts:116-198` via AgentFactory (active)
- **s-009** Multi-turn + error recovery — Error handling FSM transition ✓ confirmed in `FSMEngine.ts:11,64-66` (active)
- **s-011** Multi-state FSM intent routing — Action state + `ctx.emit` ✓ confirmed in `types/agent.ts`, `FSMEngine.ts:44-49`, `AgentRuntime.ts:223` (active)

### Partial (some requires implemented, some target only)

These work today in degraded / basic form; full story validation requires
target capabilities to land.

- **s-005** Deterministic replay — IOPort ✓, Event log ✓, Cache ✓, structural Replay ✓ (Phase 3); byte-identical replay still needs Non-determinism log (Phase 4)
- **s-006** Fork at event (current-task repair) — IOPort ✓, Event log ✓, Response cache ✓; still needs Fork primitive (Phase 5)
- **s-008** Interrupt + resume — FSM Core ✓, State stores ✓, Yield point + interrupt signal ✓ confirmed in `FSMEngine.ts:11,61-62` and `Milkie.ts:297-300` (test file exists)
- **s-010** Skill load + A/B — FSM Core ✓, working context ✓, Skill epoch loading ✓ confirmed in `ContextLayer.ts:19,40-46` (test file exists); Evolution components are target only; experiment metrics should prefer task outcome (s-016) when available
- **s-012** Batch replay + classify — Replay ✓, Cache ✓; still needs Structural diff, Suite definition + batch replay (Phase 5)
- **s-013** Variant search bounded cost — Cache ✓; still needs Fork primitive, Structural diff (Phase 5)
- **s-016** Task outcome record/query — Event log basic ✓; still needs Task outcome record API/storage (target)

### Blocked (all or most requires are target only)

Cannot be validated until target infrastructure lands. Story content stands
as design specification.

- **s-004** Lineage — Event log ✓ basic recording exists; still needs Lineage-by-typed-relations (`object.created` / `relation.created` events + graph traversal). Phase 6+
- **s-014** Reverse-reference lineage — Event log ✓; still needs Lineage-by-typed-relations (reverse direction). Phase 6+
- **s-015** Sub-agent reads in-flight trace — Event log ✓, Sub-agent as named tool ✓; still needs In-flight trace query API with stable contract over in-flight + completed runs (Phase 5)

## By status

### draft

- s-004, s-006, s-010, s-012, s-013, s-014, s-015, **s-016** (8 stories)

### active

- **s-001** ReAct + intra-agent parallel
- **s-002** Inspect a completed run (hermetic e2e with stub gateway)
- **s-003** Explain a decision with context (hermetic e2e with stub gateway)
- **s-005** Deterministic replay (structural replay implemented; byte-identical pending Phase 4 non-determinism log)
- **s-007** Inter-agent parallel via named sub-agent tools
- **s-008** Interrupt + resume (incl. Supervisor Tree interrupt propagation)
- **s-009** Multi-turn + tool error recovery (uses MemoryStore by default; Redis variant lives separately)
- **s-011** Multi-state FSM intent routing + slot filling

s-010 stays `draft` until its Evolution requires are split (current test only validates skill-epoch loading, not Experiment Registry).
s-016 stays `draft` until Outcome storage/API + e2e land (tracking #215 stage 2).

### deprecated

_(none yet)_

## By subsystem

Subsystems are **milkie-internal only**: `agent-runtime`, `agent-trace`,
`evolution`. Infrastructure layers (Context Layer, Data, Execution,
Foundation) are outside milkie's scope per `ARCHITECTURE.md` and never
appear here.

### agent-runtime

- s-001, s-005, s-006, s-007, s-008, s-009, s-010, s-011, s-015

### agent-trace

- s-001 through s-016 (every story)

### evolution

- s-010, s-013

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
| suite-replay-and-diff | s-012 |
| variant-search | s-013 |
| lineage-reverse-reference | s-014 |
| runtime-trace-consumption | s-015 |
| task-outcome | s-016 |

## Notes

- 16 stories total; 8 `draft`, 8 `active` (s-001 / s-002 / s-003 / s-005 / s-007 / s-008 / s-009 / s-011). Readiness varies — see the "By implementation readiness" view above.
- **#214 (2026-07-23)** aligned stories with ARCH minimal learning loop: new **s-016** (task outcome); **s-006** primary narrative = current-task repair (slug unchanged); pointer Out/related on s-002, s-003, s-005, s-009, s-010. Active acceptance checklists were not rewritten.
- s-012 / s-013 / s-014 / s-015 are **agent-first scenarios** added to mirror ARCHITECTURE.md invariants 12-13 (Agent Trace is agent-first; CLI is the agent-facing protocol facade). They sit alongside the existing single-run / single-consumer stories (s-002–s-006) and cover batch / runtime / reverse-graph patterns.
- The "Test" path column reserves filenames per the convention; matching E2E test files may not yet exist while stories are in `draft`.
- Parent tracking for Outcome → Fork implementation: GitHub #215.
- When code lands closing a target capability, update `ARCHITECTURE.md`'s Implementation Status first, then this index's readiness view will need to be regenerated.
