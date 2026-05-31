# milkie Roadmap

Phased plan for shipping the architecture described in `ARCHITECTURE.md`.
This file is the single source of truth for **both** "what works today"
and "what comes next" — the architecture doc deliberately stays
status-free so it reads as a timeless target. Authoritative scenario
inventory is in `docs/stories/INDEX.md`.

The roadmap moves only when concrete code closes a gap — order of operations
matters more than calendar dates, so phases are listed without dates.

When code lands that closes a gap, update this file (Completed sections,
Migration intentions, deferred items) before claiming the gap is closed
elsewhere.

Last updated: 2026-05-30

---

## TL;DR

- **Phase 1–4 + 4.5 are landed.** FSM Runtime, working context, Trajectory
  observability, IOPort, Agent Trace event log (LLM + tool I/O + lifecycle +
  clock / uuid), content-addressed cache, structural replay, and
  **byte-identical replay via non-determinism log** are all in code. State
  stores (Memory / SQLite / Redis) ship. Sub-agent as named tool, interrupt
  signal, supervisor-tree propagation, skill epoch loading are verified.
  **Phase 4.5 Context Region Substrate** replaced the `ContextLayer` shim
  with `ContextRegions` + pure `assemble()` + lifecycle engine: scratchpad
  and history are now distinct sections (scratchpad turn-local, history =
  `(user, finalAssistant)` pairs built at turn-end crystallization), skills
  declare `scope: 'turn' | 'session'` lifetime, section schema is cache-aware.
- **#73 landed — events are the single source of truth for resume.** Tool
  side-effects on working memory are event-sourced (`wm.mutated` snapshot per
  tool call), so `replay()` reconstructs tool-written WM without re-running
  handlers → cognitive-tool / WM-using runs now replay deterministically (they
  previously diverged). The stateStore **checkpoint blob is logically deleted**:
  resume state lives only in the event log as an `agent.checkpoint` event; the
  stateStore keeps a tiny `context→runId` routing pointer. `CheckpointManager`
  is archived to `.bak/store/`. This supersedes the earlier "stateStore for
  checkpoint" model and unblocks Phase 5 (fork/diff/suite need deterministic
  replay of side-effecting runs).
- **8 of 15 stories are `active`** (have green E2E tests). The 7 `draft`
  stories all depend on Phase 5–6 capabilities that haven't shipped yet.
- **Current active line:** finish the trace projection surface before
  adding heavier UI. Observable.P0 and diagnosable.P0 substrate are both
  complete (`causedBy` #30, guard evidence #31). Next sequencing is
  CLI/query projections (#29, #36) → UI projections (#26–28, #32–35).
  `Trace` remains the canonical event-sourced substrate; UI/CLI are
  projections over it, not separate products.
- **Next big rock:** Phase 5 fork / diff / suite replay. Phase 4 was its
  prerequisite for honest fork semantics — fork can now share recorded
  prefixes byte-for-byte across forks instead of structurally.
- **Invariants 12–13 landed and shipped to code.** Agent Trace is
  **agent-first**; **CLI is the canonical agent-facing protocol facade**.
  CLI surface spec
  (`docs/superpowers/specs/2026-05-24-cli-surface-design.md`) + agent
  registration spec
  (`docs/superpowers/specs/2026-05-24-agent-registration-design.md`)
  both land; `Milkie.loadManifest()` reads `.milkie/agents.json`; the
  `milkie` binary ships all 6 P0 verbs —
  `agent list / run / resume / interrupt`, `trace inspect / replay` —
  defaulting to a persistent SQLite stateStore at `.milkie/state.sqlite`
  so interrupt / resume work across CLI processes. First runnable
  example at `examples/s-005-replay/` proves SDK ↔ CLI parity end-to-end
  with zero LLM calls.
- **Lineage-by-typed-relations is still inside this repo, but must start
  explicit.** Producer code should declare objects/relations; runtime
  records typed events. Do not rebuild citation/provenance by parsing LLM
  text or UI heuristics.
- **Evolution is deferred** — there are open architectural questions before
  code work starts.
- **Reference UI projection — first probe landed.** ARCHITECTURE.md
  promotes UI from "optional" to "deferred reference projection". The
  s-002 static HTML probe is now shipped: `milkie trace report <runId>`
  (with `trace render-html` as the underlying pure-projection primitive
  and `trace inspect --include-children` for sub-agent fan-out) renders
  a completed run as a single self-contained HTML file. Full UI form
  remains TBD until Phase 5 capabilities (fork / diff / suite) land and
  CLI output contracts stabilize.

---

## Completed (Phase 1–4 + 4.5)

- **Phase 1 — FSM Core & working context.** Statechart-based runtime,
  multi-state FSMs, action / llm state types, ctx.emit transitions,
  multi-turn history, working memory, skill epoch loading.
  (`src/runtime/`, `src/fsm/`, `src/context/`)
- **Phase 2 — Trajectory observability + state stores.** Span-based
  TrajectoryStore (`llm.call` / `tool.call` / `fsm.transition` /
  `agent.spawn` / `agent.run`). MemoryStore / SQLiteStore / RedisStore
  for checkpoint / interrupt / resume — all three implement `IStateStore`
  (`src/types/store.ts`); SQLite and Redis require `init()`, MemoryStore
  is constructor-ready. (`src/trajectory/`, `src/store/`)
- **Phase 3 — Agent Trace event log + structural replay.** Append-only
  event log via `IEventStore` (Memory + JSONL implementations).
  `RecordingIOPort` decorator pattern records paired `requested` /
  `responded` events with `causedBy` chains; `agent.run.started` /
  `completed` lifecycle events. `CacheIndex` projects the log into FIFO
  response queues keyed by canonical request hash; `ReplayingIOPort`
  serves cached responses; `Milkie.replay(runId)` produces a structural
  re-run with strict `ReplayDivergenceError` on cache miss. Sub-agent
  as named tool, parent interrupt → supervisor-tree propagation
  (`src/runtime/AgentRuntime.ts:141-186`, `ChildAgentRecord` in
  `src/types/store.ts:45-51`), interrupt-and-resume across stateStores
  all working. IOPort lives at `src/runtime/IOPort.ts`; record/replay
  decorators at `src/trace/RecordingIOPort.ts` /
  `src/trace/ReplayingIOPort.ts`; cache at `src/trace/CacheIndex.ts`;
  replay entry at `src/runtime/Milkie.ts:replay`. Yield/interrupt:
  `paused` reserved FSM state + global `interrupt` event handler
  (`src/fsm/FSMEngine.ts:11,61-62`, `src/runtime/Milkie.ts:297-300`,
  `src/runtime/AgentRuntime.ts:238-242`).
  **Substrate additions on top of Phase 3 (#20 observable.P0, merged):**
  - `fsm.transition` event with explicit `FsmEventDomain` taxonomy
    (lifecycle / signal / runtime-control / business) — #21,
    commit 7857423. Closes one node-class needed by future
    diagnosable causedBy chains (#30 onward).
  - `skill.loaded` / `skill.unloaded` lifecycle events — #22.
  - Region content-addressing: `region.added` carries `contentHash` /
    `renderedHash`, content bytes offloaded to `ITraceObjectStore`
    (`MemoryTraceObjectStore` / `FileTraceObjectStore`, content-addressed
    + dedup) — #23.
  - `agent.spawned` / `agent.returned` anchors on the parent run — #24
    (PR #48). The supervisor tree is now an event, not a runtime-only
    `ChildAgentRecord` reconstruction.
- **Phase 4 — Non-determinism log + byte-identical replay.** New event
  kinds `clock.read` / `uuid.generated`. `RecordingIOPort` records every
  agent-facing `port.now()` / `port.uuid()` call via an internal pending
  buffer flushed at each async method entry (infrastructure-use bypasses
  the buffer to prevent recursion). `ReplayingIOPort` consumes from per-
  runId FIFO queues on `CacheIndex`. `Milkie.replay()` enforces strict
  P-wide under-consume check across clock / uuid / llm / tool (any
  unconsumed event → `ReplayDivergenceError`). `s-005` e2e upgraded
  from structural-only to "Phase 4 byte-identical" (asserts nondet
  events captured + 3× repeat-replay produces identical results);
  example fixtures re-recorded.
- **Phase 4.5 — Context Region Substrate.** Replaced the old `ContextLayer`
  shim with a region-based substrate (PR #6 / #7 / #8 / #9). Every piece of
  working context is now a typed `Region` in `ContextRegions`; LLM requests
  are produced by a pure `assemble(regions, scope)` function with a
  cache-aware section schema (`header` → `persistent-skills` → `tools-static`
  → `session-skills` → `state` → `tools-state` → `wm` → `footer`).
  scratchpad and history are distinct: scratchpad is `turn-local` and holds
  ReAct intermediates; history holds `(user, finalAssistant)` pairs
  produced by turn-end crystallization (`runInterTurnEngine`). Skills
  declare lifetime at request time — `skill_request({ scope: 'turn' | 'session' })`
  with default `'turn'` so the runtime auto-cleans them; there is
  intentionally no `skill_release` tool (per spec §4.3 rationale). Checkpoint
  format changed (BREAKING — pre-substrate checkpoints can't be loaded);
  format functions rehydrated on `loadCheckpoint`; crystallization runs at
  wait-for-user save AND on resume to keep message order chronological.
  Spec: `docs/superpowers/specs/2026-05-25-context-region-substrate-design.md`.
  Deferred: ToolResultStrategy (spec §4.4 — shape/visibility/target three
  axes), runIntraTurnEngine (spec §7.2 — `tool-buffer` / `one-shot`
  expiry), trace `region.added` / `region.removed` events + cache health
  span attributes (rolled into Phase 4.6 below).

- **agent-docs-qa decision-attribution viewer (#68 PR #69, #71 PR #72).**
  The #64 causal drill-down viewer (decision spine + Why panel) is now embedded
  in the example's live audit panel as a `Why` tab (lazy iframe reusing core
  `renderViewer`, zero duplicated attribution logic); `FileTraceObjectStore`
  wired into the example so #26 region composition content hydrates. Viewer UX
  polish (#71): output answer renders markdown, the causal chain is trimmed to
  decision hops, `summarizeEvent` disambiguates repeated tool calls by input,
  and `agent.run.completed.causedBy = final llm.responded` so the output ❓
  drills. Follow-up #70 (make the Execution/Steps tab projection-driven) open.
- **Phase 4.7 — Events as the single source of truth for resume (#73, PR #74).**
  Tool side-effects on working memory are now event-sourced: each tool call
  emits a frozen `wm.mutated` WM snapshot, so `replay()` — which does NOT re-run
  tool handlers (`ReplayingIOPort` serves cached output) — reconstructs
  tool-written WM by folding the snapshots. Replay is now deterministic for
  WM-using / cognitive-tool runs (previously diverged). The stateStore
  checkpoint blob is **logically deleted**: resume state lives ONLY in the event
  log as an `agent.checkpoint` event; the stateStore keeps just a tiny
  `context→runId` routing pointer (an index, not state — cannot drift).
  `CheckpointManager` archived to `.bak/store/` and removed from the public
  export; all readers (resume / multi-turn restore / CLI / child) project the
  checkpoint from events via `checkpointFromEvents`. Two latent replay aliasing
  bugs fixed: `WorkingMemory.toJSON` returned a live `log` reference;
  `MemoryEventStore` stored payloads by reference (in-place mutation rewrote
  recorded events). Supersedes the Phase 2 "stateStore for checkpoint" and
  Phase 4.5 "checkpoint format" notes — checkpoint is no longer a separate
  source of truth. **Unblocks Phase 5** (deterministic replay of side-effecting
  runs) and is the event-sourced-side-effect template for **#60** (emit-driven
  FSM transitions in replay — same root cause: replay skips the handler; the WM
  case is solved, the `ctx.emit` case can follow the same pattern).

### Stories validated by Phase 1–4 (active)

`s-001` ReAct + intra-agent parallel · `s-002` Inspect a completed run ·
`s-003` Explain a decision with context · `s-005` Deterministic replay
(byte-identical) · `s-007` Inter-agent parallel via named sub-agent tools ·
`s-008` Interrupt + resume (incl. supervisor tree) · `s-009` Multi-turn +
tool error recovery · `s-011` Multi-state FSM intent routing + slot filling.

Full readiness view: `docs/stories/INDEX.md`.

---

## In progress

**Active line: #20 trace projection and diagnostics surface.**
The observable-substrate gaps are closed; the event log is now the source
of truth for the run-level objects needed by inspection, replay, and
diagnostics. The work in front of us is to expose those facts through
agent-consumable projections before adding richer UI:

- **Merged substrate:** `fsm.transition` (#21), `skill.loaded/unloaded` (#22),
  region content-addressing (#23), `agent.spawned` / `agent.returned`
  (#24); sub-agent first-class — independent `childRunId` + nested
  sub-trace + independently replayable, "model I" (#47, PR #49);
  `tool.responded` product metadata `outputHash` / `outputBytes` +
  object-store write-through (#25, PR #50, incl. child-port wiring);
  `causedBy` densification (#30).
- **Merged diagnostics substrate:** `causedBy` densification (#30) and
  guard evidence on `fsm.transition` (#31). Diagnosable.P0 is complete.
- **Next projection surface:** #29 (`trace show/events/region`) and #36
  (`trace explain/path/why`) should land before richer UI. They define the
  agent-consumable query contract and keep UI from owning private query
  logic.
- **Next UI projections:** #26–28 and #32–35 consume the same projection
  functions as CLI. They are reference projections over Trace, not a
  separate observability product. **Landed:** the #64 causal drill-down viewer
  (decision spine + Why panel) and its embedding into the agent-docs-qa audit
  panel (#68 / #71). **Open:** #70 (make the example's Execution tab
  projection-driven instead of re-parsing events in the frontend).

**Phase 5 — fork / structural diff / suite replay** remains the next big
capability rock. Substrate prerequisites are now stronger thanks to the
region snapshot/restore primitive, byte-identical replay, and **#73**
(tool side-effects on working memory are event-sourced, so side-effecting /
cognitive-tool runs now replay deterministically — fork/diff/suite all build
on deterministic replay). Phase 5 is still a multi-PR surface area tracked by
#58 (side-effect policy), #56 (fork), #55 (diff), and #57 (suite replay).
The remaining replay determinism gap is **#60** (emit-driven FSM transitions);
#73's `wm.mutated` event-sourcing is the template to close it.

The cross-cutting work below stays valid in parallel — particularly the
smaller follow-ups inherited from earlier landings: CLI surface spec update
for the three new `trace` verbs (`render-html` / `report` /
`--include-children`); s-007-shape integration test for 3-sub-agent HTML
rendering; routing `Milkie.invoke()`'s contextId/agentRunId through
`ioPort.uuid()` to close the last byte-identical hole.

---

## Up next

Phases are listed in the order capabilities unblock each other. Each
phase ships a coherent capability surface; within a phase, items can move
in parallel.

### Phase 5 — Fork, structural diff, suite / batch replay

**Goal.** Branch a recorded run at any event with one parameter changed,
share the cache prefix so only the tail costs new LLM calls, and compare
two runs structurally to classify divergences.

**Scope.**
- **Replay side-effect policy.** Decide the per-operator policy surface
  before fork ships. Default replay stays all-from-cache; fork may need
  explicit live re-invocation hooks for selected operators. Tracked by #58.
- **Fork primitive.** Build on the existing `CacheIndex` so a fork at
  event N reuses the prefix cache for events 0..N-1 and only pays
  fresh model calls for the tail. Tracked by #56.
- **Structural diff.** Typed comparison over two event logs / projected
  graphs. Returns classified divergences (regression / improvement /
  neutral / structural-only) as structured output. Tracked by #55.
- **Suite definition + batch replay.** Saved sets of runs (e.g.
  `golden_v1`); replay the suite against a new code version; per-run
  diff reported as structured output for batch analysis. Tracked by #57.
- **In-flight trace query API.** Stable query contract over running +
  completed runs so a sub-agent can read its parent's trace mid-run.
  Today the event store is writable during a run but query semantics
  for partial logs aren't pinned down.

**Unblocks.** Closes `s-006` (fork), `s-012` (suite replay + classify),
`s-013` (variant search at bounded cost), `s-015` (sub-agent reads
parent's in-flight trace). These four stories represent the "Agent Trace
as agent-first protocol" surface — cross-cutting invariants 12–13 in
`ARCHITECTURE.md`.

**Cost contract for `s-013`.** Fork's amortization is real only when the
variant change applies to events at index N where N > 0 — typical
cases: a mid-run tool override, a synthesis-prompt-only swap, a routing
decision at a specific FSM state. **Changing anything in the system
prompt or initial messages invalidates the cache from the first LLM
request onward**, so variant search across full-prompt edits pays the
same as a from-scratch re-run. `s-013`'s contract must encode this:
its acceptance asserts a cost bound based on tail size, not raw N.

### Phase 6 — Lineage-by-typed-relations

**Goal.** Every artifact produced by an agent traces back to the
LLM calls, tool results, retrieved documents, and source versions that
produced it. Reverse: given a source identifier, return every run whose
lineage references it.

**Scope.**
- **Taxonomy first.** Define the object and relation type vocabulary before
  producers emit lineage events (#39).
- **Explicit producer API.** New event kinds `object.created` /
  `relation.created`; producers declare objects and relations, and runtime
  records typed events. Avoid recovering provenance by parsing LLM text or
  frontend citation heuristics.
- **Forward query API.** Artifact → causal chain.
- **Reverse query API.** Source version → dependent objects/runs.
- **Rebuildable index first.** Start with scan-and-build projection indexes
  over EventStore/JSONL; persistent reverse indexes can come after real
  volume proves the need.

**Unblocks.** Closes `s-004` (forward lineage) and `s-014` (reverse-reference
lineage).

### Evolution (deferred, not next)

Experiment Registry, Traffic Splitter, Outcome Collector, Promotion Gate.
Targeted but **not the next priority** — there is more upstream value in
landing Phase 5 first (Phase 4 done; fork / diff are the remaining
substrate Evolution attributes outcomes over).

`s-010` currently sits in `draft` because its `requires` lists
`Evolution: Experiment Registry`. The skill-epoch portion is already green
in code; full story validation waits for Evolution to land.

### Reference UI projection (deferred)

ARCHITECTURE.md commits to a reference UI projection for milkie — without
one, the product thesis *runs as the primary engineering product* is not
verifiable in practice by humans. Timelines, lineage DAGs, fork trees,
structural diffs, and suite replay tables don't afford discovery from a
CLI alone; visual surface is what makes a run perceptible as an object
rather than as a stream of JSONL records. Form is TBD (local web viewer
/ static HTML report generator / IDE extension) and deliberately not
picked yet.

**Kickoff condition.** Phase 5 capabilities (fork / diff / suite replay)
land in code, so the UI is designed against real data structures rather
than speculation; CLI output contracts stabilize so the projection
isn't tracking a moving substrate.

**Only thing happening before then.** The `s-002` static HTML
`trace report` probe listed under cross-cutting work — single capability,
zero framework, zero form commitment, single-digit-day cost.

**Hard constraint when it does ship.** Per ARCHITECTURE.md
`## User-facing surfaces`, a UI must remain a **projection** over CLI /
SDK output: it does not own its own query logic, fork algorithm, or
state. The moment a UI answers a question the CLI cannot, it has
turned into a parallel facade and violates invariant 12.

### External Context Layer / Data / Execution / Foundation (target)

Today there is no external Context Layer / Data / Execution / Foundation
infrastructure. After PR-C1 the old `src/context/ContextLayer.ts` shim
is gone — replaced by `ContextRegions` + `assemble` + lifecycle engine,
which form an in-Agent **context management** discipline (not a Context
Layer substitute; see ARCHITECTURE.md §Context Layer for the
distinction). The target architecture still pushes Context Layer / Data
/ Execution outside the milkie boundary as separate services; that
migration is intentionally not phased yet — it needs a concrete
consumer (e.g. KWeaver integration) to define the external interface
shape.

### Memory tools (Layer 2 external memory)

**Goal.** Give agents a `memory_read` / `memory_write` / `memory_list`
/ `memory_search` tool surface backed by an `IMemoryStore` interface
(in-memory / SQLite / Redis implementations). Persists state out-of-band
so it does not enter every LLM request; agent retrieves selectively.

**Why a separate item.** The in-context WM (`src/store/WorkingMemory.ts`
+ `wm` region) is Layer 1 — small, always-rendered state. Layer 2 is
the external memory tier described in ARCHITECTURE.md §Context Layer
"Two-tier memory pattern". Today only Layer 1 exists; Layer 2 is the
in-Agent precursor to what the future external Context Layer's
"memory lookup" responsibility would deliver.

**Estimated scope.** ~400 LOC + example. `IMemoryStore` interface,
`InMemoryMemoryStore` + `SQLiteMemoryStore` implementations, 4 tools
under `src/tools/memory.ts`, `ToolContext.memoryStore?` injection,
unit tests, a heavy-WM example (e.g. todo agent) demonstrating
cross-turn / cross-session persistence.

**Trigger to actually do it.** Any of:
- An agent on milkie needs stateful memory that exceeds the
  in-context WM size budget (a few KB)
- An agent needs cross-session persistence (state survives process
  restart / different contextId)
- A real workload appears that wants RAG-style memory_search and is
  willing to bring an embedding backend
- Two-tier memory becomes a recurring teaching point worth shipping

Until one of those triggers, this stays deferred — making it real
without a consumer would be speculative substrate.

**Relationship to External Context Layer.** Doing memory tools first
prototypes one slice of the future Context Layer's responsibility
in-Agent. When the external interface arrives, `IMemoryStore` impl
can point at the external memory service; agent / tool code stays
the same.

---

## Cross-cutting work that can move independently

These don't block any phase but pay back continuously.

- **Story coverage for already-implemented capabilities.** The remaining
  `draft` stories all need Phase 5–6 work; no current implementation gaps
  are missing tests. As capabilities land, write the story's E2E in the
  same PR per the README lifecycle.
- **CI infra for Redis-gated tests.** `tests/e2e/run-redis-e2e.sh`
  exists; wiring it into CI lets us re-introduce real-Redis variants
  of `s-008` / `s-009` without skipping by default. Currently both
  stories run against `MemoryStore` for hermetic correctness.
- **TrajectoryStore retirement decision.** ARCHITECTURE.md flags
  TrajectoryStore as the predecessor of the event-sourced log. Phase 4
  has landed and the event log now covers LLM / tool / lifecycle /
  clock / uuid — every span use-case TrajectoryStore handled. Decide
  whether to project Trajectory from the event log or retire it
  outright. Don't duplicate sources of truth long-term.
- **Public API documentation.** ARCHITECTURE.md is silent on the
  concrete public library facade by design. The first such design doc
  landed —
  `docs/superpowers/specs/2026-05-24-cli-surface-design.md` defines the
  agent-facing CLI verb surface across `agent` / `trace` / `suite`
  domains — and the P0 implementation
  (`agent list / run / resume / interrupt`, `trace inspect / replay`)
  shipped in code. **P1+ verbs follow Phase 5**: `trace fork`,
  `trace diff`, `suite create / list / replay / diff`. `milkie agent`
  could grow `inspect <contextId>` to pair with `interrupt` /
  `resume` (probe paused state, list checkpoints) but no story drives
  it yet.
- **Agent registration design.** Closed —
  `docs/superpowers/specs/2026-05-24-agent-registration-design.md`
  defines `.milkie/agents.json` manifest convention; `Milkie.loadManifest()`
  implements it; CLI auto-loads at every startup. Follow-up: a
  `milkie init` ergonomics command to scaffold `.milkie/agents.json` +
  default `.gitignore` for a new project (separate spec when needed).
- **`examples/` buildout.** Two examples landed:
  `examples/s-005-replay/` (paired `record.ts` / `replay-sdk.ts` /
  `replay-cli.sh` against a frozen JSONL fixture) demonstrates SDK ↔
  CLI parity for replay; `examples/s-002-inspect/` (`record.ts` +
  `report.sh`) demonstrates `trace report` rendering a recorded run to
  self-contained HTML. Each new example follows the same shape.
  Phase-5-gated examples: `s-006` (fork) / `s-012` (suite replay +
  classify) / `s-013` (variant search) / `s-015` (sub-agent in-flight
  trace consumption).
- **In-flight semantics spec.** `inspect` / `lineage` on a still-running
  run — return-at-snapshot vs block-until-done vs cursor-based polling.
  CLI surface spec §9 still has this open; relevant when s-015 lands.
- **`agent run` live-path coverage.** Unit tests cover wiring + error
  paths only; the happy path against a real LLM adapter is exercised by
  the existing e2e suite (s-001 / s-009 / s-011) but not yet through the
  CLI. A small CLI-level live test would close the loop.
- **Static HTML trace report (s-002 probe) — landed.** Three commands
  shipped: `trace inspect --include-children` (descendant scan),
  `trace render-html --input <file>` (pure JSONL → HTML projection,
  architectural firewall: no event-store access), `trace report <runId>`
  (sugar wrapping both). Renders one completed run as a self-contained
  HTML timeline — paired `*.requested`/`*.responded` collapsed to one
  entry via `causedBy`, sub-agent fan-out nested recursively, inline
  CSS + vanilla JS, raw events embedded as `<script
  type="application/json" id="trace-data">` so the file is its own
  re-renderable archive. `examples/s-002-inspect/` demonstrates the
  end-to-end flow. Follow-ups deferred: CLI surface spec update for the
  three new verbs; richer in-flight badge styling; an s-007-shape
  integration test (3 sub-agent fan-out).

---

## Migration intentions (not commitments)

Architectural transitions implied by the target architecture, expected to
move when a concrete consumer or capability demands them. None are scheduled.

- **Trajectory → Event log.** `TrajectoryStore` is the predecessor of the
  event-sourced Agent Trace; Phase 4 has landed and the event log now
  covers LLM / tool / lifecycle / clock / uuid / region / FSM transition.
  Decision: project Trajectory from the event log, or retire
  TrajectoryStore outright. Don't duplicate sources of truth long-term.
  (Listed in Cross-cutting work above as the operational decision; this
  entry records the architectural intent.)
- **In-Agent `ContextRegions` → External Context Layer.**
  `src/context/ContextRegions.ts` is a region-based assembly substrate
  inside Agent Runtime — **not** a local shim for the external Context
  Layer. When the external interface materializes (likely driven by a
  KWeaver-shaped integration), the read-side responsibilities
  (knowledge retrieval, memory lookup, capability resolution) move
  outward; `ContextRegions` stays in-Agent as the assembly substrate for
  whatever the Context Layer projection delivers.
- **IOPort → Execution operator.** IOPort currently passes through to
  the gateway directly. Target: route invocations through an Execution
  layer operator that itself delegates to Foundation / Model Factory.
  Waits on Execution layer materialization.

---

## Deferred small items

Substrate gaps too small to phase but worth not losing:

- **`random.consumed` non-determinism record.** Phase 4 ships
  `clock.read` and `uuid.generated`; `Math.random` has zero call sites
  in `src/` today, so the third variant is deferred until a real
  consumer appears.
- **External non-LLM tool I/O outcomes.** Beyond the already-recorded
  `tool.responded` event, file I/O / network I/O from operator
  implementations are served via the existing `tool.responded` cache
  during replay. No operator-specific policy hook yet — coupled with
  the "Replay side-effect policy" open question below.
- **6-capability vocabulary status.** `ARCHITECTURE.md` describes
  observable / diagnosable / lineage / replay / fork / diff as a
  6-capability surface. Today: **replay is complete**; **observable.P0
  substrate is complete** — `fsm.transition` (#21), `skill.loaded` (#22),
  region content hash (#23), `agent.spawned/returned` (#24), sub-agent
  first-class trace (#47), `tool.responded` metadata (#25) all merged.
  Remaining observable work is P1 consumption (CLI/query/UI, #26–29).
  **diagnosable.P0** is complete with `causedBy` (#30) and guard evidence
  (#31); **lineage / fork / diff** are absent or Phase-5/6 work.

---

## Open architectural questions

These need a decision **before** the corresponding phase starts.

- **Replay side-effect policy** (Phase 5 prerequisite). Phase 4 declared
  the simple all-from-cache policy: replay never re-invokes operators
  with live side effects. The per-operator hook (some operators served
  from cache, others re-invoked against live state for variant search)
  is to be designed alongside Phase 5 fork. Not blocking until fork
  implementation begins.
- **Deterministic-flow placement** (cross-cutting). `ARCHITECTURE.md`
  invariant #10 forbids a Workflow System. Deterministic business flows
  must either be a specific FSM configuration or live outside milkie.
  Phase 4 landed without distinguishing LLM-driven vs purely deterministic
  runs in the event schema — the question is therefore open but no longer
  blocking. Revisit if Evolution or Phase 5 fork needs the distinction.
- **Evolution outcome attribution under sub-agent fan-out** (Evolution).
  When a variant spawns sub-agents, do outcomes attribute to the parent
  variant only, or roll up from the whole subtree? Needs an explicit
  rule before Evolution ships.
- **Foundation scope under Data** (external infrastructure). Whether
  Data shares Foundation with Model Factory or has its own descent path
  for storage engines and indexing. Doesn't block any milkie phase, but
  affects integration shape with external infra (e.g. KWeaver).

---

## Out of scope (durable)

These were considered and **deliberately excluded** — not "later", but
"not at all" unless the assumptions behind the architecture change. See
`ARCHITECTURE.md` cross-cutting decisions for the why.

- A Workflow System / DAG engine inside milkie.
- An `Environment` abstraction.
- Variant proposers, root-cause analysis, learning memory store, or
  multi-objective Pareto optimization inside Evolution.
- Embedded intelligence (summarization, judgement, learning) inside
  Agent Trace.
- LLM calls from Evolution.
- Working context as an architectural layer (it stays as runtime data).
