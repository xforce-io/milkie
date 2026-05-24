# milkie Roadmap

Phased plan for shipping the architecture described in `ARCHITECTURE.md`.
This file is the **forward-looking** view; the current snapshot of "what works
today" lives in `ARCHITECTURE.md` → `## Implementation Status`, and the
authoritative scenario inventory is in `docs/stories/INDEX.md`.

The roadmap moves only when concrete code closes a gap — order of operations
matters more than calendar dates, so phases are listed without dates.

Last updated: 2026-05-24

---

## TL;DR

- **Phase 1–4 are landed.** FSM Runtime, working context, Trajectory
  observability, IOPort, Agent Trace event log (LLM + tool I/O + lifecycle +
  clock / uuid), content-addressed cache, structural replay, and
  **byte-identical replay via non-determinism log** are all in code. State
  stores (Memory / SQLite / Redis) ship. Sub-agent as named tool, interrupt
  signal, supervisor-tree propagation, skill epoch loading are verified.
- **8 of 15 stories are `active`** (have green E2E tests). The 7 `draft`
  stories all depend on Phase 5–6 capabilities that haven't shipped yet.
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
- **Evolution and Lineage-by-typed-relations are deferred** — there are
  open architectural questions before code work starts on either.
- **Reference UI projection — first probe landed.** ARCHITECTURE.md
  promotes UI from "optional" to "deferred reference projection". The
  s-002 static HTML probe is now shipped: `milkie trace report <runId>`
  (with `trace render-html` as the underlying pure-projection primitive
  and `trace inspect --include-children` for sub-agent fan-out) renders
  a completed run as a single self-contained HTML file. Full UI form
  remains TBD until Phase 5 capabilities (fork / diff / suite) land and
  CLI output contracts stabilize.

---

## Completed (Phase 1–4)

Already in code; see `ARCHITECTURE.md` → `Implementation Status` for
file pointers.

- **Phase 1 — FSM Core & working context.** Statechart-based runtime,
  multi-state FSMs, action / llm state types, ctx.emit transitions,
  multi-turn history, working memory, skill epoch loading.
- **Phase 2 — Trajectory observability + state stores.** Span-based
  TrajectoryStore (`llm.call` / `tool.call` / `fsm.transition` /
  `agent.spawn` / `agent.run`). MemoryStore / SQLiteStore / RedisStore
  for checkpoint / interrupt / resume.
- **Phase 3 — Agent Trace event log + structural replay.** Append-only
  event log via `IEventStore` (Memory + JSONL implementations).
  `RecordingIOPort` decorator pattern records paired `requested` /
  `responded` events with `causedBy` chains; `agent.run.started` /
  `completed` lifecycle events. `CacheIndex` projects the log into FIFO
  response queues keyed by canonical request hash; `ReplayingIOPort`
  serves cached responses; `Milkie.replay(runId)` produces a structural
  re-run with strict `ReplayDivergenceError` on cache miss. Sub-agent
  as named tool, parent interrupt → supervisor-tree propagation, and
  interrupt-and-resume across stateStores are all working.
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

### Stories validated by Phase 1–4 (active)

`s-001` ReAct + intra-agent parallel · `s-002` Inspect a completed run ·
`s-003` Explain a decision with context · `s-005` Deterministic replay
(byte-identical) · `s-007` Inter-agent parallel via named sub-agent tools ·
`s-008` Interrupt + resume (incl. supervisor tree) · `s-009` Multi-turn +
tool error recovery · `s-011` Multi-state FSM intent routing + slot filling.

Full readiness view: `docs/stories/INDEX.md`.

---

## In progress

**Nothing code-side actively in flight.** The Phase 4 non-determinism
log just landed (PR #2 merged); earlier in the same session the s-002
HTML report probe and the agent-first CLI wave landed too. All
substrate work for Phase 5 fork is now in place.

**Most natural next pickup is Phase 5** (fork primitive + structural
diff + suite replay + in-flight trace query API). The cross-cutting
work below stays valid in parallel — particularly the smaller
follow-ups inherited from earlier landings: CLI surface spec update
for the three new `trace` verbs (`render-html` / `report` /
`--include-children`); s-007-shape integration test for 3-sub-agent
HTML rendering; routing `Milkie.invoke()`'s contextId/agentRunId
through `ioPort.uuid()` to close the last byte-identical hole.

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
- **Fork primitive.** Build on the existing `CacheIndex` so a fork at
  event N reuses the prefix cache for events 0..N-1 and only pays
  fresh model calls for the tail.
- **Structural diff.** Typed comparison over two event logs / projected
  graphs. Returns classified divergences (regression / improvement /
  neutral / structural-only) as structured output.
- **Suite definition + batch replay.** Saved sets of runs (e.g.
  `golden_v1`); replay the suite against a new code version; per-run
  diff reported as structured output for batch analysis.
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
- New event kinds: `object.created` / `relation.created`. The event log
  already records I/O — lineage needs **typed graph events** layered on
  top.
- Forward query API: artifact → causal chain.
- Reverse query API: source version → all dependent runs.
- Index for reverse traversal (today the log is append-only without a
  reverse index).

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

Today `src/context/ContextLayer.ts` is an internal class. The target
architecture pushes it outside the milkie boundary; same with Data /
Execution / Foundation. These migrations are intentionally not phased
yet — they need a concrete consumer (e.g. KWeaver integration) to define
the external interface shape. Until then the internal shim stays.

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
