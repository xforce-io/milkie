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

- **Phase 1–3 are landed.** FSM Runtime, working context, Trajectory
  observability, IOPort, Agent Trace event log (LLM + tool I/O + lifecycle),
  content-addressed cache, and structural replay are all in code. State
  stores (Memory / SQLite / Redis) ship. Sub-agent as named tool, interrupt
  signal, supervisor-tree propagation, skill epoch loading are verified.
- **8 of 15 stories are `active`** (have green E2E tests). The 7 `draft`
  stories all depend on Phase 4–6 capabilities that haven't shipped yet.
- **Next big rock:** Phase 4 non-determinism log → unlocks byte-identical
  replay → unlocks Phase 5 fork / diff / suite replay.
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

---

## Completed (Phase 1–3)

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

### Stories validated by Phase 1–3 (active)

`s-001` ReAct + intra-agent parallel · `s-002` Inspect a completed run ·
`s-003` Explain a decision with context · `s-005` Deterministic replay
(structural) · `s-007` Inter-agent parallel via named sub-agent tools ·
`s-008` Interrupt + resume (incl. supervisor tree) · `s-009` Multi-turn +
tool error recovery · `s-011` Multi-state FSM intent routing + slot filling.

Full readiness view: `docs/stories/INDEX.md`.

---

## In progress

**Nothing code-side actively in flight at this exact moment.** The
just-landed agent-first wave covered design + scaffold + first example
end-to-end:

- ARCHITECTURE.md got invariants 12–13 (agent-first / CLI as protocol
  facade), `## User-facing surfaces` (CLI / SDK / API + UI as projection),
  `## Representative scenarios` (one entry per 6-capability surface item
  plus cross-cutting), and an expanded Implementation Status with
  `Suite definition + batch replay` and `In-flight trace query API` as
  Phase 5 targets.
- Four new agent-first stories drafted: `s-012` / `s-013` / `s-014` /
  `s-015`.
- Specs: CLI surface design + agent registration manifest convention.
- Code: `Milkie.loadManifest()`, the `milkie` CLI binary with 6 P0 verbs
  (`agent list / run / resume / interrupt`, `trace inspect / replay`),
  SQLite-backed CLI default stateStore so interrupt / resume work across
  CLI processes.
- First runnable example: `examples/s-005-replay/` with paired SDK and
  CLI scripts plus a shipped fixture; demonstrates the SDK ↔ CLI parity
  invariants 12–13 require.

**Most natural next pickup** is the cross-cutting work itemized below
(more examples following the s-005 pattern; a brief in-flight semantics
spec for `inspect` / `lineage` on a still-running run) rather than the
next phase boundary. The earlier session also closed the gap audit
(ARCHITECTURE.md ↔ code ↔ stories) and added hermetic `s-002` / `s-003`
tests. Phase 4 (non-determinism log) has not been started.

---

## Up next

Phases are listed in the order capabilities unblock each other. Each
phase ships a coherent capability surface; within a phase, items can move
in parallel.

### Phase 4 — Non-determinism log → byte-identical replay

**Goal.** Replay produces a state that is byte-identical to the original,
not just structurally equivalent. Today timestamps and UUIDs are re-sampled
on replay; clock reads, UUIDs, random values, and non-LLM external I/O
need to be recorded and replayed from the log.

**Scope.**
- Extend the event log with a positional non-determinism record (clock /
  uuid / random / external I/O outcomes).
- `RecordingIOPort` records observed values; `ReplayingIOPort` serves
  them in append order.
- `replay()` becomes byte-identical for in-scope values; out-of-scope
  external side effects need an explicit replay policy (skip / mock /
  reinvoke / require-confirmation — see Open questions).

**Unblocks.** Closes `s-005` (currently active but acknowledged as
structural-only); is a prerequisite for honest fork semantics in Phase 5.

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
landing Phase 4–5 first (replay/fork/diff is the substrate Evolution
attributes outcomes over).

`s-010` currently sits in `draft` because its `requires` lists
`Evolution: Experiment Registry`. The skill-epoch portion is already green
in code; full story validation waits for Evolution to land.

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
  `draft` stories all need Phase 4–6 work; no current implementation gaps
  are missing tests. As capabilities land, write the story's E2E in the
  same PR per the README lifecycle.
- **CI infra for Redis-gated tests.** `tests/e2e/run-redis-e2e.sh`
  exists; wiring it into CI lets us re-introduce real-Redis variants
  of `s-008` / `s-009` without skipping by default. Currently both
  stories run against `MemoryStore` for hermetic correctness.
- **TrajectoryStore retirement decision.** ARCHITECTURE.md flags
  TrajectoryStore as the predecessor of the event-sourced log. When
  Phase 4 lands and the event log covers every span use-case, decide
  whether to project Trajectory from the event log or retire it. Don't
  duplicate sources of truth long-term.
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
- **`examples/` buildout.** First example (`examples/s-005-replay/`)
  landed and demonstrates the SDK ↔ CLI parity pattern: paired
  `record.ts` / `replay-sdk.ts` / `replay-cli.sh` against a frozen
  JSONL fixture, no API key. Each new example follows the same shape.
  Likely next: `s-002` (inspect) using the new `trace inspect` verb —
  no Phase 5 dependency. Phase-5-gated examples: `s-006` (fork) /
  `s-012` (suite replay + classify) / `s-013` (variant search) /
  `s-015` (sub-agent in-flight trace consumption).
- **In-flight semantics spec.** `inspect` / `lineage` on a still-running
  run — return-at-snapshot vs block-until-done vs cursor-based polling.
  CLI surface spec §9 still has this open; relevant when s-015 lands.
- **`agent run` live-path coverage.** Unit tests cover wiring + error
  paths only; the happy path against a real LLM adapter is exercised by
  the existing e2e suite (s-001 / s-009 / s-011) but not yet through the
  CLI. A small CLI-level live test would close the loop.

---

## Open architectural questions

These need a decision **before** the corresponding phase starts.

- **Replay side-effect policy** (Phase 4). When a recorded run made an
  external side effect (e.g. wrote a file, hit an API), what does replay
  do? Skip / mock / reinvoke / require-confirmation are all plausible
  depending on the operator; we need a per-operator policy hook.
- **Deterministic-flow placement** (cross-cutting). `ARCHITECTURE.md`
  invariant #10 forbids a Workflow System. Deterministic business flows
  must either be a specific FSM configuration or live outside milkie.
  The event schema may need to distinguish LLM-driven runs from purely
  deterministic FSM runs — TBD before Phase 4 freezes the event format.
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
