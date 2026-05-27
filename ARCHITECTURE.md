# milkie Architecture

This document captures milkie's top-level architecture and the design decisions
that hold it together.

It describes the **target** architecture for milkie as a library. Many parts
lead the current implementation. See `## Implementation Status` below for a
calibrated picture of what is in code today vs. what this document anticipates;
implementation gaps should be tracked as roadmap or design-doc work rather than
treated as accidental drift by default.

## Purpose & Value

**Definition.** milkie is a TypeScript library where the agent **run** —
the full reasoning trajectory, not just the output it produces — is the
primary engineering product. Runs are addressable, reproducible, forkable,
comparable, and attributable; the agent system improves through controlled
experiments over them.

**Goal.** Move LLM agents from one-off scripts to durable engineering
systems — and elevate the **run** from byproduct to primary deliverable.
One structural model spans every pattern (intent-routed dialog, ReAct,
multi-state workflows, multi-agent orchestration); every run is
deterministically reproducible despite LLM non-determinism; every shipped
agent has a first-class path to measurably improve the runs it produces.

**Value.** A milkie agent's product is the **run** — machine-operable,
diagnosable, debuggable, auditable, and improvable as a first-class artifact.
Outputs are views over runs, not vice versa. Agents that produce only outputs
become black boxes that work until they don't.

## Implementation Status

The architecture below is a target. Below summarizes what is in code today
vs. what this document anticipates. Strong claims throughout this document
(e.g. "every run replays deterministically") should be read as **target
invariants**, not current behavior, except where this section lists them
as implemented.

### Implemented today

- **Agent Runtime FSM Core** — Statechart-based execution, transitions,
  multi-turn loops. (`src/runtime/`, `src/fsm/`)
- **working context (basic)** — Region-based substrate: every piece of
  context (header, skills, state instructions, working memory, history,
  scratchpad, current turn, tool schemas) is a typed `Region` in a
  `ContextRegions` store; a pure `assemble(regions, scope)` function
  produces the LLM request on each call. Note: this is an in-Agent
  assembly substrate, **not** an implementation of the external
  Context Layer concept described under "Infrastructure layers" — see
  §Context Layer for the distinction. (`src/context/`)
- **Trajectory / span observability** — `TrajectoryStore` collects spans
  for `llm.call`, `tool.call`, `fsm.transition`, etc. This is the
  predecessor of the event-sourced Agent Trace described below.
  (`src/trajectory/`)
- **IOPort as non-determinism boundary** — `IIOPort` / `DefaultIOPort`
  in `src/runtime/IOPort.ts`. Agent Runtime routes every LLM call, tool
  invocation, clock read, and UUID generation through it. `DefaultIOPort`
  is the live-passthrough leaf; `RecordingIOPort` and `ReplayingIOPort`
  are the record/replay decorators. (`src/runtime/IOPort.ts`,
  `src/trace/RecordingIOPort.ts`, `src/trace/ReplayingIOPort.ts`)
- **Agent Trace event log** — `Event` / `IEventStore` / `RecordingIOPort`
  in `src/trace/`. When an `eventStore` is supplied to `Milkie`, every LLM
  and tool I/O is recorded as paired `requested` / `responded` events with
  `causedBy` chains, plus `agent.run.started` / `agent.run.completed`
  lifecycle events and `clock.read` / `uuid.generated` non-determinism
  events. MemoryEventStore and JsonlEventStore implementations provided.
  (`src/trace/`)
- **Content-addressed cache + byte-identical replay** — `CacheIndex`
  projects an event log into per-hash FIFO queues for llm/tool, plus
  position-FIFO queues for clock/uuid. `ReplayingIOPort` serves every
  IIOPort method from the cache (live `inner` never called during
  replay). `Milkie.replay(runId)` re-runs a recorded run with zero
  live LLM/tool calls and byte-identical agent-observable nondet;
  strict P-wide divergence — over-consume throws immediately,
  under-consume across any of the four queues throws at the tail.
  (`src/trace/CacheIndex.ts`, `src/trace/ReplayingIOPort.ts`,
  `src/runtime/Milkie.ts:replay`)
- **State stores for checkpoint/resume** — MemoryStore / SQLiteStore /
  RedisStore for interrupt/resume scenarios. (`src/store/`) All three implement
  `IStateStore` (`src/types/store.ts`); SQLite and Redis variants require
  `init()`, MemoryStore is constructor-ready.
- **Yield point + interrupt signal** — FSM has a `paused` reserved state and
  a global `interrupt` event handler. `Milkie.interrupt(contextId)` writes an
  interrupt flag into stateStore that AgentRuntime polls on each turn boundary,
  enqueuing an `interrupt` event that transitions the FSM into `paused`. The
  paused checkpoint can be re-loaded by a later `invoke` with the same
  `contextId`. (`src/fsm/FSMEngine.ts:11,61-62`,
  `src/runtime/Milkie.ts:297-300`, `src/runtime/AgentRuntime.ts:238-242`)
- **Supervisor tree (interrupt propagation)** — When a parent agent is
  interrupted, every spawned sub-agent receives the interrupt and writes its
  own checkpoint; the parent's checkpoint records each child's `checkpointId`
  in its `children` array so the whole tree can be resumed together.
  (`src/runtime/AgentRuntime.ts:141-186`,
  `src/types/store.ts:45-51` for `ChildAgentRecord`)

### Target only (not yet in code)

- **`random.consumed` non-determinism record** — Phase 4 ships
  `clock.read` and `uuid.generated`; `Math.random` has zero call sites
  in `src/` today, so the third variant is deferred until a real
  consumer appears. External non-LLM tool I/O outcomes beyond the
  already-recorded `tool.responded` event are similarly deferred —
  Phase 4 records what `port.now()` / `port.uuid()` return and
  ReplayingIOPort serves them strictly, but file I/O / network I/O
  from operator implementations are still served via the existing
  `tool.responded` cache (no operator-specific policy hook yet —
  see Phase 5 fork follow-up).
- **Fork engine** — branch a run at any event with shared prefix from cache
  (no new model calls for the shared history). Phase 5.
- **Structural diff** — typed comparison of two event logs / projected
  graphs. Phase 5.
- **Lineage-by-typed-relations** — current event log only records I/O;
  lineage requires emitting `object.created` / `relation.created` and
  traversing causedBy chains as a graph. Both **forward queries**
  (artifact → source) and **reverse queries** (source → all dependents)
  are in scope. Phase 6 or later.
- **Suite definition + batch replay** — saved sets of runs (e.g.
  `golden_v1`) replayable as a single operation against a new code version;
  per-run divergence reported as structured output for batch analysis by
  downstream consumers. Phase 5.
- **In-flight trace query API** — query a run's event log while the run is
  still active, so sub-agents can consume their parent's trace mid-run.
  Today event stores are writable during a run; query semantics across
  in-flight + completed runs need an explicit contract. Phase 5.
- **External Context Layer / Data / Execution / Foundation** — These
  infrastructure layers are described as outside the milkie boundary.
  Today they are entirely absent: there is no Data or Execution
  abstraction (gateway substitutes for Foundation), and the read-side
  responsibilities of an external Context Layer (knowledge retrieval,
  memory lookup, capability table resolution) are not implemented. The
  in-Agent `ContextRegions` substrate is not a stand-in for them — it
  only assembles what the agent already has in scope into an LLM
  request (see §Context Layer for the distinction).
- **Evolution subsystem** — Experiment Registry, Traffic Splitter, Outcome
  Collector, Promotion Gate — none implemented.
- **Capability vocabulary in full** — observable / diagnosable / lineage /
  replay / fork / diff are described as a 6-capability surface; today only
  basic span query exists via TrajectoryStore.
- **Reference UI projection** — first probe shipped: `milkie trace report
  <runId>` renders a completed run as a self-contained HTML timeline
  (single agent + recursive sub-agent nesting via Observable). Full form
  TBD — local viewer / static report generator / IDE extension are still
  open, and Phase-5 capabilities (fork / diff / lineage / suite) have no
  visual surface yet. The product thesis "runs as the primary engineering
  product" requires a visual surface for humans to perceive runs as
  objects; the probe validates that CLI JSON output is rich enough to
  drive a projection, but the library is still incomplete without a UI
  story for the Phase 5–6 capabilities.

### Migration intentions (not commitments)

- **Trajectory → Event log**: When event-sourced Agent Trace lands,
  TrajectoryStore either becomes a projection over the event log or is
  retired in favor of the event log's native query surface.
- **In-Agent ContextRegions → External Context Layer**: The current
  `src/context/ContextRegions.ts` is a region-based assembly substrate
  that lives inside Agent Runtime — it is **not** a local shim for the
  external Context Layer. The external interface remains future work.
  When it materializes, the read-side responsibilities (knowledge
  retrieval, memory lookup, capability resolution) move outward;
  `ContextRegions` stays in-Agent as the assembly substrate for whatever
  the Context Layer projection delivers.
- **IOPort → Execution operator**: IOPort currently passes through to
  the gateway directly. The target is to route invocations through an
  Execution layer operator (which itself delegates to Foundation /
  Model Factory).

This section is the source of truth for "what works today." When code lands
that closes a gap, update this section before claiming the gap is closed
elsewhere.

## Overview

milkie is the **application layer** of an agent stack, packaged as a library.
The library exposes two parallel facades: the **SDK facade** for in-process /
programmatic consumers (applications, services, test harnesses) and the **CLI
facade** for agent consumers (sub-agents that read traces, meta-agents that
propose variants). Both are thin export surfaces over the same library
substrate.

The library contains three peer subsystems (Agent Runtime, Agent Trace,
Evolution) and sits on top of layered infrastructure (Context Layer → Data +
Execution → Foundation).

```
Applications / CLI / Services / Test Harnesses
                         │
                         ▼
┌──── milkie library ──────────────────────────────────────┐
│                                                           │
│   ┌────────────────────────┬─────────────────────────┐   │
│   │     SDK Facade         │      CLI Facade         │   │
│   │  (dev consumers)       │  (agent consumers)      │   │
│   └────────────────────────┴─────────────────────────┘   │
│                            ↓                              │
│   ┌──────────────────────────────────────────────────┐   │
│   │                  Evolution                        │   │
│   │  · Experiment Registry                            │   │
│   │  · Traffic Splitter                               │   │
│   │  · Outcome Collector                              │   │
│   │  · Promotion Gate                                 │   │
│   └──────────┬───────────────────────────┬───────────┘   │
│              ↓                            ↓                │
│   ┌──────────────────┐         ┌─────────────────────┐   │
│   │  Agent Runtime   │         │    Agent Trace      │   │
│   │  · FSM Core      │         │  · Event log        │   │
│   │  · working ctx   │         │  · Response cache   │   │
│   │    (data, not    │         │  · Replay/Fork/Diff │   │
│   │     a layer)     │         │  · Lineage/Query    │   │
│   │  · IOPort        │         │                     │   │
│   └────────┬─────────┘         └──────────▲──────────┘   │
│            │                              │              │
│            └──────── IOPort contract ─────┘              │
└────────────┼─────────────────────────────┼───────────────┘
             │                              │
   ━━━━━━━━━ ┿ ━━━━━━━ milkie boundary ━━━━┿━━━━━━━━━━━━━━
             ▼                              ▼
   ┌──────────────────────────────────────────────────────┐
   │                 Context Layer                         │
   │   Projects Data + Execution into what the agent       │
   │   currently sees and can invoke.                      │
   └────────────┬─────────────────────────┬───────────────┘
                ▼                         ▼
       ┌────────────────────┐    ┌────────────────────┐
       │       Data         │    │     Execution      │
       │  Heterogeneous     │    │  Operator registry │
       │  data aggregation: │    │  · skill           │
       │  · metadata mgmt   │    │  · function        │
       │  · connection mgmt │    │  · API / interface │
       │  · source ingestion│    │  · MCP             │
       └─────────┬──────────┘    └─────────┬──────────┘
                 ▼                         ▼
   ┌──────────────────────────────────────────────────────┐
   │                  Foundation                           │
   │  · Model Factory (model serving / lifecycle / routing)│
   │  · Storage engines, raw compute, other base services  │
   └──────────────────────────────────────────────────────┘
```

The split is deliberate: milkie owns the library API and application semantics;
entrypoints call into the library, while infrastructure layers may be provided
by milkie adapters or by an existing platform (e.g. a knowledge / serving stack
already in place at the organization).

The **SDK Facade** and **CLI Facade** are export surfaces, not subsystems.
They re-export the library's stable entrypoints and route calls to the three
subsystems below. Neither carries logic of its own. The CLI facade is
load-bearing for agent consumers (see cross-cutting decisions 12–13); concrete
shapes are a design-doc concern.

---

## Concept Model

milkie's vocabulary borrows words (event, trace, span, context, region) that
mean subtly different things across observability tools and agent frameworks.
This section gives each load-bearing term a canonical definition before the
rest of the document uses it. Each definition includes a *Not:* line naming
the concepts it is most often confused with.

### Event

An immutable, time-stamped record of one thing that happened inside a run —
the agent started, the LLM was called, a region entered context. Every Event
has a `type` (from a closed `EventKind` union), a `runId`, a `timestamp`, and
a typed `payload`. Events are append-only; nothing edits one after it is
written.

*Example:* one write of `llm.requested` to `IEventStore` is one Event.
*Not:* a span. Spans are intervals with start/end and parent/child; Events
are points.

### Trace

The ordered set of all Events belonging to one run, stored in an
`IEventStore` (today: one JSONL file per run under `.milkie/runs/`). The
Trace is the **substrate** of the Agent Trace subsystem — every higher-order
capability (replay, fork, lineage, cache lookup) derives from it, and
nothing else is canonical.

*Example:* `.milkie/runs/<runId>.jsonl` is one Trace.
*Not:* a UI timeline. A "trace timeline" panel is a *projection* of the
Trace, free to filter or group Events; the Trace itself is the raw log.

### Trajectory

A tree of `Span`s (with attached events on each span) collected during a run
by `TrajectoryStore` (`src/trajectory/`). Spans capture parent/child
structure — `llm.call` inside `tool.call` inside `fsm.step`. Trajectory is
the **older, span-shaped observability layer**; it coexists with Trace as a
peer view and is on track to become a projection over the event log (see
Implementation Status §Migration intentions).

*Example:* a `Span` for one `llm.call` with its child `tool.call`s nested
under it.
*Not:* the Trace. Trace is flat and replay-canonical; Trajectory is
hierarchical and derivable. Code calling `recorder.recordEvent(span, ...)`
writes to Trajectory but **not** to Trace — for an event to be
replay-visible or appear in event-log–backed UI, it must reach the event
log too.

### Region

An addressable chunk of the agent's working context — a system-prompt
fragment, a tool result, a retrieved document, a working-memory note. Every
Region has an identity, a kind, and content. Regions are the unit of "what
is currently in the context window."

*Example:* `region.added` with payload `{kind: 'tool_result', toolName:
'read_file', ...}` records one Region entering context.
*Not:* an Event. The Region itself is state owned by the Context Layer; the
Event is the *record* that the state changed.

### Context boundary

The rule that decides which Regions get assembled into the next LLM prompt,
and in what order. Boundaries are how milkie projects a possibly-large
working context down to a model-sized request.

*Example:* `context.boundary.applied` records one boundary decision, with a
payload describing which Regions were included, dropped, or compressed.
*Not:* the request itself. The boundary is the *policy* that produces the
request; the request lives in the subsequent `llm.requested` payload.

### IOPort

The single chokepoint through which the runtime crosses into nondeterminism
— clock, uuid, LLM calls, tool calls. Every call through IOPort is recorded
as an Event (`clock.read`, `uuid.generated`, `llm.requested/responded`,
`tool.requested/responded`); every replay reads from those Events instead of
re-calling out. This is what makes replay deterministic.

*Example:* `runtime.io.clock.now()` writes one `clock.read` Event in record
mode, reads it back in replay mode.
*Not:* the Events it produces. IOPort is the boundary; Events are its
recorded outputs. Code that bypasses IOPort (calling `Date.now()` directly)
creates nondet that replay cannot reproduce.

### SDK facade vs CLI facade

Two export surfaces over the same library substrate, distinguished by
audience: **SDK facade** for in-process consumers (apps, services, tests);
**CLI facade** for agent consumers (sub-agents that read traces,
meta-agents that propose variants). Neither carries logic of its own.

*Not:* subsystems. See the Overview diagram — facades are the top boxes;
subsystems (Agent Runtime, Agent Trace, Evolution) sit below them.

---

### Pairs that are often confused

| Pair | Key distinction |
|---|---|
| **Event** vs **Trace** | Event is one record. Trace is the ordered set of all Events in a run. |
| **Trace** vs **Trajectory** | Trace is flat, append-only, replay-canonical. Trajectory is a hierarchical span tree, derivable, currently a peer view (will be unified). |
| **IOPort** vs **Event** | IOPort is the nondet boundary. Events are its recorded outputs. |
| **Region** vs **`region.added` Event** | The Region is context state. The Event is the record that the state changed. |
| **Context boundary** vs **LLM request** | The boundary is the policy that selects/orders Regions. The request is its serialized output. |

---

## User-facing surfaces

milkie presents three abstraction layers to users, plus an optional UI
projection. They differ in audience and abstraction density, not in
capability — the same operation can be reached through any of them.

| Layer | What it is | Audience | Density |
|---|---|---|---|
| **CLI** | The verb-oriented command surface | New users, operators, agent consumers (per invariant 13) | Highest — one command per user intent |
| **SDK** | The `Milkie` class and convenience exports | TypeScript developers integrating milkie into an application | Medium — typed, ergonomic |
| **API** | Raw library exports: `IIOPort`, `IEventStore`, event types, all interfaces | Deep integrators replacing an IOPort, plugging in a storage backend, building a new event sink | Lowest — protocol contracts |

The **CLI** and **SDK** are the two explicit facades (see Overview). The
**API** is the library substrate beneath both — not a separate facade, but the
surface a user reaches when they need behavior the facades do not expose.

**Learning progression** is naturally CLI → SDK → API: try a command in a
shell, integrate into application code, drop to raw types for custom plumbing.
Documentation, examples, and onboarding should mirror this path.

**UI** is a deferred reference projection that milkie will ship — not a
fourth facade. The product thesis "runs as the primary engineering product"
requires a visual surface for humans to perceive runs (timelines, lineage
DAGs, fork trees, structural diffs, suite replay tables) as objects;
CLI / SDK alone do not afford the discovery these data structures need.
Form is TBD (local viewer / static report generator / IDE extension) and
the deliverable is deferred until the CLI output contracts stabilize, so
the UI is not pinned to a moving substrate.

The constraint is invariant: a UI must remain a **projection** over CLI /
SDK output. A UI may render trace timelines, lineage graphs, diff views,
suite replay results, etc.; user actions in the UI translate to CLI commands
or SDK calls — the UI does not own its own query logic, fork algorithm, or
state. Drift warning: when the UI starts answering questions the CLI cannot,
it has turned into a parallel facade and violates invariant 12.

The concrete CLI verb surface (commands, args, output shapes) is a
design-doc concern; see `docs/superpowers/specs/2026-05-24-cli-surface-design.md`.

---

## Subsystems (inside milkie)

### Agent Runtime

**Definition.** Agent Runtime is the execution engine that puts
**LLM-driven autonomy inside FSM structure**: each agent is a finite
state machine where the LLM reasons, picks tools, and decides
transitions. Intent-routed dialog, ReAct, multi-state workflows, and
multi-agent orchestration are not separate systems — they are
different FSM topologies on the same runtime.

**Goal.** Run every agent pattern — single-agent or multi-agent — on
one thin runtime whose primary output is a structured **run**, not a
free-form answer. The LLM's working context (instructions, history,
working memory) is explicit and managed, not buried in framework
internals; parallelism comes from the tool layer (multi-tool blocks in
one response for intra-agent, sub-agents-as-tools for inter-agent);
long-running agents survive interruption and resume from checkpoint
without losing their in-progress run.

**Value.** Adding multi-agent coordination doesn't force a framework
migration; the LLM's working context never becomes hidden state the
team has to reverse-engineer; long-running agents survive failures
without bolted-on resume logic. One runtime carries the team from
prototype to production without forcing migrations as the agent's
shape changes.

- **FSM Core** — Statechart-based runtime. Dialog and ReAct are not separate
  systems; they are different configurations of the same FSM. A continuous
  conversation loop is the degenerate case (one state, self-loop).
- **working context** — execution data held by the FSM, representing the
  agent's current working state (instructions, history, working memory,
  current turn, capability bindings). It is **data, not a layer**; it does not
  have independent architectural status. It is populated by querying the
  Context Layer.
- **IOPort** — the runtime's declared boundary for non-deterministic
  effects. Every LLM call, tool invocation, clock read, UUID, random,
  and external I/O must pass through it. The port exists for several
  legitimate reasons (test mocks, provider swaps, quota gating); Agent Trace
  is one consumer, not the reason the port exists.

The runtime is intentionally thin. Parallelism is expressed at the LLM/tool
layer (multiple `tool_use` blocks in a single response; sub-agents declared
as tools), not as built-in FSM state types.

### Agent Trace

**Definition.** Agent Trace is the subsystem where the agent **run lives
as a first-class object** — captured as an append-only event log,
queryable through inspection, derivable through replay, fork, and diff,
attributable through lineage. The log is the run's source of truth; all
operations are deterministic projections, not new computation.

**Goal.** Every agent run supports two layers of operation:
**inspection** (observable, diagnosable, lineage-traceable) and
**derivation** (replayable, forkable, diffable). Every IOPort call, FSM
transition, and non-deterministic value is captured once; operations
re-serve the shared prefix from cache and pay no LLM cost; recording the
run never changes how it executes.

**Value.**

- **Every production run replays deterministically.** State after
  replay is byte-identical to the original; bug reproduction,
  regression testing, and audit-ready evidence are routine.
- **Every decision is reconstructable.** For any decision point in
  any run, the prompt, response, and working context at that moment
  can be retrieved on demand; root-cause analysis, customer-facing
  explanations, and compliance reviews become procedural rather than
  investigative.
- **Every output is attributable.** Every claim, recommendation, or
  action in a run traces back to the LLM call, prompt, and source
  data that produced it.
- **Every proposed change can be tested against history.** Forking
  a historical run with a new configuration and diffing the outcome
  against the parent removes intuition from variant comparison.

The agent's product is the run, not its terminal output — and the run
is preserved, inspectable, and operable on the same terms as any other
production artifact.

- **Event log** — append-only sequence of events recording every IOPort call
  (request + response), every FSM transition that matters, and any
  authored side facts. The log is the substrate; every other Agent Trace
  primitive is a deterministic projection over it.
- **Response cache** — a content-addressed index projected from the event log.
  Model responses are keyed by a hash of the entire request (model id,
  messages, tool definitions, output schema). On replay, a cached response is
  served instead of calling the model.
- **Non-determinism log** — a positional projection from the event log
  capturing clock reads, UUIDs, random values, tool outputs, and other
  external I/O results, so replay can reuse recorded values rather than
  re-sampling the world.
- **Replay / Fork / Diff** — replay folds the event log back into runtime
  state; fork branches a run at any event, sharing the prefix from cache (no
  new model calls for the shared history); diff compares two runs structurally.
- **Lineage / Query API** — every produced artifact traces back to the
  IOPort call(s) and event(s) that produced it.

The relationship between these primitives mirrors event-sourced systems: the
event log is the source of truth, and Response cache, Non-determinism log, and
all query views are deterministic folds of it. Two replays of the same log
produce identical state.

Agent Trace is a **passive milkie subsystem with no embedded intelligence**.
It records, serves, and queries; it does not summarize, judge, or learn. Its
storage is provided by the Data layer; its event schema and access patterns
belong to Agent Trace itself.

**Consumers.** Agent Trace serves agent / machine consumers first:
meta-agents that read traces and propose variants, sub-agents that fork prior
runs as starting points, replay engines, evaluators, and audit automation.
The API shape is **agent-first** — uniform, structured, composable operations
on typed events — because agent consumers determine the system's scale
ceiling. Humans are indirect consumers of the same substrate through
projections (CLI rendering, dashboards, reports), not a reason to create
parallel core APIs. Agent consumers reach Agent Trace through the CLI facade
(see cross-cutting decision 13); operations like `fork` and `replay` return
run / event ids that subsequent operations key off, not in-memory state
objects.

**Implementation note.** The current codebase has `TrajectoryStore`
(`src/trajectory/`), a span-based observability store. It collects
`llm.call`, `tool.call`, `fsm.transition` spans for the run. This is the
predecessor of the event-sourced log described above — it gives some of
the same query surface (timeline inspection, attribute filtering) but
does not provide content-addressed response cache, non-determinism log,
deterministic replay, or fork. The Agent Trace section describes the
target shape; `TrajectoryStore` is the current bridge.

### Evolution

**Definition.** Evolution is the subsystem that runs controlled experiments
over agent configurations — registering variants, splitting traffic,
collecting outcomes, and applying mechanical promotion rules. It calls no
LLM and contains no embedded intelligence.

Goal and Value paragraphs are deliberately deferred — Evolution's
integration with the run-as-product stance is still being thought through.
For now, the existing component breakdown below stands as the operative
description.

- **Experiment Registry** — declares variant configurations, the metric
  to optimize, hard guardrails, and traffic split rules.
- **Traffic Splitter** — routes incoming requests to variants per the
  registered rules. Supports shadow, canary, and full A/B as first-class
  modes.
- **Outcome Collector** — pulls metric values from Agent Trace, watches
  guardrails, applies time-windowed and cohort-aware aggregation.
- **Promotion Gate** — rule-based decision (significance threshold +
  guardrail status) to promote, hold, or roll back a variant.

What Evolution **does not** contain:

- No Variant Proposer — proposing changes (prompt edits, FSM topology
  changes, skill set changes) comes from outside Evolution. The proposer
  can be a human, a script, or a **milkie meta-agent** that consumes
  Agent Trace through the CLI facade and registers experiments
  programmatically.
- No root-cause analysis — querying Agent Trace is the consumer's job
  (human or agent).
- No learning memory store — Experiment declarations live in the Experiment
  Registry; agent outcomes are read from Agent Trace; promotion/rollback
  decisions may be written as Evolution audit events.
- No multi-objective Pareto optimization — single metric + hard
  guardrails; combine objectives outside if needed.

Evolution **never calls an LLM**. All its decisions are mechanical.

---

## Infrastructure layers (outside milkie)

### Context Layer

The read-side projection. Materializes Data and Execution into what the
agent currently sees (knowledge, memory, history) and what it can invoke
(operator catalog). Responsibilities include skill loading, knowledge
retrieval, memory lookup, capability table resolution, and any
projection-side caching.

Agent Runtime depends on Context Layer for working context; it does not access
Data or Execution directly.

**Implementation note.** The external Context Layer described above is
**not implemented today** — there is no code that does knowledge
retrieval, memory lookup, or capability resolution from external sources.

What does exist in-tree is `src/context/ContextRegions.ts` plus
`src/context/assemble.ts` (and `src/context/lifecycleEngine.ts`). These
form a region-based **in-Agent assembly substrate**: each piece of
working context (header, skill, state instructions, working memory,
history pair, scratchpad, current turn, tool schema) is a typed
`Region`; the pure `assemble(regions, scope)` function produces the
LLM `ModelRequest` on every step; the lifecycle engine crystallizes
scratchpad into history at turn end.

The substrate is **not** a local shim for the external Context Layer.
Their responsibilities differ:

| Concern | External Context Layer (future) | In-Agent ContextRegions (today) |
|---|---|---|
| Knowledge retrieval from corpora / RAG | Yes | No |
| Memory lookup across sessions / agents | Yes | No |
| Capability / tool catalog resolution | Yes | No (tools wired at Agent config time) |
| Skill instruction loading | Yes (with versioned manifest) | Yes (in-Agent only, via `skill_request` system tool) |
| Per-call message / system-prompt assembly | Yes (output shape) | Yes (sole responsibility) |
| Region lifecycle (turn-local vs session-persistent vs TTL) | Not specified | Yes |
| prefix cache–aware section ordering | Not specified | Yes |

In short: the new substrate solves an in-Agent assembly problem the old
`ContextLayer` shim was beginning to grow into; it does not displace
the external Context Layer concept. When the external interface
materializes, `ContextRegions` stays where it is — Agent Runtime would
populate regions from values the external Context Layer delivers,
instead of constructing them itself.

**Two-tier memory pattern.** When the external Context Layer
materializes, "memory" naturally splits into two tiers by how it
surfaces to the LLM:

- **Layer 1: in-context (WM-style).** Small, always-relevant state —
  current intent, current plan, last decision. Rendered into every
  LLM request via the `wm` region. Should be **session-stable** to
  preserve prefix cache; mutations during a session cost cache hits.
- **Layer 2: external (tool-mediated).** Large, occasionally-relevant
  state — research findings, conversation summaries, user preferences,
  long task state. Persisted out-of-band; LLM accesses via
  `memory_read` / `memory_search` / `memory_write` tools. Tool results
  enter scratchpad (turn-local), not the in-context WM region.

Today only Layer 1 exists (`src/store/WorkingMemory.ts` + `wm` region).
Layer 2 — `IMemoryStore` interface + memory tool surface — is
roadmap.md "Memory tools (Layer 2)" and is one of the natural ways the
external Context Layer's "memory lookup" responsibility could be
prototyped before the full external infrastructure arrives.

The substrate's full design spec lives at
`docs/superpowers/specs/2026-05-25-context-region-substrate-design.md`.

### Data

Multi-source heterogeneous data aggregation. Not a generic store — its
job is to make varied backends look uniform to the layers above.

- **Metadata management** — catalogs, schemas, lineage at the data level
- **Connection management** — connectors, credentials, query routing
- **Source ingestion** — bringing external sources into the federation

Agent Trace's event log, non-determinism log, and response cache are persisted
*through* Data (connection routing), but the schema and operations belong to
Agent Trace.

### Execution

Registry of executable operators. Anything callable lives here under a
uniform invocation interface.

- skill
- function
- API / interface
- MCP servers
- (extensible)

**Skill belongs in Execution, not Data.** A skill is a callable
capability, not a configuration document.

IOPort, when it needs to invoke something, calls into Execution. It
does not need to know how that operator is implemented underneath.

### Foundation

The base layer beneath Data and Execution. A single layer that hosts
all foundational services rather than two separate descent paths.

- **Model Factory** — model serving, lifecycle, adapter routing.
  Execution holds an "LLM invocation" operator; the operator delegates
  here. Execution does not know what model serves the call, what
  version, or how it is routed.
- Storage engines, compute pools, and other base services live here too.

Foundation implementations are replaceable behind stable Data/Execution-facing
interfaces.

---

## Cross-cutting decisions (invariants)

These are the rules the architecture rests on. Violating any of them
typically signals a structural mistake, not a tradeoff.

1. **The run is the product; the output is a view.** An agent's
   deliverable is its full reasoning trajectory — decisions, claims,
   evidence, intermediate state, and lineage — not the terminal output
   alone. Storage, sharing, evaluation, and improvement all target the
   run; the output is one projection among many.
2. **IOPort is part of Agent Runtime's design**, not an Agent Trace-imposed
   hook. The port exists because the runtime declares its own
   non-determinism boundary. Agent Trace is a decorator implementation of
   that port, alongside test mocks and pass-through impls.
3. **Agent Runtime does not depend on Agent Trace.** Runtime depends only
   on its IOPort contract. Agent Trace is one IOPort decorator or
   implementation that observes and records calls.
4. **working context is data, not a layer.** What lives inside Agent Runtime
   under that name is execution state held by the FSM. It has no
   architectural standing of its own. The real Context Layer is below
   the milkie boundary.
5. **Read/write separation.** Context Layer is the read side (it
   answers "what does the agent know and can do?"). IOPort is the
   write side (it invokes operators in Execution). The two paths share
   underlying Data/Execution services through their respective interfaces;
   neither path subsumes the other.
6. **Skill is a capability, not a configuration.** It lives in
   Execution. Loading a skill means binding an Execution operator into
   the current turn's capability set; it does not mean copying a
   configuration document into context.
7. **Evolution is deterministic.** It never invokes an LLM. Any
   open-ended intelligence (proposing variants, clustering failures,
   judging outputs) is provided by external systems.
8. **Agent Trace does not contain intelligence.** It records, serves, and
   queries. It does not summarize, judge, or learn.
9. **No "Environment" concept.** Environment is too narrow or too
   broad to be a useful abstraction. Touchpoints attach to specific
   layers: users cross the milkie boundary as callers; external data
   enters through Data; external models/tools enter through Execution
   (and Foundation underneath); feedback enters through Agent Trace and
   Evolution.
10. **No Workflow System.** milkie does not include a deterministic
    workflow runtime. Deterministic flow, where required, is expressed
    as a specific FSM configuration in Agent Runtime, or delegated to
    an external system.
11. **Sub-agents are tools.** Inter-agent parallelism uses the
    sub-agent-as-named-tool pattern; it is not a special runtime
    construct. A sub-agent appears in the parent's Agent Trace as one
    event plus a nested sub-trace.
12. **milkie core is agent-first.** Non-UI milkie surfaces are designed for
    machine / agent consumers first: typed events, stable ids, structured
    inputs and outputs, deterministic operations, and composable commands.
    Humans consume projections over this substrate (dashboards, reports,
    pretty CLI rendering), not separate core APIs.
13. **CLI is the canonical agent-facing protocol facade.** Agent consumers
    reach milkie capabilities (especially Agent Trace) through CLI commands,
    not bespoke tool schemas. The CLI contract is machine-readable,
    non-interactive by default, id-oriented, bounded, and stable enough to
    serve as a protocol surface. Data goes to stdout; diagnostics go to
    stderr; structured output is the default for non-TTY consumers. Human
    friendly rendering is an optional projection. A new consumer-facing
    capability ships with a CLI entry point, or it is not reachable by agent
    consumers.

---

## Representative scenarios

Concrete uses each subsystem is designed to enable. Two entries for the
Agent Runtime substrate (the FSM-driven execution that emits everything
downstream observes), one entry per Agent Trace capability (the
6-capability surface — observable / diagnosable / lineage / replay / fork
/ diff), and the cross-cutting patterns that combine traces under
invariants 12–13. Stories under `docs/stories/` are the authoritative
spec; this section keeps the architecture self-explanatory without
depending on the story system to load.

### Runtime — ReAct agent with intra-agent parallel tools

A single agent runs a plan-and-act loop: the LLM emits a multi-step plan
that issues several `tool_use` blocks in one response; Agent Runtime
dispatches them concurrently and merges observations before the next FSM
step. The canonical exercise of FSM Core + intra-agent parallelism +
cognitive toolbox state on `workingMemory`, without sub-agents.
Story: [s-001](docs/stories/s-001-react-with-intra-agent-parallel-tools.md).

### Runtime — interrupt a long-running agent and resume from checkpoint

Mid-run an external interrupt signal reaches Agent Runtime at the next
yield point: state is checkpointed to the State Store, FSM moves to
`paused`, and `InterruptSignal` propagates through the supervisor tree
to any in-flight sub-agents. `milkie.resume(checkpointId)` continues
from the checkpoint as a single trajectory across the interrupt, with no
work duplicated. The canonical exercise of yield points + state stores +
supervisor-tree interrupt propagation.
Story: [s-008](docs/stories/s-008-long-task-interrupt-and-resume.md).

### Observable — inspect a completed run

Given a `runId`, return the timeline of every event (FSM transitions, LLM
calls, tool calls, working context updates) with filters by event type and
time window. Pure read, no replay, no causal traversal.
Story: [s-002](docs/stories/s-002-inspect-a-completed-run.md).

### Diagnosable — explain a specific decision

For any decision point in any run, return the working context at that
moment, the prompt sent to the LLM, the response received, and the
capability bindings in scope. Answers "why did the agent do this" without
re-execution.
Story: [s-003](docs/stories/s-003-explain-a-decision-with-context.md).

### Lineage — attribute an artifact (forward) and assess impact (reverse)

**Forward**: given a claim, recommendation, or output, return the causal
chain back to LLM calls, tool results, retrieved documents, and source
versions. **Reverse**: given a source identifier (content hash, retracted
document version), return every run whose lineage references it. Both
directions are typed-graph queries, not log searches.
Stories: [s-004](docs/stories/s-004-lineage-from-artifact-to-source.md)
(forward), [s-014](docs/stories/s-014-reverse-reference-lineage-query.md)
(reverse).

### Replay — reproduce a production run deterministically

Re-run a recorded run end-to-end with zero live LLM/tool calls; state
after replay matches the original. Local repro of prod failures, examples
that ship without API keys, regression baselines.
Story: [s-005](docs/stories/s-005-deterministic-replay.md).

### Fork — branch a run at any event

Continue a run from an arbitrary event with one parameter changed (prompt,
tool choice, configuration). Prefix is served from the response cache, so
the shared history is paid for once across the parent and all forks. The
single-fork building block for counterfactual analysis and variant search.
Story: [s-006](docs/stories/s-006-fork-at-event-for-what-if.md).

### Diff — structurally compare two runs at scale

Typed comparison of two event logs / projected graphs. The substrate for
suite replay: replay a saved suite of real production runs against a new
code version and classify each divergence (regression / improvement /
neutral) from structured output, not log diffing.
Story: [s-012](docs/stories/s-012-batch-replay-suite-and-classify-divergences.md).

### Cross-cutting — variant search at bounded cost

Fork × diff applied across N variant configurations. Cost is amortized via
the shared cache prefix: actual LLM calls scale as N × tail_size, not
N × full_run. This is the bridge between Agent Trace and Evolution — the
mechanism that makes systematic variant exploration economically viable.
Story: [s-013](docs/stories/s-013-variant-search-with-bounded-cost.md).

### Cross-cutting — runtime trace consumption by a sub-agent

A sub-agent spawned mid-run queries its parent's **in-flight** trace
through the CLI facade to verify, second-guess, or augment the parent's
reasoning — without redoing retrieval or re-calling the LLM for evidence
already produced. The canonical use of invariants 12–13: trace is a
substrate operable by other agents, not an after-the-fact artifact.
Story: [s-015](docs/stories/s-015-subagent-reads-parent-trace-runtime.md).

---

## One Turn Execution Path

For reference, what happens on a single agent step:

1. FSM enters a state that requires LLM input.
2. Agent Runtime asks Context Layer for the current working context
   (data + capabilities).
3. Context Layer pulls from Data (knowledge/memory) and Execution
   (capability catalog), returns assembled execution data.
4. Agent Runtime serializes the request and sends it through IOPort.
5. IOPort (Agent Trace decorator in effect):
   a. Hashes the request, writes an `llm.requested` event.
   b. If replay and the hash hits the cache: serves the cached response.
   c. Otherwise: calls Execution's LLM operator (which delegates to
      Foundation / Model Factory), writes an `llm.responded` event,
      returns the response.
   d. For non-LLM non-determinism (clock, UUID, random, tool I/O), records
      the observed result so replay can reuse it according to replay policy.
6. Agent Runtime consumes the response, FSM transitions, possibly
   emits tool calls (back through IOPort to Execution operators), and
   the cycle continues.

Agent outcomes flow into Agent Trace; Evolution observes Agent Trace;
Variant Proposers (external) read Agent Trace and register new experiments.

---

## Open questions

- **Foundation scope under Data.** It is assumed Data has Foundation
  dependencies (storage engines, indexing) in the same Foundation layer
  as Model Factory, rather than a separate descent path. Confirm as
  the storage side is built out.
- **Deterministic flow placement.** With Workflow System excluded,
  deterministic business flows live either as a specific Agent FSM
  configuration or in an external system. The event schema may need
  to distinguish LLM-driven runs from deterministic FSM configurations if
  this affects Agent Trace.
- **Evolution outcome attribution under sub-agent fan-out.** When a
  variant spawns sub-agents, attributing outcomes back to the variant
  needs explicit rules (whole subtree vs. only top-level events).
- **Replay side-effect policy.** Agent Trace records non-deterministic results,
  but the architecture still needs explicit policies for replaying external
  side effects: skip, mock, reinvoke, or require confirmation.

---

## Document scope

This file describes the architecture, not the concrete public API. Concrete
interface shapes (public library facade, IOPort signatures, Agent Trace event
schema, Context Layer query shape, Evolution experiment declaration) belong in
design docs under `docs/`. When those are written, link them here.
