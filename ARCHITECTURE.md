# milkie Architecture

This document captures milkie's top-level architecture and the design decisions
that hold it together.

It describes the **target** architecture for milkie as a library.

> **Target document.** Strong claims throughout (e.g. "every run replays
> deterministically") are **target invariants** — not necessarily current
> behavior. For what's implemented today, what's still target-only, and
> migration intentions, see `roadmap.md`.

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

### FSM

The finite state machine that structures an agent's control flow: each agent
*is* an FSM whose states are phases of work (`classify`, `handle_a`, …) and
whose transitions move between them. The LLM reasons *inside* a state, picks
tools, and emits the events that drive transitions — autonomy lives in the
state, structure lives in the machine. Dialog, ReAct, and multi-state
workflows are different FSM topologies on one runtime, not separate systems; a
continuous conversation loop is the degenerate one-state self-loop. Implemented
by `FSMEngine` (`src/fsm/`); a state is `{name, type: 'llm'|'action', on:
Record<eventName, targetState>, …}`.

*Example:* a routing agent `classify → {handle_a, handle_b}`, each state's
`on:` mapping an emitted event name to its target.
*Not:* the `llm.call` / `tool.call` inside a state. Those are IOPort effects
nested within one `fsm.step` (`llm.call inside tool.call inside fsm.step`), not
machine states — the FSM owns *which state*, never *how one LLM/tool call
executes*. *Not:* the working context; that is the data a state holds (see
State, by level), not the state itself.

Two axes classify a state. By **execution type** (the `type` field): an `llm`
state lets the LLM reason, pick tools, and emit the transition event; an
`action` state runs a named deterministic `handler` with no LLM call. By
**role**:

| Kind | `type` | Example | Notes |
|---|---|---|---|
| Initial | either | first entry in `states[]` | where every run starts |
| LLM work | `llm` | `classify` (router), a ReAct `think` self-loop, a `chat` conversation self-loop | reasons and routes |
| Action work | `action` | `handle_a` (`handler: doA`), `format_output` | deterministic, no LLM |
| Terminal | either, `terminal: true` | `done` | FSM halts here |
| Reserved (framework-injected, not in user config) | `action` | `paused` (terminal, on `interrupt`), `error_handling` (on `error`), `failed` (terminal) | global signals jump here, overriding `on:` |

### Transition

A move from one FSM state to the next, selected by the **name** of an emitted
event through the current state's `on:` map. Today this is a pure name lookup;
conditional guards (evaluate a predicate before allowing the move) are a
planned extension (#31). Global signals (`interrupt`, `error`) override
state-local `on:` mappings.

*Example:* `fsm.transition` with payload `{from: 'classify', to: 'handle_b',
trigger: {name: 'INTENT_B', …}}`, written when a tool calls
`ctx.emit('INTENT_B')`.
*Not:* the event that triggered it. The emitted event is the *cause*; the
transition is the *effect* recorded as `fsm.transition`. *Not:* the decision of
*which* event to emit — that judgement happens inside the state (today inside
the LLM/tool) and is itself not yet traced; surfacing it is the subject of #31
(guard evaluation).

### State, by level

"State" is overloaded in milkie — three distinct things live on three distinct
mechanisms. Naming the level removes most of the ambiguity.

| Level | What it is | Mechanism / owner | Lifetime | Recorded as |
|---|---|---|---|---|
| **Effect** | a single LLM or tool call | IOPort | within one `fsm.step` | `llm.requested/responded`, `tool.requested/responded` |
| **FSM state** | the current machine state — *where in the flow* the agent is | FSM Core (`FSMEngine`) | across turns; survives interrupt via checkpoint | `fsm.transition` |
| **Working context** | session data held while in a state — history, working memory, scratchpad, current turn | Context Layer / `ContextRegions` | turn-local (scratchpad) vs session-persistent (history, WM) vs TTL | `region.added/removed` |

*Example:* in one turn the agent sits in FSM state `classify` (level 2), makes
an `llm.call` then a `tool.call` (level 1) whose result lands in the scratchpad
(level 3, turn-local); at turn end the lifecycle engine crystallizes scratchpad
into history (level 3, session-persistent).
*Not:* one thing. Unqualified "the agent's state" is ambiguous — an effect, a
machine state, and session data are different concerns; name the level.

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

### Event-sourced runtime state

Runtime state used for observability, diagnosis, replay, fork, and lineage is
reconstructed from the append-only event log plus content-addressed object
stores. Snapshots may exist as checkpoints, resume artifacts, or acceleration
indexes, but they are derived artifacts, not canonical truth.

For context Regions specifically, `region.added` / `region.removed` Events
describe lifecycle changes, while content hashes point to immutable Region
content objects. A "context at Event X" view is produced by folding those
Events up to X. Persisting complete `ContextRegions` snapshots at every
moment is not the model; if a snapshot is used to speed up a query, it must be
rebuildable from the event log plus the content-addressed store.

*Example:* a trace inspector reconstructs the active Region map before an
`llm.requested` Event by folding prior Region lifecycle Events and fetching
their content by hash.
*Not:* checkpoint restore. Checkpoints are runtime recovery artifacts; they
may contain Region snapshots, but they are not the audit source for explaining
historical context.

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
| **FSM state** vs **`llm.call` / `tool.call`** | The FSM state is the flow position the machine owns. The call is an IOPort effect nested inside one `fsm.step`, not a machine state. |
| **FSM state** vs **working context** | The state is *where in the flow*; the working context is the *data held while there* (history, WM, scratchpad). |
| **Transition** vs **the event that triggers it** | The event (emitted by a tool) is the cause; the transition (`fsm.transition`) is the effect. |
| **Context boundary** vs **LLM request** | The boundary is the policy that selects/orders Regions. The request is its serialized output. |
| **Snapshot** vs **Event-sourced view** | A snapshot is a checkpoint or acceleration artifact. The view is derived from Events plus content-addressed objects and remains reconstructable without the snapshot. |

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
- **Region content store** — a content-addressed object store for Region
  content and, where needed, rendered Region output. Region lifecycle Events
  carry hashes into this store; context-at-time and why-this-call queries fold
  the event log and dereference those hashes rather than reading live runtime
  state or relying on per-moment full snapshots.
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
