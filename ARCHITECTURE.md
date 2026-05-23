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

**Value.** A milkie agent's product is the **run** — diagnosable,
debuggable, auditable, and improvable as a first-class artifact. Outputs
are views over runs, not vice versa. Agents that produce only outputs
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
- **working context (basic)** — Bucket structure (system / instructions /
  history / working memory / current turn) assembled by an internal
  `ContextLayer` class. (`src/context/`)
- **Trajectory / span observability** — `TrajectoryStore` collects spans
  for `llm.call`, `tool.call`, `fsm.transition`, etc. This is the
  predecessor of the event-sourced Agent Trace described below.
  (`src/trajectory/`)
- **IOPort as non-determinism boundary** — `IIOPort` / `DefaultIOPort`
  in `src/runtime/IOPort.ts`. Agent Runtime routes every LLM call, tool
  invocation, clock read, and UUID generation through it. Current
  `DefaultIOPort` is a passthrough; recording / cache / replay variants
  are target. (`src/runtime/IOPort.ts`)
- **State stores for checkpoint/resume** — MemoryStore / SQLiteStore /
  RedisStore for interrupt/resume scenarios. (`src/store/`)

### Target only (not yet in code)

- **Event-sourced Agent Trace** — Including content-addressed response
  cache, non-determinism log, deterministic replay, fork, structural diff,
  and event-log-based lineage. Current Trajectory is span-based; event log
  is target.
- **External Context Layer / Data / Execution / Foundation** — These
  infrastructure layers are described as outside the milkie boundary.
  Today they are partially internal (`src/context/ContextLayer.ts`) or
  absent (no Data / Execution abstraction; gateway substitutes for
  Foundation).
- **Evolution subsystem** — Experiment Registry, Traffic Splitter, Outcome
  Collector, Promotion Gate — none implemented.
- **Capability vocabulary in full** — observable / diagnosable / lineage /
  replay / fork / diff are described as a 6-capability surface; today only
  basic span query exists via TrajectoryStore.

### Migration intentions (not commitments)

- **Trajectory → Event log**: When event-sourced Agent Trace lands,
  TrajectoryStore either becomes a projection over the event log or is
  retired in favor of the event log's native query surface.
- **Internal ContextLayer → External Context Layer**: The current
  `src/context/ContextLayer.ts` is treated as a local adapter / shim that
  anticipates the external interface. When external infrastructure exists,
  responsibilities migrate outward.
- **IOPort → Execution operator**: IOPort currently passes through to
  the gateway directly. The target is to route invocations through an
  Execution layer operator (which itself delegates to Foundation /
  Model Factory).

This section is the source of truth for "what works today." When code lands
that closes a gap, update this section before claiming the gap is closed
elsewhere.

## Overview

milkie is the **application layer** of an agent stack, packaged as a library.
Its public API is the stable entrypoint for applications, CLIs, services, and
test harnesses; CLI is a consumer of the library, not a core subsystem.

The library contains three peer subsystems (Agent Runtime, Agent Trace,
Evolution) and sits on top of layered infrastructure (Context Layer → Data +
Execution → Foundation).

```
Applications / CLI / Services / Test Harnesses
                         │
                         ▼
┌──── milkie library ──────────────────────────────────────┐
│                                                           │
│   ┌──────────────────────────────────────────────────┐   │
│   │              Public API / SDK Facade              │   │
│   └────────────────────────┬─────────────────────────┘   │
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

The **Public API / SDK Facade** is an export surface, not a subsystem. It
re-exports the library's stable types and entrypoints and routes calls to the
three subsystems below it. It carries no logic of its own and has no separate
section in this document; its concrete shape is a design-doc concern.

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
  changes, skill set changes) is external (human, script, or another
  milkie agent that consumes Agent Trace and registers experiments).
- No root-cause analysis — querying Agent Trace is the user's job.
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

**Implementation note.** Today, `src/context/ContextLayer.ts` is an
**internal** class instantiated directly by `AgentRuntime`. It assembles
working-context buckets (system / history / tools / working memory) but
does not yet pull from a distinct Data or Execution layer — those layers
do not exist as separate runtime concepts. The internal class is treated
as a **local adapter / shim** that anticipates the external Context Layer
interface; when external infrastructure lands, responsibilities migrate
outward. New contributors should not infer from the current file location
that Context Layer is permanently inside milkie's boundary; the target
position is below it.

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
