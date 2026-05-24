# milkie

**A TypeScript library for LLM agents — where every run is a first-class engineering artifact.**

milkie is a TypeScript library for building LLM-powered agents. Its central
commitment: **every agent run — the full reasoning trajectory, not just
the output — is a first-class engineering product**: addressable,
reproducible, forkable, comparable, and attributable.

Under the hood, every agent pattern (dialog, ReAct, multi-state workflows,
multi-agent orchestration) is a finite state machine on one thin runtime.
The runtime, an event-sourced **Agent Trace**, and a deterministic
**Evolution** experiment subsystem together form three peer subsystems.

[中文文档](./README_CN.md) · [Usage Guide](./docs/en/guide.md) · [Architecture](./ARCHITECTURE.md)

---

## Features (implemented today)

- **Agent = FSM** — two state types (`llm` / `action`) compose every agent pattern without special-casing
- **Intra-agent parallelism** — LLM emits multiple `tool_use` blocks in one response; runtime executes them concurrently with `allSettled` join
- **Inter-agent parallelism** — sub-agents declared as named tools; orchestrators spawn them in parallel across independent FSM + context instances
- **Interrupt / Resume** — cooperative yield points save checkpoints; any interrupted run resumes from the exact state
- **Multi-turn conversations** — share a `contextId` across `invoke()` calls to accumulate history
- **Event-sourced Agent Trace** — append-only event log records every LLM / tool I/O with `causedBy` chains; `Milkie.replay(runId)` re-runs a recorded run from the log with zero live LLM calls (Phase 3 structural replay)
- **`milkie` CLI** — `agent list / run / resume / interrupt` and `trace inspect / replay` over a `.milkie/agents.json` manifest; SQLite-backed default state store so interrupt / resume work across CLI processes. The CLI is the canonical agent-facing surface (ARCHITECTURE.md invariants 12–13) and every verb maps 1:1 to an SDK call
- **Pluggable backends** — swap state stores (Memory / SQLite / Redis) and trajectory recorders (JSONL / in-memory / console) without touching agent logic
- **Provider-agnostic** — Anthropic and any OpenAI-compatible endpoint out of the box

**Target capabilities** still in development — fork / diff / lineage as
first-class operations over the event log, a non-determinism log for
byte-identical replay, and the Evolution experiment subsystem. See
[ARCHITECTURE.md → Implementation Status](./ARCHITECTURE.md#implementation-status)
for what's in code today vs. what's target architecture.

---

## Install

```bash
npm install milkie
```

Node.js ≥ 20 required.

---

## Quickstart

```typescript
import { Milkie, MemoryStore, TrajectoryStore } from 'milkie'
import type { AgentConfig, ToolDefinition } from 'milkie'

// 1. Define a tool
const webSearch: ToolDefinition = {
  name: 'web_search',
  description: 'Search the web for information.',
  inputSchema: {
    type: 'object',
    properties: { query: { type: 'string' } },
    required: ['query'],
  },
  parallelSafe: true,
  handler: async (input) => {
    const { query } = input as { query: string }
    return { result: `Results for "${query}"` }  // replace with real search
  },
}

// 2. Configure an agent
const researchAgent: AgentConfig = {
  agentId: 'researcher',
  version: '1.0.0',
  systemPrompt: 'You are a research assistant. Use web_search to answer questions accurately.',
  fsm: {
    states: [{ name: 'react', type: 'llm', max_iterations: 10 }],
  },
  model: {
    provider: 'volcengine',       // or 'anthropic'
    model: 'doubao-seed-2.0-lite',
    adapter: 'openai-compatible',
    baseUrl: process.env['VOLCENGINE_API_BASE'],
  },
}

// 3. Run
const milkie = new Milkie({
  stateStore: new MemoryStore(),
  tools: [webSearch],
})
milkie.registerAgent(researchAgent)

const result = await milkie.invoke({
  agentId: 'researcher',
  goal: 'Summarize the key features of TypeScript 5.0',
  input: 'What are the main new features in TypeScript 5.0?',
})

console.log(result.output)
// result.status: 'completed' | 'interrupted' | 'error'
```

---

## CLI

Once the package is installed, the same agent runs from the shell. Drop a
manifest under `.milkie/agents.json` and call the CLI from any directory
inside the project:

```bash
$ cat .milkie/agents.json
{ "agents": [{ "id": "researcher", "file": "../agents/researcher.md" }] }

# 1. List registered agents (auto-loads the manifest on startup)
$ milkie agent list
{"id":"researcher","source":"manifest"}

# 2. Run an agent — records to .milkie/runs/<runId>.jsonl
$ milkie agent run researcher --input "What's new in TypeScript 5.0?"
{"runId":"...","contextId":"...","status":"completed","lastOutput":"..."}

# 3. Replay the recorded run — zero live LLM calls
$ milkie trace replay <runId>
{"newRunId":"...","status":"completed","output":"..."}

# 4. Inspect every event in the run as JSONL
$ milkie trace inspect <runId>
{"id":"...","runId":"...","type":"agent.run.started",...}
{"id":"...","runId":"...","type":"llm.requested",...}
...
```

Available verbs: `agent list / run / resume / interrupt`,
`trace inspect / replay`. See
[CLI surface design](./docs/superpowers/specs/2026-05-24-cli-surface-design.md)
for the full contract and
[agent registration design](./docs/superpowers/specs/2026-05-24-agent-registration-design.md)
for the `.milkie/agents.json` manifest convention.

---

## Core Concept: Agent = FSM

Every agent is described by a list of **states**. There are only two state types:

| Type | Behavior |
|------|----------|
| `llm` | Calls the LLM in a loop. Exits when a tool emits an FSM event, or on plain text output (`DONE`). |
| `action` | Runs a deterministic handler (e.g. spawn a sub-agent). No LLM call. |

States declare transitions in an `on` map:

```typescript
fsm: {
  states: [
    {
      name: 'classify',
      type: 'llm',
      tools: ['classify_intent'],   // tool emits INTENT_ORDER or ESCALATE
      on: {
        INTENT_ORDER: 'collect_slots',
        ESCALATE:     'escalated',
      },
    },
    {
      name: 'collect_slots',
      type: 'llm',
      tools: ['collect_slot'],      // tool emits SLOTS_COMPLETE when all filled
      on: { SLOTS_COMPLETE: 'confirm' },
    },
    {
      name:     'escalated',
      type:     'llm',
      terminal: true,               // no exit — produces final message
    },
  ],
}
```

A tool triggers a transition by calling `ctx.emit()` in its handler:

```typescript
handler: async (input, ctx) => {
  const { intent } = input as { intent: string }
  ctx.emit(intent === 'order' ? 'INTENT_ORDER' : 'ESCALATE')
  return { intent }
}
```

See the [Usage Guide](./docs/en/guide.md) for multi-agent orchestration, interrupt/resume, and the full API reference.

---

## Architecture

milkie is structured as three peer subsystems:

- **Agent Runtime** — execution engine that puts LLM-driven autonomy inside FSM structure
- **Agent Trace** — preserves every agent run as a first-class object; supports inspection, replay, fork, diff, and lineage as deterministic projections over the event log
- **Evolution** — deterministic experiment subsystem for iterating agent configurations

The full target architecture, cross-cutting invariants, and an
[Implementation Status](./ARCHITECTURE.md#implementation-status) section
calibrating current code vs. target are documented in
[ARCHITECTURE.md](./ARCHITECTURE.md).

User-facing scenarios are tracked under [docs/stories/](./docs/stories/) with
their own [README](./docs/stories/README.md) and
[INDEX](./docs/stories/INDEX.md).

---

## Examples

Runnable demos paired with their stories live under
[examples/](./examples/). Each example ships an SDK script and the
equivalent CLI invocation over a frozen fixture — no API key required.

- [`s-005-replay`](./examples/s-005-replay/) — deterministic replay
  (Phase 3): record a run with an in-process stub gateway, then replay
  the recorded run twice (once via SDK, once via CLI) and observe
  identical output with zero live LLM calls.

---

## License

MIT
