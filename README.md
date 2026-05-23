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
- **Pluggable backends** — swap state stores (Memory / SQLite / Redis) and trajectory recorders (JSONL / in-memory / console) without touching agent logic
- **Provider-agnostic** — Anthropic and any OpenAI-compatible endpoint out of the box

**Target capabilities** under development — event-sourced Agent Trace
(replay / fork / diff / lineage), IOPort non-determinism boundary,
Evolution experiment subsystem. See
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

## License

MIT
