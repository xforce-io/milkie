# milkie

**TypeScript Agent framework — Agent = FSM**

[![npm](https://img.shields.io/npm/v/milkie)](https://www.npmjs.com/package/milkie)
[![build](https://img.shields.io/github/actions/workflow/status/milkie/milkie/ci.yml)](https://github.com/milkie/milkie/actions)
[![license](https://img.shields.io/badge/license-MIT-blue)](#license)

milkie is a TypeScript framework for building LLM-powered agents. Its core insight: **every agent pattern is a Finite State Machine (FSM)**. ReAct loops, intent routing, slot filling, multi-turn conversations — all are different FSM configurations running on the same runtime.

[中文文档](./README_CN.md) · [Usage Guide](./docs/en/guide.md)

---

## Features

- **Agent = FSM** — two state types (`llm` / `action`) compose every agent pattern without special-casing
- **Intra-agent parallelism** — LLM emits multiple `tool_use` blocks in one response; runtime executes them concurrently with `allSettled` join
- **Inter-agent parallelism** — sub-agents declared as named tools; orchestrators spawn them in parallel across independent FSM + context instances
- **Interrupt / Resume** — cooperative yield points save checkpoints; any interrupted run resumes from the exact state
- **Multi-turn conversations** — share a `contextId` across `invoke()` calls to accumulate history
- **Pluggable backends** — swap state stores (Memory / SQLite / Redis) and trajectory recorders (JSONL / in-memory / console) without touching agent logic
- **Provider-agnostic** — Anthropic and any OpenAI-compatible endpoint out of the box

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

## License

MIT
