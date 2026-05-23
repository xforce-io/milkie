# milkie Usage Guide

- [1. Core Concepts](#1-core-concepts)
- [2. Agent Configuration](#2-agent-configuration)
- [3. State Stores](#3-state-stores)
- [4. Model Adapters](#4-model-adapters)
- [5. Built-in Tools](#5-built-in-tools)
- [6. Multi-Agent Patterns](#6-multi-agent-patterns)
- [7. Interrupt & Resume](#7-interrupt--resume)
- [8. Trajectory & Observability](#8-trajectory--observability)
- [9. API Reference](#9-api-reference)

---

## 1. Core Concepts

### Agent = FSM

Every agent in milkie is a Finite State Machine. The FSM has two state types:

**`type: llm`** — Calls the LLM in a loop until one of:
- A tool calls `ctx.emit(event)` → FSM transitions to the target state declared in `on`
- The LLM produces plain text output → FSM emits `DONE` internally; transitions to `on.DONE` if declared, otherwise waits for the next user message (multi-turn mode)
- `max_iterations` is reached → throws `MaxIterationsError`; the agent run returns `status: 'error'`

**`type: action`** — Executes a deterministic handler synchronously, then transitions on `DONE` or `ERROR`. No LLM call.

```
type: llm state                     type: action state
─────────────────                   ──────────────────
  LLM call                            handler()
     ↓                                    ↓
  tool_use? → execute tools          emit DONE / ERROR
     ↓                                    ↓
  ctx.emit(event)?                   on.DONE → next state
     ↓ yes → transition
  plain text? → emit DONE
     ↓
  on.DONE → next state (or wait)
```

### FSM Global Transitions

Two reserved states are automatically available — you do not need to declare them:

| Reserved state | Trigger | Behavior |
|----------------|---------|----------|
| `error_handling` | A tool raises an error with `retryable: true` | The runtime transitions here temporarily, waits 500 ms, then retries the same tool call (up to 3 attempts). Non-retryable errors skip `error_handling` and return `isError: true` as an observation to the LLM. |
| `paused` | `milkie.interrupt(contextId)` is called | Saves a checkpoint and stops execution cleanly; `AgentResult.status` is `'interrupted'`. |

### Parallel Execution

**Intra-agent parallelism:** When the LLM produces multiple `tool_use` blocks in a single response, the runtime executes all `parallelSafe: true` tools concurrently. Non-parallel-safe tools are executed sequentially. Results are joined with `allSettled` — a failure in one tool does not cancel others.

**Inter-agent parallelism:** When an orchestrator agent calls multiple sub-agent tools in a single LLM response, the runtime spawns independent agent instances concurrently. Each sub-agent has its own FSM and context; their internal state is never visible to the parent.

### Context Buckets

> **Conceptual model.** The five-bucket ordering reflects how `ContextLayer` constructs the LLM request. Provider-level prefix caching and history compression are not implemented in v1.

Each agent's LLM context is organized into five ordered buckets:

```
[STABLE]
  system_prompt       never changes across the agent's lifetime
  instructions        loaded skill instructions (epoch-gated)

[DYNAMIC — rebuilt each turn]
  history             accumulated conversation history
  working_memory      intra-turn intermediate state
  current_turn        current user input (always last)
```

### Goal vs. Input

- **`goal`** — the agent run's immutable intent. Written to the `agent.run` span, used for A/B experiment comparison. Does not change across multi-turn continuations.
- **`input`** — the current turn's dynamic input. Changes each time you call `invoke()` with the same `contextId`.

---

## 2. Agent Configuration

### TypeScript Object

```typescript
import type { AgentConfig } from 'milkie'

const config: AgentConfig = {
  agentId:      'my-agent',
  version:      '1.0.0',
  systemPrompt: 'You are a helpful assistant.',
  fsm: {
    states: [
      {
        name:           'react',
        type:           'llm',
        max_iterations: 15,
        // tools: ['web_search']  // omit to inherit all registered tools
      },
    ],
  },
  model: {
    provider: 'anthropic',
    model:    'claude-sonnet-4-6',
    adapter:  'anthropic',
  },
  subAgents: {
    'researcher': '1.0.0',   // agentId → pinned version
  },
}
```

### Markdown Frontmatter File

Agents can also be defined as `.md` files with YAML frontmatter. The file body becomes `systemPrompt`.

```markdown
---
agentId: my-agent
version: "1.0.0"
fsm:
  states:
    - name: react
      type: llm
      max_iterations: 15
model:
  provider: anthropic
  model: claude-sonnet-4-6
  adapter: anthropic
sub_agents:
  researcher: "1.0.0"
---

You are a helpful assistant.
```

Load the file with:

```typescript
const config = milkie.loadAgentFile('./agents/my-agent.md')
```

### FSMState Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `string` | yes | Unique state name within the FSM |
| `type` | `'llm' \| 'action'` | yes | Execution type |
| `instructions` | `string` | no | Extra instructions injected for this state only (type: llm) |
| `tools` | `string[]` | no | Tool names available in this state; omit to inherit all agent tools |
| `on` | `Record<string, string>` | no | Event → target state transitions |
| `handler` | `string` | no | Handler name for type: action (typically a sub-agent agentId) |
| `terminal` | `boolean` | no | If true, FSM stops after this state produces output |
| `max_iterations` | `number` | no | Max LLM loop iterations (type: llm); defaults to unlimited |

---

## 3. State Stores

State stores persist checkpoints for interrupt/resume and multi-turn conversation history. All stores implement `IStateStore`.

### MemoryStore (default)

In-process, no dependencies. State is lost when the process restarts.

```typescript
import { Milkie, MemoryStore } from 'milkie'

const milkie = new Milkie({ stateStore: new MemoryStore() })
```

### SQLiteStore

Local persistent storage. No external service required.

```typescript
import { Milkie, SQLiteStore } from 'milkie'

const store = new SQLiteStore({ path: './data/state.db' })
await store.init()   // creates the table if it does not exist

const milkie = new Milkie({ stateStore: store })
// ...
store.close()        // call when done
```

### RedisStore

Cross-process, cross-session. Required for horizontally scaled deployments.

```typescript
import { Milkie, RedisStore } from 'milkie'

const store = new RedisStore({
  host: 'localhost',
  port: 6379,
  // password: '...',
  // db: 0,
})
await store.init()   // must be called before use

const milkie = new Milkie({ stateStore: store })
// ...
await store.disconnect()
```

### IStateStore Interface

You can implement a custom store by satisfying this interface:

```typescript
interface IStateStore {
  set(key: string, value: unknown, ttl?: number): Promise<void>
  get(key: string): Promise<unknown>
  delete(key: string): Promise<void>
  exists(key: string): Promise<boolean>
}
```

---

## 4. Model Adapters

### Anthropic

```typescript
model: {
  provider: 'anthropic',
  model:    'claude-sonnet-4-6',
  adapter:  'anthropic',
  // baseUrl: 'https://api.anthropic.com',  // optional override
}
```

Requires `ANTHROPIC_API_KEY` in the environment.

### OpenAI-Compatible

Works with any OpenAI-compatible endpoint: OpenAI, Azure OpenAI, Volcengine, DeepSeek, local Ollama instances, etc.

```typescript
model: {
  provider: 'openai',
  model:    'gpt-4o',
  adapter:  'openai-compatible',
  baseUrl:  'https://api.openai.com/v1',
}
```

```typescript
// Volcengine (Doubao)
model: {
  provider: 'volcengine',
  model:    'doubao-seed-2.0-lite',
  adapter:  'openai-compatible',
  baseUrl:  process.env['VOLCENGINE_API_BASE'],
}
```

The adapter reads API keys from environment variables in this order: `VOLCENGINE_TOKEN`, then `OPENAI_API_KEY`. Set the appropriate variable before running:

```bash
export OPENAI_API_KEY=sk-...
# or
export VOLCENGINE_TOKEN=your-token
export VOLCENGINE_API_BASE=https://your-endpoint/v1
```

### Custom Gateway

Inject any gateway that implements `IModelGateway` to override all agents:

```typescript
import type { IModelGateway, ModelRequest, ModelResponse } from 'milkie'

class MockGateway implements IModelGateway {
  async complete(req: ModelRequest): Promise<ModelResponse> {
    return {
      content:      [{ type: 'text', text: 'mock response' }],
      toolCalls:    [],
      finishReason: 'end_turn',
    }
  }

  async *stream(req: ModelRequest): AsyncIterable<never> {
    yield* []
  }
}

const milkie = new Milkie({ gateway: new MockGateway() })
```

This is useful for testing without making real API calls.

---

## 5. Built-in Tools

### Cognitive Toolbox

`cognitiveTools` and `systemTools` are **automatically registered** by the runtime for every agent — you do not need to pass them to `Milkie`. They are available in every agent's tool registry out of the box.

| Tool | Description |
|------|-------------|
| `think` | Records a reasoning step in working memory. No side effects. `parallelSafe: true`. |
| `create_plan` | Creates a checklist of steps stored in working memory. Call once at the start of a multi-step task. |
| `update_step` | Marks a step as `done` or `failed`. If failed, call `create_plan` again to revise. |

**System prompt guidance for create_plan / update_step:**

```
When given a multi-step task:
1. Call create_plan with all steps listed upfront.
2. Execute steps, calling update_step after each one completes.
3. After all steps are done, produce the final output.
```

### System Tools

These are also auto-registered. No manual setup required.

| Tool | Description |
|------|-------------|
| `skill_list` | Returns a list of available skills. (v1 stub — always returns an empty list; full registry support is planned.) |
| `skill_request` | Requests a skill to be loaded in the next context epoch. Requires the skill to be declared in `AgentConfig.skills` and its instructions provided via `AgentConfig.skillInstructions`. The instructions appear in the LLM context from the next turn onward. |

To wire up a skill, declare it in the agent config:

```typescript
const config: AgentConfig = {
  // ...
  skills: { research: '1.1.0' },
  skillInstructions: {
    research: `## Research Guidelines
Search for information and summarize findings with citations.`,
  },
}
```

### Defining Custom Tools

```typescript
import type { ToolDefinition } from 'milkie'

const myTool: ToolDefinition = {
  name:        'query_database',
  description: 'Query records from the database by ID.',
  inputSchema: {
    type:       'object',
    properties: {
      id: { type: 'string', description: 'Record ID' },
    },
    required: ['id'],
  },
  parallelSafe: true,   // safe to run concurrently with other tools
  handler: async (input, ctx) => {
    const { id } = input as { id: string }

    // ctx.workingMemory — read/write intra-turn state
    // ctx.emit(event)   — trigger an FSM state transition
    // ctx.stateStore    — read/write persistent state

    const record = await fetchRecord(id)
    ctx.workingMemory.set('lastQueried', id)

    if (!record) {
      ctx.emit('NOT_FOUND')
      return { found: false }
    }

    return { found: true, record }
  },
}
```

**`ToolContext` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `workingMemory` | `WorkingMemory` | Intra-turn scratchpad; persisted in checkpoints |
| `emit` | `(event, payload?) => void` | Trigger an FSM state transition |
| `stateStore` | `IStateStore` | Persistent key-value store |
| `agentFactory` | `AgentFactory` | Spawn child agents programmatically |

---

## 6. Multi-Agent Patterns

### Declaring Sub-Agents

Add sub-agents to `AgentConfig.subAgents`. The runtime automatically generates a named tool for each one. The orchestrator LLM calls them like any other tool.

```typescript
const orchestratorConfig: AgentConfig = {
  agentId:      'orchestrator',
  version:      '1.0.0',
  systemPrompt: `You coordinate research tasks.
Call researcher and writer in parallel, then combine their outputs.`,
  fsm: {
    states: [{ name: 'react', type: 'llm', max_iterations: 10 }],
  },
  model: { /* ... */ },
  subAgents: {
    'researcher': '1.0.0',
    'writer':     '1.0.0',
  },
}

milkie.registerAgent(researcherConfig)
milkie.registerAgent(writerConfig)
milkie.registerAgent(orchestratorConfig)
```

When the orchestrator LLM calls `researcher` and `writer` in a single response, they run concurrently in independent FSM + context instances.

### Parallel Sub-Agent Invocation

The orchestrator's system prompt should instruct the LLM to call sub-agents in a single response:

```
Call researcher and writer simultaneously in one response — do not call them one at a time.
```

The framework handles concurrency automatically. Results arrive as observations once all sub-agents complete (`allSettled` semantics).

### Sub-Agent Tool Schema

Each auto-generated sub-agent tool accepts:

```typescript
{
  goal:  string   // the sub-agent's immutable intent for this run
  input: string   // the current turn's dynamic input
}
```

### Action State for Routing

Use `type: action` to route to a sub-agent deterministically (no LLM required):

```typescript
{
  name:    'route_to_billing',
  type:    'action',
  handler: 'billing-specialist',   // matches a registered agentId
  on: { DONE: 'completed' },
}
```

### Context Isolation

Sub-agents cannot read or write the parent agent's working memory or conversation history. Information passes explicitly:
- **Parent → child:** via the `goal` and `input` arguments
- **Child → parent:** via the tool result returned to the parent's context

---

## 7. Interrupt & Resume

### Interrupting a Running Agent

```typescript
const contextId = `ctx-${Date.now()}`

// Start a long-running agent
const runPromise = milkie.invoke({
  agentId: 'analyst',
  goal:    'Process 1000 records',
  input:   'Begin processing',
  contextId,
})

// Interrupt at any time
setTimeout(() => milkie.interrupt(contextId), 5000)

const result = await runPromise
// result.status === 'interrupted'
// checkpoint is stored at: context:{contextId}:checkpoint:latest
```

Interruption is cooperative: the runtime checks for the interrupt signal at yield points (before/after each tool call, before each LLM call). In-flight tool calls complete before the interrupt is processed.

When interrupted, the checkpoint key follows the pattern:
```
context:{contextId}:checkpoint:latest
```

### Resuming

```typescript
const checkpointKey = `context:${contextId}:checkpoint:latest`

const result = await milkie.resume(
  checkpointKey,
  'analyst',
  'Process 1000 records',    // same goal
  'Continue from where you left off',
)
// result.status === 'completed'
```

The resumed run continues with the same `contextId` and `agentRunId`, and trajectory spans are appended to the same trace.

### Multi-Turn Conversations (without interrupt)

Reuse a `contextId` across multiple `invoke()` calls to carry conversation history forward:

```typescript
// Turn 1
const turn1 = await milkie.invoke({
  agentId: 'assistant',
  goal:    'Help the user with order #12345',
  input:   'My order seems delayed.',
})

// Turn 2 — same contextId continues the conversation
const turn2 = await milkie.invoke({
  agentId:   'assistant',
  goal:      'Help the user with order #12345',
  input:     'Can you check the shipping status?',
  contextId: turn1.contextId,
})
```

---

## 8. Trajectory & Observability

### Recording Spans

```typescript
import { Milkie, TrajectoryStore } from 'milkie'

const trajectoryStore = new TrajectoryStore({
  jsonlDir: './trajectories',   // writes one JSONL file per run
})

const milkie = new Milkie({ trajectoryStore })
```

### Span Types

| Span | When | Attributes |
|------|------|------------|
| `agent.run` | Entire agent execution (root span) | `agentId`, `goal`, `contextId` |
| `fsm.transition` | Each FSM state change | `fromState`, `toState`, `event` |
| `llm.call` | Each model API call | `provider`, `model`, `turn`, `state`, `loadedSkills`, `contextEpoch`; token usage is recorded as a `usage` event (not a span attribute) |
| `tool.call` | Each tool execution | `toolName`, `toolCallId`, `input`, `turn`, `attempt`, `parallelBatchId`; `output` added on success; duration = `span.endTime - span.startTime` |
| `agent.spawn` | Each sub-agent launch | `childAgentId`, `taskId`, `childTraceId`, `childContextId`, `turn`; `resultStatus` and `checkpointId` added on completion |

### Querying Trajectories

```typescript
// By run ID (most precise)
const traj = await trajectoryStore.getByRunId(result.agentRunId)

// By context ID (all runs in a conversation)
const traj = await trajectoryStore.getByContextId(contextId)

// All spans of a given type
const toolSpans = traj.spans.filter(s => s.name === 'tool.call')
const llmCalls  = traj.spans.filter(s => s.name === 'llm.call')
```

### ResolvedManifest

Every trajectory captures a full dependency snapshot at run time:

```typescript
traj.resolvedManifest.agentVersion   // '1.2.0'
traj.resolvedManifest.model.model    // 'doubao-seed-2.0-lite'
traj.resolvedManifest.skills         // { research: { version: '1.1.0' } }
traj.resolvedManifest.subAgents      // { researcher: { version: '1.0.0' } }
```

### A/B Experiments

Run the same goal against two agent versions and compare trajectories:

```typescript
const [r1, r2] = await Promise.all([
  milkie.invoke({ agentId: 'agent-v1', goal, input: goal }),
  milkie.invoke({ agentId: 'agent-v2', goal, input: goal }),
])

const [t1, t2] = await Promise.all([
  trajectoryStore.getByRunId(r1.agentRunId),
  trajectoryStore.getByRunId(r2.agentRunId),
])

// Compare outputs, token usage, tool call counts, etc.
console.log(r1.output, r2.output)
console.log(t1.resolvedManifest.skills, t2.resolvedManifest.skills)
```

---

## 9. API Reference

### `new Milkie(options?)`

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `stateStore` | `IStateStore` | `new MemoryStore()` | Persists checkpoints and multi-turn history |
| `gateway` | `IModelGateway` | `undefined` | Override gateway for all agents (useful for testing) |
| `tools` | `ToolDefinition[]` | `[]` | Tools available to all registered agents |
| `trajectoryStore` | `TrajectoryStore` | `undefined` | Records spans; omit to disable tracing |

### `milkie.registerAgent(config)`

Registers an `AgentConfig`. Must be called before `invoke()`. Registered agents are also available as sub-agents to any orchestrator.

### `milkie.loadAgentFile(filePath)`

Loads an agent from a Markdown file with YAML frontmatter. Returns the parsed `AgentConfig` and registers it.

### `milkie.invoke(request)`

```typescript
interface AgentInvokeRequest {
  agentId:    string    // registered agent ID
  goal:       string    // immutable run intent
  input:      string    // current turn input
  contextId?: string    // omit to start a new conversation
}

interface AgentResult {
  agentRunId: string
  contextId:  string
  output:     string
  status:     'completed' | 'interrupted' | 'error'
}
// When status is 'interrupted', the checkpoint is saved at:
// context:{result.contextId}:checkpoint:latest
```

### `milkie.resume(checkpointKey, agentId, goal, input)`

Resumes an interrupted run from a saved checkpoint. Continues with the same `agentRunId` and `contextId`.

```typescript
const result = await milkie.resume(
  'context:ctx-abc:checkpoint:latest',
  'my-agent',
  'original goal',
  'continue processing',
)
```

### `milkie.interrupt(contextId)`

Signals the agent bound to `contextId` to stop at the next yield point. Also propagates the interrupt to any running sub-agents. Returns immediately; the actual stop happens asynchronously.

### `milkie.registerTool(tool)`

Adds a single tool to the runtime after construction. Equivalent to passing it in `options.tools`.

### `milkie.getAgent(agentId)`

Returns the registered `AgentConfig` for the given ID, or `undefined` if not found.
