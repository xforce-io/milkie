# Phase 3 Cache + Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add content-addressed cache + `Milkie.replay(runId)` that re-runs a recorded agent run from the Phase 2 event log without making any real LLM/tool calls; result is structurally equivalent to the original run.

**Architecture:** event log = single source of truth. Replay loads a run's events, builds two projections (`RunSnapshot` for lifecycle identity, `CacheIndex` for I/O queues keyed by content hash), then runs a fresh `AgentRuntime` whose `IOPort` serves LLM/tool calls from `CacheIndex` and fail-fasts on cache miss. Same-host / same-code / same-registered-agent replay only — no portable replay, no sub-agent fail-fast detection, no fork.

**Tech Stack:** TypeScript, Node built-in `crypto`, Jest. No new external deps.

**Spec:** `docs/superpowers/specs/2026-05-24-phase-3-cache-and-replay-design.md`

## File Structure

**Created:**
- `src/trace/hash.ts` — `hashModelRequest`, `hashToolCall`, canonical JSON helper
- `src/trace/CacheIndex.ts` — FIFO queues per hash, `consumeLLM` / `consumeTool` / `remaining` / `allHashes`
- `src/trace/RunSnapshot.ts` — `extractRunSnapshot(events)` function
- `src/trace/ReplayingIOPort.ts` — `IIOPort` impl that serves from `CacheIndex`
- `src/trace/ReplayDivergenceError.ts` + `src/trace/ReplayError.ts` — typed errors
- `src/__tests__/Hash.test.ts`
- `src/__tests__/CacheIndex.test.ts`
- `src/__tests__/ReplayingIOPort.test.ts`
- `src/__tests__/Replay.test.ts`
- `tests/e2e/s-005-deterministic-replay.e2e.test.ts`

**Modified:**
- `src/trace/types.ts` — upgrade `ToolRespondedPayload.error` to structured; add `agent.run.started/completed` event kinds + payloads; add `requestHash` to LLM/tool payloads
- `src/trace/RecordingIOPort.ts` — compute `requestHash` on LLM/tool emit; add `attach` / `detach` methods; structured tool error
- `src/runtime/Milkie.ts` — call `attach` before run / `detach` in finally; add `replay(runId)` method
- `src/__tests__/Trace.test.ts` — extend for `requestHash`, lifecycle events, structured error
- `ARCHITECTURE.md` — move cache + replay from Target to Implemented
- `docs/stories/INDEX.md` — s-005 readiness update
- `docs/stories/s-005-deterministic-replay.md` — status: draft → active

---

## Conventions

- All file paths are absolute from repo root `/Users/xupeng/dev/github/milkie/`.
- `npx tsc --noEmit` must pass after each step that touches a `.ts` file.
- `npm run test:unit` covers unit tests; `npm run test:e2e:live` covers e2e (live uses no API keys for our seqential-gateway fixtures).
- After every commit: `git log --oneline -3` to verify chain.
- All commit messages end with the `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` trailer.

---

## Task 1: Canonical hashing for LLM/tool requests

**Files:**
- Create: `src/trace/hash.ts`
- Test: `src/__tests__/Hash.test.ts`

- [ ] **Step 1.1: Write failing tests**

Create `src/__tests__/Hash.test.ts`:

```typescript
import { hashModelRequest, hashToolCall, canonicalize } from '../trace/hash'
import type { ModelRequest } from '../types/model'

describe('canonicalize', () => {
  it('sorts object keys recursively', () => {
    const a = canonicalize({ b: 1, a: { d: 2, c: 3 } })
    const b = canonicalize({ a: { c: 3, d: 2 }, b: 1 })
    expect(a).toBe(b)
  })

  it('preserves array order', () => {
    expect(canonicalize([1, 2, 3])).toBe(JSON.stringify([1, 2, 3]))
  })

  it('treats undefined as missing key', () => {
    expect(canonicalize({ a: 1, b: undefined })).toBe(canonicalize({ a: 1 }))
  })

  it('distinguishes null from missing', () => {
    expect(canonicalize({ a: null })).not.toBe(canonicalize({}))
  })
})

const reqA = (): ModelRequest => ({
  model:       'm1',
  messages:    [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
  systemPrompt: 'sys',
  tools:       [],
})

describe('hashModelRequest', () => {
  it('returns stable 64-hex SHA-256', () => {
    const h = hashModelRequest(reqA())
    expect(h).toMatch(/^[0-9a-f]{64}$/)
  })

  it('is stable under key reordering', () => {
    const h1 = hashModelRequest(reqA())
    const reordered: ModelRequest = {
      tools:       [],
      systemPrompt: 'sys',
      messages:    [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
      model:       'm1',
    }
    expect(hashModelRequest(reordered)).toBe(h1)
  })

  it('changes when any field changes', () => {
    const h1 = hashModelRequest(reqA())
    const mutated: ModelRequest = { ...reqA(), model: 'm2' }
    expect(hashModelRequest(mutated)).not.toBe(h1)
  })
})

describe('hashToolCall', () => {
  it('returns stable 64-hex', () => {
    expect(hashToolCall('grep', { pattern: 'x' })).toMatch(/^[0-9a-f]{64}$/)
  })

  it('changes when name or input differs', () => {
    const a = hashToolCall('grep', { pattern: 'x' })
    const b = hashToolCall('grep', { pattern: 'y' })
    const c = hashToolCall('rg',   { pattern: 'x' })
    expect(a).not.toBe(b)
    expect(a).not.toBe(c)
  })

  it('is stable under input key reordering', () => {
    const a = hashToolCall('t', { x: 1, y: 2 })
    const b = hashToolCall('t', { y: 2, x: 1 })
    expect(a).toBe(b)
  })
})
```

- [ ] **Step 1.2: Run test, verify it fails**

Run: `npx jest src/__tests__/Hash.test.ts --runInBand`
Expected: FAIL with "Cannot find module '../trace/hash'"

- [ ] **Step 1.3: Implement hash.ts**

Create `src/trace/hash.ts`:

```typescript
import { createHash } from 'crypto'
import type { ModelRequest } from '../types/model.js'

/**
 * Canonical JSON serialization: object keys sorted recursively,
 * undefined values omitted, arrays preserve order. Used so two
 * structurally-equal payloads always produce the same hash.
 */
export function canonicalize(value: unknown): string {
  return JSON.stringify(normalize(value))
}

function normalize(value: unknown): unknown {
  if (value === null) return null
  if (Array.isArray(value)) return value.map(normalize)
  if (typeof value === 'object') {
    const out: Record<string, unknown> = {}
    const keys = Object.keys(value as object).sort()
    for (const k of keys) {
      const v = (value as Record<string, unknown>)[k]
      if (v === undefined) continue
      out[k] = normalize(v)
    }
    return out
  }
  return value
}

function sha256Hex(s: string): string {
  return createHash('sha256').update(s).digest('hex')
}

export function hashModelRequest(req: ModelRequest): string {
  return sha256Hex(canonicalize(req))
}

export function hashToolCall(toolName: string, input: unknown): string {
  return sha256Hex(canonicalize({ toolName, input }))
}
```

- [ ] **Step 1.4: Run tests, verify they pass**

Run: `npx jest src/__tests__/Hash.test.ts --runInBand`
Expected: PASS, 8/8

- [ ] **Step 1.5: Verify typecheck**

Run: `npx tsc --noEmit`
Expected: no output (success)

- [ ] **Step 1.6: Commit**

```bash
git add src/trace/hash.ts src/__tests__/Hash.test.ts
git commit -m "$(cat <<'EOF'
feat(trace): add canonical hashing for LLM/tool requests

hashModelRequest and hashToolCall produce stable SHA-256 hex digests
over canonical JSON (keys sorted, undefined omitted). Foundation for
Phase 3 content-addressed cache.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Structured tool error + lifecycle event kinds + requestHash on I/O events

**Files:**
- Modify: `src/trace/types.ts` (upgrade ToolRespondedPayload; add 2 event kinds + payload interfaces)
- Modify: `src/trace/RecordingIOPort.ts` (compute requestHash; add attach/detach; structured tool error)
- Modify: `src/runtime/Milkie.ts` (call attach before run, detach in finally)
- Modify: `src/__tests__/Trace.test.ts` (extend coverage)

- [ ] **Step 2.1: Write failing tests for new event kinds + requestHash + structured error**

Append to `src/__tests__/Trace.test.ts` (inside the existing top-level describe; before the final closing brace):

```typescript
describe('RecordingIOPort — Phase 3 additions', () => {
  it('LLM events carry requestHash matching hashModelRequest(request)', async () => {
    const { hashModelRequest } = await import('../trace/hash')
    const store = new MemoryEventStore()
    const gateway = new StubGateway([textResponse('hi')])
    const port = new RecordingIOPort(new DefaultIOPort(gateway), store, 'r1')

    const req: ModelRequest = {
      model:        'm1',
      messages:     [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
      systemPrompt: 'sys',
      tools:        [],
    }
    await port.invokeLLM(req)

    const events = await store.readByRunId('r1')
    const requested = events.find(e => e.type === 'llm.requested')!
    expect((requested.payload as { requestHash: string }).requestHash)
      .toBe(hashModelRequest(req))
  })

  it('tool error is recorded as structured payload preserving retryable/code/name', async () => {
    const store = new MemoryEventStore()
    const port = new RecordingIOPort(new DefaultIOPort(new StubGateway([])), store, 'r1')

    const err = Object.assign(new Error('boom'), { retryable: true, code: 'EBUSY', name: 'BusyError' })
    await expect(port.invokeTool('t', { x: 1 }, async () => { throw err })).rejects.toThrow('boom')

    const events = await store.readByRunId('r1')
    const responded = events.find(e => e.type === 'tool.responded')!
    const payload = responded.payload as { error?: { message: string; retryable?: boolean; code?: string; name?: string } }
    expect(payload.error).toEqual({ message: 'boom', retryable: true, code: 'EBUSY', name: 'BusyError' })
  })

  it('attach emits agent.run.started with lifecycle identity payload', async () => {
    const store = new MemoryEventStore()
    const port = new RecordingIOPort(new DefaultIOPort(new StubGateway([])), store, 'r1')

    port.attach({
      agentId:   'a1',
      goal:      'do the thing',
      input:     'go',
      contextId: 'ctx-1',
      parentId:  undefined,
    })

    const events = await store.readByRunId('r1')
    const started = events.find(e => e.type === 'agent.run.started')!
    expect(started.payload).toEqual({
      agentId: 'a1', goal: 'do the thing', input: 'go', contextId: 'ctx-1', parentId: undefined,
    })
  })

  it('detach emits agent.run.completed with terminal status', async () => {
    const store = new MemoryEventStore()
    const port = new RecordingIOPort(new DefaultIOPort(new StubGateway([])), store, 'r1')

    port.attach({ agentId: 'a1', goal: 'g', input: 'i', contextId: 'c1' })
    port.detach({ status: 'completed', lastTextOutput: 'done' })

    const events = await store.readByRunId('r1')
    const completed = events.find(e => e.type === 'agent.run.completed')!
    expect(completed.payload).toEqual({ status: 'completed', lastTextOutput: 'done' })
  })

  it('Milkie.invoke wraps run with attach/detach around the run', async () => {
    const store = new MemoryEventStore()
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      gateway:    new StubGateway([textResponse('hello')]),
      eventStore: store,
    })
    milkie.registerAgent(minimalAgentConfig('a1'))

    const result = await milkie.invoke({ agentId: 'a1', goal: 'g', input: 'i' })

    const events = await store.readByRunId(result.agentRunId)
    const kinds = events.map(e => e.type)
    expect(kinds[0]).toBe('agent.run.started')
    expect(kinds[kinds.length - 1]).toBe('agent.run.completed')
    expect((events[0].payload as { agentId: string }).agentId).toBe('a1')
  })
})

// Helper used by the last test above. Put near other helpers at top of file.
function minimalAgentConfig(agentId: string): AgentConfig {
  return {
    agentId,
    version: '0.0.0',
    systemPrompt: 'system',
    fsm: {
      initial: 's0',
      states: [
        { name: 's0', kind: 'llm', instructions: 'reply hello', tools: [], transitions: [{ when: 'always', to: 'end' }] },
        { name: 'end', kind: 'terminal' },
      ],
    },
    model: { provider: 'stub', model: 'stub', adapter: 'stub' },
  } as AgentConfig
}
```

(If `minimalAgentConfig` already exists in the file from prior tests, skip the helper definition.)

- [ ] **Step 2.2: Run tests, verify they fail**

Run: `npx jest src/__tests__/Trace.test.ts --runInBand`
Expected: FAIL — `requestHash` undefined, `attach` not a function, error is string not object, no lifecycle events.

- [ ] **Step 2.3: Update `src/trace/types.ts`**

Read current file first, then replace the type definitions. Final content of the file:

```typescript
import type { ModelRequest, ModelResponse } from '../types/model.js'

/**
 * Agent Trace event types.
 *
 * Phase 3 adds lifecycle events and content-addressed cache fields on I/O
 * payloads. Spawn/fork events are deferred to Phase 5.
 */
export type EventKind =
  | 'llm.requested'
  | 'llm.responded'
  | 'tool.requested'
  | 'tool.responded'
  | 'agent.run.started'
  | 'agent.run.completed'

export interface Event<P = unknown> {
  id: string
  runId: string
  type: EventKind
  actor: string
  causedBy?: string
  timestamp: number
  payload: P
}

// ---- I/O payloads (Phase 2 shapes + Phase 3 requestHash) ----

export interface LlmRequestedPayload {
  request: ModelRequest
  /** Phase 3: hash of canonicalized request; cache key for replay. */
  requestHash: string
}

export interface LlmRespondedPayload {
  response: ModelResponse
  /** Mirrors the requested-event hash so consumers don't need to re-join. */
  requestHash: string
}

export interface ToolRequestedPayload {
  toolName: string
  input: unknown
  /** Phase 3: hash of canonicalized (toolName + input); cache key for replay. */
  requestHash: string
}

export interface ToolRespondedPayload {
  toolName: string
  output?: unknown
  /** Phase 3: structured to preserve retryable/code/name; replay rebuilds Error. */
  error?: {
    message:    string
    retryable?: boolean
    code?:      string
    name?:      string
  }
  /** Mirrors the requested-event hash. */
  requestHash: string
}

// ---- Lifecycle payloads (Phase 3) ----

export interface AgentRunStartedPayload {
  agentId:    string
  goal:       string
  input:      string
  contextId:  string
  parentId?:  string
}

export interface AgentRunCompletedPayload {
  status:           'completed' | 'interrupted' | 'error'
  lastTextOutput?:  string
  error?:           string
}

// ---- Typed event aliases ----

export type LlmRequestedEvent       = Event<LlmRequestedPayload>       & { type: 'llm.requested' }
export type LlmRespondedEvent       = Event<LlmRespondedPayload>       & { type: 'llm.responded' }
export type ToolRequestedEvent      = Event<ToolRequestedPayload>      & { type: 'tool.requested' }
export type ToolRespondedEvent      = Event<ToolRespondedPayload>      & { type: 'tool.responded' }
export type AgentRunStartedEvent    = Event<AgentRunStartedPayload>    & { type: 'agent.run.started' }
export type AgentRunCompletedEvent  = Event<AgentRunCompletedPayload>  & { type: 'agent.run.completed' }

export type AnyEvent =
  | LlmRequestedEvent
  | LlmRespondedEvent
  | ToolRequestedEvent
  | ToolRespondedEvent
  | AgentRunStartedEvent
  | AgentRunCompletedEvent
```

- [ ] **Step 2.4: Update `src/trace/RecordingIOPort.ts`**

Read current file, then rewrite. Final content:

```typescript
import type { ModelRequest, ModelResponse } from '../types/model.js'
import type { IIOPort } from '../runtime/IOPort.js'
import type { IEventStore } from './EventStore.js'
import type {
  LlmRequestedPayload,
  LlmRespondedPayload,
  ToolRequestedPayload,
  ToolRespondedPayload,
  AgentRunStartedPayload,
  AgentRunCompletedPayload,
} from './types.js'
import { hashModelRequest, hashToolCall } from './hash.js'

/**
 * RecordingIOPort — decorates an inner IOPort to emit Agent Trace events.
 *
 * Phase 3 additions:
 *  - LLM/tool events carry requestHash (cache key)
 *  - attach()/detach() emit agent.run.started/completed lifecycle events
 *  - Tool errors are recorded as structured payloads (preserve retryable/code/name)
 *
 * Clock (now()) and UUID generation pass through to inner — non-determinism
 * log is Phase 4.
 */
export class RecordingIOPort implements IIOPort {
  constructor(
    private readonly inner: IIOPort,
    private readonly store: IEventStore,
    private readonly runId: string,
    private readonly actor: string = 'runtime',
  ) {}

  attach(payload: AgentRunStartedPayload): void {
    // Fire-and-forget: store.append is async but lifecycle event ordering only
    // matters relative to subsequent I/O on the same store. Errors surface via
    // the store implementation's own error handling.
    void this.store.append({
      id:        this.inner.uuid(),
      runId:     this.runId,
      type:      'agent.run.started',
      actor:     this.actor,
      timestamp: this.inner.now(),
      payload,
    })
  }

  detach(payload: AgentRunCompletedPayload): void {
    void this.store.append({
      id:        this.inner.uuid(),
      runId:     this.runId,
      type:      'agent.run.completed',
      actor:     this.actor,
      timestamp: this.inner.now(),
      payload,
    })
  }

  async invokeLLM(request: ModelRequest): Promise<ModelResponse> {
    const requestHash = hashModelRequest(request)
    const reqEventId  = this.inner.uuid()
    await this.store.append({
      id:        reqEventId,
      runId:     this.runId,
      type:      'llm.requested',
      actor:     this.actor,
      timestamp: this.inner.now(),
      payload:   { request, requestHash } satisfies LlmRequestedPayload,
    })

    const response = await this.inner.invokeLLM(request)

    await this.store.append({
      id:        this.inner.uuid(),
      runId:     this.runId,
      type:      'llm.responded',
      actor:     this.actor,
      causedBy:  reqEventId,
      timestamp: this.inner.now(),
      payload:   { response, requestHash } satisfies LlmRespondedPayload,
    })

    return response
  }

  async invokeTool(
    toolName: string,
    input: unknown,
    execute: () => Promise<unknown>,
  ): Promise<unknown> {
    const requestHash = hashToolCall(toolName, input)
    const reqEventId  = this.inner.uuid()
    await this.store.append({
      id:        reqEventId,
      runId:     this.runId,
      type:      'tool.requested',
      actor:     this.actor,
      timestamp: this.inner.now(),
      payload:   { toolName, input, requestHash } satisfies ToolRequestedPayload,
    })

    try {
      const output = await this.inner.invokeTool(toolName, input, execute)
      await this.store.append({
        id:        this.inner.uuid(),
        runId:     this.runId,
        type:      'tool.responded',
        actor:     this.actor,
        causedBy:  reqEventId,
        timestamp: this.inner.now(),
        payload:   { toolName, output, requestHash } satisfies ToolRespondedPayload,
      })
      return output
    } catch (err) {
      const e = err as { message?: string; retryable?: boolean; code?: string; name?: string }
      const errorPayload: NonNullable<ToolRespondedPayload['error']> = {
        message: e.message ?? String(err),
      }
      if (typeof e.retryable === 'boolean') errorPayload.retryable = e.retryable
      if (typeof e.code === 'string')       errorPayload.code      = e.code
      if (typeof e.name === 'string' && e.name !== 'Error') errorPayload.name = e.name

      await this.store.append({
        id:        this.inner.uuid(),
        runId:     this.runId,
        type:      'tool.responded',
        actor:     this.actor,
        causedBy:  reqEventId,
        timestamp: this.inner.now(),
        payload:   { toolName, error: errorPayload, requestHash } satisfies ToolRespondedPayload,
      })
      throw err
    }
  }

  now(): number {
    return this.inner.now()
  }

  uuid(): string {
    return this.inner.uuid()
  }
}
```

- [ ] **Step 2.5: Update `src/runtime/Milkie.ts` — call attach/detach**

In the `invoke` method (around line 116-134), replace the `const runtime = new AgentRuntime(...)` through `return runtime.run(request.input)` block with:

```typescript
    const ioPort = this.wrapIOPort(gateway, agentRunId)

    const runtime = new AgentRuntime({
      config,
      goal:            request.goal,
      input:           request.input,
      contextId,
      agentRunId,
      stateStore:      this.stateStore,
      recorder,
      ioPort,
      extraTools:      this.extraTools,
      subAgentConfigs: this.agents,
      childRecorderFactory,
    })

    if (restoredCheckpoint) {
      await runtime.loadCheckpoint(restoredCheckpoint)
    }

    if (ioPort instanceof RecordingIOPort) {
      ioPort.attach({
        agentId:   config.agentId,
        goal:      request.goal,
        input:     request.input,
        contextId,
        parentId:  undefined,
      })
    }

    try {
      const result = await runtime.run(request.input)
      if (ioPort instanceof RecordingIOPort) {
        ioPort.detach({ status: result.status, lastTextOutput: result.output })
      }
      return result
    } catch (err) {
      if (ioPort instanceof RecordingIOPort) {
        ioPort.detach({ status: 'error', error: err instanceof Error ? err.message : String(err) })
      }
      throw err
    }
```

**Do NOT modify `resume`.** Per spec §5, Phase 3 only wraps `invoke()` with attach/detach. The `resume()` path will be handled in a later phase together with checkpoint / non-determinism log design.

Note: `RecordingIOPort` is already imported in `src/runtime/Milkie.ts` from Phase 2. No new import needed for this task.

- [ ] **Step 2.6: Run Trace tests, verify they pass**

Run: `npx jest src/__tests__/Trace.test.ts --runInBand`
Expected: all tests PASS (Phase 2 tests + 5 new Phase 3 tests)

- [ ] **Step 2.7: Run typecheck**

Run: `npx tsc --noEmit`
Expected: no output

- [ ] **Step 2.8: Run full unit suite to verify no regression**

Run: `npm run test:unit`
Expected: 100% pass

- [ ] **Step 2.9: Commit**

```bash
git add src/trace/types.ts src/trace/RecordingIOPort.ts src/runtime/Milkie.ts src/__tests__/Trace.test.ts
git commit -m "$(cat <<'EOF'
feat(trace): structured tool error + lifecycle events + requestHash on I/O

- ToolRespondedPayload.error upgraded from string to structured
  { message, retryable?, code?, name? } so replay can rebuild errors
  with the fields AgentRuntime depends on (retryable controls retry).
- New event kinds agent.run.started/completed carry lifecycle identity
  (agentId, goal, input, contextId, parentId?) — no configSnapshot.
- RecordingIOPort.attach/detach emit lifecycle events; Milkie invoke/
  resume wraps run() with attach + finally-detach.
- LLM/tool requested/responded payloads now carry requestHash (cache key).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: CacheIndex + ReplayingIOPort + ReplayDivergenceError + extractRunSnapshot

**Files:**
- Create: `src/trace/CacheIndex.ts`
- Create: `src/trace/RunSnapshot.ts`
- Create: `src/trace/ReplayingIOPort.ts`
- Create: `src/trace/ReplayDivergenceError.ts`
- Create: `src/trace/ReplayError.ts`
- Test: `src/__tests__/CacheIndex.test.ts`
- Test: `src/__tests__/ReplayingIOPort.test.ts`

- [ ] **Step 3.1: Create `src/trace/ReplayError.ts` + `src/trace/ReplayDivergenceError.ts`**

`src/trace/ReplayError.ts`:

```typescript
/**
 * Thrown when a replay cannot proceed for structural reasons:
 * missing lifecycle event, unknown agentId, malformed cached response,
 * empty event log, etc.
 *
 * Distinct from ReplayDivergenceError which fires on cache miss.
 */
export class ReplayError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ReplayError'
  }
}
```

`src/trace/ReplayDivergenceError.ts`:

```typescript
/**
 * Thrown when the replayed agent issues an LLM or tool call whose
 * canonical hash does not appear (or appears too few times) in the
 * recorded event log. Replay is strict — divergence is fail-fast.
 */
export class ReplayDivergenceError extends Error {
  constructor(
    public readonly kind:            'llm' | 'tool',
    public readonly actualHash:      string,
    public readonly summary:         string,
    public readonly availableHashes: string[],
  ) {
    super(`Replay divergence (${kind}): hash ${actualHash.slice(0, 12)}… not in cache. ${summary}`)
    this.name = 'ReplayDivergenceError'
  }
}
```

- [ ] **Step 3.2: Write failing CacheIndex tests**

Create `src/__tests__/CacheIndex.test.ts`:

```typescript
import { CacheIndex } from '../trace/CacheIndex'
import type { Event, LlmRespondedPayload, ToolRespondedPayload } from '../trace/types'
import type { ModelResponse } from '../types/model'

const llmResp = (text: string): ModelResponse => ({
  content:      [{ type: 'text', text }],
  toolCalls:    [],
  finishReason: 'end_turn',
})

const mkLlmResponded = (hash: string, text: string): Event<LlmRespondedPayload> => ({
  id:        `e-${Math.random()}`,
  runId:     'r1',
  type:      'llm.responded',
  actor:     'runtime',
  timestamp: 1,
  payload:   { response: llmResp(text), requestHash: hash },
})

const mkToolResponded = (hash: string, output?: unknown, error?: NonNullable<ToolRespondedPayload['error']>): Event<ToolRespondedPayload> => ({
  id:        `e-${Math.random()}`,
  runId:     'r1',
  type:      'tool.responded',
  actor:     'runtime',
  timestamp: 1,
  payload:   { toolName: 't', output, error, requestHash: hash },
})

describe('CacheIndex', () => {
  it('fromEvents builds empty index for empty events', () => {
    const idx = CacheIndex.fromEvents([])
    expect(idx.remaining()).toEqual({ llm: 0, tool: 0 })
  })

  it('consumeLLM serves cached responses in FIFO order per hash', () => {
    const idx = CacheIndex.fromEvents([
      mkLlmResponded('h1', 'first'),
      mkLlmResponded('h1', 'second'),
      mkLlmResponded('h2', 'other'),
    ])
    expect(idx.consumeLLM('h1').content[0]).toMatchObject({ text: 'first' })
    expect(idx.consumeLLM('h1').content[0]).toMatchObject({ text: 'second' })
    expect(idx.consumeLLM('h2').content[0]).toMatchObject({ text: 'other' })
    expect(idx.remaining()).toEqual({ llm: 0, tool: 0 })
  })

  it('consumeLLM throws when queue exhausted', () => {
    const idx = CacheIndex.fromEvents([mkLlmResponded('h1', 'x')])
    idx.consumeLLM('h1')
    expect(() => idx.consumeLLM('h1')).toThrow(/queue empty/)
  })

  it('consumeLLM throws when hash never seen', () => {
    const idx = CacheIndex.fromEvents([])
    expect(() => idx.consumeLLM('nope')).toThrow(/queue empty/)
  })

  it('consumeTool returns output for successful tool', () => {
    const idx = CacheIndex.fromEvents([mkToolResponded('h1', { ok: true })])
    expect(idx.consumeTool('h1')).toEqual({ ok: true })
  })

  it('consumeTool rethrows Error with retryable/code/name preserved', () => {
    const idx = CacheIndex.fromEvents([
      mkToolResponded('h1', undefined, { message: 'boom', retryable: true, code: 'EBUSY', name: 'BusyError' }),
    ])
    try {
      idx.consumeTool('h1')
      throw new Error('expected throw')
    } catch (err) {
      const e = err as Error & { retryable?: boolean; code?: string }
      expect(e.message).toBe('boom')
      expect(e.retryable).toBe(true)
      expect(e.code).toBe('EBUSY')
      expect(e.name).toBe('BusyError')
    }
  })

  it('remaining tracks unconsumed counts', () => {
    const idx = CacheIndex.fromEvents([
      mkLlmResponded('h1', 'a'),
      mkLlmResponded('h1', 'b'),
      mkToolResponded('h2', 'out'),
    ])
    expect(idx.remaining()).toEqual({ llm: 2, tool: 1 })
    idx.consumeLLM('h1')
    expect(idx.remaining()).toEqual({ llm: 1, tool: 1 })
  })

  it('allHashes returns all unique hashes seen', () => {
    const idx = CacheIndex.fromEvents([
      mkLlmResponded('h1', 'a'),
      mkLlmResponded('h1', 'b'),
      mkLlmResponded('h2', 'c'),
      mkToolResponded('th1', 'out'),
    ])
    expect(idx.allHashes().llm.sort()).toEqual(['h1', 'h2'])
    expect(idx.allHashes().tool).toEqual(['th1'])
  })
})
```

- [ ] **Step 3.3: Run, verify fail**

Run: `npx jest src/__tests__/CacheIndex.test.ts --runInBand`
Expected: FAIL with "Cannot find module '../trace/CacheIndex'"

- [ ] **Step 3.4: Implement `src/trace/CacheIndex.ts`**

```typescript
import type { Event, LlmRespondedPayload, ToolRespondedPayload } from './types.js'
import type { ModelResponse } from '../types/model.js'

/**
 * In-memory projection of LLM/tool response events keyed by canonical
 * request hash, with one FIFO queue per hash. Drives strict structural
 * replay: consume calls dequeue; replay throws when a queue empties
 * or the hash was never recorded.
 */
export class CacheIndex {
  private readonly llm:  Map<string, ModelResponse[]>
  private readonly tool: Map<string, ToolOutcome[]>

  private constructor(
    llm:  Map<string, ModelResponse[]>,
    tool: Map<string, ToolOutcome[]>,
  ) {
    this.llm  = llm
    this.tool = tool
  }

  static fromEvents(events: Event[]): CacheIndex {
    const llm:  Map<string, ModelResponse[]> = new Map()
    const tool: Map<string, ToolOutcome[]>   = new Map()

    for (const ev of events) {
      if (ev.type === 'llm.responded') {
        const p = ev.payload as LlmRespondedPayload
        push(llm, p.requestHash, p.response)
      } else if (ev.type === 'tool.responded') {
        const p = ev.payload as ToolRespondedPayload
        push(tool, p.requestHash, { output: p.output, error: p.error })
      }
    }

    return new CacheIndex(llm, tool)
  }

  consumeLLM(hash: string): ModelResponse {
    const q = this.llm.get(hash)
    if (!q || q.length === 0) throw new Error(`CacheIndex: LLM queue empty for hash ${hash}`)
    return q.shift()!
  }

  consumeTool(hash: string): unknown {
    const q = this.tool.get(hash)
    if (!q || q.length === 0) throw new Error(`CacheIndex: tool queue empty for hash ${hash}`)
    const outcome = q.shift()!
    if (outcome.error) {
      const err = new Error(outcome.error.message) as Error & { retryable?: boolean; code?: string }
      if (outcome.error.retryable !== undefined) err.retryable = outcome.error.retryable
      if (outcome.error.code !== undefined)      err.code      = outcome.error.code
      if (outcome.error.name !== undefined)      err.name      = outcome.error.name
      throw err
    }
    return outcome.output
  }

  remaining(): { llm: number; tool: number } {
    let llmCount = 0, toolCount = 0
    for (const q of this.llm.values())  llmCount  += q.length
    for (const q of this.tool.values()) toolCount += q.length
    return { llm: llmCount, tool: toolCount }
  }

  allHashes(): { llm: string[]; tool: string[] } {
    return { llm: [...this.llm.keys()], tool: [...this.tool.keys()] }
  }
}

interface ToolOutcome {
  output?: unknown
  error?:  ToolRespondedPayload['error']
}

function push<K, V>(map: Map<K, V[]>, key: K, value: V): void {
  const q = map.get(key)
  if (q) q.push(value)
  else   map.set(key, [value])
}
```

- [ ] **Step 3.5: Run CacheIndex tests, verify pass**

Run: `npx jest src/__tests__/CacheIndex.test.ts --runInBand`
Expected: PASS, 8/8

- [ ] **Step 3.6: Write failing ReplayingIOPort tests**

Create `src/__tests__/ReplayingIOPort.test.ts`:

```typescript
import { ReplayingIOPort } from '../trace/ReplayingIOPort'
import { CacheIndex } from '../trace/CacheIndex'
import { DefaultIOPort } from '../runtime/IOPort'
import { ReplayDivergenceError } from '../trace/ReplayDivergenceError'
import { hashModelRequest, hashToolCall } from '../trace/hash'
import type { IIOPort } from '../runtime/IOPort'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'
import type { Event, LlmRespondedPayload, ToolRespondedPayload } from '../trace/types'

class FailingGateway implements IModelGateway {
  complete(): Promise<ModelResponse> { throw new Error('inner gateway must not be called during replay') }
  async *stream(): AsyncIterable<never> { yield* [] }
}

const innerNeverCalled = (): IIOPort => new DefaultIOPort(new FailingGateway())

const llmResp = (text: string): ModelResponse => ({
  content: [{ type: 'text', text }], toolCalls: [], finishReason: 'end_turn',
})

describe('ReplayingIOPort', () => {
  it('serves LLM from cache; never calls inner', async () => {
    const req: ModelRequest = { model: 'm', messages: [], systemPrompt: '', tools: [] }
    const h = hashModelRequest(req)
    const ev: Event<LlmRespondedPayload> = {
      id: 'e1', runId: 'r1', type: 'llm.responded', actor: 'runtime', timestamp: 1,
      payload: { response: llmResp('cached'), requestHash: h },
    }
    const port = new ReplayingIOPort(CacheIndex.fromEvents([ev]), innerNeverCalled())
    const resp = await port.invokeLLM(req)
    expect(resp.content[0]).toMatchObject({ text: 'cached' })
  })

  it('throws ReplayDivergenceError on LLM cache miss', async () => {
    const req: ModelRequest = { model: 'm', messages: [], systemPrompt: '', tools: [] }
    const port = new ReplayingIOPort(CacheIndex.fromEvents([]), innerNeverCalled())
    await expect(port.invokeLLM(req)).rejects.toBeInstanceOf(ReplayDivergenceError)
  })

  it('divergence error carries kind and actualHash', async () => {
    const req: ModelRequest = { model: 'm', messages: [], systemPrompt: '', tools: [] }
    const port = new ReplayingIOPort(CacheIndex.fromEvents([]), innerNeverCalled())
    try {
      await port.invokeLLM(req)
    } catch (err) {
      const e = err as ReplayDivergenceError
      expect(e.kind).toBe('llm')
      expect(e.actualHash).toBe(hashModelRequest(req))
      expect(e).toBeInstanceOf(ReplayDivergenceError)
    }
  })

  it('serves tool output from cache; execute thunk never runs', async () => {
    const input = { x: 1 }
    const h = hashToolCall('t', input)
    const ev: Event<ToolRespondedPayload> = {
      id: 'e1', runId: 'r1', type: 'tool.responded', actor: 'runtime', timestamp: 1,
      payload: { toolName: 't', output: { ok: true }, requestHash: h },
    }
    const port = new ReplayingIOPort(CacheIndex.fromEvents([ev]), innerNeverCalled())
    let executeCalled = false
    const out = await port.invokeTool('t', input, async () => { executeCalled = true; return 'should not run' })
    expect(out).toEqual({ ok: true })
    expect(executeCalled).toBe(false)
  })

  it('tool error rethrows with retryable preserved', async () => {
    const input = { x: 1 }
    const h = hashToolCall('t', input)
    const ev: Event<ToolRespondedPayload> = {
      id: 'e1', runId: 'r1', type: 'tool.responded', actor: 'runtime', timestamp: 1,
      payload: { toolName: 't', error: { message: 'boom', retryable: true }, requestHash: h },
    }
    const port = new ReplayingIOPort(CacheIndex.fromEvents([ev]), innerNeverCalled())
    try {
      await port.invokeTool('t', input, async () => 'unused')
    } catch (err) {
      const e = err as Error & { retryable?: boolean }
      expect(e.message).toBe('boom')
      expect(e.retryable).toBe(true)
    }
  })

  it('now/uuid passthrough to inner', () => {
    const port = new ReplayingIOPort(CacheIndex.fromEvents([]), new DefaultIOPort(new FailingGateway()))
    expect(typeof port.now()).toBe('number')
    expect(port.uuid()).toMatch(/^[0-9a-f-]{36}$/i)
  })
})
```

- [ ] **Step 3.7: Run, verify fail**

Run: `npx jest src/__tests__/ReplayingIOPort.test.ts --runInBand`
Expected: FAIL with "Cannot find module '../trace/ReplayingIOPort'"

- [ ] **Step 3.8: Implement `src/trace/ReplayingIOPort.ts`**

```typescript
import type { IIOPort } from '../runtime/IOPort.js'
import type { ModelRequest, ModelResponse } from '../types/model.js'
import type { CacheIndex } from './CacheIndex.js'
import { hashModelRequest, hashToolCall } from './hash.js'
import { ReplayDivergenceError } from './ReplayDivergenceError.js'

/**
 * IOPort implementation that serves LLM/tool calls from a CacheIndex
 * built from a recorded run's events. Cache miss → ReplayDivergenceError.
 * inner is used only for now()/uuid() passthrough — its invokeLLM /
 * invokeTool are never called during replay.
 */
export class ReplayingIOPort implements IIOPort {
  constructor(
    private readonly cache: CacheIndex,
    private readonly inner: IIOPort,
  ) {}

  async invokeLLM(request: ModelRequest): Promise<ModelResponse> {
    const hash = hashModelRequest(request)
    try {
      return this.cache.consumeLLM(hash)
    } catch {
      const lastUserMessage = request.messages
        .filter(m => m.role === 'user')
        .flatMap(m => m.content)
        .map(c => c.type === 'text' ? c.text : `[${c.type}]`)
        .pop() ?? ''
      const summary = `model=${request.model} lastUser=${lastUserMessage.slice(0, 80)}`
      throw new ReplayDivergenceError('llm', hash, summary, this.cache.allHashes().llm.slice(0, 5))
    }
  }

  async invokeTool(
    toolName: string,
    input: unknown,
    _execute: () => Promise<unknown>,
  ): Promise<unknown> {
    const hash = hashToolCall(toolName, input)
    try {
      return this.cache.consumeTool(hash)
    } catch (err) {
      // consumeTool throws a normal Error for "queue empty"; rethrows reconstructed
      // tool errors for recorded failures. Distinguish by message prefix.
      if (err instanceof Error && err.message.startsWith('CacheIndex: tool queue empty')) {
        const summary = `toolName=${toolName} input=${JSON.stringify(input).slice(0, 80)}`
        throw new ReplayDivergenceError('tool', hash, summary, this.cache.allHashes().tool.slice(0, 5))
      }
      throw err
    }
  }

  now(): number {
    return this.inner.now()
  }

  uuid(): string {
    return this.inner.uuid()
  }
}
```

- [ ] **Step 3.9: Run, verify pass**

Run: `npx jest src/__tests__/ReplayingIOPort.test.ts --runInBand`
Expected: PASS, 6/6

- [ ] **Step 3.10: Implement `src/trace/RunSnapshot.ts`**

```typescript
import type { Event, AgentRunStartedPayload, AgentRunCompletedPayload } from './types.js'
import { ReplayError } from './ReplayError.js'

export interface RunSnapshot {
  agentId:        string
  goal:           string
  input:          string
  contextId:      string
  parentId?:      string
  terminalStatus?: AgentRunCompletedPayload['status']
}

/**
 * Pure projection: pulls the run's lifecycle identity from
 * agent.run.started and (optionally) agent.run.completed. Throws
 * ReplayError if the started event is missing — Phase 2 runs (no
 * lifecycle events) cannot be replayed.
 */
export function extractRunSnapshot(events: Event[]): RunSnapshot {
  if (events.length === 0) throw new ReplayError('no events for this run')

  const started = events.find(e => e.type === 'agent.run.started')
  if (!started) throw new ReplayError('no lifecycle start event; run was recorded before Phase 3')

  const startPayload = started.payload as AgentRunStartedPayload

  const completed = events.find(e => e.type === 'agent.run.completed')
  const terminalStatus = completed
    ? (completed.payload as AgentRunCompletedPayload).status
    : undefined

  return {
    agentId:        startPayload.agentId,
    goal:           startPayload.goal,
    input:          startPayload.input,
    contextId:      startPayload.contextId,
    parentId:       startPayload.parentId,
    terminalStatus,
  }
}
```

- [ ] **Step 3.11: Run typecheck**

Run: `npx tsc --noEmit`
Expected: no output

- [ ] **Step 3.12: Run full unit suite to verify no regression**

Run: `npm run test:unit`
Expected: all pass

- [ ] **Step 3.13: Commit**

```bash
git add src/trace/CacheIndex.ts src/trace/ReplayingIOPort.ts src/trace/ReplayDivergenceError.ts src/trace/ReplayError.ts src/trace/RunSnapshot.ts src/__tests__/CacheIndex.test.ts src/__tests__/ReplayingIOPort.test.ts
git commit -m "$(cat <<'EOF'
feat(trace): add CacheIndex + ReplayingIOPort + run snapshot projection

CacheIndex is a FIFO per-hash projection of LLM/tool responded events;
ReplayingIOPort serves invokeLLM/invokeTool from it and fail-fasts on
cache miss with ReplayDivergenceError. extractRunSnapshot reads
lifecycle identity from agent.run.started. ReplayError surfaces
structural failures (no start event, unknown run).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Milkie.replay(runId) API

**Files:**
- Modify: `src/runtime/Milkie.ts`
- Create: `src/__tests__/Replay.test.ts`

- [ ] **Step 4.1: Write failing Replay integration tests**

Create `src/__tests__/Replay.test.ts`:

```typescript
import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { ReplayError } from '../trace/ReplayError'
import { ReplayDivergenceError } from '../trace/ReplayDivergenceError'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'
import type { AgentConfig } from '../types/agent'

class SequentialGateway implements IModelGateway {
  public callCount = 0
  constructor(private readonly responses: ModelResponse[]) {}
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    this.callCount++
    const r = this.responses.shift()
    if (!r) throw new Error('SequentialGateway exhausted')
    return r
  }
  async *stream(_req: ModelRequest): AsyncIterable<never> { yield* [] }
}

const text = (s: string): ModelResponse => ({
  content: [{ type: 'text', text: s }], toolCalls: [], finishReason: 'end_turn',
})

const oneShotAgent = (agentId = 'a1'): AgentConfig => ({
  agentId,
  version: '0.0.0',
  systemPrompt: 'sys',
  fsm: {
    initial: 's0',
    states: [
      { name: 's0', kind: 'llm', instructions: 'say hi', tools: [], transitions: [{ when: 'always', to: 'end' }] },
      { name: 'end', kind: 'terminal' },
    ],
  },
  model: { provider: 'stub', model: 'stub', adapter: 'stub' },
} as AgentConfig)

describe('Milkie.replay', () => {
  it('replays a recorded run with identical result and zero LLM calls', async () => {
    const store = new MemoryEventStore()
    const replayGateway = new SequentialGateway([text('this would be wrong')])
    // First run records
    const recordGateway = new SequentialGateway([text('hello world')])
    const recordMilkie = new Milkie({ stateStore: new MemoryStore(), gateway: recordGateway, eventStore: store })
    recordMilkie.registerAgent(oneShotAgent())
    const original = await recordMilkie.invoke({ agentId: 'a1', goal: 'g', input: 'i' })

    // Replay reuses the same store + agent config but a different gateway
    const replayMilkie = new Milkie({ stateStore: new MemoryStore(), gateway: replayGateway, eventStore: store })
    replayMilkie.registerAgent(oneShotAgent())
    const replayed = await replayMilkie.replay(original.agentRunId)

    expect(replayed.status).toBe(original.status)
    expect(replayed.output).toBe(original.output)
    expect(replayGateway.callCount).toBe(0)  // I5/I6: cache served everything
  })

  it('throws ReplayError when runId has no events', async () => {
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      gateway:    new SequentialGateway([]),
      eventStore: new MemoryEventStore(),
    })
    await expect(milkie.replay('nonexistent')).rejects.toBeInstanceOf(ReplayError)
  })

  it('throws ReplayError when run has no lifecycle start (Phase 2 run)', async () => {
    const store = new MemoryEventStore()
    // Manually append only an llm.responded — no agent.run.started
    await store.append({
      id: 'e1', runId: 'r-old', type: 'llm.responded', actor: 'runtime', timestamp: 1,
      payload: { response: text('x'), requestHash: 'h' },
    })
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      gateway:    new SequentialGateway([]),
      eventStore: store,
    })
    await expect(milkie.replay('r-old')).rejects.toThrow(/no lifecycle start/)
  })

  it('throws ReplayError when agentId is not registered', async () => {
    const store = new MemoryEventStore()
    const recordMilkie = new Milkie({ stateStore: new MemoryStore(), gateway: new SequentialGateway([text('x')]), eventStore: store })
    recordMilkie.registerAgent(oneShotAgent('a1'))
    const original = await recordMilkie.invoke({ agentId: 'a1', goal: 'g', input: 'i' })

    const replayMilkie = new Milkie({ stateStore: new MemoryStore(), gateway: new SequentialGateway([]), eventStore: store })
    // intentionally do NOT register a1
    await expect(replayMilkie.replay(original.agentRunId)).rejects.toThrow(/not registered/)
  })

  it('throws ReplayDivergenceError when replay agent diverges from recorded I/O', async () => {
    const store = new MemoryEventStore()
    const recordGateway = new SequentialGateway([text('original')])
    const recordMilkie = new Milkie({ stateStore: new MemoryStore(), gateway: recordGateway, eventStore: store })
    recordMilkie.registerAgent(oneShotAgent())
    const original = await recordMilkie.invoke({ agentId: 'a1', goal: 'g', input: 'i' })

    // Replay registers a *changed* agent so the LLM request differs
    const replayMilkie = new Milkie({ stateStore: new MemoryStore(), gateway: new SequentialGateway([]), eventStore: store })
    const mutated = oneShotAgent()
    ;(mutated.fsm.states[0] as { instructions: string }).instructions = 'say goodbye'  // changes ModelRequest → hash mismatch
    replayMilkie.registerAgent(mutated)

    await expect(replayMilkie.replay(original.agentRunId)).rejects.toBeInstanceOf(ReplayDivergenceError)
  })
})
```

- [ ] **Step 4.2: Run, verify fail**

Run: `npx jest src/__tests__/Replay.test.ts --runInBand`
Expected: FAIL with "milkie.replay is not a function"

- [ ] **Step 4.3: Implement `Milkie.replay`**

In `src/runtime/Milkie.ts`, after the `resume` method (around line 193), add:

```typescript
  /**
   * Re-run a recorded agent run from its event log; all LLM/tool I/O
   * is served from the event-derived CacheIndex — no live calls. Result
   * is structurally equivalent to the original run (status, output);
   * timestamps and UUIDs are not guaranteed identical (byte-identical
   * replay is Phase 4).
   *
   * Phase 3 constraints:
   *  - Requires this Milkie has an eventStore configured
   *  - Requires this Milkie has the original agentId registered
   *  - Throws ReplayError on structural failures (missing run, missing
   *    lifecycle event, unknown agentId)
   *  - Throws ReplayDivergenceError when the replayed agent issues an
   *    LLM/tool call whose hash is not in the recorded cache
   */
  async replay(runId: string): Promise<AgentResult> {
    if (!this.eventStore) {
      throw new ReplayError('Milkie has no eventStore; cannot replay')
    }

    const events = await this.eventStore.readByRunId(runId)
    const snapshot = extractRunSnapshot(events)

    const config = this.agents.get(snapshot.agentId)
    if (!config) {
      throw new ReplayError(`agentId "${snapshot.agentId}" not registered on this Milkie instance`)
    }

    const cache  = CacheIndex.fromEvents(events)
    const inner  = new DefaultIOPort(this.gatewayOverride ?? createGateway(config.model))
    const ioPort = new ReplayingIOPort(cache, inner)

    const recorder = new InMemoryRecorder(undefined, config.agentId)

    const runtime = new AgentRuntime({
      config,
      goal:            snapshot.goal,
      input:           snapshot.input,
      contextId:       snapshot.contextId,
      agentRunId:      runId,
      parentId:        snapshot.parentId,
      stateStore:      new MemoryStore(),  // ephemeral
      recorder,
      ioPort,                                // NOT wrapped — replay writes no events
      extraTools:      this.extraTools,
      subAgentConfigs: this.agents,
      childRecorderFactory: undefined,
    })

    return runtime.run(snapshot.input)
  }
```

Add these imports at the top of `src/runtime/Milkie.ts` (next to existing imports):

```typescript
import { CacheIndex } from '../trace/CacheIndex.js'
import { ReplayingIOPort } from '../trace/ReplayingIOPort.js'
import { ReplayError } from '../trace/ReplayError.js'
import { extractRunSnapshot } from '../trace/RunSnapshot.js'
import { MemoryStore } from '../store/MemoryStore.js'
```

(`AgentResult` and `AgentRuntime` and `DefaultIOPort` and `InMemoryRecorder` and `createGateway` are already imported.)

- [ ] **Step 4.4: Run Replay tests, verify pass**

Run: `npx jest src/__tests__/Replay.test.ts --runInBand`
Expected: PASS, 5/5

- [ ] **Step 4.5: Run typecheck**

Run: `npx tsc --noEmit`
Expected: no output

- [ ] **Step 4.6: Run full unit suite to verify no regression**

Run: `npm run test:unit`
Expected: all pass

- [ ] **Step 4.7: Commit**

```bash
git add src/runtime/Milkie.ts src/__tests__/Replay.test.ts
git commit -m "$(cat <<'EOF'
feat(runtime): add Milkie.replay(runId) API

Replays a recorded agent run from its event log. CacheIndex serves all
LLM and tool I/O; live gateway is never called. ReplayError on
structural failure (no run / no lifecycle start / unknown agentId);
ReplayDivergenceError when the replayed agent issues a call whose
hash isn't in the recorded cache.

Phase 3 scope: same-host / same-code / same-registered-agent replay.
Sub-agent runs containing spawn may fail with cache miss; not actively
detected (Phase 5).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: E2E test + ARCHITECTURE.md + story readiness

**Files:**
- Create: `tests/e2e/s-005-deterministic-replay.e2e.test.ts`
- Modify: `package.json` (add the new e2e test to `test:e2e:live` and `test:e2e:deterministic` if appropriate)
- Modify: `ARCHITECTURE.md`
- Modify: `docs/stories/INDEX.md`
- Modify: `docs/stories/s-005-deterministic-replay.md`

- [ ] **Step 5.1: Inspect an existing e2e test for patterns**

Run: `head -80 tests/e2e/s-001-react-with-intra-agent-parallel-tools.e2e.test.ts`

This is read-only — no action required. Use the file's gateway/agent setup as a template for the new test.

- [ ] **Step 5.2: Write the E2E test**

Create `tests/e2e/s-005-deterministic-replay.e2e.test.ts`:

```typescript
import { Milkie } from '../../src/runtime/Milkie'
import { MemoryStore } from '../../src/store/MemoryStore'
import { MemoryEventStore } from '../../src/trace/MemoryEventStore'
import type { IModelGateway, ModelRequest, ModelResponse } from '../../src/types/model'
import type { AgentConfig } from '../../src/types/agent'

/**
 * s-005: Deterministically replay a recorded agent run.
 *
 * Phase 3 scope: structural replay. Result must equal original on
 * status + output; LLM gateway must not be called during replay.
 */

class SequentialGateway implements IModelGateway {
  public callCount = 0
  constructor(private readonly responses: ModelResponse[]) {}
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    this.callCount++
    const r = this.responses.shift()
    if (!r) throw new Error('SequentialGateway exhausted')
    return r
  }
  async *stream(_req: ModelRequest): AsyncIterable<never> { yield* [] }
}

const text = (s: string): ModelResponse => ({
  content: [{ type: 'text', text: s }], toolCalls: [], finishReason: 'end_turn',
})

const twoStepAgent: AgentConfig = {
  agentId: 'replay-demo',
  version: '0.0.0',
  systemPrompt: 'you are a friendly bot',
  fsm: {
    initial: 'greet',
    states: [
      { name: 'greet',   kind: 'llm', instructions: 'say hello',          tools: [], transitions: [{ when: 'always', to: 'farewell' }] },
      { name: 'farewell',kind: 'llm', instructions: 'say goodbye briefly',tools: [], transitions: [{ when: 'always', to: 'end' }] },
      { name: 'end',     kind: 'terminal' },
    ],
  },
  model: { provider: 'stub', model: 'stub', adapter: 'stub' },
} as AgentConfig

describe('s-005 deterministic replay (Phase 3 structural)', () => {
  test('record a 2-step run, then replay it without a live gateway', async () => {
    const eventStore = new MemoryEventStore()

    // ---- Record ----
    const recordGateway = new SequentialGateway([text('Hello!'), text('Goodbye!')])
    const recordMilkie = new Milkie({
      stateStore: new MemoryStore(),
      gateway:    recordGateway,
      eventStore,
    })
    recordMilkie.registerAgent(twoStepAgent)
    const original = await recordMilkie.invoke({
      agentId: 'replay-demo', goal: 'demo replay', input: 'start',
    })

    expect(original.status).toBe('completed')
    expect(recordGateway.callCount).toBe(2)

    // ---- Replay ----
    const replayGateway = new SequentialGateway([])  // empty: must not be called
    const replayMilkie = new Milkie({
      stateStore: new MemoryStore(),
      gateway:    replayGateway,
      eventStore,
    })
    replayMilkie.registerAgent(twoStepAgent)

    const replayed = await replayMilkie.replay(original.agentRunId)

    expect(replayed.status).toBe(original.status)
    expect(replayed.output).toBe(original.output)
    expect(replayGateway.callCount).toBe(0)
  })
})
```

- [ ] **Step 5.3: Add e2e to `package.json` scripts**

Read `package.json`. In the `scripts` section, extend `test:e2e:live` to include the new file. The current script ends with `...s-011-...e2e.test.ts --runInBand`. Append the new file path before `--runInBand`:

```
"test:e2e:live": "jest tests/e2e/s-001-react-with-intra-agent-parallel-tools.e2e.test.ts tests/e2e/s-005-deterministic-replay.e2e.test.ts tests/e2e/s-007-inter-agent-parallel-code-review.e2e.test.ts tests/e2e/s-008-long-task-interrupt-and-resume.e2e.test.ts tests/e2e/s-009-multi-turn-with-tool-error-recovery.e2e.test.ts tests/e2e/s-010-skill-versioned-load-and-ab-experiment.e2e.test.ts tests/e2e/s-011-multi-state-fsm-intent-routing-and-slot-filling.e2e.test.ts --runInBand"
```

- [ ] **Step 5.4: Run the new e2e in isolation**

Run: `npx jest tests/e2e/s-005-deterministic-replay.e2e.test.ts --runInBand`
Expected: PASS, 1/1

- [ ] **Step 5.5: Run the full live e2e suite to verify no regression**

Run: `npm run test:e2e:live`
Expected: all pass (Redis-gated cases skipped is OK).

- [ ] **Step 5.6: Update story status — `docs/stories/s-005-deterministic-replay.md`**

Read the file, change the frontmatter `status: draft` to `status: active`. Add a line under it: `# Phase 3 provides structural replay; byte-identical pending Phase 4 non-determinism log.`

- [ ] **Step 5.7: Update `docs/stories/INDEX.md`**

Read the file. Update the s-005 line in the "By status" Notes section and the readiness table to reflect status change. Also update the "Last updated" line:

- Change `Last updated: ...` to `Last updated: 2026-05-24 (Phase 3: cache + structural replay implemented; s-005 active)`
- Change the s-005 Notes line from `s-005 — ... still needs Response cache, Non-determinism log, Replay engine` to `s-005 — IOPort ✓, Event log ✓, Cache ✓, Replay engine ✓; still needs Non-determinism log for byte-identical replay`

- [ ] **Step 5.8: Update `ARCHITECTURE.md`**

Read the file. In the "Implemented today" section, after the Agent Trace event log bullet (basic recording), add:

```markdown
- **Content-addressed cache + structural replay** — `CacheIndex` projects an
  event log into FIFO response queues keyed by canonical request hash;
  `ReplayingIOPort` serves LLM and tool calls from it. `Milkie.replay(runId)`
  re-runs a recorded run with zero live LLM/tool calls; cache miss = strict
  `ReplayDivergenceError`. Phase 3 scope: same-host / same-code /
  same-registered-agent replay; timestamps and UUIDs not byte-identical
  (Phase 4 non-determinism log). (`src/trace/CacheIndex.ts`,
  `src/trace/ReplayingIOPort.ts`, `src/runtime/Milkie.ts:replay`)
```

In the "Target only" section, remove the bullet "**Content-addressed response cache** — request hash → cached response; required for deterministic replay. Phase 3." and the "**Replay and Fork engines**" bullet should be edited to remove "Phase 3" — leave fork as Phase 5.

- [ ] **Step 5.9: Verify everything once more end-to-end**

Run in sequence:
```bash
npx tsc --noEmit
npm run test:unit
npm run test:e2e:live
```
Expected: all pass.

- [ ] **Step 5.10: Commit**

```bash
git add tests/e2e/s-005-deterministic-replay.e2e.test.ts package.json ARCHITECTURE.md docs/stories/INDEX.md docs/stories/s-005-deterministic-replay.md
git commit -m "$(cat <<'EOF'
test(e2e): add s-005 deterministic replay + mark story active

End-to-end test records a 2-step LLM run via Milkie + MemoryEventStore,
then replays the run with an empty gateway and asserts (a) status and
output match, (b) gateway is never called. Story s-005 moves from
draft to active (structural replay; byte-identical pending Phase 4).
ARCHITECTURE.md moves cache + replay from Target to Implemented.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

After completing all 5 tasks, verify against the spec:

| Spec requirement | Task |
|---|---|
| §4 `hashModelRequest` / `hashToolCall` (canonical JSON + SHA-256) | Task 1 |
| §4 `CacheIndex` with FIFO queues + `consumeLLM/consumeTool/remaining/allHashes` | Task 3 |
| §4 `extractRunSnapshot(events)` (no class) | Task 3 |
| §4 `ReplayingIOPort` + cache miss → divergence error | Task 3 |
| §4 `ReplayDivergenceError { kind, actualHash, summary, availableHashes }` | Task 3 |
| §4 New event kinds `agent.run.started/completed` with payload `{ agentId, goal, input, contextId, parentId? }` / `{ status, lastTextOutput?, error? }` | Task 2 |
| §4 `Milkie.replay(runId)` API | Task 4 |
| §4 Structured tool error `{ message, retryable?, code?, name? }` | Task 2 |
| §5 `RecordingIOPort` adds `requestHash` field + `attach` / `detach` | Task 2 |
| §5 `Milkie.invoke` / `resume` wrap with attach/detach | Task 2 |
| §6 I1: exactly one `agent.run.started` per replayable run | Implicit (Task 4 throws if missing); add stricter test if desired |
| §6 I3: `requestHash === hashModelRequest(request)` | Task 2 test |
| §6 I5: ReplayingIOPort cache-hit does not call inner | Task 3 test (FailingGateway) |
| §6 I6: strict miss = throw | Task 3 test |
| §6 I7: replay does not write new events | Task 4 (ioPort is NOT wrapped) — add explicit assertion if not already covered |
| §6 I8: tool error preserves retryable in replay | Task 3 test |
| §8 Hash / CacheIndex / ReplayingIOPort / Trace / Replay tests | Tasks 1-4 |
| §8 e2e s-005 | Task 5 |
| §10 Story readiness update | Task 5 |

**Gaps to close if reviewer flags them:**
- I1 explicit test: append to Task 2 a test that calls `attach` twice in a row and asserts the store has 2 `agent.run.started` events (current behavior — invariant is about replayable runs, not RecordingIOPort enforcing uniqueness). Document this as a structural promise from Milkie, not an enforced check.
- I7 explicit test: in Task 4, add an assertion that during `replay`, the source `eventStore` has the same count of events before and after. Easy to add:
  ```typescript
  const before = (await store.readByRunId(original.agentRunId)).length
  await replayMilkie.replay(original.agentRunId)
  const after  = (await store.readByRunId(original.agentRunId)).length
  expect(after).toBe(before)
  ```

---

## Done Criteria

- [ ] All 5 tasks committed
- [ ] `npx tsc --noEmit` passes
- [ ] `npm run test:unit` passes
- [ ] `npm run test:e2e:live` passes
- [ ] Spec gap table (above) has no unaddressed rows
- [ ] `git log --oneline -6` shows 5 Phase 3 commits on top of `65e4781`
- [ ] Push happens only on explicit user instruction
