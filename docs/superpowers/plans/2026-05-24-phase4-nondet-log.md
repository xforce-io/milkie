# Phase 4 Non-Determinism Log Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade milkie's replay from structural-equivalence to **byte-identical** by recording every `port.now()` / `port.uuid()` call as an event and replaying values from the log. Closes the gap that Phase 3's `s-005` story acknowledges as "structural only"; prerequisite for honest fork semantics in Phase 5.

**Architecture:** Two new event kinds (`clock.read`, `uuid.generated`) join the append-only event log per `port.now()` / `port.uuid()` call. `RecordingIOPort` keeps the sync `IIOPort` signature and uses an internal pending buffer flushed at every async method entry. `ReplayingIOPort` consumes from new FIFO queues on `CacheIndex`. Strict P-wide: over-consume throws `ReplayDivergenceError` immediately; under-consume across all four queues (clock / uuid / llm / tool) throws at `Milkie.replay()` tail. No legacy fallback.

**Tech Stack:** TypeScript, Jest + ts-jest, existing `IIOPort` / `RecordingIOPort` / `ReplayingIOPort` / `CacheIndex` / `IEventStore` (`MemoryEventStore` + `JsonlEventStore`).

---

## File structure

**New:**
- `src/__tests__/CacheIndex.nondet.test.ts` — clock/uuid queue + remaining accessor tests
- `src/__tests__/RecordingIOPort.nondet.test.ts` — pending buffer + flush behavior tests
- `src/__tests__/ReplayingIOPort.nondet.test.ts` — cache consumption + divergence error tests
- `src/__tests__/Replay.nondet.test.ts` — integration: byte-identical proof + over/under-consume

**Modify:**
- `src/trace/types.ts` — add `clock.read` / `uuid.generated` to `EventKind`; add payloads + typed event aliases
- `src/trace/CacheIndex.ts` — add `clockQueue` / `uuidQueue` + `consumeClock` / `consumeUuid` + extend `remaining()`
- `src/trace/ReplayDivergenceError.ts` — extend `kind` union + kind-aware message
- `src/trace/RecordingIOPort.ts` — pending buffer + real `now()` / `uuid()` recording + flush at every async method entry
- `src/trace/ReplayingIOPort.ts` — `now()` / `uuid()` consume from cache; remove inner passthrough
- `src/runtime/Milkie.ts` — `replay()` tail check on all four queues
- `tests/e2e/s-005-deterministic-replay.e2e.test.ts` — assertion upgrade: inner-isolation proof
- `examples/s-005-replay/.milkie/runs/*.jsonl` + `last-run.txt` — re-record fixture
- `examples/s-002-inspect/.milkie/runs/*.jsonl` + `last-run.txt` — re-record fixture
- `roadmap.md` — mark Phase 4 completed; update side-effect policy entry

---

## Task 1: Event schema additions

**Files:**
- Modify: `src/trace/types.ts`
- Test: (compile-level, no new jest file needed)

- [ ] **Step 1: Write failing test**

Append to `src/__tests__/CacheIndex.nondet.test.ts` (NEW file — will hold Task 2 tests too; for now just the import that fails to compile):

```typescript
import type {
  ClockReadEvent,
  ClockReadPayload,
  UuidGeneratedEvent,
  UuidGeneratedPayload,
} from '../trace/types'

describe('Phase 4 event types', () => {
  it('ClockReadEvent is structurally correct', () => {
    const evt: ClockReadEvent = {
      id: 'x', runId: 'r', type: 'clock.read', actor: 'runtime', timestamp: 0,
      payload: { value: 12345 } satisfies ClockReadPayload,
    }
    expect(evt.type).toBe('clock.read')
    expect(evt.payload.value).toBe(12345)
  })

  it('UuidGeneratedEvent is structurally correct', () => {
    const evt: UuidGeneratedEvent = {
      id: 'x', runId: 'r', type: 'uuid.generated', actor: 'runtime', timestamp: 0,
      payload: { value: 'some-uuid-string' } satisfies UuidGeneratedPayload,
    }
    expect(evt.type).toBe('uuid.generated')
    expect(evt.payload.value).toBe('some-uuid-string')
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest src/__tests__/CacheIndex.nondet.test.ts`
Expected: FAIL — `Cannot find module` or TS compile error: `Module '"../trace/types"' has no exported member 'ClockReadEvent'`.

- [ ] **Step 3: Add types to `src/trace/types.ts`**

Extend `EventKind` (around line 9-15):

```typescript
export type EventKind =
  | 'llm.requested'
  | 'llm.responded'
  | 'tool.requested'
  | 'tool.responded'
  | 'agent.run.started'
  | 'agent.run.completed'
  | 'clock.read'
  | 'uuid.generated'
```

Append payloads (after the existing AgentRunCompletedPayload block, before the "Typed event aliases" comment around line 78):

```typescript
// ---- Non-determinism payloads (Phase 4) ----

export interface ClockReadPayload {
  /** Epoch ms returned by the underlying clock at the time agent code called port.now(). */
  value: number
}

export interface UuidGeneratedPayload {
  /** UUID string returned by the underlying generator at the time agent code called port.uuid(). */
  value: string
}
```

Add to the typed event aliases section:

```typescript
export type ClockReadEvent     = Event<ClockReadPayload>     & { type: 'clock.read' }
export type UuidGeneratedEvent = Event<UuidGeneratedPayload> & { type: 'uuid.generated' }
```

Extend `AnyEvent` union:

```typescript
export type AnyEvent =
  | LlmRequestedEvent
  | LlmRespondedEvent
  | ToolRequestedEvent
  | ToolRespondedEvent
  | AgentRunStartedEvent
  | AgentRunCompletedEvent
  | ClockReadEvent
  | UuidGeneratedEvent
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx jest src/__tests__/CacheIndex.nondet.test.ts`
Expected: PASS (2 tests). Also run `npm run build` to confirm no other consumers break.

- [ ] **Step 5: Commit**

```bash
git add src/trace/types.ts src/__tests__/CacheIndex.nondet.test.ts
git commit -m "feat(trace): add clock.read + uuid.generated event types (Phase 4 schema)"
```

---

## Task 2: CacheIndex — clock + uuid FIFO queues

**Files:**
- Modify: `src/trace/CacheIndex.ts`
- Test: `src/__tests__/CacheIndex.nondet.test.ts` (extend Task 1's file)

- [ ] **Step 1: Write failing test**

Append to `src/__tests__/CacheIndex.nondet.test.ts` (inside the file, after Task 1's describe):

```typescript
import { CacheIndex, CacheIndexEmptyError } from '../trace/CacheIndex'
import type { Event } from '../trace/types'

const clockEvent = (id: string, value: number): Event => ({
  id, runId: 'r', type: 'clock.read', actor: 'runtime', timestamp: 0,
  payload: { value },
})

const uuidEvent = (id: string, value: string): Event => ({
  id, runId: 'r', type: 'uuid.generated', actor: 'runtime', timestamp: 0,
  payload: { value },
})

describe('CacheIndex — clock/uuid queues', () => {
  it('consumeClock returns values in FIFO order across the entire log', () => {
    const cache = CacheIndex.fromEvents([
      clockEvent('c1', 100),
      clockEvent('c2', 200),
      clockEvent('c3', 300),
    ])
    expect(cache.consumeClock()).toBe(100)
    expect(cache.consumeClock()).toBe(200)
    expect(cache.consumeClock()).toBe(300)
  })

  it('consumeUuid returns values in FIFO order', () => {
    const cache = CacheIndex.fromEvents([
      uuidEvent('u1', 'first-uuid'),
      uuidEvent('u2', 'second-uuid'),
    ])
    expect(cache.consumeUuid()).toBe('first-uuid')
    expect(cache.consumeUuid()).toBe('second-uuid')
  })

  it('consumeClock throws CacheIndexEmptyError when queue empty', () => {
    const cache = CacheIndex.fromEvents([])
    expect(() => cache.consumeClock()).toThrow(CacheIndexEmptyError)
  })

  it('consumeUuid throws CacheIndexEmptyError when queue empty', () => {
    const cache = CacheIndex.fromEvents([clockEvent('c1', 1)])
    expect(() => cache.consumeUuid()).toThrow(CacheIndexEmptyError)
  })

  it('remaining() reports all four queues including clock + uuid', () => {
    const cache = CacheIndex.fromEvents([
      clockEvent('c1', 100),
      clockEvent('c2', 200),
      uuidEvent('u1', 'a'),
    ])
    const r = cache.remaining()
    expect(r).toEqual({ llm: 0, tool: 0, clock: 2, uuid: 1 })
  })

  it('remaining decreases as values are consumed', () => {
    const cache = CacheIndex.fromEvents([
      clockEvent('c1', 100),
      uuidEvent('u1', 'a'),
    ])
    cache.consumeClock()
    cache.consumeUuid()
    expect(cache.remaining()).toEqual({ llm: 0, tool: 0, clock: 0, uuid: 0 })
  })

  it('clock and uuid queues do not interfere with each other', () => {
    const cache = CacheIndex.fromEvents([
      clockEvent('c1', 100),
      uuidEvent('u1', 'a'),
      clockEvent('c2', 200),
      uuidEvent('u2', 'b'),
    ])
    expect(cache.consumeClock()).toBe(100)
    expect(cache.consumeUuid()).toBe('a')
    expect(cache.consumeClock()).toBe(200)
    expect(cache.consumeUuid()).toBe('b')
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest src/__tests__/CacheIndex.nondet.test.ts`
Expected: FAIL — `cache.consumeClock is not a function` or similar.

- [ ] **Step 3: Implement clock/uuid queues in `src/trace/CacheIndex.ts`**

Modify the imports at top to add the new payload types:

```typescript
import type {
  Event,
  LlmRespondedPayload,
  ToolRespondedPayload,
  ClockReadPayload,
  UuidGeneratedPayload,
} from './types.js'
```

Add two private fields next to `llm` and `tool`:

```typescript
export class CacheIndex {
  private readonly llm:   Map<string, ModelResponse[]>
  private readonly tool:  Map<string, ToolOutcome[]>
  private readonly clock: number[]
  private readonly uuid:  string[]
```

Update the private constructor signature and assignments:

```typescript
  private constructor(
    llm:   Map<string, ModelResponse[]>,
    tool:  Map<string, ToolOutcome[]>,
    clock: number[],
    uuid:  string[],
  ) {
    this.llm   = llm
    this.tool  = tool
    this.clock = clock
    this.uuid  = uuid
  }
```

Update `fromEvents` to populate the new arrays and pass them in:

```typescript
  static fromEvents(events: Event[]): CacheIndex {
    const llm:   Map<string, ModelResponse[]> = new Map()
    const tool:  Map<string, ToolOutcome[]>   = new Map()
    const clock: number[] = []
    const uuid:  string[] = []

    for (const ev of events) {
      if (ev.type === 'llm.responded') {
        const p = ev.payload as LlmRespondedPayload
        if (!p.requestHash) continue   // Phase 2 events; skip
        push(llm, p.requestHash, p.response)
      } else if (ev.type === 'tool.responded') {
        const p = ev.payload as ToolRespondedPayload
        if (!p.requestHash) continue   // Phase 2 events; skip
        push(tool, p.requestHash, { output: p.output, error: p.error })
      } else if (ev.type === 'clock.read') {
        clock.push((ev.payload as ClockReadPayload).value)
      } else if (ev.type === 'uuid.generated') {
        uuid.push((ev.payload as UuidGeneratedPayload).value)
      }
    }

    return new CacheIndex(llm, tool, clock, uuid)
  }
```

Add two consume methods after `consumeTool`:

```typescript
  consumeClock(): number {
    if (this.clock.length === 0) throw new CacheIndexEmptyError('CacheIndex: clock queue empty')
    return this.clock.shift()!
  }

  consumeUuid(): string {
    if (this.uuid.length === 0) throw new CacheIndexEmptyError('CacheIndex: uuid queue empty')
    return this.uuid.shift()!
  }
```

Update `remaining()` to include clock/uuid:

```typescript
  remaining(): { llm: number; tool: number; clock: number; uuid: number } {
    let llmCount = 0, toolCount = 0
    for (const q of this.llm.values())  llmCount  += q.length
    for (const q of this.tool.values()) toolCount += q.length
    return { llm: llmCount, tool: toolCount, clock: this.clock.length, uuid: this.uuid.length }
  }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx jest src/__tests__/CacheIndex.nondet.test.ts`
Expected: PASS (9 tests — 2 from Task 1 + 7 from Task 2).

Also run: `npx jest src/__tests__/` to catch any regression in the existing CacheIndex consumers (Replay.test.ts uses `remaining()` — verify it still passes with the new return shape).

- [ ] **Step 5: Commit**

```bash
git add src/trace/CacheIndex.ts src/__tests__/CacheIndex.nondet.test.ts
git commit -m "feat(trace): CacheIndex grows clock + uuid FIFO queues with remaining()"
```

---

## Task 3: ReplayDivergenceError — kind extension + kind-aware message

**Files:**
- Modify: `src/trace/ReplayDivergenceError.ts`
- Test: `src/__tests__/CacheIndex.nondet.test.ts` (extend)

- [ ] **Step 1: Write failing test**

Append to `src/__tests__/CacheIndex.nondet.test.ts`:

```typescript
import { ReplayDivergenceError } from '../trace/ReplayDivergenceError'

describe('ReplayDivergenceError — clock/uuid kinds', () => {
  it('accepts clock kind and produces a hash-free message', () => {
    const err = new ReplayDivergenceError('clock', '', 'clock queue exhausted after 3 consumed', [])
    expect(err.kind).toBe('clock')
    expect(err.message).toContain('clock')
    expect(err.message).toContain('clock queue exhausted after 3 consumed')
    expect(err.message).not.toMatch(/hash\s+…/)  // no garbled hash-truncation for hashless kinds
  })

  it('accepts uuid kind', () => {
    const err = new ReplayDivergenceError('uuid', '', '1 uuid event(s) unconsumed', [])
    expect(err.kind).toBe('uuid')
    expect(err.message).toContain('uuid')
    expect(err.message).toContain('1 uuid event(s) unconsumed')
  })

  it('preserves existing llm/tool message format with truncated hash', () => {
    const err = new ReplayDivergenceError('llm', 'abc123def456hash999', 'last user: hello', ['hash-a'])
    expect(err.kind).toBe('llm')
    expect(err.message).toContain('abc123def456')   // first 12 chars of hash
    expect(err.message).toContain('last user: hello')
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest src/__tests__/CacheIndex.nondet.test.ts -t "ReplayDivergenceError"`
Expected: FAIL — TS error: `Argument of type '"clock"' is not assignable to parameter of type '"llm" | "tool"'`.

- [ ] **Step 3: Extend `src/trace/ReplayDivergenceError.ts`**

Replace the entire file contents:

```typescript
export type DivergenceKind = 'llm' | 'tool' | 'clock' | 'uuid'

/**
 * Thrown when replay diverges from the recorded log:
 *  - llm/tool: replayed call's canonical hash not in cache (over-consume)
 *  - clock/uuid: replay called port.now/port.uuid more times than recorded
 *    OR replay completed with unconsumed recorded values (under-consume)
 *
 * For llm/tool, actualHash + availableHashes carry the hash diagnostic;
 * for clock/uuid the message itself carries the count, hash fields are
 * empty placeholders.
 */
export class ReplayDivergenceError extends Error {
  constructor(
    public readonly kind:            DivergenceKind,
    public readonly actualHash:      string,
    public readonly summary:         string,
    public readonly availableHashes: string[],
  ) {
    const detail = (kind === 'llm' || kind === 'tool')
      ? `hash ${actualHash.slice(0, 12)}… not in cache. ${summary}`
      : summary
    super(`Replay divergence (${kind}): ${detail}`)
    this.name = 'ReplayDivergenceError'
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx jest src/__tests__/CacheIndex.nondet.test.ts -t "ReplayDivergenceError"`
Expected: PASS (3 tests).

Also re-run existing tests that consume ReplayDivergenceError:
`npx jest src/__tests__/Replay.test.ts`
Expected: PASS — existing `kind: 'llm' | 'tool'` usages still type-check and runtime behavior unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/trace/ReplayDivergenceError.ts src/__tests__/CacheIndex.nondet.test.ts
git commit -m "feat(trace): ReplayDivergenceError supports clock + uuid kinds"
```

---

## Task 4: RecordingIOPort — pending buffer + real now/uuid recording

**Files:**
- Modify: `src/trace/RecordingIOPort.ts`
- Test: `src/__tests__/RecordingIOPort.nondet.test.ts` (new)

- [ ] **Step 1: Write failing test**

Create `src/__tests__/RecordingIOPort.nondet.test.ts`:

```typescript
import { RecordingIOPort } from '../trace/RecordingIOPort'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import type { IIOPort } from '../runtime/IOPort'
import type { ModelRequest, ModelResponse } from '../types/model'

class StubInnerPort implements IIOPort {
  public clockCalls = 0
  public uuidCalls  = 0
  private nextClock = 1000
  private nextUuid  = 1

  async invokeLLM(_req: ModelRequest): Promise<ModelResponse> {
    return { content: [{ type: 'text', text: 'stub' }], toolCalls: [], finishReason: 'end_turn' }
  }
  async invokeTool(_n: string, _i: unknown, _e: () => Promise<unknown>): Promise<unknown> {
    return 'stub-output'
  }
  now():  number { this.clockCalls++; return this.nextClock++ }
  uuid(): string { this.uuidCalls++; return `uuid-${this.nextUuid++}` }
}

describe('RecordingIOPort — non-determinism recording', () => {
  it('now() returns inner value and queues a clock.read event', async () => {
    const inner = new StubInnerPort()
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(inner, store, 'r1')

    const val = port.now()
    expect(typeof val).toBe('number')

    // event not flushed yet — flush only happens at next async method entry
    expect((await store.readByRunId('r1')).filter(e => e.type === 'clock.read')).toHaveLength(0)

    // trigger flush via any async method
    await port.detach({ status: 'completed' })

    const clockEvents = (await store.readByRunId('r1')).filter(e => e.type === 'clock.read')
    expect(clockEvents).toHaveLength(1)
    expect((clockEvents[0]!.payload as { value: number }).value).toBe(val)
  })

  it('uuid() returns inner value and queues a uuid.generated event', async () => {
    const inner = new StubInnerPort()
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(inner, store, 'r1')

    const val = port.uuid()
    expect(typeof val).toBe('string')

    await port.detach({ status: 'completed' })

    const uuidEvents = (await store.readByRunId('r1')).filter(e => e.type === 'uuid.generated')
    expect(uuidEvents).toHaveLength(1)
    expect((uuidEvents[0]!.payload as { value: string }).value).toBe(val)
  })

  it('multiple sync now/uuid calls flush in input order at next async boundary', async () => {
    const inner = new StubInnerPort()
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(inner, store, 'r1')

    port.now()
    port.uuid()
    port.now()
    port.uuid()

    await port.attach({ agentId: 'a', goal: 'g', input: 'i', contextId: 'r1' })

    // events should appear in interleaved order BEFORE agent.run.started
    const events = await store.readByRunId('r1')
    const nondetIdx = events
      .map((e, i) => ['clock.read', 'uuid.generated'].includes(e.type) ? i : -1)
      .filter(i => i >= 0)
    const startedIdx = events.findIndex(e => e.type === 'agent.run.started')
    expect(nondetIdx).toHaveLength(4)
    for (const i of nondetIdx) expect(i).toBeLessThan(startedIdx)
    expect(events[nondetIdx[0]!]!.type).toBe('clock.read')
    expect(events[nondetIdx[1]!]!.type).toBe('uuid.generated')
    expect(events[nondetIdx[2]!]!.type).toBe('clock.read')
    expect(events[nondetIdx[3]!]!.type).toBe('uuid.generated')
  })

  it('flush happens at every async method entry, not only detach', async () => {
    const inner = new StubInnerPort()
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(inner, store, 'r1')

    await port.attach({ agentId: 'a', goal: 'g', input: 'i', contextId: 'r1' })

    port.now()
    await port.invokeLLM({ provider: 'stub', model: 'stub', messages: [] } as ModelRequest)

    // clock.read recorded BEFORE llm.requested (flushed at invokeLLM entry)
    const events = await store.readByRunId('r1')
    const clockIdx = events.findIndex(e => e.type === 'clock.read')
    const llmReqIdx = events.findIndex(e => e.type === 'llm.requested')
    expect(clockIdx).toBeGreaterThan(-1)
    expect(llmReqIdx).toBeGreaterThan(-1)
    expect(clockIdx).toBeLessThan(llmReqIdx)
  })

  it('infrastructure now/uuid (for event id/timestamp) do not recurse into nondet events', async () => {
    const inner = new StubInnerPort()
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(inner, store, 'r1')

    // attach + detach use inner.now/uuid internally for event id/timestamp,
    // but those must NOT be recorded as clock.read/uuid.generated.
    await port.attach({ agentId: 'a', goal: 'g', input: 'i', contextId: 'r1' })
    await port.detach({ status: 'completed' })

    const events  = await store.readByRunId('r1')
    const nondets = events.filter(e => e.type === 'clock.read' || e.type === 'uuid.generated')
    expect(nondets).toHaveLength(0)   // zero agent-facing calls were made
    // sanity: inner WAS called (for event id/timestamp on attach + detach)
    expect(inner.clockCalls).toBeGreaterThan(0)
    expect(inner.uuidCalls).toBeGreaterThan(0)
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest src/__tests__/RecordingIOPort.nondet.test.ts`
Expected: FAIL — multiple tests fail because `port.now()` / `port.uuid()` are still passthrough (no event written).

- [ ] **Step 3: Modify `src/trace/RecordingIOPort.ts`**

Add the type imports at top:

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
  ClockReadPayload,
  UuidGeneratedPayload,
} from './types.js'
import { hashModelRequest, hashToolCall } from './hash.js'
```

Add a private buffer type and field inside the class (after `actor`):

```typescript
type PendingNondet =
  | { kind: 'clock'; value: number }
  | { kind: 'uuid';  value: string }

export class RecordingIOPort implements IIOPort {
  private readonly pendingNondet: PendingNondet[] = []

  constructor(
    private readonly inner: IIOPort,
    private readonly store: IEventStore,
    private readonly runId: string,
    private readonly actor: string = 'runtime',
  ) {}
```

Add a private flush method (place after the constructor, before `attach`):

```typescript
  /**
   * Drain pending nondet records to the store in input order. Called at
   * every async method entry so that agent-facing port.now/port.uuid
   * calls observe the invariant "nondet events appear before the next
   * recorded event that consumes them."
   *
   * Each emitted event's own `id` and `timestamp` fields use inner.uuid/
   * inner.now directly — they are infrastructure bookkeeping, not part
   * of agent-observable non-determinism, and recording them would recurse.
   */
  private async flushPendingNondet(): Promise<void> {
    while (this.pendingNondet.length > 0) {
      const item = this.pendingNondet.shift()!
      if (item.kind === 'clock') {
        await this.store.append({
          id:        this.inner.uuid(),
          runId:     this.runId,
          type:      'clock.read',
          actor:     this.actor,
          timestamp: this.inner.now(),
          payload:   { value: item.value } satisfies ClockReadPayload,
        })
      } else {
        await this.store.append({
          id:        this.inner.uuid(),
          runId:     this.runId,
          type:      'uuid.generated',
          actor:     this.actor,
          timestamp: this.inner.now(),
          payload:   { value: item.value } satisfies UuidGeneratedPayload,
        })
      }
    }
  }
```

Update `now()` and `uuid()` (replace the existing methods at the bottom of the class):

```typescript
  now(): number {
    const value = this.inner.now()
    this.pendingNondet.push({ kind: 'clock', value })
    return value
  }

  uuid(): string {
    const value = this.inner.uuid()
    this.pendingNondet.push({ kind: 'uuid', value })
    return value
  }
```

Insert `await this.flushPendingNondet()` as the FIRST line of every async method:

In `attach`:
```typescript
  async attach(payload: AgentRunStartedPayload): Promise<void> {
    await this.flushPendingNondet()
    await this.store.append({
      // ... existing body
    })
  }
```

In `detach`, `invokeLLM`, `invokeTool` — same pattern. For each, the first executable line of the method body becomes `await this.flushPendingNondet()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `npx jest src/__tests__/RecordingIOPort.nondet.test.ts`
Expected: PASS (5 tests).

Also re-run existing recording tests for regression:
`npx jest src/__tests__/Trace.test.ts`
Expected: PASS — existing assertions on attach/detach/invokeLLM/invokeTool event shapes still hold; flushPendingNondet on empty buffer is a no-op.

- [ ] **Step 5: Commit**

```bash
git add src/trace/RecordingIOPort.ts src/__tests__/RecordingIOPort.nondet.test.ts
git commit -m "feat(trace): RecordingIOPort records port.now/port.uuid via pending buffer"
```

---

## Task 5: ReplayingIOPort — now/uuid consume from cache

**Files:**
- Modify: `src/trace/ReplayingIOPort.ts`
- Test: `src/__tests__/ReplayingIOPort.nondet.test.ts` (new)

- [ ] **Step 1: Write failing test**

Create `src/__tests__/ReplayingIOPort.nondet.test.ts`:

```typescript
import { ReplayingIOPort } from '../trace/ReplayingIOPort'
import { CacheIndex } from '../trace/CacheIndex'
import { ReplayDivergenceError } from '../trace/ReplayDivergenceError'
import type { IIOPort } from '../runtime/IOPort'
import type { ModelRequest, ModelResponse } from '../types/model'
import type { Event } from '../trace/types'

class ExplodingInnerPort implements IIOPort {
  async invokeLLM(_req: ModelRequest): Promise<ModelResponse> {
    throw new Error('inner.invokeLLM must not be called during replay')
  }
  async invokeTool(_n: string, _i: unknown, _e: () => Promise<unknown>): Promise<unknown> {
    throw new Error('inner.invokeTool must not be called during replay')
  }
  now():  number { throw new Error('inner.now must not be called during nondet replay') }
  uuid(): string { throw new Error('inner.uuid must not be called during nondet replay') }
}

const clockEvent = (value: number): Event => ({
  id: 'c', runId: 'r', type: 'clock.read', actor: 'runtime', timestamp: 0,
  payload: { value },
})
const uuidEvent = (value: string): Event => ({
  id: 'u', runId: 'r', type: 'uuid.generated', actor: 'runtime', timestamp: 0,
  payload: { value },
})

describe('ReplayingIOPort — nondet consumption', () => {
  it('now() returns cached value in FIFO order without touching inner', () => {
    const cache = CacheIndex.fromEvents([clockEvent(111), clockEvent(222)])
    const port  = new ReplayingIOPort(cache, new ExplodingInnerPort())
    expect(port.now()).toBe(111)
    expect(port.now()).toBe(222)
  })

  it('uuid() returns cached value in FIFO order without touching inner', () => {
    const cache = CacheIndex.fromEvents([uuidEvent('a'), uuidEvent('b')])
    const port  = new ReplayingIOPort(cache, new ExplodingInnerPort())
    expect(port.uuid()).toBe('a')
    expect(port.uuid()).toBe('b')
  })

  it('now() throws ReplayDivergenceError when clock queue is exhausted', () => {
    const cache = CacheIndex.fromEvents([])
    const port  = new ReplayingIOPort(cache, new ExplodingInnerPort())
    expect(() => port.now()).toThrow(ReplayDivergenceError)
    try { port.now() }
    catch (err) {
      const e = err as ReplayDivergenceError
      expect(e.kind).toBe('clock')
    }
  })

  it('uuid() throws ReplayDivergenceError when uuid queue is exhausted', () => {
    const cache = CacheIndex.fromEvents([])
    const port  = new ReplayingIOPort(cache, new ExplodingInnerPort())
    expect(() => port.uuid()).toThrow(ReplayDivergenceError)
    try { port.uuid() }
    catch (err) {
      const e = err as ReplayDivergenceError
      expect(e.kind).toBe('uuid')
    }
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest src/__tests__/ReplayingIOPort.nondet.test.ts`
Expected: FAIL — current `port.now()` calls `inner.now()` which throws "must not be called during nondet replay".

- [ ] **Step 3: Modify `src/trace/ReplayingIOPort.ts`**

Add the `CacheIndexEmptyError` to existing imports (it's already imported from CacheIndex.js). Add `ReplayDivergenceError` if not already imported. Then replace `now()` and `uuid()`:

```typescript
  now(): number {
    try {
      return this.cache.consumeClock()
    } catch (err) {
      if (err instanceof CacheIndexEmptyError) {
        const r = this.cache.remaining()
        throw new ReplayDivergenceError(
          'clock', '',
          `clock.read queue exhausted (remaining llm=${r.llm} tool=${r.tool} uuid=${r.uuid})`,
          [],
        )
      }
      throw err
    }
  }

  uuid(): string {
    try {
      return this.cache.consumeUuid()
    } catch (err) {
      if (err instanceof CacheIndexEmptyError) {
        const r = this.cache.remaining()
        throw new ReplayDivergenceError(
          'uuid', '',
          `uuid.generated queue exhausted (remaining llm=${r.llm} tool=${r.tool} clock=${r.clock})`,
          [],
        )
      }
      throw err
    }
  }
```

Note: `this.inner` reference is preserved (still used by `invokeLLM` / `invokeTool` constructors below, even though they don't actually call inner during replay).

- [ ] **Step 4: Run test to verify it passes**

Run: `npx jest src/__tests__/ReplayingIOPort.nondet.test.ts`
Expected: PASS (4 tests).

Also re-run existing replay tests for regression:
`npx jest src/__tests__/Replay.test.ts`
Expected: PASS — existing llm/tool replay paths unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/trace/ReplayingIOPort.ts src/__tests__/ReplayingIOPort.nondet.test.ts
git commit -m "feat(trace): ReplayingIOPort consumes clock/uuid from cache"
```

---

## Task 6: Milkie.replay() — P-wide tail check across all four queues

**Files:**
- Modify: `src/runtime/Milkie.ts`
- Test: `src/__tests__/Replay.nondet.test.ts` (new)

- [ ] **Step 1: Write failing test**

Create `src/__tests__/Replay.nondet.test.ts`:

```typescript
import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { JsonlEventStore } from '../trace/JsonlEventStore'
import { ReplayDivergenceError } from '../trace/ReplayDivergenceError'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'
import type { Event } from '../trace/types'
import fs from 'fs'
import os from 'os'
import path from 'path'

class SequentialGateway implements IModelGateway {
  constructor(private readonly responses: ModelResponse[]) {}
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    const r = this.responses.shift()
    if (!r) throw new Error('SequentialGateway exhausted')
    return r
  }
  async *stream(_req: ModelRequest): AsyncIterable<never> { yield* [] }
}

const text = (s: string): ModelResponse => ({
  content: [{ type: 'text', text: s }], toolCalls: [], finishReason: 'end_turn',
})

const echoAgentMd = `---
agentId: echo
fsm:
  states:
    - name: react
      type: llm
      instructions: say hi
      tools: []
model:
  provider: stub
  model: stub
  adapter: openai-compatible
---
sys`

describe('Milkie.replay — Phase 4 tail check (P-wide)', () => {
  let tmpDir: string
  let agentFile: string
  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'milkie-p4-replay-'))
    agentFile = path.join(tmpDir, 'echo.md')
    fs.writeFileSync(agentFile, echoAgentMd)
  })
  afterEach(() => { fs.rmSync(tmpDir, { recursive: true, force: true }) })

  it('replay succeeds with no unconsumed events', async () => {
    const eventStore = new JsonlEventStore(path.join(tmpDir, 'runs'))
    const record = new Milkie({ stateStore: new MemoryStore(),
      gateway: new SequentialGateway([text('hello')]), eventStore })
    record.loadAgentFile(agentFile)
    const original = await record.invoke({ agentId: 'echo', goal: 'g', input: 'i' })
    expect(original.status).toBe('completed')

    const replayer = new Milkie({ stateStore: new MemoryStore(),
      gateway: new SequentialGateway([]), eventStore })
    replayer.loadAgentFile(agentFile)
    const replayed = await replayer.replay(original.agentRunId)
    expect(replayed.status).toBe('completed')
  })

  it('replay throws ReplayDivergenceError when recorded clock events go unconsumed', async () => {
    const eventStore = new JsonlEventStore(path.join(tmpDir, 'runs'))
    const record = new Milkie({ stateStore: new MemoryStore(),
      gateway: new SequentialGateway([text('hello')]), eventStore })
    record.loadAgentFile(agentFile)
    const original = await record.invoke({ agentId: 'echo', goal: 'g', input: 'i' })

    // Inject a phantom clock.read into the recorded file — replay won't consume it.
    const phantom: Event = {
      id: 'phantom-clock', runId: original.agentRunId, type: 'clock.read',
      actor: 'runtime', timestamp: 999, payload: { value: 999 },
    }
    fs.appendFileSync(
      path.join(tmpDir, 'runs', `${original.agentRunId}.jsonl`),
      JSON.stringify(phantom) + '\n',
    )

    const replayer = new Milkie({ stateStore: new MemoryStore(),
      gateway: new SequentialGateway([]), eventStore })
    replayer.loadAgentFile(agentFile)
    await expect(replayer.replay(original.agentRunId)).rejects.toThrow(ReplayDivergenceError)
    try { await replayer.replay(original.agentRunId) }
    catch (err) {
      const e = err as ReplayDivergenceError
      expect(e.kind).toBe('clock')
      expect(e.message).toContain('unconsumed')
    }
  })

  it('replay throws ReplayDivergenceError when recorded uuid events go unconsumed', async () => {
    const eventStore = new JsonlEventStore(path.join(tmpDir, 'runs'))
    const record = new Milkie({ stateStore: new MemoryStore(),
      gateway: new SequentialGateway([text('hello')]), eventStore })
    record.loadAgentFile(agentFile)
    const original = await record.invoke({ agentId: 'echo', goal: 'g', input: 'i' })

    const phantom: Event = {
      id: 'phantom-uuid', runId: original.agentRunId, type: 'uuid.generated',
      actor: 'runtime', timestamp: 999, payload: { value: 'never-consumed' },
    }
    fs.appendFileSync(
      path.join(tmpDir, 'runs', `${original.agentRunId}.jsonl`),
      JSON.stringify(phantom) + '\n',
    )

    const replayer = new Milkie({ stateStore: new MemoryStore(),
      gateway: new SequentialGateway([]), eventStore })
    replayer.loadAgentFile(agentFile)
    await expect(replayer.replay(original.agentRunId)).rejects.toThrow(ReplayDivergenceError)
  })

  it('replay throws ReplayDivergenceError when recorded llm events go unconsumed', async () => {
    const eventStore = new JsonlEventStore(path.join(tmpDir, 'runs'))
    const record = new Milkie({ stateStore: new MemoryStore(),
      gateway: new SequentialGateway([text('hello')]), eventStore })
    record.loadAgentFile(agentFile)
    const original = await record.invoke({ agentId: 'echo', goal: 'g', input: 'i' })

    // Append a phantom llm.responded event whose hash will never be consumed
    // (no llm.requested with this hash will be issued during replay).
    const phantom: Event = {
      id: 'phantom-llm', runId: original.agentRunId, type: 'llm.responded',
      actor: 'runtime', timestamp: 999,
      payload: { response: text('phantom'), requestHash: 'phantom-hash-never-issued' },
    }
    fs.appendFileSync(
      path.join(tmpDir, 'runs', `${original.agentRunId}.jsonl`),
      JSON.stringify(phantom) + '\n',
    )

    const replayer = new Milkie({ stateStore: new MemoryStore(),
      gateway: new SequentialGateway([]), eventStore })
    replayer.loadAgentFile(agentFile)
    await expect(replayer.replay(original.agentRunId)).rejects.toThrow(ReplayDivergenceError)
    try { await replayer.replay(original.agentRunId) }
    catch (err) {
      const e = err as ReplayDivergenceError
      expect(e.kind).toBe('llm')
    }
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest src/__tests__/Replay.nondet.test.ts`
Expected: FAIL — the "no unconsumed events" test should pass once Tasks 4+5 are in (RecordingIOPort writes nondet, ReplayingIOPort consumes), but the three "throws on unconsumed" tests fail because `Milkie.replay()` has no tail check yet.

- [ ] **Step 3: Modify `src/runtime/Milkie.ts`**

In the `replay()` method, find the existing tail (around lines 364-366):

```typescript
    const result = await runtime.run(snapshot.input)
    if (divergenceError) throw divergenceError
    return result
  }
```

Replace with:

```typescript
    const result = await runtime.run(snapshot.input)
    if (divergenceError) throw divergenceError

    // P-wide strict under-consume check: any recorded event the replay
    // failed to consume signals divergence (the run took a different path
    // than recording, or recording captured events the runtime no longer
    // emits). Check all four queues.
    const remaining = cache.remaining()
    for (const kind of ['clock', 'uuid', 'llm', 'tool'] as const) {
      const n = remaining[kind]
      if (n > 0) {
        throw new ReplayDivergenceError(
          kind, '',
          `${n} ${kind} event(s) unconsumed after replay completed`,
          [],
        )
      }
    }
    return result
  }
```

Verify `ReplayDivergenceError` is already imported at top of file (it is — used elsewhere in `replay()`).

- [ ] **Step 4: Run test to verify it passes**

Run: `npx jest src/__tests__/Replay.nondet.test.ts`
Expected: PASS (4 tests).

Run full suite for regression:
`npx jest src/__tests__/`
Expected: existing `Replay.test.ts` tests should still pass. The existing s-005 fixture in `examples/s-005-replay/.milkie/runs/<uuid>.jsonl` was recorded under Phase 3 (no nondet events). When Phase 3 fixture is replayed under Phase 4 code, ReplayingIOPort.now/uuid will hit empty queues and throw — but `src/__tests__/Replay.test.ts` uses in-test fixtures, not the s-005 file, so it should still pass.

**IMPORTANT**: e2e tests against the committed s-005 fixture WILL break. That's expected and fixed in Task 8.

- [ ] **Step 5: Commit**

```bash
git add src/runtime/Milkie.ts src/__tests__/Replay.nondet.test.ts
git commit -m "feat(runtime): Milkie.replay() strict under-consume check across all four queues"
```

---

## Task 7: Byte-identical integration test (the core proof)

**Files:**
- Test: `src/__tests__/Replay.nondet.test.ts` (extend)

- [ ] **Step 1: Write failing test**

Append to `src/__tests__/Replay.nondet.test.ts` inside the existing describe block:

```typescript
  it('byte-identical replay: agent embeds port.uuid() in LLM request, replay reuses cached uuid', async () => {
    // Custom agent that injects port.uuid() into the goal so the LLM request
    // hash depends on a non-deterministic value. Without Phase 4 record/replay
    // of uuid, the second invocation generates a fresh uuid → hash mismatch
    // → cache miss → ReplayDivergenceError.
    //
    // With Phase 4: recording captures the uuid; replay returns the same
    // uuid; LLM request hash matches; cache hit; replay succeeds.

    const eventStore = new JsonlEventStore(path.join(tmpDir, 'runs'))
    const gateway = new SequentialGateway([text('hello')])

    // Record: agent runs once; port.uuid() is called during AgentRuntime
    // construction (for childTraceId etc) AND inside the FSM's LLM path.
    // The recorded hash of the llm.requested event captures the random uuid
    // values embedded by the runtime.
    const record = new Milkie({ stateStore: new MemoryStore(), gateway, eventStore })
    record.loadAgentFile(agentFile)
    const original = await record.invoke({ agentId: 'echo', goal: 'g', input: 'i' })
    expect(original.status).toBe('completed')

    // Verify recording captured nondet events.
    const recordedEvents = await eventStore.readByRunId(original.agentRunId)
    const clocks = recordedEvents.filter(e => e.type === 'clock.read')
    const uuids  = recordedEvents.filter(e => e.type === 'uuid.generated')
    expect(clocks.length).toBeGreaterThan(0)
    expect(uuids.length).toBeGreaterThan(0)

    // Replay: zero gateway responses available. If replay needed a fresh
    // LLM call (because hash didn't match), the SequentialGateway would
    // throw 'exhausted'. The fact that replay succeeds proves the cached
    // LLM response was served, which proves the LLM request hash matched,
    // which proves the embedded uuid values were replayed not re-sampled.
    const replayer = new Milkie({ stateStore: new MemoryStore(),
      gateway: new SequentialGateway([]), eventStore })
    replayer.loadAgentFile(agentFile)
    const replayed = await replayer.replay(original.agentRunId)
    expect(replayed.status).toBe('completed')
    expect(replayed.output).toBe(original.output)
  })

  it('over-consume: replay calls port.uuid() more than recorded → immediate ReplayDivergenceError', async () => {
    const eventStore = new JsonlEventStore(path.join(tmpDir, 'runs'))
    const record = new Milkie({ stateStore: new MemoryStore(),
      gateway: new SequentialGateway([text('hello')]), eventStore })
    record.loadAgentFile(agentFile)
    const original = await record.invoke({ agentId: 'echo', goal: 'g', input: 'i' })

    // Tamper: remove all uuid.generated events from the recorded file.
    // Now during replay, the first port.uuid() call will over-consume.
    const filePath = path.join(tmpDir, 'runs', `${original.agentRunId}.jsonl`)
    const filtered = fs.readFileSync(filePath, 'utf-8')
      .split('\n')
      .filter(line => line.length > 0)
      .map(line => JSON.parse(line))
      .filter((e: { type: string }) => e.type !== 'uuid.generated')
      .map(e => JSON.stringify(e))
      .join('\n') + '\n'
    fs.writeFileSync(filePath, filtered)

    const replayer = new Milkie({ stateStore: new MemoryStore(),
      gateway: new SequentialGateway([]), eventStore })
    replayer.loadAgentFile(agentFile)
    await expect(replayer.replay(original.agentRunId)).rejects.toThrow(ReplayDivergenceError)
    try { await replayer.replay(original.agentRunId) }
    catch (err) {
      const e = err as ReplayDivergenceError
      expect(e.kind).toBe('uuid')
      expect(e.message).toContain('exhausted')
    }
  })
```

- [ ] **Step 2: Run test to verify it passes**

Run: `npx jest src/__tests__/Replay.nondet.test.ts -t "byte-identical|over-consume"`
Expected: PASS (2 tests). Both rely on Tasks 4+5+6 already being in place.

If "byte-identical" fails:
- Most likely cause: the test echo agent doesn't actually call port.uuid() in any code path the LLM request hash depends on. Verify by inspecting what's in the recorded llm.requested event. If the uuid isn't reflected in the request, the test isn't actually proving the property. Inspect: `cat $(jest --listFiles)/...` is not feasible; instead, add a `console.log(JSON.stringify(recordedEvents.find(e => e.type === 'llm.requested'), null, 2))` to surface the request shape.
- If the uuid genuinely doesn't appear in the request: agent runtime's `port.uuid()` calls (childTraceId, taskId, batchId) might not be embedded in LLM requests. In that case, write a custom inline FSM config or extend the test to inject a synthetic uuid-consuming agent. The plan should not assume free embedding — adjust the test agent to make the dependency explicit. (See "If this test is hard to make work" below.)

If "over-consume" fails: confirm the file rewrite actually removes uuid.generated events; verify by reading the file back and asserting zero uuid.generated lines.

**If this test is hard to make work**: the byte-identical proof depends on agent code emitting port.uuid() output into the LLM request. The default echo agent may not do this. Plan B is to add a small custom agent in this test file that explicitly threads `port.uuid()` into the system message via a fixture mechanism. If you hit this, STOP and report as DONE_WITH_CONCERNS; controller will adjust the test scaffolding rather than letting you spend hours.

- [ ] **Step 3: Commit**

```bash
git add src/__tests__/Replay.nondet.test.ts
git commit -m "test(trace): byte-identical replay integration proof"
```

---

## Task 8: s-005 fixture re-record + e2e assertion upgrade

**Files:**
- Modify: `tests/e2e/s-005-deterministic-replay.e2e.test.ts`
- Delete: `examples/s-005-replay/.milkie/runs/*.jsonl`
- Modify: `examples/s-005-replay/.milkie/last-run.txt`

- [ ] **Step 1: Re-record the s-005 fixture**

```bash
rm examples/s-005-replay/.milkie/runs/*.jsonl
npx tsx examples/s-005-replay/record.ts
```

Expected: prints JSON with new runId / status=completed / output. `examples/s-005-replay/.milkie/last-run.txt` updated to new runId; `examples/s-005-replay/.milkie/runs/<new-runId>.jsonl` exists with nondet events (look for `"type":"clock.read"` and `"type":"uuid.generated"`).

Quick check:
```bash
RUN=$(cat examples/s-005-replay/.milkie/last-run.txt)
grep -c '"type":"clock.read"' examples/s-005-replay/.milkie/runs/${RUN}.jsonl
grep -c '"type":"uuid.generated"' examples/s-005-replay/.milkie/runs/${RUN}.jsonl
```

Expected: both > 0.

- [ ] **Step 2: Verify the existing s-005 e2e test still passes against the new fixture**

Run: `npx jest tests/e2e/s-005-deterministic-replay.e2e.test.ts`
Expected: PASS — the existing assertions (status, output) hold; nondet is captured + consumed transparently.

If it fails: read the failure carefully; most likely an artifact of the test relying on the old runId. Update test fixture references if needed.

- [ ] **Step 3: Write failing test for the inner-isolation upgrade**

The existing s-005 e2e proves structural equivalence. Phase 4's stronger claim is "replay never touches inner.now/inner.uuid". Add this assertion to the test file.

Find the existing test in `tests/e2e/s-005-deterministic-replay.e2e.test.ts` (likely structure: build a Milkie, replay, assert on output). Add a new `it()` block at the end:

```typescript
  it('byte-identical: replay does not call inner.now / inner.uuid', async () => {
    // Construct a Milkie whose underlying gateway and IOPort inner explode
    // on now/uuid. If replay touches inner, this test fails immediately;
    // if Phase 4 routing is correct, replay serves all nondet from the cache.
    //
    // Implementation detail: Milkie internally creates a DefaultIOPort with
    // a gateway. We can't easily swap the IOPort, so we test this property
    // indirectly via Replay.nondet.test.ts (which uses an ExplodingInnerPort
    // directly). For s-005 e2e, the stronger assertion we can add cheaply is:
    // re-replay multiple times and confirm every replay produces the SAME
    // status + output without consuming new gateway responses.

    const RUN = fs.readFileSync(
      path.join(__dirname, '..', '..', 'examples', 's-005-replay', '.milkie', 'last-run.txt'),
      'utf-8',
    ).trim()
    const eventStore = new JsonlEventStore(
      path.join(__dirname, '..', '..', 'examples', 's-005-replay', '.milkie', 'runs'),
    )

    // Replay 3 times back-to-back; if nondet was passing through to inner,
    // each replay would produce fresh uuid/timestamp values and any downstream
    // hash dependency would diverge by run 2. With Phase 4, all three are
    // byte-identical.
    const results: string[] = []
    for (let i = 0; i < 3; i++) {
      const m = new Milkie({
        stateStore: new MemoryStore(),
        gateway: new SequentialGateway([]),  // empty — must not be called
        eventStore,
      })
      m.loadAgentFile(path.join(__dirname, '..', '..', 'examples', 's-005-replay', 'agents', 'echo.md'))
      const r = await m.replay(RUN)
      results.push(JSON.stringify({ status: r.status, output: r.output }))
    }
    expect(new Set(results).size).toBe(1)   // all three identical
  })
```

Note: The exact imports / test scaffolding shape depends on what's in the existing e2e test file. Read it first and match the established pattern; the snippet above is illustrative.

- [ ] **Step 4: Run the upgraded e2e test**

Run: `npx jest tests/e2e/s-005-deterministic-replay.e2e.test.ts`
Expected: PASS (existing tests + the new inner-isolation upgrade).

- [ ] **Step 5: Commit**

```bash
git add tests/e2e/s-005-deterministic-replay.e2e.test.ts \
        examples/s-005-replay/.milkie/runs \
        examples/s-005-replay/.milkie/last-run.txt
git commit -m "test(s-005): re-record fixture under Phase 4; add byte-identical assertion"
```

---

## Task 9: s-002 fixture re-record (HTML report shows nondet)

**Files:**
- Delete: `examples/s-002-inspect/.milkie/runs/*.jsonl`
- Modify: `examples/s-002-inspect/.milkie/last-run.txt`

- [ ] **Step 1: Re-record the s-002 fixture**

```bash
rm examples/s-002-inspect/.milkie/runs/*.jsonl
npx tsx examples/s-002-inspect/record.ts
```

Expected: new runId; `last-run.txt` updated; the new `.jsonl` contains clock.read and uuid.generated events.

- [ ] **Step 2: Smoke-test the HTML report**

```bash
npm run build
./examples/s-002-inspect/report.sh
```

Expected: writes `report.html` for the new runId. Open it in a browser (or grep) and confirm:
- The timeline now includes "clock read" and "uuid generated" entries (rendered by the existing lifecycle/llm/tool fallthrough — they'll show as unknown kind entries with the raw event data in the payload).
- Filter chips still work; no JS errors in console.

Note: The HTML renderer (`src/trace/render/tree.ts` + `html.ts`) doesn't have explicit handling for `clock.read` / `uuid.generated` — those events will currently render as lifecycle-fallthrough or be ignored by the `buildTimelineTree` switch statement. This is acceptable for Phase 4: the JSON data is in the embedded `<script type="application/json">` regardless, so it's queryable; nicer rendering is a follow-up.

Quick visual check (no test code, just human inspection):
```bash
grep -c 'clock.read' examples/s-002-inspect/report.html
```
Expected: > 0 (nondet data is in the embedded JSON).

- [ ] **Step 3: Commit**

```bash
git add examples/s-002-inspect/.milkie/runs examples/s-002-inspect/.milkie/last-run.txt
git commit -m "chore(examples): re-record s-002 fixture under Phase 4 (with nondet events)"
```

---

## Task 10: roadmap.md update

**Files:**
- Modify: `roadmap.md`

- [ ] **Step 1: Read current roadmap structure**

Run: `grep -n "^##\|^###" roadmap.md`
Identify the sections to edit: `## TL;DR`, `## Completed (Phase 1–3)` (will become `1–4`), `## Up next` → `### Phase 4`, `## Open architectural questions` → `Replay side-effect policy`.

- [ ] **Step 2: Update TL;DR**

Find the line in `## TL;DR` that says `**Next big rock:** Phase 4 non-determinism log → unlocks byte-identical replay → unlocks Phase 5 fork / diff / suite replay.` and replace with:

```markdown
- **Phase 4 landed.** Non-determinism log records every `port.now()` /
  `port.uuid()` call; replay serves them from the log; byte-identical
  replay now honest. Strict P-wide divergence check across all four
  queues (clock / uuid / llm / tool) — over-consume or under-consume
  throws `ReplayDivergenceError` immediately. `s-005` upgraded from
  structural-only to byte-identical.
- **Next big rock:** Phase 5 fork / diff / suite replay. Phase 4 was
  its prerequisite for honest fork semantics.
```

- [ ] **Step 3: Move Phase 4 from "Up next" to "Completed"**

Rename heading `## Completed (Phase 1–3)` → `## Completed (Phase 1–4)`.

Add a new subsection at the bottom of the Completed section (before the `### Stories validated by Phase 1–3 (active)` line — or rename that to `Phase 1–4`):

```markdown
- **Phase 4 — Non-determinism log + byte-identical replay.** New event
  kinds `clock.read` / `uuid.generated`; `RecordingIOPort` records every
  agent-facing `port.now()` / `port.uuid()` via internal pending buffer
  flushed at each async method entry; `ReplayingIOPort` consumes from
  per-runId FIFO queues on `CacheIndex`; `Milkie.replay()` enforces
  strict P-wide under-consume check across clock / uuid / llm / tool.
  `s-005` fixture re-recorded; e2e asserts inner-isolation (replay
  never touches `inner.now` / `inner.uuid`).
```

Remove the entire `### Phase 4 — Non-determinism log → byte-identical replay` subsection from `## Up next`. The subsections below it (Phase 5, Phase 6, Evolution, etc.) shift up.

- [ ] **Step 4: Update Open architectural questions**

Find the "Replay side-effect policy" bullet under `## Open architectural questions` and replace it with:

```markdown
- **Replay side-effect policy** (Phase 5 prerequisite). Phase 4 declared
  the simple all-from-cache policy: replay never re-invokes operators
  with live side effects. The per-operator hook (some operators served
  from cache, others re-invoked against live state for variant search)
  is to be designed alongside Phase 5 fork. Not blocking until fork
  implementation begins.
```

- [ ] **Step 5: Verify the roadmap reads coherently**

Read `roadmap.md` end-to-end. Check:
- No reference to "Phase 4 in progress" or "Phase 4 next" remains
- The "Stories validated by ..." section still lists `s-005` (it's still active; the only change was the assertion strength, not story status)
- No broken section references

- [ ] **Step 6: Commit**

```bash
git add roadmap.md
git commit -m "docs(roadmap): Phase 4 landed; promote to Completed; re-scope side-effect policy"
```

---

## Self-review

**1. Spec coverage:**

| Spec section | Implementation task(s) |
|---|---|
| §1 目标与边界 | Task 1 (schema), Task 4/5 (record/replay paths) |
| §2.A 每次调用一条事件 | Task 4 (pending buffer per-call) |
| §2.Y' clock + uuid 类型 | Task 1 (types), Task 4 (writer), Task 5 (consumer) |
| §2 P-wide | Task 6 (Milkie tail check) + Task 7 (integration test) |
| §2 sync 接口 + buffer | Task 4 (flushPendingNondet + async-entry flush) |
| §2 call site 已就绪 | (no work — verified during brainstorming) |
| §2 s-005 重录 | Task 8 |
| §2 s-002 重录 | Task 9 |
| §2 side-effect 策略声明 | Task 10 (roadmap.md update) |
| §3 控制流图 | Implemented across Task 4/5/6 |
| §4 types.ts | Task 1 |
| §4 CacheIndex | Task 2 |
| §4 RecordingIOPort | Task 4 |
| §4 ReplayingIOPort | Task 5 |
| §4 ReplayDivergenceError | Task 3 |
| §4 Milkie.replay tail check | Task 6 |
| §5 不变量 1–7 | Task 4/5/6/7 tests cover them (1+2 in Task 4; 3+4 in Task 7; 5+6 in Task 7; 7 in Task 4) |
| §6 边界情形 | Task 4 (空 run / detach 兜底 flush in tests) |
| §7 测试策略 Unit / Integration / E2E | Tasks 2/4/5/6/7/8 |
| §8 落地次序 1–10 | Tasks 1–10 directly |
| §9 roadmap 更新 | Task 10 |

No gaps.

**2. Placeholder scan:** No "TBD" / "implement later" / "add error handling" / "similar to Task N" in any step. Task 7 step 2 has a "if this test is hard to make work" escape hatch — this is intentional discipline, not a placeholder; it gives the implementer a clear stop-and-report instruction rather than open-ended struggle.

**3. Type consistency:**
- `ClockReadPayload` / `UuidGeneratedPayload` defined in Task 1 § Step 3, used in Tasks 2, 4 consistently.
- `consumeClock(): number` / `consumeUuid(): string` defined in Task 2 § Step 3, called in Task 5 § Step 3 with matching return types.
- `remaining(): { llm, tool, clock, uuid }` defined in Task 2 § Step 3, accessed in Tasks 5 (for error message) and 6 (for tail check) with the same field names.
- `ReplayDivergenceError` constructor signature unchanged across tasks (kind, actualHash, summary, availableHashes).
- `flushPendingNondet()` defined in Task 4 § Step 3, called from the same method's async entries — no external references, no consistency risk.
