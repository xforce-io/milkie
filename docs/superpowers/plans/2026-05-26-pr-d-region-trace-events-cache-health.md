# PR-D: Region trace events + cache health observability — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the Context Region Substrate's lifecycle and cache behavior to the trace event log + adapter layer so agent designers can observe (and provider-level cache can act on) what the substrate is doing.

**Architecture:** Three orthogonal additions: (1) `region.added` / `region.removed` / `context.boundary.applied` event types emitted by `ContextRegions` + `lifecycleEngine` via injected callbacks, keeping those modules pure-ish; (2) `cache_read_tokens` / `cache_creation_tokens` extracted from LLM provider response.usage into a new `cacheStats` block on `llm.responded` + as span attributes on `llm.call`; (3) optional `cacheBreakpoint` field on `ModelRequest` that the Anthropic adapter translates into `cache_control: {type:'ephemeral'}` on the appropriate block. PR-D Phase 1 supports a SINGLE breakpoint per request (system-end); multi-breakpoint placement is Phase 2.

**Tech Stack:** TypeScript, jest (`ts-jest`), Anthropic SDK (already a dependency).

**Spec:** `docs/superpowers/specs/2026-05-25-context-region-substrate-design.md` §8 (trace + cache health) + §6 (cache breakpoint mention).

---

## File Structure

**Modify:**
- `src/trace/types.ts` — add 3 EventKind literals + 3 payload interfaces + 3 typed event aliases + AnyEvent additions
- `src/types/model.ts` — extend `ModelUsage` with optional `cacheReadTokens` / `cacheCreationTokens`; add optional `cacheBreakpoint` to `ModelRequest`
- `src/trace/RecordingIOPort.ts` — populate `cacheStats` on `llm.responded` payload from response.usage
- `src/runtime/AgentRuntime.ts` — add cache stats to `llm.call` span attributes; wire region/boundary callbacks; thread `cacheBreakpoint` into `ModelRequest`
- `src/context/ContextRegions.ts` — accept optional `onChange` callback at construction; fire on every `set` / `delete`
- `src/context/lifecycleEngine.ts` — `runInterTurnEngine` accepts optional `onBoundary` callback; fires after crystallization with summary
- `src/context/assemble.ts` — compute single `cacheBreakpoint` if any region has `cacheBreakpoint: true` and is in the stable region of system
- `src/gateway/AnthropicAdapter.ts` — when `request.cacheBreakpoint === 'system-end'`, wrap the system string as a single block array with `cache_control: { type: 'ephemeral' }`
- `src/trace/render/tree.ts` — add `RegionEntry` to `TimelineEntry` union; map `region.*` events
- `src/trace/render/html.ts` — render `RegionEntry` with `region` chip + icon

**Test:**
- `src/__tests__/ContextRegions.test.ts` — add tests for `onChange` callback (fires on set/delete, payload shape, no-fire when callback omitted)
- `src/__tests__/lifecycleEngine.test.ts` — add tests for `onBoundary` callback (fires once per turn-end with summary; not fired when callback omitted)
- `src/__tests__/assemble.test.ts` — add tests for `cacheBreakpoint` computation
- `src/__tests__/AnthropicAdapter.test.ts` (CREATE) — unit tests for cache_control injection

**Fixtures:**
- `examples/s-005-replay/.milkie/runs/*.jsonl` — re-record (new region/boundary events shift event log byte structure)
- `examples/s-002-inspect/.milkie/runs/*.jsonl` — re-record (same)

---

## Task 1: Trace event type extensions

**Files:**
- Modify: `src/trace/types.ts`

**Why:** Additive type-only change. Establishes the contract before anything emits the events. All later tasks reference these types.

- [ ] **Step 1: Extend EventKind union**

Locate the `EventKind` union (lines 9-17). Add three literals at the end:

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
  | 'region.added'                // NEW
  | 'region.removed'              // NEW
  | 'context.boundary.applied'    // NEW
```

- [ ] **Step 2: Add payload interfaces (append at end of `// ---- Non-determinism payloads (Phase 4) ----` block, before the typed aliases)**

```typescript
// ---- Region / context boundary payloads (Phase 4.6) ----

export interface RegionAddedPayload {
  /** Region id (e.g. 'header', 'skill:verifier', 'scratch:abc123'). */
  id:        string
  /** Stable identifier for which substrate section/target this region targets. */
  target:    'system' | 'message' | 'tool'
  section:   string
  stability: 'immutable' | 'session-stable' | 'turn-stable' | 'volatile'
  /** Why this region appeared (e.g. 'agent-set', 'turn-archived', 'promoted-to-wm'). */
  reason:    string
}

export interface RegionRemovedPayload {
  id:     string
  /** Why this region was removed (e.g. 'turn-local-released', 'ttl-expired', 'promoted-source-removed'). */
  reason: string
}

export interface ContextBoundaryAppliedPayload {
  /** Which boundary engine fired: 'turn-end' (crystallization) for now. */
  boundary: 'turn-end' | 'turn-start' | 'fsm-step'
  /** epoch of the regions Map AFTER the boundary engine ran. */
  epoch:    number
  /** Summary of crystallization activity (omitted when boundary is non-turn-end). */
  crystallization?: {
    kept:         number   // count, not full ids (full ids in region.added/removed events)
    dropped:      number
    promoted:     number
    archivedPair: string | undefined   // id of the new history pair region, if any
  }
}
```

- [ ] **Step 3: Add typed event aliases (in the `// ---- Typed event aliases ----` block, after `UuidGeneratedEvent`)**

```typescript
export type RegionAddedEvent             = Event<RegionAddedPayload>             & { type: 'region.added' }
export type RegionRemovedEvent           = Event<RegionRemovedPayload>           & { type: 'region.removed' }
export type ContextBoundaryAppliedEvent  = Event<ContextBoundaryAppliedPayload>  & { type: 'context.boundary.applied' }
```

- [ ] **Step 4: Extend AnyEvent union (add three lines before the closing of `AnyEvent`)**

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
  | RegionAddedEvent              // NEW
  | RegionRemovedEvent            // NEW
  | ContextBoundaryAppliedEvent   // NEW
```

- [ ] **Step 5: TypeScript check + commit**

```bash
cd /Users/xupeng/dev/github/milkie
npx tsc --noEmit -p .
```
Expected: zero errors (purely additive type change).

```bash
git add src/trace/types.ts
git commit -m "feat(trace): add region.added / region.removed / context.boundary.applied event types (PR-D step 1/9)"
```

---

## Task 2: Cache stats on ModelUsage + LlmRespondedPayload

**Files:**
- Modify: `src/types/model.ts`
- Modify: `src/trace/types.ts`

**Why:** Type-additive precursor to actually populating these values. Optional fields so existing callers don't break.

- [ ] **Step 1: Extend `ModelUsage` in `src/types/model.ts`**

Locate lines 38-42:
```typescript
export interface ModelUsage {
  inputTokens:  number
  outputTokens: number
  cost?:        number
}
```

Replace with:
```typescript
export interface ModelUsage {
  inputTokens:        number
  outputTokens:       number
  cost?:              number
  /** PR-D: tokens served from provider prefix cache (Anthropic). */
  cacheReadTokens?:     number
  /** PR-D: tokens written to provider prefix cache (Anthropic). */
  cacheCreationTokens?: number
}
```

- [ ] **Step 2: Extend `LlmRespondedPayload` in `src/trace/types.ts`**

Locate lines 37-41:
```typescript
export interface LlmRespondedPayload {
  response: ModelResponse
  /** Mirrors the requested-event hash so consumers don't need to re-join. */
  requestHash: string
}
```

Replace with:
```typescript
export interface LlmRespondedPayload {
  response: ModelResponse
  /** Mirrors the requested-event hash so consumers don't need to re-join. */
  requestHash: string
  /** PR-D: cache-health snapshot lifted from response.usage; null when provider does not report. */
  cacheStats?: {
    readTokens:       number
    creationTokens:   number
    totalInputTokens: number
    /** readTokens / totalInputTokens, [0, 1]. 0 when totalInputTokens === 0. */
    hitRate:          number
  }
}
```

- [ ] **Step 3: TypeScript check + commit**

```bash
npx tsc --noEmit -p .
```
Expected: zero errors.

```bash
git add src/types/model.ts src/trace/types.ts
git commit -m "feat(trace): cache stats on ModelUsage + llm.responded payload (PR-D step 2/9)"
```

---

## Task 3: RecordingIOPort populates cacheStats

**Files:**
- Modify: `src/trace/RecordingIOPort.ts`

**Why:** Wire the model response's cache numbers into the event payload at the recording boundary.

- [ ] **Step 1: Locate the llm.responded emission**

Run:
```bash
grep -n "'llm.responded'" src/trace/RecordingIOPort.ts
```
Expected: one match around line 122-127. Read the surrounding block.

- [ ] **Step 2: Add a helper to compute cacheStats from ModelResponse.usage**

In `src/trace/RecordingIOPort.ts`, near the top of the file (after imports), add:

```typescript
import type { ModelResponse } from '../types/model.js'

function cacheStatsFrom(response: ModelResponse): {
  readTokens:       number
  creationTokens:   number
  totalInputTokens: number
  hitRate:          number
} | undefined {
  const usage = response.usage
  if (!usage || usage.cacheReadTokens === undefined) return undefined
  const readTokens     = usage.cacheReadTokens
  const creationTokens = usage.cacheCreationTokens ?? 0
  const totalInputTokens = usage.inputTokens
  return {
    readTokens,
    creationTokens,
    totalInputTokens,
    hitRate: totalInputTokens > 0 ? readTokens / totalInputTokens : 0,
  }
}
```

(The `import type { ModelResponse }` line may already exist; if it does, do not duplicate.)

- [ ] **Step 3: Use the helper at the llm.responded append site**

Replace the existing `llm.responded` payload construction:
```typescript
payload: { response, requestHash } satisfies LlmRespondedPayload,
```
with:
```typescript
payload: {
  response,
  requestHash,
  ...(cacheStatsFrom(response) ? { cacheStats: cacheStatsFrom(response) } : {}),
} satisfies LlmRespondedPayload,
```

(Calling `cacheStatsFrom(response)` twice is fine — pure function on already-computed values.)

- [ ] **Step 4: Run unit tests**

```bash
npm run test:unit
```
Expected: 30/30 pass (existing tests don't set cacheReadTokens; cacheStats stays undefined; no behavior change for them).

- [ ] **Step 5: Commit**

```bash
git add src/trace/RecordingIOPort.ts
git commit -m "feat(trace): RecordingIOPort lifts cacheStats into llm.responded payload (PR-D step 3/9)"
```

---

## Task 4: AgentRuntime exposes cache stats on llm.call span

**Files:**
- Modify: `src/runtime/AgentRuntime.ts`

**Why:** Span attributes are what trajectory consumers (HTML report, downstream analysis) read. Without this, cache health is only on raw events.

- [ ] **Step 1: Locate the llm.call span site**

Run:
```bash
grep -n "llm.call" src/runtime/AgentRuntime.ts
```
Expected: a `startSpan('llm.call', ...)` around line 511 + a `recordEvent(llmSpan, 'usage', ...)` shortly after.

- [ ] **Step 2: Extend the usage recordEvent payload**

Locate the existing block (around line 525):
```typescript
this.recorder.recordEvent(llmSpan, 'usage', {
  inputTokens:  response.usage?.inputTokens,
  outputTokens: response.usage?.outputTokens,
})
```

Replace with:
```typescript
this.recorder.recordEvent(llmSpan, 'usage', {
  inputTokens:         response.usage?.inputTokens,
  outputTokens:        response.usage?.outputTokens,
  cacheReadTokens:     response.usage?.cacheReadTokens,
  cacheCreationTokens: response.usage?.cacheCreationTokens,
  cacheHitRate:        response.usage?.cacheReadTokens !== undefined && (response.usage?.inputTokens ?? 0) > 0
                         ? response.usage.cacheReadTokens / response.usage.inputTokens
                         : undefined,
})
```

- [ ] **Step 3: Verify**

```bash
npm run test:unit && npm run test:e2e:deterministic
```
Expected: 30 + 7 pass. Existing tests don't assert on `cacheReadTokens` / `cacheHitRate`; undefined values do not affect assertions.

- [ ] **Step 4: Commit**

```bash
git add src/runtime/AgentRuntime.ts
git commit -m "feat(runtime): llm.call span exposes cache hit rate (PR-D step 4/9)"
```

---

## Task 5: ContextRegions onChange callback (region.added / region.removed emission)

**Files:**
- Modify: `src/context/ContextRegions.ts`
- Modify: `src/__tests__/ContextRegions.test.ts`

**Why:** ContextRegions today is pure substrate. PR-D needs to emit events on mutations without sacrificing purity. Solution: accept an optional callback at construction; when present, fire on `set`/`delete` with delta info. Tests omit the callback (substrate stays unit-testable without trace dependency).

- [ ] **Step 1: Write failing tests (append to `src/__tests__/ContextRegions.test.ts`)**

```typescript
describe('ContextRegions — onChange callback', () => {
  test('fires on set with added=true and the new region', () => {
    const calls: Array<{ kind: string; id: string; section?: string }> = []
    const store = new ContextRegions(() => 0, {
      onChange: (delta) => calls.push(delta),
    })
    store.set('a', regionInput({ content: 'hi', section: 'header' }))
    expect(calls).toHaveLength(1)
    expect(calls[0]).toMatchObject({ kind: 'added', id: 'a', section: 'header' })
  })

  test('fires on delete with removed=true', () => {
    const calls: Array<{ kind: string; id: string }> = []
    const store = new ContextRegions(() => 0, {
      onChange: (delta) => calls.push(delta),
    })
    store.set('a', regionInput())
    calls.length = 0
    store.delete('a')
    expect(calls).toHaveLength(1)
    expect(calls[0]).toMatchObject({ kind: 'removed', id: 'a' })
  })

  test('does NOT fire on delete of non-existent id', () => {
    const calls: unknown[] = []
    const store = new ContextRegions(() => 0, {
      onChange: (delta) => calls.push(delta),
    })
    store.delete('nope')
    expect(calls).toEqual([])
  })

  test('fires on upsert (re-set on existing id) with kind=added', () => {
    const calls: Array<{ kind: string }> = []
    const store = new ContextRegions(() => 0, {
      onChange: (delta) => calls.push(delta),
    })
    store.set('a', regionInput())
    calls.length = 0
    store.set('a', regionInput({ content: 'v2' }))
    expect(calls).toEqual([{ kind: 'added', id: 'a', section: expect.any(String) }])
  })

  test('no callback in constructor → no error on set/delete', () => {
    const store = new ContextRegions(() => 0)   // no second arg
    expect(() => store.set('a', regionInput())).not.toThrow()
    expect(() => store.delete('a')).not.toThrow()
  })

  test('restore() does NOT fire onChange per region', () => {
    const calls: unknown[] = []
    const src = new ContextRegions(() => 0)
    src.set('a', regionInput())
    src.set('b', regionInput())
    const snap = src.snapshot()

    const dst = new ContextRegions(() => 0, { onChange: (delta) => calls.push(delta) })
    dst.restore(snap)
    // restore is a bulk reset; per-region events would be misleading replay noise.
    expect(calls).toEqual([])
  })
})
```

- [ ] **Step 2: Run tests — expect FAIL (option arg not supported)**

```bash
npx jest src/__tests__/ContextRegions.test.ts
```
Expected: FAIL with `Expected 1 arguments, but got 2` or similar.

- [ ] **Step 3: Add the optional second constructor argument + emission**

In `src/context/ContextRegions.ts`, locate the existing constructor:
```typescript
constructor(private readonly clock: Clock) {}
```

Replace with:
```typescript
export interface RegionChangeDelta {
  kind:    'added' | 'removed'
  id:      string
  /** Present on 'added' deltas only (so consumers can index into section without re-reading the region). */
  section?: string
  /** Present on 'added' deltas only. */
  target?:  'system' | 'message' | 'tool'
  /** Present on 'added' deltas only. */
  stability?: 'immutable' | 'session-stable' | 'turn-stable' | 'volatile'
}

export interface ContextRegionsOptions {
  /** Fired on every successful set/delete. Optional — substrate stays pure-ish without it. */
  onChange?: (delta: RegionChangeDelta) => void
}

export class ContextRegions {
  private readonly regions = new Map<string, Region>()
  private epoch = 0

  constructor(
    private readonly clock: Clock,
    private readonly options: ContextRegionsOptions = {},
  ) {}
  // ...
}
```

Then update `set`:
```typescript
set(id: string, input: RegionInput): void {
  const existing = this.regions.get(id)
  const createdAt = existing?.createdAt ?? this.clock()
  const region: Region = { id, createdAt, ...input }
  this.regions.set(id, region)
  this.epoch++
  this.options.onChange?.({
    kind:      'added',
    id,
    section:   region.section,
    target:    region.target,
    stability: region.stability,
  })
}
```

And `delete`:
```typescript
delete(id: string): boolean {
  const existed = this.regions.delete(id)
  if (existed) {
    this.epoch++
    this.options.onChange?.({ kind: 'removed', id })
  }
  return existed
}
```

Note: `restore()` MUST NOT fire onChange. Verify the existing implementation:
```typescript
restore(snap: RegionSnapshot): void {
  this.regions.clear()
  for (const r of snap.regions) {
    this.regions.set(r.id, r)
  }
  this.epoch = snap.epoch
}
```
This uses `this.regions.set` (the Map method) directly, NOT `this.set`. Good — no event fires. Verify with the test from Step 1.

- [ ] **Step 4: Run tests — expect 6 new pass**

```bash
npx jest src/__tests__/ContextRegions.test.ts
```
Expected: 25 + 6 = 31 pass.

- [ ] **Step 5: Commit**

```bash
git add src/context/ContextRegions.ts src/__tests__/ContextRegions.test.ts
git commit -m "feat(context): ContextRegions onChange callback (PR-D step 5/9)"
```

---

## Task 6: lifecycleEngine onBoundary callback

**Files:**
- Modify: `src/context/lifecycleEngine.ts`
- Modify: `src/__tests__/lifecycleEngine.test.ts`

**Why:** Crystallization is the inter-turn boundary. Emit a single `context.boundary.applied` event per crystallize with summary counts.

- [ ] **Step 1: Write failing tests (append to lifecycleEngine.test.ts inside the existing `describe('runInterTurnEngine — turn-end crystallization')` block)**

```typescript
  test('onBoundary callback fires once with summary counts', () => {
    const r = new ContextRegions(() => 100)
    r.set('current', makeCurrentTurnRegion('q'))
    r.set('s-final', makeScratchpadAssistantRegion(
      [{ type: 'text', text: 'a' }], false,
    ))
    r.set('hdr', makeHeaderRegion('h'))   // session-persistent, will be kept

    const events: Array<{ kept: number; dropped: number; promoted: number; archivedPair?: string }> = []
    runInterTurnEngine(r, {
      boundary: 'turn-end',
      userInput: 'q',
      now: 999,
      onBoundary: (summary) => events.push(summary),
    })

    expect(events).toHaveLength(1)
    expect(events[0]!.dropped).toBeGreaterThanOrEqual(2)   // at least current + scratch
    expect(events[0]!.kept).toBeGreaterThanOrEqual(1)      // at least header
    expect(events[0]!.archivedPair).toMatch(/^history:turn-/)
  })

  test('onBoundary not fired when boundary !== turn-end', () => {
    const r = new ContextRegions(() => 0)
    const events: unknown[] = []
    runInterTurnEngine(r, {
      boundary: 'turn-start',
      now: 1,
      onBoundary: (s) => events.push(s),
    })
    expect(events).toEqual([])
  })

  test('runInterTurnEngine works without onBoundary callback (existing tests still pass)', () => {
    const r = new ContextRegions(() => 0)
    r.set('s', makeScratchpadAssistantRegion([{type:'text',text:'a'}], false))
    expect(() => runInterTurnEngine(r, { boundary: 'turn-end', userInput: 'q', now: 1 })).not.toThrow()
  })
```

- [ ] **Step 2: Run tests — expect FAIL (onBoundary not in InterTurnContext type)**

```bash
npx jest src/__tests__/lifecycleEngine.test.ts -t "onBoundary"
```
Expected: TS error or runtime failure.

- [ ] **Step 3: Extend InterTurnContext + fire callback**

In `src/context/lifecycleEngine.ts`, locate `InterTurnContext`:
```typescript
export interface InterTurnContext {
  boundary:   'turn-end' | 'turn-start'
  userInput?: string
  now:        number
}
```

Replace with:
```typescript
export interface InterTurnContext {
  boundary:   'turn-end' | 'turn-start'
  userInput?: string
  now:        number
  /** PR-D: fires once after crystallization completes (turn-end only) with the summary. */
  onBoundary?: (summary: CrystallizationSummary) => void
}
```

At the end of `runInterTurnEngine`, before `return { crystallization: summary }`, add:

```typescript
ctx.onBoundary?.(summary)
return { crystallization: summary }
```

- [ ] **Step 4: Run tests**

```bash
npx jest src/__tests__/lifecycleEngine.test.ts
```
Expected: 21 + 3 = 24 pass.

- [ ] **Step 5: Commit**

```bash
git add src/context/lifecycleEngine.ts src/__tests__/lifecycleEngine.test.ts
git commit -m "feat(context): lifecycleEngine onBoundary callback (PR-D step 6/9)"
```

---

## Task 7: AgentRuntime wires region + boundary callbacks to emit trace events

**Files:**
- Modify: `src/runtime/AgentRuntime.ts`

**Why:** This is the gluing task — connects the substrate's now-available callbacks to actual event emission via the recorder + ioPort/eventStore.

- [ ] **Step 1: Add a private event-emission helper**

In `src/runtime/AgentRuntime.ts`, add this private method anywhere in the class (suggested: near the other private region helpers):

```typescript
private emitRegionDelta(delta: import('../context/ContextRegions.js').RegionChangeDelta, reason: string): void {
  // PR-D: emit region.added or region.removed event into the trace.
  // The recorder span layer doesn't have a "regions" namespace, so we record
  // as a custom span event. Downstream HTML report reads from these.
  this.recorder.recordEvent(this.rootSpan, delta.kind === 'added' ? 'region.added' : 'region.removed', {
    id:        delta.id,
    section:   delta.section,
    target:    delta.target,
    stability: delta.stability,
    reason,
  })
}

private emitBoundaryApplied(summary: import('../context/lifecycleEngine.js').CrystallizationSummary): void {
  // PR-D: emit context.boundary.applied event into the trace.
  this.recorder.recordEvent(this.rootSpan, 'context.boundary.applied', {
    boundary: 'turn-end',
    epoch:    this.regions.getEpoch(),
    crystallization: {
      kept:         summary.kept.length,
      dropped:      summary.dropped.length,
      promoted:     summary.promoted.length,
      archivedPair: summary.archivedPair,
    },
  })
}
```

Note: `this.rootSpan` exists when `run()` has been called; emitRegionDelta might be called during `loadCheckpoint` before run(). Add a guard:

```typescript
private emitRegionDelta(delta, reason): void {
  if (!this.rootSpan) return   // pre-run mutations (e.g. checkpoint restore) not yet attributable
  // ...
}
```

Same guard on `emitBoundaryApplied`.

- [ ] **Step 2: Wire `onChange` at ContextRegions construction**

Locate the constructor block (around line 91):
```typescript
this.regions = new ContextRegions(() => this.ioPort.now())
```

Replace with:
```typescript
this.regions = new ContextRegions(
  () => this.ioPort.now(),
  { onChange: (delta) => this.emitRegionDelta(delta, 'agent-set') },
)
```

(The "agent-set" string is a coarse reason; refining it per-callsite is a future improvement. PR-D ships the wire-up, not the rich reason taxonomy.)

- [ ] **Step 3: Wire `onBoundary` at crystallizeTurn call**

Locate `crystallizeTurn` (around line 289-298):
```typescript
private crystallizeTurn(userInput?: string): void {
  const input = userInput ?? (this.regions.get('current-turn')?.content as string | undefined)
  if (input === undefined) return
  runInterTurnEngine(this.regions, {
    boundary:  'turn-end',
    userInput: input,
    now:       this.ioPort.now(),
  })
}
```

Replace with:
```typescript
private crystallizeTurn(userInput?: string): void {
  const input = userInput ?? (this.regions.get('current-turn')?.content as string | undefined)
  if (input === undefined) return
  runInterTurnEngine(this.regions, {
    boundary:   'turn-end',
    userInput:  input,
    now:        this.ioPort.now(),
    onBoundary: (summary) => this.emitBoundaryApplied(summary),
  })
}
```

- [ ] **Step 4: Verify**

```bash
npm run test:unit
```
Expected: 30 pass (existing tests don't observe region/boundary events; emission is fire-and-forget).

```bash
npm run test:e2e:deterministic
```
Expected: 7 pass + 15 skipped.

```bash
npx jest 2>&1 | tail -7
```
Expected: all suites pass (replay fixtures may or may not break — see Task 9 if so).

- [ ] **Step 5: Commit**

```bash
git add src/runtime/AgentRuntime.ts
git commit -m "feat(runtime): wire region + boundary events into trace (PR-D step 7/9)"
```

---

## Task 8: cacheBreakpoint plumbing — assemble → ModelRequest → Anthropic adapter

**Files:**
- Modify: `src/types/model.ts`
- Modify: `src/context/assemble.ts`
- Modify: `src/runtime/AgentRuntime.ts`
- Modify: `src/gateway/AnthropicAdapter.ts`
- Modify: `src/__tests__/assemble.test.ts`
- Create: `src/__tests__/AnthropicAdapter.test.ts`

**Why:** Phase 1 supports a single breakpoint at the END of the system string. When any system region has `cacheBreakpoint: true`, the Anthropic adapter wraps the system as `[{type:'text', text, cache_control:{type:'ephemeral'}}]`. OpenAI adapter ignores (provider does automatic prefix caching).

**Scope reduction explicit:** spec §4.1 / §6 envisions THREE cache breakpoint positions (stable-cut, session-cut, turn-cut). PR-D Phase 1 implements **only end-of-system** (one breakpoint). Multi-breakpoint placement is deferred until trace data tells us whether one breakpoint is enough.

- [ ] **Step 1: Extend ModelRequest with optional cacheBreakpoint**

In `src/types/model.ts`, locate the `ModelRequest` interface (lines 11-20):
```typescript
export interface ModelRequest {
  model:           string
  system?:         string
  messages:        Message[]
  tools?:          ToolSchema[]
  toolChoice?:     unknown
  responseFormat?: unknown
  reasoning?:      ReasoningOptions
  metadata?:       Record<string, unknown>
}
```

Add a new optional field:
```typescript
export interface ModelRequest {
  model:           string
  system?:         string
  messages:        Message[]
  tools?:          ToolSchema[]
  toolChoice?:     unknown
  responseFormat?: unknown
  reasoning?:      ReasoningOptions
  metadata?:       Record<string, unknown>
  /** PR-D Phase 1: when 'system-end', adapter wraps system block with cache_control. */
  cacheBreakpoint?: 'system-end'
}
```

- [ ] **Step 2: Write failing assemble test (append to `src/__tests__/assemble.test.ts` inside a new describe block)**

```typescript
describe('assemble — cacheBreakpoint computation', () => {
  test('returns no cacheBreakpoint when no region marks one', () => {
    const store = new ContextRegions(() => 0)
    store.set('h', systemRegion())
    const out = assemble(store, defaultScope())
    expect(out.cacheBreakpoint).toBeUndefined()
  })

  test('returns "system-end" when any system region has cacheBreakpoint=true', () => {
    const store = new ContextRegions(() => 0)
    store.set('h', systemRegion())
    store.set('skill', systemRegion({
      section:        'persistent-skills',
      content:        'I',
      cacheBreakpoint: true,
    }))
    const out = assemble(store, defaultScope())
    expect(out.cacheBreakpoint).toBe('system-end')
  })

  test('cacheBreakpoint stays undefined if only message/tool regions mark it (Phase 1 only handles system-end)', () => {
    const store = new ContextRegions(() => 0)
    store.set('h', systemRegion())
    store.set('hist', messageRegion({ section: 'history', cacheBreakpoint: true }))
    const out = assemble(store, defaultScope())
    expect(out.cacheBreakpoint).toBeUndefined()
  })
})
```

- [ ] **Step 3: Run tests — expect FAIL (cacheBreakpoint not on AssembledContext)**

```bash
npx jest src/__tests__/assemble.test.ts -t cacheBreakpoint
```
Expected: FAIL with TS error.

- [ ] **Step 4: Extend AssembledContext + compute cacheBreakpoint in assemble**

In `src/context/assemble.ts`, locate `AssembledContext` (around line 24-28):
```typescript
export interface AssembledContext {
  system:   string
  messages: Message[]
  tools?:   ToolSchema[]
}
```

Replace with:
```typescript
export interface AssembledContext {
  system:           string
  messages:         Message[]
  tools?:           ToolSchema[]
  /** PR-D Phase 1: 'system-end' when any active system region declared cacheBreakpoint=true. */
  cacheBreakpoint?: 'system-end'
}
```

In the `assemble` function, after computing `systemBlocks` and before the `return`, compute the breakpoint flag:

```typescript
const hasSystemBreakpoint = active.some(r => r.target === 'system' && r.cacheBreakpoint === true)
```

Update the return:
```typescript
return {
  system:   systemBlocks.join('\n'),
  messages,
  ...(tools.length > 0 ? { tools } : {}),
  ...(hasSystemBreakpoint ? { cacheBreakpoint: 'system-end' as const } : {}),
}
```

- [ ] **Step 5: Run tests — expect GREEN**

```bash
npx jest src/__tests__/assemble.test.ts
```
Expected: 22 + 3 = 25 pass.

- [ ] **Step 6: Thread cacheBreakpoint into ModelRequest in AgentRuntime**

In `src/runtime/AgentRuntime.ts`, locate the request construction (around line 506):
```typescript
const request   = {
  model:    this.config.model.model,
  system:   assembled.system,
  messages: assembled.messages,
  ...(assembled.tools ? { tools: assembled.tools } : {}),
}
```

Add the cacheBreakpoint pass-through:
```typescript
const request: ModelRequest = {
  model:    this.config.model.model,
  system:   assembled.system,
  messages: assembled.messages,
  ...(assembled.tools ? { tools: assembled.tools } : {}),
  ...(assembled.cacheBreakpoint ? { cacheBreakpoint: assembled.cacheBreakpoint } : {}),
}
```

(`ModelRequest` import is already at the top of the file.)

- [ ] **Step 7: Write Anthropic adapter test (create `src/__tests__/AnthropicAdapter.test.ts`)**

```typescript
import { AnthropicAdapter } from '../gateway/AnthropicAdapter'
import type { ModelRequest } from '../types/model'

// Internal type access: cast the adapter so we can call buildParams.
// buildParams is private, so we exercise via a focused test fixture that doesn't hit the network.
//
// Strategy: instantiate the adapter; call (adapter as any).buildParams(request)
// and assert on the returned params shape. No network call.

function buildParamsOf(adapter: AnthropicAdapter, request: ModelRequest): Record<string, unknown> {
  return (adapter as unknown as { buildParams(r: ModelRequest): Record<string, unknown> }).buildParams(request)
}

describe('AnthropicAdapter — cacheBreakpoint translation', () => {
  const adapter = new AnthropicAdapter({ apiKey: 'sk-test' })

  test('no cacheBreakpoint → system stays as string', () => {
    const params = buildParamsOf(adapter, {
      model:    'claude-sonnet-4-6',
      system:   'You are an agent.',
      messages: [],
    })
    expect(params['system']).toBe('You are an agent.')
  })

  test('cacheBreakpoint=system-end → system becomes block array with cache_control', () => {
    const params = buildParamsOf(adapter, {
      model:           'claude-sonnet-4-6',
      system:          'You are an agent. + persistent skills...',
      messages:        [],
      cacheBreakpoint: 'system-end',
    })
    expect(params['system']).toEqual([
      {
        type:          'text',
        text:          'You are an agent. + persistent skills...',
        cache_control: { type: 'ephemeral' },
      },
    ])
  })

  test('cacheBreakpoint=system-end with empty system → no system field emitted', () => {
    const params = buildParamsOf(adapter, {
      model:           'claude-sonnet-4-6',
      messages:        [],
      cacheBreakpoint: 'system-end',
    })
    expect(params['system']).toBeUndefined()
  })

  test('cacheBreakpoint plays nicely with tools (both fields emitted)', () => {
    const params = buildParamsOf(adapter, {
      model:           'claude-sonnet-4-6',
      system:          'sys',
      messages:        [],
      tools:           [{ name: 'echo', description: 'e', inputSchema: {} }],
      cacheBreakpoint: 'system-end',
    })
    expect(params['system']).toEqual([
      { type: 'text', text: 'sys', cache_control: { type: 'ephemeral' } },
    ])
    expect(params['tools']).toHaveLength(1)
  })
})
```

- [ ] **Step 8: Run new adapter tests — expect FAIL**

```bash
npx jest src/__tests__/AnthropicAdapter.test.ts
```
Expected: FAIL because adapter ignores cacheBreakpoint.

- [ ] **Step 9: Implement cache_control injection in Anthropic adapter**

In `src/gateway/AnthropicAdapter.ts`, locate `buildParams` (line 40):

Replace the current system handling:
```typescript
if (request.system) {
  params['system'] = request.system
}
```

with:
```typescript
if (request.system) {
  if (request.cacheBreakpoint === 'system-end') {
    params['system'] = [{
      type:          'text',
      text:          request.system,
      cache_control: { type: 'ephemeral' },
    }]
  } else {
    params['system'] = request.system
  }
}
```

- [ ] **Step 10: Run all tests**

```bash
npx jest src/__tests__/AnthropicAdapter.test.ts
```
Expected: 4 pass.

```bash
npm run test:unit && npm run test:e2e:deterministic
```
Expected: 30 + 7 pass.

- [ ] **Step 11: Commit**

```bash
git add src/types/model.ts src/context/assemble.ts src/runtime/AgentRuntime.ts \
        src/gateway/AnthropicAdapter.ts src/__tests__/assemble.test.ts \
        src/__tests__/AnthropicAdapter.test.ts
git commit -m "feat(adapter): cacheBreakpoint=system-end → Anthropic cache_control (PR-D step 8/9)"
```

---

## Task 9: HTML report renders region events + full suite + re-record fixtures + PR

**Files:**
- Modify: `src/trace/render/tree.ts`
- Modify: `src/trace/render/html.ts`
- Re-record: `examples/s-005-replay/.milkie/runs/*.jsonl`
- Re-record: `examples/s-002-inspect/.milkie/runs/*.jsonl`

- [ ] **Step 1: Extend TimelineEntry union with RegionEntry**

In `src/trace/render/tree.ts`, locate the `TimelineEntry` union (around line 19-26).

Add:
```typescript
export interface RegionEntry {
  kind:      'region'
  eventId:   string
  eventType: 'region.added' | 'region.removed' | 'context.boundary.applied'
  timestamp: number
  /** Free-text summary for the timeline row. */
  summary:   string
  /** Raw payload for click-through. */
  payload:   unknown
}

export type TimelineEntry = LlmEntry | ToolEntry | LifecycleEntry | RegionEntry
```

In the `buildTimelineTree` function (or wherever event type → entry mapping happens), add cases for the three new event types. Look for the existing switch / if-else chain handling `llm.requested` / `tool.requested` etc. Add:

```typescript
if (evt.type === 'region.added') {
  const p = evt.payload as { id: string; section?: string; reason?: string }
  out.push({
    kind:      'region',
    eventId:   evt.id,
    eventType: 'region.added',
    timestamp: evt.timestamp,
    summary:   `+ region ${p.id} (${p.section ?? 'unknown section'}, ${p.reason ?? 'no reason'})`,
    payload:   evt.payload,
  })
  continue
}
if (evt.type === 'region.removed') {
  const p = evt.payload as { id: string; reason?: string }
  out.push({
    kind:      'region',
    eventId:   evt.id,
    eventType: 'region.removed',
    timestamp: evt.timestamp,
    summary:   `- region ${p.id} (${p.reason ?? 'no reason'})`,
    payload:   evt.payload,
  })
  continue
}
if (evt.type === 'context.boundary.applied') {
  const p = evt.payload as { boundary: string; epoch: number; crystallization?: { kept: number; dropped: number; promoted: number; archivedPair?: string } }
  const cryst = p.crystallization
  const detail = cryst
    ? ` (kept=${cryst.kept} dropped=${cryst.dropped} promoted=${cryst.promoted}${cryst.archivedPair ? ' archived=' + cryst.archivedPair : ''})`
    : ''
  out.push({
    kind:      'region',
    eventId:   evt.id,
    eventType: 'context.boundary.applied',
    timestamp: evt.timestamp,
    summary:   `boundary ${p.boundary} @ epoch ${p.epoch}${detail}`,
    payload:   evt.payload,
  })
  continue
}
```

(Adapt `continue` to match the surrounding loop control.)

- [ ] **Step 2: Render RegionEntry in html.ts**

In `src/trace/render/html.ts`, locate `renderEntry`. Add a branch:

```typescript
if (entry.kind === 'region') {
  const icon = entry.eventType === 'context.boundary.applied' ? '⌖' :
               entry.eventType === 'region.added' ? '＋' : '－'
  return `<li class="entry region" data-event="${entry.eventId}">
    <span class="icon">${icon}</span>
    <span class="summary">${escapeHtml(entry.summary)}</span>
    <span class="ts">${formatTs(entry.timestamp)}</span>
  </li>`
}
```

(If the existing renderEntry returns strings without `<li>` wrappers, match its convention. Read the file first.)

Locate the filter chip block (around line 92-95). Add a region chip:
```html
<span class="chip" data-kind="llm">LLM</span>
<span class="chip" data-kind="tool">tool</span>
<span class="chip" data-kind="lifecycle">lifecycle</span>
<span class="chip" data-kind="region">region</span>
```

If there's a CSS block in the same file, add:
```css
.entry.region .icon { color: #888; }
.entry.region .summary { font-family: ui-monospace, SFMono-Regular, monospace; font-size: 12px; }
```

- [ ] **Step 3: Verify HTML report tests still pass**

```bash
npx jest src/__tests__/CliTraceReport.test.ts src/__tests__/CliTrace.test.ts
```
Expected: existing tests pass (additive). If a test asserts specific entry count and the recorded fixtures contain region events, that may need updating — proceed to Step 4.

- [ ] **Step 4: Re-record example fixtures**

Examples were re-recorded in PR-C1 Task 12. PR-D adds region/boundary events to the recorded stream → fixtures' event count and ids shift.

```bash
cd /Users/xupeng/dev/github/milkie/examples/s-005-replay
rm .milkie/runs/*.jsonl
npx tsx record.ts
ls .milkie/runs/

cd /Users/xupeng/dev/github/milkie/examples/s-002-inspect
rm .milkie/runs/*.jsonl
npx tsx record.ts
ls .milkie/runs/
```

Update `.milkie/last-run.txt` files if record.ts didn't (it usually does).

- [ ] **Step 5: Run full suite**

```bash
cd /Users/xupeng/dev/github/milkie
npm run build && npx jest
```
Expected: all suites pass (additive change; existing assertions don't depend on absence of region events).

If any test fails because it asserted a specific event count or specific event[0] type, update the assertion to use `find` by type rather than positional indexing.

- [ ] **Step 6: Verify the new example replay flow**

```bash
cd examples/s-005-replay
npx tsx replay-sdk.ts
bash replay-cli.sh
```
Both should return `{status: "completed"}`.

```bash
cd examples/s-002-inspect
bash report.sh
ls -la report.html   # should regenerate successfully
```

- [ ] **Step 7: Commit + push + open PR**

```bash
cd /Users/xupeng/dev/github/milkie
git add -A
git commit -m "feat(render): HTML report renders region + boundary events + fixture re-record (PR-D step 9/9)"
git push -u origin feat/pr-d-region-trace-events-cache-health
```

```bash
gh pr create --title "feat(trace): region events + cache health observability (PR-D)" --body "$(cat <<'EOF'
## Summary

Phase 4.6 of the substrate work. Adds observability for the Context Region Substrate's lifecycle + provider cache behavior.

Three additions, in increasing order of risk:

1. **Trace events for region lifecycle**: \`region.added\` / \`region.removed\` fire on every \`ContextRegions.set/delete\`; \`context.boundary.applied\` fires once per crystallize. Injected via optional callbacks at \`ContextRegions\` and \`runInterTurnEngine\` construction so those modules stay test-isolated.

2. **Cache stats lifted from provider response**: \`ModelUsage\` gains \`cacheReadTokens\` / \`cacheCreationTokens\` (optional). \`RecordingIOPort\` aggregates these into \`LlmRespondedPayload.cacheStats\` (with computed hitRate). \`AgentRuntime\` exposes them as \`llm.call\` span attributes for downstream trajectory consumers.

3. **Adapter cache_control translation (Phase 1: single breakpoint)**: \`ModelRequest\` gets optional \`cacheBreakpoint?: 'system-end'\`. \`assemble\` emits this when any active system region has \`cacheBreakpoint: true\`. Anthropic adapter wraps the system block with \`cache_control: { type: 'ephemeral' }\`. OpenAI adapter ignores (provider does automatic prefix caching).

**Scope reduction explicit**: spec §6 envisions three cache cuts (stable / session / turn). PR-D Phase 1 ships ONE (system-end). Multi-breakpoint placement deferred until trace data shows whether one is enough.

## Test results

- Unit: 30 + new region/boundary/cacheBreakpoint/adapter tests
- e2e deterministic: 7 + 15 skipped
- Full suite: green (replay fixtures re-recorded)

## Not in PR-D

- Cache-aware section reordering beyond the existing PR-B schema
- Multiple cacheBreakpoint positions (Phase 2; needs trace data)
- HTML report cache-health overlay (would benefit from accumulated trace data)
- §4.5 ARCHITECTURE.md chapter (separate small PR after this lands; per chat decision)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Spec coverage self-check

| Spec section | Status in PR-D |
|---|---|
| §8.1 New event types (region.added/removed, context.boundary.applied) | ✅ Tasks 1, 5, 6, 7 |
| §8.2 Cache health on llm.responded + span | ✅ Tasks 2, 3, 4 |
| §8.3 Trace cache breakpoint adapter translation | ✅ Task 8 (Phase 1 single-breakpoint scope) |
| §6 multi-breakpoint placement (3 cuts) | Deferred to Phase 2 (explicitly scoped down) |

## Type consistency check

- `RegionChangeDelta` defined in Task 5; used in Task 7 helper signature ✓
- `CrystallizationSummary` (already in code) used in Task 6 callback + Task 7 helper ✓
- `cacheBreakpoint?: 'system-end'` consistent across types/model.ts (Task 8.1), AssembledContext (Task 8.4), AnthropicAdapter (Task 8.9) ✓
- `cacheStats` shape consistent in trace/types.ts (Task 2) and computed by RecordingIOPort (Task 3) ✓

## Placeholder scan

None — every step shows concrete code or commands.
