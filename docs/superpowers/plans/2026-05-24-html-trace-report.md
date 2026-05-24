# HTML Trace Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `milkie trace report <runId>` that renders an event log as a self-contained HTML timeline, plus the underlying primitive `trace render-html` and the descendant-collecting flag `--include-children` on `trace inspect`. Validates the architecture claim "UI is a pure projection over CLI / SDK output" (ARCHITECTURE.md `## User-facing surfaces`) with the smallest possible probe; covers stories `s-002` (single-agent observe), `s-007` (sub-agent fan-out), `s-008` (interrupt + resume).

**Architecture:** Three layered commands. `trace inspect <runId> [--include-children]` reads the event store and emits JSONL — gains a flag to transitively follow `agent.run.started.parentId` so a sub-agent fan-out renders as one report. `trace render-html --input <file>` is a **pure JSONL→HTML projection** with zero event-store access — this is the architectural firewall that enforces "UI doesn't own query logic." `trace report <runId>` is sugar: in-process it does the equivalent of `inspect --include-children | render-html`. The renderer first builds a `TimelineNode` tree (pairing each `*.requested`/`*.responded` causal pair into one entry; nesting sub-agent runs by `parentId`) then formats it into a self-contained HTML file with inline CSS, vanilla JS for fold/unfold + type filter, and the raw events embedded as `<script type="application/json" id="trace-data">` so the file is its own re-renderable archive.

**Tech Stack:** TypeScript, commander (already a dependency, used by existing CLI), Jest. No new runtime dependencies. Renderer uses string templating — no template library.

---

## File structure

**New:**
- `src/trace/render/tree.ts` — `TimelineNode` types + `buildTimelineTree(events): TimelineNode[]` (pure, recursive over sub-agents)
- `src/trace/render/html.ts` — `renderHtml(events: Event[]): string` (calls tree builder, formats)
- `src/trace/render/template.ts` — inline CSS + vanilla JS constants used by `html.ts`
- `src/trace/render/children.ts` — `findDescendantRuns(baseDir, rootRunId): Promise<string[]>` (directory scan; not on `IEventStore` interface — only `JsonlEventStore` consumers need it)
- `src/__tests__/render-tree.test.ts` — unit tests for `buildTimelineTree`
- `src/__tests__/render-html.test.ts` — unit tests for `renderHtml`
- `src/__tests__/render-children.test.ts` — unit tests for `findDescendantRuns`
- `src/__tests__/CliTraceReport.test.ts` — CLI integration tests for `trace render-html` and `trace report`
- `examples/s-002-inspect/README.md` — example walkthrough
- `examples/s-002-inspect/agents/echo.md` — single-LLM-state agent
- `examples/s-002-inspect/record.ts` — produces a sample run
- `examples/s-002-inspect/report.sh` — `milkie trace report <runId> > report.html` driver
- `examples/s-002-inspect/.milkie/agents.json` — manifest
- `examples/s-002-inspect/.gitignore` — ignores `.milkie/runs/`, `.milkie/state.sqlite`, `report.html`

**Modify:**
- `src/cli/main.ts` — add `--include-children` flag to existing `trace inspect`, add new `trace render-html` and `trace report` commands
- `src/__tests__/CliTrace.test.ts` — add a test for `--include-children`

---

## Task 1: `findDescendantRuns` helper (directory scan by parentId)

**Files:**
- Create: `src/trace/render/children.ts`
- Test: `src/__tests__/render-children.test.ts`

- [ ] **Step 1: Write failing test**

```typescript
// src/__tests__/render-children.test.ts
import fs from 'fs'
import os from 'os'
import path from 'path'
import { findDescendantRuns } from '../trace/render/children'

function writeRun(baseDir: string, runId: string, parentId?: string): void {
  const startedEvent = {
    id: `${runId}-started`,
    runId,
    type: 'agent.run.started',
    actor: 'runtime',
    timestamp: 1,
    payload: { agentId: 'a', goal: 'g', input: 'i', contextId: runId, parentId },
  }
  fs.writeFileSync(
    path.join(baseDir, `${runId}.jsonl`),
    JSON.stringify(startedEvent) + '\n',
  )
}

describe('findDescendantRuns', () => {
  let tmpDir: string
  beforeEach(() => { tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'milkie-children-')) })
  afterEach(() => { fs.rmSync(tmpDir, { recursive: true, force: true }) })

  it('returns empty when no children exist', async () => {
    writeRun(tmpDir, 'root')
    expect(await findDescendantRuns(tmpDir, 'root')).toEqual([])
  })

  it('finds direct children', async () => {
    writeRun(tmpDir, 'root')
    writeRun(tmpDir, 'child-a', 'root')
    writeRun(tmpDir, 'child-b', 'root')
    const ids = await findDescendantRuns(tmpDir, 'root')
    expect(ids.sort()).toEqual(['child-a', 'child-b'])
  })

  it('finds transitive grandchildren', async () => {
    writeRun(tmpDir, 'root')
    writeRun(tmpDir, 'child', 'root')
    writeRun(tmpDir, 'grand', 'child')
    const ids = await findDescendantRuns(tmpDir, 'root')
    expect(ids.sort()).toEqual(['child', 'grand'])
  })

  it('returns empty when baseDir does not exist', async () => {
    expect(await findDescendantRuns(path.join(tmpDir, 'nope'), 'root')).toEqual([])
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest src/__tests__/render-children.test.ts`
Expected: FAIL with `Cannot find module '../trace/render/children'`.

- [ ] **Step 3: Implement `findDescendantRuns`**

```typescript
// src/trace/render/children.ts
import { promises as fs } from 'fs'
import path from 'path'

/**
 * Scan a JsonlEventStore base directory for runs whose first event is an
 * `agent.run.started` with `parentId` in the descendant closure of `rootRunId`.
 * Returns descendant runIds (not including the root). Directory may not exist.
 */
export async function findDescendantRuns(baseDir: string, rootRunId: string): Promise<string[]> {
  let entries: string[]
  try {
    entries = await fs.readdir(baseDir)
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') return []
    throw err
  }

  // Map every runId in the dir → its parentId (or undefined).
  const parentOf = new Map<string, string | undefined>()
  for (const entry of entries) {
    if (!entry.endsWith('.jsonl')) continue
    const runId = entry.slice(0, -'.jsonl'.length)
    try {
      const content = await fs.readFile(path.join(baseDir, entry), 'utf-8')
      const firstLine = content.split('\n').find(l => l.length > 0)
      if (!firstLine) continue
      const evt = JSON.parse(firstLine) as { type?: string, payload?: { parentId?: string } }
      if (evt.type !== 'agent.run.started') continue
      parentOf.set(runId, evt.payload?.parentId)
    } catch { /* skip unparseable files */ }
  }

  // BFS over parentOf, gathering anyone in the closure.
  const result: string[] = []
  const frontier = [rootRunId]
  while (frontier.length > 0) {
    const current = frontier.shift()!
    for (const [runId, parentId] of parentOf) {
      if (parentId === current && !result.includes(runId)) {
        result.push(runId)
        frontier.push(runId)
      }
    }
  }
  return result
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx jest src/__tests__/render-children.test.ts`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/trace/render/children.ts src/__tests__/render-children.test.ts
git commit -m "feat(trace): findDescendantRuns helper for cross-run scans"
```

---

## Task 2: `--include-children` flag on `trace inspect`

**Files:**
- Modify: `src/cli/main.ts` (extend the existing `trace inspect` action)
- Test: `src/__tests__/CliTrace.test.ts` (add one test)

- [ ] **Step 1: Write failing test**

Append to `src/__tests__/CliTrace.test.ts` inside the existing `describe` block:

```typescript
  it('inspect --include-children emits parent + descendant events as JSONL', async () => {
    const parentId = 'parent-run'
    const childId  = 'child-run'
    fs.writeFileSync(
      path.join(tmpDir, '.milkie', 'runs', `${parentId}.jsonl`),
      JSON.stringify({ id: 'p1', runId: parentId, type: 'agent.run.started', actor: 'runtime', timestamp: 1,
        payload: { agentId: 'p', goal: 'g', input: 'i', contextId: parentId } }) + '\n',
    )
    fs.writeFileSync(
      path.join(tmpDir, '.milkie', 'runs', `${childId}.jsonl`),
      JSON.stringify({ id: 'c1', runId: childId, type: 'agent.run.started', actor: 'runtime', timestamp: 2,
        payload: { agentId: 'c', goal: 'g', input: 'i', contextId: childId, parentId } }) + '\n',
    )

    const cwdSpy = jest.spyOn(process, 'cwd').mockReturnValue(tmpDir)
    try {
      const without = await main(['trace', 'inspect', parentId])
      expect(without.stdout.trim().split('\n')).toHaveLength(1)

      const withFlag = await main(['trace', 'inspect', parentId, '--include-children'])
      const runIds = withFlag.stdout.trim().split('\n').map(l => (JSON.parse(l) as { runId: string }).runId)
      expect(runIds.sort()).toEqual([childId, parentId])
    } finally {
      cwdSpy.mockRestore()
    }
  })
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest src/__tests__/CliTrace.test.ts -t "include-children"`
Expected: FAIL — second `main` call returns same length as the first because the flag is unknown to commander or ignored.

- [ ] **Step 3: Extend the `trace inspect` command in `src/cli/main.ts`**

Replace the existing `trace inspect` command block (currently lines ~137–150) with:

```typescript
  trace
    .command('inspect <runId>')
    .description('Print every event in a recorded run as JSONL')
    .option('--include-children', 'also emit events from descendant sub-agent runs')
    .action(async (runId: string, opts: { includeChildren?: boolean }) => {
      const milkieDir = findMilkieDir(process.cwd())
      if (!milkieDir) {
        throw new Error('no .milkie/ directory found upward from cwd')
      }
      const runsDir = path.join(milkieDir, 'runs')
      const eventStore = new JsonlEventStore(runsDir)

      const runIds = [runId]
      if (opts.includeChildren) {
        const { findDescendantRuns } = await import('../trace/render/children.js')
        runIds.push(...(await findDescendantRuns(runsDir, runId)))
      }
      for (const id of runIds) {
        for (const event of await eventStore.readByRunId(id)) {
          stdout.push(JSON.stringify(event) + '\n')
        }
      }
    })
```

- [ ] **Step 4: Run tests to verify all CliTrace tests still pass**

Run: `npx jest src/__tests__/CliTrace.test.ts`
Expected: PASS (existing tests + the new one — 5 total).

- [ ] **Step 5: Commit**

```bash
git add src/cli/main.ts src/__tests__/CliTrace.test.ts
git commit -m "feat(cli): trace inspect --include-children follows sub-agent runs"
```

---

## Task 3: `buildTimelineTree` — pair requested/responded, nest sub-agents

**Files:**
- Create: `src/trace/render/tree.ts`
- Test: `src/__tests__/render-tree.test.ts`

- [ ] **Step 1: Write failing test**

```typescript
// src/__tests__/render-tree.test.ts
import { buildTimelineTree } from '../trace/render/tree'
import type { Event } from '../trace/types'

const e = (over: Partial<Event> & { id: string, runId: string, type: Event['type'] }): Event => ({
  actor: 'runtime', timestamp: 0, payload: {}, ...over,
})

describe('buildTimelineTree', () => {
  it('returns one root per run, ordered by agent.run.started.timestamp', () => {
    const events: Event[] = [
      e({ id: 'b', runId: 'r2', type: 'agent.run.started', timestamp: 20 }),
      e({ id: 'a', runId: 'r1', type: 'agent.run.started', timestamp: 10 }),
    ]
    const tree = buildTimelineTree(events)
    expect(tree.map(n => n.runId)).toEqual(['r1', 'r2'])
  })

  it('pairs an llm.requested/responded into one entry via causedBy', () => {
    const events: Event[] = [
      e({ id: 's', runId: 'r1', type: 'agent.run.started', timestamp: 1 }),
      e({ id: 'q', runId: 'r1', type: 'llm.requested',  timestamp: 2,
          payload: { request: {}, requestHash: 'h' } }),
      e({ id: 'a', runId: 'r1', type: 'llm.responded',  timestamp: 3, causedBy: 'q',
          payload: { response: {}, requestHash: 'h' } }),
    ]
    const tree = buildTimelineTree(events)
    expect(tree).toHaveLength(1)
    const entries = tree[0]!.entries
    const llmEntries = entries.filter(en => en.kind === 'llm')
    expect(llmEntries).toHaveLength(1)
    expect(llmEntries[0]!.requestedId).toBe('q')
    expect(llmEntries[0]!.respondedId).toBe('a')
  })

  it('orphan llm.requested without a response stays as one entry (in-flight or error)', () => {
    const events: Event[] = [
      e({ id: 's', runId: 'r1', type: 'agent.run.started', timestamp: 1 }),
      e({ id: 'q', runId: 'r1', type: 'llm.requested',  timestamp: 2,
          payload: { request: {}, requestHash: 'h' } }),
    ]
    const tree = buildTimelineTree(events)
    const llmEntries = tree[0]!.entries.filter(en => en.kind === 'llm')
    expect(llmEntries).toHaveLength(1)
    expect(llmEntries[0]!.respondedId).toBeUndefined()
  })

  it('nests a child run under its parent via agent.run.started.parentId', () => {
    const events: Event[] = [
      e({ id: 'ps', runId: 'parent', type: 'agent.run.started', timestamp: 1,
          payload: { agentId: 'p', goal: 'g', input: 'i', contextId: 'parent' } }),
      e({ id: 'cs', runId: 'child', type: 'agent.run.started', timestamp: 2,
          payload: { agentId: 'c', goal: 'g', input: 'i', contextId: 'child', parentId: 'parent' } }),
    ]
    const tree = buildTimelineTree(events)
    expect(tree.map(n => n.runId)).toEqual(['parent'])
    expect(tree[0]!.children.map(n => n.runId)).toEqual(['child'])
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest src/__tests__/render-tree.test.ts`
Expected: FAIL — `Cannot find module '../trace/render/tree'`.

- [ ] **Step 3: Implement `buildTimelineTree`**

```typescript
// src/trace/render/tree.ts
import type { Event } from '../types.js'

export interface LlmEntry {
  kind:          'llm'
  requestedId:   string
  respondedId?:  string
  timestamp:     number
  requestHash?:  string
}

export interface ToolEntry {
  kind:          'tool'
  requestedId:   string
  respondedId?:  string
  timestamp:     number
  toolName:      string
}

export interface LifecycleEntry {
  kind:        'lifecycle'
  eventId:     string
  eventType:   'agent.run.started' | 'agent.run.completed'
  timestamp:   number
}

export type TimelineEntry = LlmEntry | ToolEntry | LifecycleEntry

export interface TimelineNode {
  runId:     string
  startedAt: number
  agentId?:  string
  status?:   string
  parentId?: string
  entries:   TimelineEntry[]
  children:  TimelineNode[]
}

/**
 * Build a tree of TimelineNodes from a flat event array spanning one or more
 * runs. Pairs `*.requested` with its `*.responded` (via causedBy). Nests child
 * runs under parents by `agent.run.started.payload.parentId`.
 *
 * Pure function — no I/O, no clock, no randomness.
 */
export function buildTimelineTree(events: Event[]): TimelineNode[] {
  // Group events by runId.
  const byRun = new Map<string, Event[]>()
  for (const evt of events) {
    const bucket = byRun.get(evt.runId)
    if (bucket) bucket.push(evt)
    else byRun.set(evt.runId, [evt])
  }

  // Build a node per run.
  const nodes = new Map<string, TimelineNode>()
  for (const [runId, evts] of byRun) {
    const sorted = [...evts].sort((a, b) => a.timestamp - b.timestamp)
    const started   = sorted.find(e => e.type === 'agent.run.started') as
      (Event & { payload: { agentId: string, parentId?: string } }) | undefined
    const completed = sorted.find(e => e.type === 'agent.run.completed') as
      (Event & { payload: { status: string } }) | undefined

    const entries: TimelineEntry[] = []
    const requestedById = new Map<string, Event>()
    for (const evt of sorted) requestedById.set(evt.id, evt)

    // Pair *.responded back to *.requested via causedBy.
    const consumed = new Set<string>()
    for (const evt of sorted) {
      if (evt.type === 'llm.responded' || evt.type === 'tool.responded') {
        if (evt.causedBy && requestedById.has(evt.causedBy)) consumed.add(evt.id)
      }
    }
    for (const evt of sorted) {
      if (consumed.has(evt.id)) continue
      if (evt.type === 'llm.requested') {
        const paired = sorted.find(o =>
          o.type === 'llm.responded' && o.causedBy === evt.id) as Event | undefined
        entries.push({
          kind: 'llm', requestedId: evt.id, respondedId: paired?.id,
          timestamp: evt.timestamp,
          requestHash: (evt.payload as { requestHash?: string }).requestHash,
        })
      } else if (evt.type === 'tool.requested') {
        const paired = sorted.find(o =>
          o.type === 'tool.responded' && o.causedBy === evt.id) as Event | undefined
        entries.push({
          kind: 'tool', requestedId: evt.id, respondedId: paired?.id,
          timestamp: evt.timestamp,
          toolName: (evt.payload as { toolName: string }).toolName,
        })
      } else if (evt.type === 'agent.run.started' || evt.type === 'agent.run.completed') {
        entries.push({ kind: 'lifecycle', eventId: evt.id, eventType: evt.type, timestamp: evt.timestamp })
      }
    }

    nodes.set(runId, {
      runId,
      startedAt: started?.timestamp ?? sorted[0]?.timestamp ?? 0,
      agentId:   started?.payload.agentId,
      status:    completed?.payload.status,
      parentId:  started?.payload.parentId,
      entries,
      children:  [],
    })
  }

  // Wire children under parents.
  const roots: TimelineNode[] = []
  for (const node of nodes.values()) {
    if (node.parentId && nodes.has(node.parentId)) {
      nodes.get(node.parentId)!.children.push(node)
    } else {
      roots.push(node)
    }
  }
  // Stable ordering by startedAt at each level.
  const sortRec = (ns: TimelineNode[]): void => {
    ns.sort((a, b) => a.startedAt - b.startedAt)
    for (const n of ns) sortRec(n.children)
  }
  sortRec(roots)
  return roots
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx jest src/__tests__/render-tree.test.ts`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/trace/render/tree.ts src/__tests__/render-tree.test.ts
git commit -m "feat(trace): buildTimelineTree pairs req/resp and nests sub-agents"
```

---

## Task 4: `renderHtml` — pure JSON-array → self-contained HTML

**Files:**
- Create: `src/trace/render/html.ts`
- Create: `src/trace/render/template.ts`
- Test: `src/__tests__/render-html.test.ts`

- [ ] **Step 1: Write failing test**

```typescript
// src/__tests__/render-html.test.ts
import { renderHtml } from '../trace/render/html'
import type { Event } from '../trace/types'

const e = (over: Partial<Event> & { id: string, runId: string, type: Event['type'] }): Event => ({
  actor: 'runtime', timestamp: 0, payload: {}, ...over,
})

describe('renderHtml', () => {
  it('produces a complete HTML document with doctype and trace-data script', () => {
    const events: Event[] = [
      e({ id: 's', runId: 'r1', type: 'agent.run.started', timestamp: 1,
          payload: { agentId: 'echo', goal: 'g', input: 'i', contextId: 'c' } }),
      e({ id: 'c', runId: 'r1', type: 'agent.run.completed', timestamp: 9,
          payload: { status: 'completed', lastTextOutput: 'hi' } }),
    ]
    const html = renderHtml(events)
    expect(html.startsWith('<!doctype html>')).toBe(true)
    expect(html).toContain('</html>')
    expect(html).toContain('<script type="application/json" id="trace-data">')
    // Embedded JSON must be valid and contain the events.
    const m = html.match(/<script type="application\/json" id="trace-data">([\s\S]*?)<\/script>/)
    expect(m).not.toBeNull()
    const embedded = JSON.parse(m![1]!) as Event[]
    expect(embedded).toHaveLength(2)
    expect(embedded[0]!.id).toBe('s')
  })

  it('renders one timeline section per root run, shows runId in header', () => {
    const events: Event[] = [
      e({ id: 's1', runId: 'parent', type: 'agent.run.started', timestamp: 1,
          payload: { agentId: 'p', goal: 'g', input: 'i', contextId: 'parent' } }),
      e({ id: 's2', runId: 'child',  type: 'agent.run.started', timestamp: 2,
          payload: { agentId: 'c', goal: 'g', input: 'i', contextId: 'child', parentId: 'parent' } }),
    ]
    const html = renderHtml(events)
    expect(html).toContain('parent')
    expect(html).toContain('child')
    // child should appear nested inside parent's section — assert child's
    // marker comes after parent's section start.
    const parentIdx = html.indexOf('data-run-id="parent"')
    const childIdx  = html.indexOf('data-run-id="child"')
    expect(parentIdx).toBeGreaterThan(-1)
    expect(childIdx).toBeGreaterThan(parentIdx)
  })

  it('escapes HTML-special characters in user payload (XSS guard)', () => {
    const events: Event[] = [
      e({ id: 's', runId: 'r1', type: 'agent.run.started', timestamp: 1,
          payload: { agentId: '<script>alert(1)</script>', goal: 'g', input: 'i', contextId: 'c' } }),
    ]
    const html = renderHtml(events)
    expect(html).not.toContain('<script>alert(1)</script>')
    expect(html).toContain('&lt;script&gt;alert(1)&lt;/script&gt;')
  })

  it('returns valid HTML for an empty event array', () => {
    const html = renderHtml([])
    expect(html.startsWith('<!doctype html>')).toBe(true)
    expect(html).toContain('</html>')
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest src/__tests__/render-html.test.ts`
Expected: FAIL — `Cannot find module '../trace/render/html'`.

- [ ] **Step 3: Implement `template.ts` and `html.ts`**

```typescript
// src/trace/render/template.ts
export const STYLES = `
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0; padding: 24px; background: #f7f7f8; color: #1c1c1e; }
  h1 { font-size: 18px; margin: 0 0 16px 0; font-weight: 600; }
  .run { background: white; border: 1px solid #e5e5e7; border-radius: 8px;
         margin-bottom: 16px; padding: 16px; }
  .run-head { display: flex; gap: 12px; align-items: baseline; margin-bottom: 12px; }
  .run-id { font-family: ui-monospace, SFMono-Regular, monospace; font-size: 12px;
            color: #6e6e73; }
  .badge { font-size: 11px; padding: 2px 6px; border-radius: 4px;
           background: #e8f5e8; color: #2d6a2d; }
  .badge.error { background: #fde8e8; color: #a13; }
  .badge.interrupted { background: #fff4e0; color: #8a5a00; }
  .entry { display: flex; gap: 8px; padding: 6px 0; border-bottom: 1px solid #f0f0f2;
           cursor: pointer; }
  .entry:last-child { border-bottom: none; }
  .entry .icon { width: 16px; text-align: center; }
  .entry.llm .icon { color: #5b3ec9; }
  .entry.tool .icon { color: #2563eb; }
  .entry.lifecycle .icon { color: #6e6e73; }
  .entry .summary { flex: 1; font-size: 13px; }
  .entry .ts { font-family: ui-monospace, monospace; font-size: 11px; color: #6e6e73; }
  .child-run { margin-left: 24px; margin-top: 8px; border-left: 2px solid #e5e5e7;
               padding-left: 12px; }
  .filters { margin-bottom: 16px; display: flex; gap: 8px; flex-wrap: wrap; }
  .chip { font-size: 12px; padding: 4px 10px; border-radius: 999px; cursor: pointer;
          background: white; border: 1px solid #d1d1d6; user-select: none; }
  .chip.active { background: #1c1c1e; color: white; border-color: #1c1c1e; }
  .payload { display: none; margin-top: 6px; background: #fafafa; padding: 8px;
             font-family: ui-monospace, monospace; font-size: 11px; white-space: pre-wrap;
             border-radius: 4px; max-height: 320px; overflow: auto; }
  .entry.open .payload { display: block; }
`

export const SCRIPT = `
  (function () {
    document.addEventListener('click', function (ev) {
      var entry = ev.target.closest('.entry');
      if (entry) entry.classList.toggle('open');
      var chip = ev.target.closest('.chip');
      if (chip) {
        chip.classList.toggle('active');
        var kinds = Array.from(document.querySelectorAll('.chip.active'))
          .map(function (c) { return c.dataset.kind; });
        document.querySelectorAll('.entry').forEach(function (e) {
          var k = e.dataset.kind;
          e.style.display = (kinds.length === 0 || kinds.indexOf(k) >= 0) ? '' : 'none';
        });
      }
    });
  })();
`
```

```typescript
// src/trace/render/html.ts
import type { Event } from '../types.js'
import { buildTimelineTree, type TimelineEntry, type TimelineNode } from './tree.js'
import { STYLES, SCRIPT } from './template.js'

function esc(s: string): string {
  return s
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;')
}

function summaryFor(entry: TimelineEntry): string {
  if (entry.kind === 'llm')       return 'LLM call' + (entry.respondedId ? '' : ' (no response)')
  if (entry.kind === 'tool')      return 'tool: ' + esc(entry.toolName) + (entry.respondedId ? '' : ' (no response)')
  return entry.eventType === 'agent.run.started' ? 'run started' : 'run completed'
}

function renderEntry(entry: TimelineEntry): string {
  return `<div class="entry ${entry.kind}" data-kind="${entry.kind}">`
       + `<span class="icon">${entry.kind === 'llm' ? '◆' : entry.kind === 'tool' ? '▣' : '●'}</span>`
       + `<span class="summary">${summaryFor(entry)}</span>`
       + `<span class="ts">${entry.timestamp}</span>`
       + `</div>`
}

function renderNode(node: TimelineNode): string {
  const status = node.status ?? 'in-flight'
  const badgeClass = status === 'completed' ? '' : ' ' + status
  return `<section class="run" data-run-id="${esc(node.runId)}">`
       + `<div class="run-head">`
       + `<strong>${esc(node.agentId ?? '(unknown)')}</strong>`
       + `<span class="run-id">${esc(node.runId)}</span>`
       + `<span class="badge${badgeClass}">${esc(status)}</span>`
       + `</div>`
       + node.entries.map(renderEntry).join('')
       + (node.children.length > 0
           ? `<div class="child-run">${node.children.map(renderNode).join('')}</div>`
           : '')
       + `</section>`
}

/**
 * Pure projection: flat events → self-contained HTML report.
 *
 * The output is a single HTML document with inline CSS, vanilla JS for
 * fold/unfold + type filter, and the raw events embedded as a JSON script
 * tag so the file is its own re-renderable archive.
 *
 * No I/O, no clock, no randomness. The renderer cannot reach into the event
 * store — this is the architectural firewall behind "UI is a pure projection
 * over CLI / SDK output" (ARCHITECTURE.md `## User-facing surfaces`).
 */
export function renderHtml(events: Event[]): string {
  const tree = buildTimelineTree(events)
  const dataJson = JSON.stringify(events)
    // close-tag-safe inlining: prevent the JSON from ending the script element.
    .replace(/<\/script/gi, '<\\/script')
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>milkie trace report</title>
<style>${STYLES}</style>
</head>
<body>
<h1>milkie trace report</h1>
<div class="filters">
  <span class="chip" data-kind="llm">LLM</span>
  <span class="chip" data-kind="tool">tool</span>
  <span class="chip" data-kind="lifecycle">lifecycle</span>
</div>
${tree.map(renderNode).join('')}
<script type="application/json" id="trace-data">${dataJson}</script>
<script>${SCRIPT}</script>
</body>
</html>`
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx jest src/__tests__/render-html.test.ts`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/trace/render/html.ts src/trace/render/template.ts src/__tests__/render-html.test.ts
git commit -m "feat(trace): renderHtml pure projection — JSONL → self-contained report"
```

---

## Task 5: `trace render-html --input <file>` CLI command

**Files:**
- Modify: `src/cli/main.ts`
- Test: `src/__tests__/CliTraceReport.test.ts` (new file)

- [ ] **Step 1: Write failing test**

```typescript
// src/__tests__/CliTraceReport.test.ts
import { main } from '../cli/main'
import fs from 'fs'
import os from 'os'
import path from 'path'

describe('CLI: trace render-html', () => {
  let tmpDir: string
  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'milkie-render-'))
  })
  afterEach(() => { fs.rmSync(tmpDir, { recursive: true, force: true }) })

  it('reads JSONL from --input file and writes self-contained HTML to stdout', async () => {
    const events = [
      { id: 's', runId: 'r1', type: 'agent.run.started', actor: 'runtime', timestamp: 1,
        payload: { agentId: 'echo', goal: 'g', input: 'i', contextId: 'c' } },
      { id: 'c', runId: 'r1', type: 'agent.run.completed', actor: 'runtime', timestamp: 9,
        payload: { status: 'completed', lastTextOutput: 'hi' } },
    ]
    const input = path.join(tmpDir, 'events.jsonl')
    fs.writeFileSync(input, events.map(e => JSON.stringify(e)).join('\n') + '\n')

    const result = await main(['trace', 'render-html', '--input', input])
    expect(result.exitCode).toBe(0)
    expect(result.stdout.startsWith('<!doctype html>')).toBe(true)
    expect(result.stdout).toContain('echo')
    expect(result.stdout).toContain('r1')
  })

  it('exits non-zero with diagnostic when --input file is missing', async () => {
    const result = await main(['trace', 'render-html', '--input', path.join(tmpDir, 'nope.jsonl')])
    expect(result.exitCode).not.toBe(0)
    expect(result.stderr).toMatch(/ENOENT|not found|no such file/i)
  })

  it('handles empty JSONL gracefully (still emits valid HTML)', async () => {
    const input = path.join(tmpDir, 'empty.jsonl')
    fs.writeFileSync(input, '')
    const result = await main(['trace', 'render-html', '--input', input])
    expect(result.exitCode).toBe(0)
    expect(result.stdout).toContain('<!doctype html>')
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest src/__tests__/CliTraceReport.test.ts -t "render-html"`
Expected: FAIL — unknown command `render-html`.

- [ ] **Step 3: Add the `trace render-html` command in `src/cli/main.ts`**

Insert this block in `src/cli/main.ts` after the existing `trace inspect` command and before `trace replay`:

```typescript
  trace
    .command('render-html')
    .description('Render trace JSONL into a self-contained HTML report (reads --input file, writes HTML to stdout)')
    .requiredOption('--input <path>', 'JSONL file produced by `trace inspect` (or any equivalent source)')
    .action(async (opts: { input: string }) => {
      const { renderHtml } = await import('../trace/render/html.js')
      const content = fs.readFileSync(opts.input, 'utf-8')
      const events = content.split('\n')
        .filter(l => l.length > 0)
        .map(l => JSON.parse(l))
      stdout.push(renderHtml(events))
    })
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx jest src/__tests__/CliTraceReport.test.ts -t "render-html"`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/cli/main.ts src/__tests__/CliTraceReport.test.ts
git commit -m "feat(cli): trace render-html — pure JSONL → HTML projection command"
```

---

## Task 6: `trace report <runId>` convenience command (sugar over inspect+render-html)

**Files:**
- Modify: `src/cli/main.ts`
- Test: `src/__tests__/CliTraceReport.test.ts` (extend existing file)

- [ ] **Step 1: Write failing test**

Append to `src/__tests__/CliTraceReport.test.ts` (inside the existing `describe`):

```typescript
  it('trace report <runId> renders the run from .milkie/runs/ as HTML', async () => {
    fs.mkdirSync(path.join(tmpDir, '.milkie', 'runs'), { recursive: true })
    const runId = 'demo-run'
    const events = [
      { id: 's', runId, type: 'agent.run.started', actor: 'runtime', timestamp: 1,
        payload: { agentId: 'echo', goal: 'g', input: 'i', contextId: runId } },
      { id: 'c', runId, type: 'agent.run.completed', actor: 'runtime', timestamp: 9,
        payload: { status: 'completed', lastTextOutput: 'hi' } },
    ]
    fs.writeFileSync(
      path.join(tmpDir, '.milkie', 'runs', `${runId}.jsonl`),
      events.map(e => JSON.stringify(e)).join('\n') + '\n',
    )

    const cwdSpy = jest.spyOn(process, 'cwd').mockReturnValue(tmpDir)
    try {
      const result = await main(['trace', 'report', runId])
      expect(result.exitCode).toBe(0)
      expect(result.stdout.startsWith('<!doctype html>')).toBe(true)
      expect(result.stdout).toContain('echo')
      expect(result.stdout).toContain(runId)
    } finally {
      cwdSpy.mockRestore()
    }
  })

  it('trace report includes descendant sub-agent runs in one HTML', async () => {
    fs.mkdirSync(path.join(tmpDir, '.milkie', 'runs'), { recursive: true })
    const parent = 'parent-run'
    const child  = 'child-run'
    fs.writeFileSync(
      path.join(tmpDir, '.milkie', 'runs', `${parent}.jsonl`),
      JSON.stringify({ id: 'p', runId: parent, type: 'agent.run.started', actor: 'runtime', timestamp: 1,
        payload: { agentId: 'p', goal: 'g', input: 'i', contextId: parent } }) + '\n',
    )
    fs.writeFileSync(
      path.join(tmpDir, '.milkie', 'runs', `${child}.jsonl`),
      JSON.stringify({ id: 'c', runId: child, type: 'agent.run.started', actor: 'runtime', timestamp: 2,
        payload: { agentId: 'c', goal: 'g', input: 'i', contextId: child, parentId: parent } }) + '\n',
    )

    const cwdSpy = jest.spyOn(process, 'cwd').mockReturnValue(tmpDir)
    try {
      const result = await main(['trace', 'report', parent])
      expect(result.exitCode).toBe(0)
      const parentIdx = result.stdout.indexOf('data-run-id="parent-run"')
      const childIdx  = result.stdout.indexOf('data-run-id="child-run"')
      expect(parentIdx).toBeGreaterThan(-1)
      expect(childIdx).toBeGreaterThan(parentIdx)
    } finally {
      cwdSpy.mockRestore()
    }
  })
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest src/__tests__/CliTraceReport.test.ts -t "report"`
Expected: FAIL — unknown command `report`.

- [ ] **Step 3: Add the `trace report` command in `src/cli/main.ts`**

Insert this block after `trace render-html` and before `trace replay`:

```typescript
  trace
    .command('report <runId>')
    .description('Render <runId> (and any descendant sub-agent runs) as a self-contained HTML report to stdout')
    .action(async (runId: string) => {
      const milkieDir = findMilkieDir(process.cwd())
      if (!milkieDir) {
        throw new Error('no .milkie/ directory found upward from cwd')
      }
      const runsDir = path.join(milkieDir, 'runs')
      const eventStore = new JsonlEventStore(runsDir)
      const { findDescendantRuns } = await import('../trace/render/children.js')
      const { renderHtml } = await import('../trace/render/html.js')

      const runIds = [runId, ...(await findDescendantRuns(runsDir, runId))]
      const events = []
      for (const id of runIds) events.push(...(await eventStore.readByRunId(id)))
      stdout.push(renderHtml(events))
    })
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx jest src/__tests__/CliTraceReport.test.ts`
Expected: PASS (5 tests in this file).

- [ ] **Step 5: Commit**

```bash
git add src/cli/main.ts src/__tests__/CliTraceReport.test.ts
git commit -m "feat(cli): trace report <runId> — sugar wrapping inspect+render-html"
```

---

## Task 7: Full suite + lint check

**Files:** (no edits, just verification)

- [ ] **Step 1: Run the unit suite**

Run: `npm run test:unit`
Expected: PASS — existing unit tests still green.

- [ ] **Step 2: Run all jest tests in `src/__tests__/`**

Run: `npx jest src/__tests__/`
Expected: PASS — all existing CLI / runtime tests plus the four new test files (render-children, render-tree, render-html, CliTraceReport) pass.

- [ ] **Step 3: Run lint**

Run: `npm run lint`
Expected: clean exit. If any errors surface, fix them (most likely: unused imports, ESLint rule about explicit any in render code).

- [ ] **Step 4: Build the dist/ output**

Run: `npm run build`
Expected: clean tsc compile.

- [ ] **Step 5: Smoke-test the binary end-to-end against the s-005 example fixture**

```bash
# Use the s-005 example's already-recorded run as a real input.
node dist/cli/index.js trace inspect $(cat examples/s-005-replay/.milkie/last-run.txt) > /tmp/events.jsonl
node dist/cli/index.js trace render-html --input /tmp/events.jsonl > /tmp/report.html
test -s /tmp/report.html && head -1 /tmp/report.html
# Expected: prints `<!doctype html>` and exits 0. No commit — this is a smoke check.
```

(If `examples/s-005-replay/.milkie/last-run.txt` doesn't exist yet, run `npx tsx examples/s-005-replay/record.ts` first.)

---

## Task 8: `examples/s-002-inspect/` example

**Files:**
- Create: `examples/s-002-inspect/README.md`
- Create: `examples/s-002-inspect/agents/echo.md`
- Create: `examples/s-002-inspect/record.ts`
- Create: `examples/s-002-inspect/report.sh`
- Create: `examples/s-002-inspect/.milkie/agents.json`
- Create: `examples/s-002-inspect/.gitignore`

- [ ] **Step 1: Create the agent definition**

```markdown
<!-- examples/s-002-inspect/agents/echo.md -->
---
agentId: echo
fsm:
  states:
    - name: react
      type: llm
      instructions: greet the user briefly
      tools: []
model:
  provider: stub
  model: stub
  adapter: openai-compatible
---
You are a friendly greeter.
```

- [ ] **Step 2: Create the manifest**

```json
// examples/s-002-inspect/.milkie/agents.json
{ "agents": [{ "id": "echo", "file": "../agents/echo.md" }] }
```

- [ ] **Step 3: Create `.gitignore`**

```gitignore
# examples/s-002-inspect/.gitignore
.milkie/runs/
.milkie/state.sqlite
.milkie/last-run.txt
report.html
```

- [ ] **Step 4: Create `record.ts` (mirrors `examples/s-005-replay/record.ts` shape)**

```typescript
// examples/s-002-inspect/record.ts
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'
import { Milkie } from '../../src/runtime/Milkie.js'
import { SQLiteStore } from '../../src/store/SQLiteStore.js'
import { JsonlEventStore } from '../../src/trace/JsonlEventStore.js'
import type { IModelGateway, ModelRequest, ModelResponse } from '../../src/types/model.js'

class StubGateway implements IModelGateway {
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    return { content: [{ type: 'text', text: 'hello, milkie!' }], toolCalls: [], finishReason: 'end_turn' }
  }
  async *stream(_req: ModelRequest): AsyncIterable<never> { yield* [] }
}

async function main(): Promise<void> {
  const here = path.dirname(fileURLToPath(import.meta.url))
  const milkieDir = path.join(here, '.milkie')
  fs.mkdirSync(path.join(milkieDir, 'runs'), { recursive: true })

  const stateStore = new SQLiteStore({ path: path.join(milkieDir, 'state.sqlite') })
  await stateStore.init()
  const eventStore = new JsonlEventStore(path.join(milkieDir, 'runs'))
  const milkie = new Milkie({ stateStore, gateway: new StubGateway(), eventStore })
  milkie.loadAgentFile(path.join(here, 'agents', 'echo.md'))

  const result = await milkie.invoke({ agentId: 'echo', goal: 'say hi', input: 'hi there' })
  fs.writeFileSync(path.join(milkieDir, 'last-run.txt'), result.agentRunId)
  process.stdout.write(JSON.stringify({
    runId: result.agentRunId, status: result.status, output: result.output,
    eventFile: path.join(milkieDir, 'runs', result.agentRunId + '.jsonl'),
  }, null, 2) + '\n')
}

main().catch(err => { process.stderr.write(String(err) + '\n'); process.exit(1) })
```

- [ ] **Step 5: Create `report.sh`**

```bash
#!/usr/bin/env bash
# examples/s-002-inspect/report.sh
set -euo pipefail
cd "$(dirname "$0")"
RUN_ID="$(cat .milkie/last-run.txt)"
node ../../dist/cli/index.js trace report "$RUN_ID" > report.html
echo "wrote report.html for run $RUN_ID"
```

- [ ] **Step 6: Create README**

```markdown
<!-- examples/s-002-inspect/README.md -->
# s-002 — Inspect a Completed Run (HTML report)

Runnable example for story
[`s-002-inspect-a-completed-run`](../../docs/stories/s-002-inspect-a-completed-run.md).

Demonstrates **`trace report`** — the HTML projection over an event log.
The same data you can read via `milkie trace inspect <runId>` (JSONL) is
rendered into a self-contained HTML file that opens in any browser,
without a server or framework.

This example exists to validate two architectural claims:

1. Visual rendering changes trace **affordance** vs. CLI output alone —
   a folded timeline with click-to-expand payloads makes "what did the
   agent do" discoverable in a way `inspect` JSONL is not.
2. The CLI JSON output is rich enough to **fully drive** a UI projection
   (ARCHITECTURE.md `## User-facing surfaces` — UI is a pure projection
   over CLI / SDK output, never a parallel facade).

## Files

```
.milkie/
  agents.json       # manifest
  runs/             # JsonlEventStore base (filled by record.ts)
  last-run.txt      # runId of the most recent recording
agents/
  echo.md
record.ts           # records a sample run with a stub gateway (no API key)
report.sh           # `milkie trace report <runId> > report.html`
report.html         # generated artifact (gitignored)
README.md
```

## Run it

```bash
# 1. Build once so the CLI binary exists.
$ npm run build

# 2. Record a sample run.
$ npx tsx examples/s-002-inspect/record.ts
{
  "runId": "...",
  "status": "completed",
  "output": "hello, milkie!",
  "eventFile": ".../examples/s-002-inspect/.milkie/runs/....jsonl"
}

# 3. Render the report.
$ ./examples/s-002-inspect/report.sh
wrote report.html for run ...

# 4. Open the report in a browser.
$ open examples/s-002-inspect/report.html
```

## What's in the report

- Run header: agent id, runId, status badge.
- Event timeline: one entry per LLM call / tool call / lifecycle event;
  paired `*.requested` / `*.responded` events collapse to a single entry
  via their `causedBy` chain.
- Click any entry to expand its payload.
- Type-filter chips (LLM / tool / lifecycle) at the top.
- Sub-agent runs (when present) nest under their parent as indented
  child timelines — same layout, recursively.

## What this proves

- **CLI ↔ projection parity.** `report.html` consumes nothing beyond
  the data `trace inspect --include-children` already emits; the renderer
  cannot reach into the event store. If the report can show it, the CLI
  can output it.
- **Self-contained artifact.** The HTML file embeds its own raw events
  as `<script type="application/json" id="trace-data">`, making it a
  re-renderable archive: future renderer versions can re-render the
  same file without re-running the agent.

## What this does NOT prove yet

- **Fork / diff / lineage / suite views.** These require Phase 5–6
  capabilities that aren't in code yet; the report scope is intentionally
  limited to the Observable surface.
- **In-flight rendering.** The report is for completed runs. Mid-run
  updates require the Phase 5 in-flight trace query API.

## Related

- Story: [s-002](../../docs/stories/s-002-inspect-a-completed-run.md)
- Spec: [CLI surface](../../docs/superpowers/specs/2026-05-24-cli-surface-design.md)
- Architecture: [`Observable` capability](../../ARCHITECTURE.md#representative-scenarios)
```

- [ ] **Step 7: Make `report.sh` executable and run the full example end-to-end**

```bash
chmod +x examples/s-002-inspect/report.sh
npm run build
npx tsx examples/s-002-inspect/record.ts
./examples/s-002-inspect/report.sh
# Expected: report.html exists, starts with <!doctype html>, can be opened
# in a browser and shows one run with the agent.run.started/completed lifecycle.
test -s examples/s-002-inspect/report.html
head -1 examples/s-002-inspect/report.html
open examples/s-002-inspect/report.html  # visual check
```

- [ ] **Step 8: Commit**

```bash
git add examples/s-002-inspect/
git commit -m "feat(examples): s-002 inspect — HTML trace report end-to-end"
```

---

## Self-review checklist

1. **Spec coverage.** Confirmed decisions traced to tasks:
   - "Candidate A + B, both sharing renderer" → Tasks 4–6 (renderer in `html.ts`, `trace render-html` and `trace report` both consume it).
   - "Embed raw JSON as re-extractable script tag" → Task 4 step 3 (`<script type="application/json" id="trace-data">`) + Task 4 step 1 test (parses the script content back).
   - "Sub-agent nesting recursive" → Tasks 1, 3, 4, 6 (descendant scan + tree nesting + recursive renderNode + CLI integration test).
2. **Placeholder scan.** No "TBD" / "add error handling later" / "similar to Task N" — every code step has full code.
3. **Type consistency.** `TimelineNode`, `TimelineEntry`, `LlmEntry`, `ToolEntry`, `LifecycleEntry` defined in Task 3; consumed in Task 4 by exact field names. `findDescendantRuns(baseDir, rootRunId)` signature matches between Task 1 definition and Tasks 2 + 6 imports.
4. **No backwards-compat shim.** `trace inspect` keeps prior behavior when `--include-children` is absent (verified by existing test still asserting one event without the flag); new commands are additive.
