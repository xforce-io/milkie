# self-explain MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让业务 agent 在对话中溯源自己最近 N 轮执行 —— 复用 inspection 层(①Observable + ③Lineage),自绑定窗口,不造 self 专用 tool。

**Architecture:** 唯一硬新增是把 `previousRunId` 注入 `ToolContext`(D1);① 改造现有 `get_execution`(runId 可选 + lookback + 自视图丢 prompt 正文),③ 新增 `get_lineage` tool(按 claim 在窗口内搜 `cites` 边)。两个 tool 共用一个有界链遍历 helper `walkRunWindow`,投影逻辑为纯函数。event log 只进 tool handler 进程内存,不进 context。

**Tech Stack:** TypeScript (ESM, `.js` import 后缀), jest, 现有 `IEventStore` / `buildExecutionProjection` / lineage 事件(`object.created` / `relation.created`)。

**Spec:** xforce-io/milkie#189(issue body 为单一事实源)。

## Global Constraints

- 语言/风格:TypeScript ESM,import 必须带 `.js` 后缀;匹配周边代码缩进与命名。
- 测试:jest;单测跑 `npx jest <file>`;类型检查 `npm run build`(tsc)。测试框架用全局 `describe/it/expect`(`@types/jest`),不 import。
- 窗口参数:`lookback` 默认 **3**,clamp 到 **[1, 10]**。
- 自视图(runId 省略时)**必须丢掉 llm step 的 `prompt` / `response` 正文**(避免泄露 system prompt);显式传 runId 的诊断路径保持全量,向后兼容。
- DRY / YAGNI / TDD / 每个 task 末尾 commit。
- 不做:②Diagnosable、Replay/Fork/Diff、无界整会话回溯、完整 D3 对外视图框架。

---

### Task 1: D1 —— 注入 `ToolContext.previousRunId`

唯一硬新增。`previousRunId` 在 `Milkie.ts:312` 已算出(会话链 `checkpoint-run:latest`),只是没穿到 runtime 与 ToolContext。

**Files:**
- Modify: `src/types/tool.ts`(ToolContext 加字段)
- Modify: `src/runtime/AgentRuntime.ts`(opts 加字段 + 存字段 + buildToolContext 注入)
- Modify: `src/runtime/Milkie.ts`(`new AgentRuntime({...})` 透传)
- Test: `src/__tests__/AgentRuntime.previousRunId.test.ts`

**Interfaces:**
- Produces: `ToolContext.previousRunId?: string` —— Task 3 / Task 5 的 tool handler 在 `runId` 省略时读它定位窗口起点。

- [ ] **Step 1: Write the failing test**(照搬 `AgentRuntime.currentTurn.test.ts` 的 fixtures)

```typescript
// src/__tests__/AgentRuntime.previousRunId.test.ts
import { AgentRuntime } from '../runtime/AgentRuntime'
import { DefaultIOPort } from '../runtime/IOPort'
import { MemoryStore } from '../store/MemoryStore'
import { InMemoryRecorder } from '../trajectory/InMemoryRecorder'
import type { AgentConfig } from '../types/agent'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'
import type { ToolDefinition } from '../types/tool'

function makeConfig(): AgentConfig {
  return {
    agentId: 'test-agent', version: '1.0.0', systemPrompt: 'test',
    fsm: { states: [{ name: 'react', type: 'llm' }] },
    model: { provider: 'test', model: 'test-model', adapter: 'test' },
  }
}
class SequentialGateway implements IModelGateway {
  private i = 0
  constructor(private responses: ModelResponse[]) {}
  async complete(_r: ModelRequest): Promise<ModelResponse> {
    const r = this.responses[this.i++]; if (!r) throw new Error('no more'); return r
  }
  async *stream(_r: ModelRequest): AsyncIterable<never> { yield* [] }
}

describe('#189 D1 ToolContext.previousRunId', () => {
  it('LLM-state tool handler receives ctx.previousRunId from runtime opts', async () => {
    let captured: string | undefined
    const probe: ToolDefinition = {
      name: 'probe', description: 'capture previousRunId',
      inputSchema: { type: 'object', properties: {} },
      handler: async (_input, ctx) => { captured = ctx?.previousRunId; return { ok: true } },
    }
    const gateway = new SequentialGateway([
      { content: [{ type: 'tool_use', id: 't1', name: 'probe', input: {} }],
        toolCalls: [{ id: 't1', name: 'probe', input: {} }], finishReason: 'tool_use' },
      { content: [{ type: 'text', text: 'done' }], toolCalls: [], finishReason: 'end_turn' },
    ])
    const runtime = new AgentRuntime({
      config: makeConfig(), goal: 'g', input: 'hi',
      previousRunId: 'run-prev-123',
      stateStore: new MemoryStore(),
      recorder: new InMemoryRecorder(),
      ioPort: new DefaultIOPort(gateway),
      extraTools: [probe],
      config2: undefined as never,
    } as never)
    await runtime.run('hi')
    expect(captured).toBe('run-prev-123')
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest src/__tests__/AgentRuntime.previousRunId.test.ts`
Expected: FAIL —— `captured` is `undefined`(字段未注入);或 TS 报 `previousRunId` 不在 options 上。

- [ ] **Step 3: Add field to ToolContext type**

`src/types/tool.ts`,在 `currentTurn?: string` 之后加:

```typescript
  /** #189: the previous run of this session (D1). Lets a self-explain tool
   * resolve "my last turn(s)" without the agent passing a runId. Unset on a
   * session's first turn. */
  previousRunId?: string
```

- [ ] **Step 4: Thread through AgentRuntime**

`src/runtime/AgentRuntime.ts`:
1. 在 `AgentRuntimeOptions` 接口里 `parentId?: string` 之后加:
```typescript
  /** #189 D1: previous run of this session, for self-explain window. */
  previousRunId?:    string
```
2. 字段声明(在 `private readonly extraTools` 附近)加:
```typescript
  private readonly previousRunId?:  string
```
3. 构造函数里(`this.parentId = opts.parentId` 之后)加:
```typescript
    this.previousRunId   = opts.previousRunId
```
4. `buildToolContext` 的 ctx 字面量里(`currentTurn: this.currentTurnRaw,` 之后)加:
```typescript
      previousRunId: this.previousRunId,
```

- [ ] **Step 5: Thread through Milkie**

`src/runtime/Milkie.ts`,`new AgentRuntime({ ... })`(约 354 行,`agentRunId,` 附近)加一行:

```typescript
      ...(previousRunId ? { previousRunId } : {}),
```

- [ ] **Step 6: Run test to verify it passes**

Run: `npx jest src/__tests__/AgentRuntime.previousRunId.test.ts`
Expected: PASS

- [ ] **Step 7: Typecheck + commit**

```bash
npm run build
git add src/types/tool.ts src/runtime/AgentRuntime.ts src/runtime/Milkie.ts src/__tests__/AgentRuntime.previousRunId.test.ts
git commit -m "feat(#189): inject ToolContext.previousRunId (D1)"
```

---

### Task 2: `walkRunWindow` —— 有界会话链遍历 helper

① 和 ③ 共用。镜像 `Milkie.getSessionHistory`(Milkie.ts:711)的链遍历,但有界到 N 跳。

**Files:**
- Create: `src/trace/diagnostics/walkRunWindow.ts`
- Test: `src/__tests__/walkRunWindow.test.ts`

**Interfaces:**
- Consumes: `IEventStore.readByRunId`(`src/trace/EventStore.ts`)、`AgentRunStartedPayload.previousRunId`(`src/trace/types.ts:121`)。
- Produces: `walkRunWindow(eventStore: IEventStore, startRunId: string | undefined, lookback: number): Promise<{ runId: string; events: Event[] }[]>` —— newest→oldest,最多 `lookback` 个 run;`startRunId` 为 undefined 返回 `[]`;链断/缺失/成环时优雅停止。

- [ ] **Step 1: Write the failing test**

```typescript
// src/__tests__/walkRunWindow.test.ts
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { walkRunWindow } from '../trace/diagnostics/walkRunWindow'
import type { Event } from '../trace/types'

function startedEvent(runId: string, previousRunId?: string): Event {
  return {
    id: `${runId}-start`, runId, type: 'agent.run.started', ts: 0,
    payload: { agentId: 'a', goal: 'g', input: 'i', contextId: 'c',
               ...(previousRunId ? { previousRunId } : {}) },
  } as Event
}

describe('#189 walkRunWindow', () => {
  it('walks the previousRunId chain newest→oldest, bounded by lookback', async () => {
    const store = new MemoryEventStore()
    await store.append(startedEvent('run1'))
    await store.append(startedEvent('run2', 'run1'))
    await store.append(startedEvent('run3', 'run2'))

    const w3 = await walkRunWindow(store, 'run3', 3)
    expect(w3.map(r => r.runId)).toEqual(['run3', 'run2', 'run1'])

    const w2 = await walkRunWindow(store, 'run3', 2)
    expect(w2.map(r => r.runId)).toEqual(['run3', 'run2'])
  })

  it('returns [] for undefined start and stops at a missing run', async () => {
    const store = new MemoryEventStore()
    await store.append(startedEvent('run2', 'run-missing'))
    expect(await walkRunWindow(store, undefined, 3)).toEqual([])
    const w = await walkRunWindow(store, 'run2', 3)
    expect(w.map(r => r.runId)).toEqual(['run2']) // run-missing 无事件,优雅停止
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest src/__tests__/walkRunWindow.test.ts`
Expected: FAIL —— `Cannot find module '../trace/diagnostics/walkRunWindow'`

- [ ] **Step 3: Implement walkRunWindow**

```typescript
// src/trace/diagnostics/walkRunWindow.ts
import type { IEventStore } from '../EventStore.js'
import type { Event, AgentRunStartedPayload } from '../types.js'

/**
 * #189: walk the session run-chain newest→oldest from `startRunId`, bounded to
 * `lookback` runs. Mirrors Milkie.getSessionHistory's walk (each run's
 * agent.run.started.previousRunId links to the prior run) but bounded, so a
 * self-explain tool can read "my last N turns" without an external index.
 * Stops on undefined start, missing run (no events), or a cycle.
 */
export async function walkRunWindow(
  eventStore: IEventStore,
  startRunId: string | undefined,
  lookback: number,
): Promise<{ runId: string; events: Event[] }[]> {
  const out: { runId: string; events: Event[] }[] = []
  const seen = new Set<string>()
  let runId = startRunId
  while (runId && !seen.has(runId) && out.length < lookback) {
    seen.add(runId)
    const events = await eventStore.readByRunId(runId)
    if (events.length === 0) break
    out.push({ runId, events })
    const started = events.find(e => e.type === 'agent.run.started')
    runId = (started?.payload as AgentRunStartedPayload | undefined)?.previousRunId
  }
  return out
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx jest src/__tests__/walkRunWindow.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/trace/diagnostics/walkRunWindow.ts src/__tests__/walkRunWindow.test.ts
git commit -m "feat(#189): bounded session run-chain walk (walkRunWindow)"
```

---

### Task 3: ①Observable —— `get_execution` 加自视图(runId 可选 + lookback)

**Files:**
- Modify: `src/tools/trace.ts`
- Test: `src/__tests__/traceTools.selfView.test.ts`

**Interfaces:**
- Consumes: `walkRunWindow`(Task 2)、`ToolContext.previousRunId`(Task 1)、`buildExecutionProjection`(已存在)。
- Produces: `get_execution` 行为扩展 —— 入参 `{ runId?: string, lookback?: number }`;`runId` 给出时返回原 `ExecutionProjection`(诊断路径,含 prompt,不变);`runId` 省略时返回 `{ turns: { runId: string; toolSteps: ToolStep[]; llmStepCount: number }[] }`(自视图,丢 prompt/response 正文)。

- [ ] **Step 1: Write the failing test**

```typescript
// src/__tests__/traceTools.selfView.test.ts
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { makeTraceTools } from '../tools/trace'
import type { Event } from '../trace/types'

function startedEvent(runId: string, previousRunId?: string): Event {
  return { id: `${runId}-s`, runId, type: 'agent.run.started', ts: 0,
    payload: { agentId: 'a', goal: 'g', input: 'i', contextId: 'c',
               ...(previousRunId ? { previousRunId } : {}) } } as Event
}
function llmReq(runId: string): Event {
  return { id: `${runId}-llm`, runId, type: 'llm.requested', ts: 1,
    payload: { requestHash: 'h1', request: { system: 'SECRET PROMPT', messages: [{}], tools: [] } } } as Event
}
function toolReq(runId: string, name: string): Event {
  return { id: `${runId}-tr`, runId, type: 'tool.requested', ts: 2,
    payload: { requestHash: 'h2', toolName: name, input: { query: 'q' } } } as Event
}
function toolResp(runId: string): Event {
  return { id: `${runId}-tp`, runId, type: 'tool.responded', ts: 3,
    payload: { requestHash: 'h2', output: { hits: 37 } } } as Event
}

describe('#189 get_execution self view', () => {
  it('runId omitted → windowed tool steps, no prompt bodies', async () => {
    const store = new MemoryEventStore()
    for (const e of [startedEvent('run1'), llmReq('run1'), toolReq('run1', 'fetch_news'), toolResp('run1')]) await store.append(e)
    const tool = makeTraceTools(store).find(t => t.name === 'get_execution')!
    const res = await tool.handler({}, { previousRunId: 'run1' } as never) as { turns: { runId: string; toolSteps: unknown[]; llmStepCount: number }[] }
    expect(res.turns).toHaveLength(1)
    expect(res.turns[0].runId).toBe('run1')
    expect(res.turns[0].toolSteps).toHaveLength(1)
    expect(res.turns[0].llmStepCount).toBe(1)
    expect(JSON.stringify(res.turns[0])).not.toContain('SECRET PROMPT')
  })

  it('explicit runId → full projection unchanged (diagnoser path)', async () => {
    const store = new MemoryEventStore()
    for (const e of [startedEvent('run1'), llmReq('run1')]) await store.append(e)
    const tool = makeTraceTools(store).find(t => t.name === 'get_execution')!
    const res = await tool.handler({ runId: 'run1' }, {} as never) as { steps: unknown[] }
    expect(res.steps).toBeDefined()
    expect(JSON.stringify(res)).toContain('SECRET PROMPT') // 诊断路径保留全量
  })

  it('no previousRunId (first turn) → empty turns', async () => {
    const store = new MemoryEventStore()
    const tool = makeTraceTools(store).find(t => t.name === 'get_execution')!
    const res = await tool.handler({}, {} as never) as { turns: unknown[] }
    expect(res.turns).toEqual([])
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest src/__tests__/traceTools.selfView.test.ts`
Expected: FAIL —— 自视图分支不存在(`res.turns` undefined)。

- [ ] **Step 3: Implement self view in get_execution**

`src/tools/trace.ts`:顶部加 import:
```typescript
import { walkRunWindow } from '../trace/diagnostics/walkRunWindow.js'
import type { ToolContext } from '../types/tool.js'
import type { ExecutionStep, ToolStep } from '../trace/diagnostics/buildExecutionProjection.js'
```
把 `get_execution` 整体替换为(保留诊断路径,新增自视图):
```typescript
  const LOOKBACK_DEFAULT = 3
  const LOOKBACK_MAX = 10

  /** Self view: drop llm step prompt/response bodies, keep tool steps. */
  function selfShape(steps: ExecutionStep[]): { toolSteps: ToolStep[]; llmStepCount: number } {
    const toolSteps = steps.filter(s => s.kind === 'tool' && s.tool).map(s => s.tool!) as ToolStep[]
    const llmStepCount = steps.filter(s => s.kind === 'llm').length
    return { toolSteps, llmStepCount }
  }

  async function regionContentFor(events: Awaited<ReturnType<typeof eventStore.readByRunId>>) {
    const regionContent = new Map<string, string>()
    if (objectStore) {
      for (const h of regionReuseCounts(events).keys()) {
        const c = await objectStore.getCanonical(h)
        if (c !== undefined) regionContent.set(h, c)
      }
    }
    return regionContent
  }

  const get_execution: ToolDefinition = {
    name: 'get_execution',
    description: '取执行投影:步骤序列(LLM/工具调用、工具 query、命中证据、region 组成)。' +
      '诊断:传 { runId } 取该 run 全量投影(steps)。自溯源:不传 runId,取自己最近 N 轮(默认 3)的工具步骤摘要(turns;不含 prompt 正文),可加 { lookback }。',
    inputSchema: { type: 'object', properties: {
      runId:    { type: 'string' },
      lookback: { type: 'number', description: '自溯源回看的轮数,默认 3,上限 10' },
    } },
    handler: async (input, ctx) => {
      const { runId, lookback } = (input ?? {}) as { runId?: string; lookback?: number }
      if (runId) {
        const events = await eventStore.readByRunId(runId)
        return buildExecutionProjection(events, { regionContent: await regionContentFor(events) })
      }
      const n = Math.max(1, Math.min(LOOKBACK_MAX, lookback ?? LOOKBACK_DEFAULT))
      const window = await walkRunWindow(eventStore, (ctx as ToolContext | undefined)?.previousRunId, n)
      const turns = []
      for (const { runId: rid, events } of window) {
        const proj = buildExecutionProjection(events, { regionContent: await regionContentFor(events) })
        turns.push({ runId: rid, ...selfShape(proj.steps) })
      }
      return { turns }
    },
  }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx jest src/__tests__/traceTools.selfView.test.ts`
Expected: PASS（三个用例全绿）

- [ ] **Step 5: Typecheck + commit**

```bash
npm run build
git add src/tools/trace.ts src/__tests__/traceTools.selfView.test.ts
git commit -m "feat(#189): get_execution self view (windowed, prompt-stripped) — ①Observable"
```

---

### Task 4: `buildLineageProjection` —— claim→源 解析(纯函数)

**Files:**
- Create: `src/trace/diagnostics/buildLineageProjection.ts`
- Test: `src/__tests__/buildLineageProjection.test.ts`

**Interfaces:**
- Consumes: `object.created`(`ObjectCreatedPayload {objectId,type,meta}`)、`relation.created`(`RelationCreatedPayload {type:'cites',fromObjectId,toObjectId}`),`src/trace/types.ts:288-311`。claim 对象为 `type:'claim'`、`meta.text` 为结论文本(见 `src/tools/lineage.ts` citeHandler)。
- Produces: `resolveClaimSources(events: Event[], query?: string): { claim: string; sources: { objectId: string; type: string; meta?: Record<string, unknown> }[] }[]` —— query 为子串过滤 claim.text;省略则返回全部 claim。

- [ ] **Step 1: Write the failing test**

```typescript
// src/__tests__/buildLineageProjection.test.ts
import { resolveClaimSources } from '../trace/diagnostics/buildLineageProjection'
import type { Event } from '../trace/types'

function objCreated(objectId: string, type: string, meta?: Record<string, unknown>): Event {
  return { id: `${objectId}-c`, runId: 'r', type: 'object.created', ts: 0,
    payload: { objectId, type, producerEventId: 'p', ...(meta ? { meta } : {}) } } as Event
}
function citesRel(from: string, to: string): Event {
  return { id: `${from}-${to}-rel`, runId: 'r', type: 'relation.created', ts: 0,
    payload: { relationId: `${from}-${to}`, type: 'cites', fromObjectId: from, toObjectId: to, causedByEventId: 'x' } } as Event
}

describe('#189 resolveClaimSources', () => {
  const events: Event[] = [
    objCreated('src1', 'passage', { file: 'news.json', source: 'reuters' }),
    objCreated('claim1', 'claim', { text: '实时抓取 228 条' }),
    citesRel('claim1', 'src1'),
  ]
  it('matches a claim by query substring and returns its cited sources', () => {
    const r = resolveClaimSources(events, '228')
    expect(r).toHaveLength(1)
    expect(r[0].claim).toContain('228')
    expect(r[0].sources).toEqual([{ objectId: 'src1', type: 'passage', meta: { file: 'news.json', source: 'reuters' } }])
  })
  it('returns [] for a non-matching query', () => {
    expect(resolveClaimSources(events, '999')).toEqual([])
  })
  it('returns all claims when query omitted', () => {
    expect(resolveClaimSources(events)).toHaveLength(1)
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest src/__tests__/buildLineageProjection.test.ts`
Expected: FAIL —— 模块不存在。

- [ ] **Step 3: Implement resolveClaimSources**

```typescript
// src/trace/diagnostics/buildLineageProjection.ts
import type { Event, ObjectCreatedPayload, RelationCreatedPayload } from '../types.js'

export interface ClaimLineage {
  claim:   string
  sources: { objectId: string; type: string; meta?: Record<string, unknown> }[]
}

/**
 * #189 ③Lineage read: fold a run's object.created / relation.created into
 * claim→source attributions. Pure (no IO). A claim is an object of type 'claim'
 * whose meta.text holds the conclusion (minted by the `cite` tool); a 'cites'
 * relation points claim→source. `query` filters claims by text substring; omit
 * to return all. Walks the explicit cite graph, NOT event.causedBy.
 */
export function resolveClaimSources(events: Event[], query?: string): ClaimLineage[] {
  const objects = new Map<string, ObjectCreatedPayload>()
  const cites: RelationCreatedPayload[] = []
  for (const e of events) {
    if (e.type === 'object.created') {
      const p = e.payload as ObjectCreatedPayload
      objects.set(p.objectId, p)
    } else if (e.type === 'relation.created') {
      const p = e.payload as RelationCreatedPayload
      if (p.type === 'cites') cites.push(p)
    }
  }
  const out: ClaimLineage[] = []
  for (const [objectId, obj] of objects) {
    if (obj.type !== 'claim') continue
    const text = String(obj.meta?.text ?? '')
    if (query && !text.includes(query)) continue
    const sources = cites
      .filter(c => c.fromObjectId === objectId)
      .map(c => objects.get(c.toObjectId))
      .filter((s): s is ObjectCreatedPayload => !!s)
      .map(s => ({ objectId: s.objectId, type: s.type, ...(s.meta ? { meta: s.meta } : {}) }))
    out.push({ claim: text, sources })
  }
  return out
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx jest src/__tests__/buildLineageProjection.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/trace/diagnostics/buildLineageProjection.ts src/__tests__/buildLineageProjection.test.ts
git commit -m "feat(#189): resolveClaimSources — claim→source over cite graph"
```

---

### Task 5: ③Lineage —— 新增 `get_lineage` tool(窗口 + query)

**Files:**
- Modify: `src/tools/trace.ts`
- Test: `src/__tests__/traceTools.lineage.test.ts`

**Interfaces:**
- Consumes: `walkRunWindow`(Task 2)、`resolveClaimSources`(Task 4)、`ToolContext.previousRunId`(Task 1)。
- Produces: `makeTraceTools(...)` 返回数组多一个 `get_lineage` —— 入参 `{ runId?: string; lookback?: number; query?: string }`;返回 `{ matches: { runId: string; claim: string; sources: {...}[] }[] }`。`runId` 给出→单 run;省略→从 `ctx.previousRunId` 起的窗口。

- [ ] **Step 1: Write the failing test**

```typescript
// src/__tests__/traceTools.lineage.test.ts
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { makeTraceTools } from '../tools/trace'
import type { Event } from '../trace/types'

function startedEvent(runId: string, previousRunId?: string): Event {
  return { id: `${runId}-s`, runId, type: 'agent.run.started', ts: 0,
    payload: { agentId: 'a', goal: 'g', input: 'i', contextId: 'c',
               ...(previousRunId ? { previousRunId } : {}) } } as Event
}
function objCreated(runId: string, objectId: string, type: string, meta?: Record<string, unknown>): Event {
  return { id: `${objectId}-c`, runId, type: 'object.created', ts: 1,
    payload: { objectId, type, producerEventId: 'p', ...(meta ? { meta } : {}) } } as Event
}
function citesRel(runId: string, from: string, to: string): Event {
  return { id: `${from}-${to}`, runId, type: 'relation.created', ts: 2,
    payload: { relationId: `${from}-${to}`, type: 'cites', fromObjectId: from, toObjectId: to, causedByEventId: 'x' } } as Event
}

describe('#189 get_lineage', () => {
  it('finds a claim in an earlier window run by query, tagged with its runId', async () => {
    const store = new MemoryEventStore()
    // run1 produced the cited number; run2 is the immediately previous run
    for (const e of [startedEvent('run1'),
      objCreated('run1', 'src1', 'passage', { source: 'reuters' }),
      objCreated('run1', 'claim1', 'claim', { text: '抓取 228 条' }),
      citesRel('run1', 'claim1', 'src1')]) await store.append(e)
    await store.append(startedEvent('run2', 'run1'))

    const tool = makeTraceTools(store).find(t => t.name === 'get_lineage')!
    const res = await tool.handler({ query: '228', lookback: 3 }, { previousRunId: 'run2' } as never) as
      { matches: { runId: string; claim: string; sources: { objectId: string }[] }[] }
    expect(res.matches).toHaveLength(1)
    expect(res.matches[0].runId).toBe('run1')
    expect(res.matches[0].sources[0].objectId).toBe('src1')
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest src/__tests__/traceTools.lineage.test.ts`
Expected: FAIL —— `get_lineage` 不存在(`.find` 返回 undefined)。

- [ ] **Step 3: Implement get_lineage and add to makeTraceTools**

`src/tools/trace.ts`:加 import:
```typescript
import { resolveClaimSources } from '../trace/diagnostics/buildLineageProjection.js'
```
在 `return [get_run_io, get_execution]` 之前定义:
```typescript
  const get_lineage: ToolDefinition = {
    name: 'get_lineage',
    description: '溯源:某条结论/数字引用了哪条源。传 { query } 按结论文本子串匹配(如对话里出现的数字),' +
      '默认在自己最近 N 轮(lookback,默认 3)里搜;也可传 { runId } 限定单轮。返回 matches:{ runId, claim, sources }。',
    inputSchema: { type: 'object', properties: {
      runId:    { type: 'string' },
      lookback: { type: 'number', description: '回看轮数,默认 3,上限 10' },
      query:    { type: 'string', description: '要溯源的结论/数字文本' },
    } },
    handler: async (input, ctx) => {
      const { runId, lookback, query } = (input ?? {}) as { runId?: string; lookback?: number; query?: string }
      const window = runId
        ? [{ runId, events: await eventStore.readByRunId(runId) }]
        : await walkRunWindow(eventStore, (ctx as ToolContext | undefined)?.previousRunId,
            Math.max(1, Math.min(LOOKBACK_MAX, lookback ?? LOOKBACK_DEFAULT)))
      const matches = []
      for (const { runId: rid, events } of window) {
        for (const m of resolveClaimSources(events, query)) {
          matches.push({ runId: rid, claim: m.claim, sources: m.sources })
        }
      }
      return { matches }
    },
  }
```
把返回改为:
```typescript
  return [get_run_io, get_execution, get_lineage]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx jest src/__tests__/traceTools.lineage.test.ts`
Expected: PASS

- [ ] **Step 5: Typecheck + commit**

```bash
npm run build
git add src/tools/trace.ts src/__tests__/traceTools.lineage.test.ts
git commit -m "feat(#189): get_lineage tool (windowed claim→source) — ③Lineage"
```

---

### Task 6: 回归 + 收尾

**Files:** 无新增。

- [ ] **Step 1: 跑全量相关测试**

Run: `npx jest src/__tests__/AgentRuntime.previousRunId.test.ts src/__tests__/walkRunWindow.test.ts src/__tests__/traceTools.selfView.test.ts src/__tests__/buildLineageProjection.test.ts src/__tests__/traceTools.lineage.test.ts src/__tests__/traceTools.test.ts src/__tests__/AgentRuntime.test.ts`
Expected: 全 PASS（含既有 `traceTools.test.ts` / `AgentRuntime.test.ts` 未被破坏)。

- [ ] **Step 2: 全量类型检查**

Run: `npm run build`
Expected: 无 TS 错误。

- [ ] **Step 3: 更新 #189 实现状态(可选)**

在 #189 加一条 comment:MVP(D1 + ①get_execution 自视图 + ③get_lineage + walkRunWindow + resolveClaimSources)已实现,附测试文件路径。

```bash
git log --oneline -6   # 确认 6 个 task 的提交都在
```

## Self-Review

**1. Spec coverage(对 #189 §七 MVP):**
- D1 注入 `previousRunId` → Task 1 ✅
- ①Observable 复用 get_execution + runId 可选 + lookback + 自视图丢 prompt → Task 3 ✅
- ③Lineage 新增引用图遍历(非 walkCausedBy)按 claim 跨窗口 → Task 4(纯函数)+ Task 5(tool)✅
- 最近 N 轮有界窗口(默认 3,上限 10)→ Task 2(walkRunWindow)+ Task 3/5 clamp ✅
- 查询式(event log 进 handler 进程内存,不进 context;返回摘要)→ Task 3 selfShape / Task 5 matches ✅
- 不造 self 专用 tool(复用 get_execution + 通用 get_lineage)✅
- 后置项(②Diagnosable / Replay/Fork/Diff / 无界回溯 / 完整 D3)→ 不在任何 task,符合预期 ✅

**2. Placeholder scan:** 无 TBD/TODO;每个 code step 给了完整代码与命令。

**3. Type consistency:**
- `walkRunWindow(eventStore, startRunId, lookback)` 返回 `{runId,events}[]` —— Task 3/5 一致消费。
- `resolveClaimSources(events, query?)` 返回 `{claim, sources}[]` —— Task 5 映射为 `{runId, claim, sources}`。
- `ToolContext.previousRunId`(Task 1)—— Task 3/5 读取一致。
- `selfShape` 返回 `{toolSteps, llmStepCount}`,`ToolStep` 来自 buildExecutionProjection 导出。

**已知边界(写入 #189,非本计划缺陷):** ①③ 的语义粒度 = 抓取被建模为 milkie tool 的粒度;host 若用单个 run_command 跑黑盒脚本则拆不出 per-源(alfred gated on #87)。
