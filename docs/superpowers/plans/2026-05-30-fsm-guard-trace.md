# FSM Guard 评估留痕 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让工具在触发 FSM 转移时可选地自报"判断依据"(guard evaluation),依据被捕获进 `fsm.transition` 事件、可在 trace 中排查,且不破坏 replay 确定性。

**Architecture:** 选项 A(内嵌)。新增 `GuardEvaluation` 类型;`ctx.emit` 加第三参把依据透传到 `FSMEngine` → `onTransition` 回调 → `setupFSMCallbacks` 写进 `fsm.transition.payload.guardEvaluations`。框架只搬运,不产生/不求值 guard。复用 #30 的写入路径(绕过 IOPort,零新增 IOPort 调用 → replay 不破)。

**Tech Stack:** TypeScript、Vitest/Jest 风格测试(`describe/it/expect`)、现有 `Milkie`/`AgentRuntime`/`FSMEngine`/`MemoryEventStore` 测试设施。

参考 spec:`docs/superpowers/specs/2026-05-30-fsm-guard-trace-design.md`

---

## File Structure

| 文件 | 责任 | 改动 |
|---|---|---|
| `src/trace/types.ts` | 事件类型契约 | +`GuardEvaluation`;`FsmTransitionPayload` 加 `guardEvaluations?` |
| `src/fsm/FSMEngine.ts` | FSM 引擎 | `FSMEvent` 加 `guard?`;`emitEvent` 加第三参并归一化为数组存入 `pendingEvent` |
| `src/types/tool.ts` | 工具上下文契约 | `ToolContext.emit` 签名加第三参 `guard?` |
| `src/runtime/AgentRuntime.ts` | 运行时 | `buildToolContext` 透传 guard;两个 emit 闭包传 guard;`setupFSMCallbacks` 写 `guardEvaluations` |
| `tests/e2e/s-011-*.e2e.test.ts` | 真实 producer | 3 个工具补上报 + live 断言 |
| `src/__tests__/FSMEngine.test.ts` | 单测 | guard 透传 |
| `src/__tests__/AgentRuntime.test.ts` | 单测 | 捕获进 fsm.transition |
| `src/__tests__/Replay.test.ts` | 单测 | replay 确定性 |
| (可选/建议延后) `src/trace/render/tree.ts` `html.ts` | HTML 时间线 | 新增 fsm.transition 渲染分支 + guard |

> **说明**:`fsm.transition` 当前**根本未在 HTML 时间线(`tree.ts`)渲染**。把它(连同 guard)加进时间线是一个独立的小特性(#21 渲染缺口),因此放在 **Task 5,建议延后到 #33 或单独 issue**。guard 数据本身在事件 payload 里,已满足 #31 验收("事件流可还原"),可经原始事件日志 / CLI trace 查看。

---

## Task 1: GuardEvaluation 类型 + FSMEngine 透传 guard

**Files:**
- Modify: `src/trace/types.ts`(`FsmTransitionPayload` 附近,约 `:192`)
- Modify: `src/fsm/FSMEngine.ts:4-14`(`FSMEvent`)与 `:55-61`(`emitEvent`)
- Test: `src/__tests__/FSMEngine.test.ts`

- [ ] **Step 1: 写失败测试**

在 `src/__tests__/FSMEngine.test.ts` 的 `describe('ctx.emit() / processPendingEvent()')` 块内追加:

```ts
it('carries guard evaluations through to the onTransition callback', () => {
  const fsm = new FSMEngine(routingFSM)   // classify, on INTENT_A: handle_a
  let received: import('../fsm/FSMEngine').FSMEvent | undefined
  fsm.onTransitionCallback((_from, _to, event) => { received = event })

  fsm.emitEvent('INTENT_A', undefined, {
    guardId: 'g', result: 'INTENT_A', contextSlice: { s: 1 },
  })
  fsm.processPendingEvent()

  expect(received?.guard).toEqual([
    { guardId: 'g', result: 'INTENT_A', contextSlice: { s: 1 } },
  ])
})

it('normalizes a guard array argument as-is', () => {
  const fsm = new FSMEngine(routingFSM)
  let received: import('../fsm/FSMEngine').FSMEvent | undefined
  fsm.onTransitionCallback((_f, _t, e) => { received = e })
  const arr = [{ guardId: 'a', result: 1, contextSlice: {} }, { guardId: 'b', result: 2, contextSlice: {} }]
  fsm.emitEvent('INTENT_A', undefined, arr)
  fsm.processPendingEvent()
  expect(received?.guard).toEqual(arr)
})
```

- [ ] **Step 2: 运行,确认失败**

Run: `npx vitest run src/__tests__/FSMEngine.test.ts -t "carries guard"`
Expected: FAIL —— `emitEvent` 不接受第三参 / `event.guard` 为 undefined（类型错误或断言失败）。

- [ ] **Step 3: 加类型**(`src/trace/types.ts`,在 `FsmTransitionPayload` 之前)

```ts
export interface GuardEvaluation {
  /** 判断标识,如 'intent-threshold'。 */
  guardId:      string
  /** 判断结果:产出的事件名或布尔/任意值。 */
  result:       unknown
  /** 决定结果真假的最小输入切片(约定最小化,框架不强制)。 */
  contextSlice: unknown
}
```

并在 `FsmTransitionPayload` 末尾加字段:

```ts
export interface FsmTransitionPayload {
  from: string
  to:   string
  trigger: {
    domain: FsmEventDomain
    name:   string
    payload?: unknown
  }
  /** #31:本次转移背后的判断依据(工具自报,可选)。 */
  guardEvaluations?: GuardEvaluation[]
}
```

- [ ] **Step 4: 改 FSMEngine**(`src/fsm/FSMEngine.ts`)

顶部 import 处加:

```ts
import type { FsmEventDomain, GuardEvaluation } from '../trace/types.js'
```

`FSMEvent` 接口加字段(在 `domain?` 之后):

```ts
  /** #31:触发本次转移的判断依据(由 ctx.emit 第三参带入)。 */
  guard?: GuardEvaluation[]
```

`emitEvent` 改签名与实现:

```ts
emitEvent(
  event: string,
  payload?: unknown,
  guard?: GuardEvaluation | GuardEvaluation[],
): void {
  if (this.pendingEvent) {
    // First event wins within a single tool execution
    return
  }
  this.pendingEvent = {
    name:    event,
    payload,
    domain:  'business',
    ...(guard ? { guard: Array.isArray(guard) ? guard : [guard] } : {}),
  }
}
```

- [ ] **Step 5: 运行,确认通过**

Run: `npx vitest run src/__tests__/FSMEngine.test.ts`
Expected: PASS(含原有用例,无回归)。

- [ ] **Step 6: 提交**

```bash
git add src/trace/types.ts src/fsm/FSMEngine.ts src/__tests__/FSMEngine.test.ts
git commit -m "feat(#31): GuardEvaluation type + FSMEngine carries guard to onTransition"
```

---

## Task 2: ctx.emit 第三参 + 捕获进 fsm.transition

**Files:**
- Modify: `src/types/tool.ts`(`ToolContext.emit`)
- Modify: `src/runtime/AgentRuntime.ts`(`buildToolContext` 约 `:815`;两个 emit 闭包 `:1017` 与 `:1092`;`setupFSMCallbacks` `:170-211`)
- Test: `src/__tests__/AgentRuntime.test.ts`

- [ ] **Step 1: 写失败测试**

在 `src/__tests__/AgentRuntime.test.ts` 末尾合适 `describe` 内追加(复用文件已有的 `routingConfig`、`SequentialGateway`、`toolCallResponse`、`InMemoryRecorder`、`DefaultIOPort`、`MemoryEventStore`、`MemoryStore` 导入):

```ts
describe('#31 guard evaluation capture', () => {
  it('writes guardEvaluations onto the fsm.transition event', async () => {
    const eventStore = new MemoryEventStore()
    const guardTool: ToolDefinition = {
      name:        'classify_intent',
      description: 'classify',
      inputSchema: { type: 'object', properties: {} },
      handler:     async (_input, ctx) => {
        ctx.emit('INTENT_DONE', undefined, {
          guardId: 'intent-threshold', result: 'INTENT_DONE',
          contextSlice: { confidence: 0.9, threshold: 0.75 },
        })
        return { ok: true }
      },
    }
    const runtime = new AgentRuntime({
      config:     routingConfig,
      goal:       'classify',
      input:      'hello',
      stateStore: new MemoryStore(),
      recorder:   new InMemoryRecorder(),
      ioPort:     new DefaultIOPort(new SequentialGateway([
        toolCallResponse('tc-1', 'classify_intent', {}),
      ])),
      extraTools: [guardTool],
      eventStore,
    })

    const result = await runtime.run('hello')
    const events = await eventStore.readByRunId(result.agentRunId)
    const transitions = events.filter(e => e.type === 'fsm.transition')
    const withGuard = transitions.find(
      t => (t.payload as import('../trace/types').FsmTransitionPayload).guardEvaluations,
    )
    expect((withGuard!.payload as import('../trace/types').FsmTransitionPayload).guardEvaluations)
      .toEqual([{ guardId: 'intent-threshold', result: 'INTENT_DONE', contextSlice: { confidence: 0.9, threshold: 0.75 } }])
  })

  it('omits guardEvaluations when the tool does not report one', async () => {
    const eventStore = new MemoryEventStore()
    const plainTool: ToolDefinition = {
      name:        'classify_intent',
      description: 'classify',
      inputSchema: { type: 'object', properties: {} },
      handler:     async (_input, ctx) => { ctx.emit('INTENT_DONE'); return {} },
    }
    const runtime = new AgentRuntime({
      config:     routingConfig,
      goal:       'classify',
      input:      'hello',
      stateStore: new MemoryStore(),
      recorder:   new InMemoryRecorder(),
      ioPort:     new DefaultIOPort(new SequentialGateway([
        toolCallResponse('tc-1', 'classify_intent', {}),
      ])),
      extraTools: [plainTool],
      eventStore,
    })
    const result = await runtime.run('hello')
    const events = await eventStore.readByRunId(result.agentRunId)
    const transition = events.find(
      e => e.type === 'fsm.transition'
        && (e.payload as import('../trace/types').FsmTransitionPayload).trigger.name === 'INTENT_DONE',
    )
    expect((transition!.payload as import('../trace/types').FsmTransitionPayload).guardEvaluations)
      .toBeUndefined()
  })
})
```

- [ ] **Step 2: 运行,确认失败**

Run: `npx vitest run src/__tests__/AgentRuntime.test.ts -t "#31 guard"`
Expected: FAIL —— `ctx.emit` 不接受第三参（类型错误）/ `guardEvaluations` 为 undefined。

- [ ] **Step 3: 改 ToolContext.emit 签名**（`src/types/tool.ts`）

顶部 import 加:

```ts
import type { GuardEvaluation } from '../trace/types.js'
```

`emit` 字段改为:

```ts
  emit: (event: string, payload?: unknown, guard?: GuardEvaluation | GuardEvaluation[]) => void
```

- [ ] **Step 4: 改 AgentRuntime 透传与写入**（`src/runtime/AgentRuntime.ts`）

`buildToolContext` 的形参类型改为带 guard:

```ts
private buildToolContext(
  emitFn: (event: string, payload?: unknown, guard?: import('../trace/types.js').GuardEvaluation | import('../trace/types.js').GuardEvaluation[]) => void,
): ToolContext {
```

两个构造 ctx 的闭包(`runActionState` 约 `:1017`、`executeSingleTool` 约 `:1092`)各改为透传第三参:

```ts
const ctx = this.buildToolContext((event, payload, guard) => {
  this.fsm.emitEvent(event, payload, guard)
})
```

`setupFSMCallbacks` 里组装 `fsm.transition` payload 处(`:198-208`),在 `trigger` 之后加入 guardEvaluations:

```ts
payload: {
  from,
  to,
  trigger: {
    domain:  event.domain ?? 'business',
    name:    event.name,
    ...(event.payload !== undefined ? { payload: event.payload } : {}),
  },
  ...(event.guard?.length ? { guardEvaluations: event.guard } : {}),
},
```

- [ ] **Step 5: 运行,确认通过**

Run: `npx vitest run src/__tests__/AgentRuntime.test.ts`
Expected: PASS（两个新用例 + 无回归）。

- [ ] **Step 6: 提交**

```bash
git add src/types/tool.ts src/runtime/AgentRuntime.ts src/__tests__/AgentRuntime.test.ts
git commit -m "feat(#31): ctx.emit 3rd arg captures guardEvaluations into fsm.transition"
```

---

## Task 3: Replay 确定性(byte-identical 验收)

**Files:**
- Test: `src/__tests__/Replay.test.ts`

- [ ] **Step 1: 写测试**

在 `src/__tests__/Replay.test.ts` 的 `describe('Milkie.replay')` 内追加。复用文件已有 `Milkie`/`MemoryStore`/`MemoryEventStore`/`SequentialGateway`/`text`/`toolCallResponse`:

```ts
it('replays a run whose tool reported a guard, identically and with zero gateway calls (#31)', async () => {
  const agentWithGuard: AgentConfig = {
    agentId: 'guard-agent', version: '0.0.0', systemPrompt: 'sys',
    fsm: { states: [
      { name: 's0', type: 'llm', tools: ['classify_intent'], on: { INTENT_DONE: 'end' } },
      { name: 'end', type: 'action', terminal: true },
    ] },
    model: { provider: 'stub', model: 'stub', adapter: 'stub' },
  }
  const guardTool: ToolDefinition = {
    name: 'classify_intent', description: 'c', inputSchema: { type: 'object', properties: {} },
    handler: async (_i, ctx) => {
      ctx.emit('INTENT_DONE', undefined, { guardId: 'g', result: 'INTENT_DONE', contextSlice: { confidence: 0.9 } })
      return {}
    },
  }
  // record
  const store = new MemoryEventStore()
  const recordGateway = new SequentialGateway([
    toolCallResponse('tc-1', 'classify_intent', {}),
    text('done'),
  ])
  const recordMilkie = new Milkie({ stateStore: new MemoryStore(), gateway: recordGateway, eventStore: store, tools: [guardTool] })
  recordMilkie.registerAgent(agentWithGuard)
  const original = await recordMilkie.invoke({ agentId: 'guard-agent', goal: 'g', input: 'i' })

  // assert guard was recorded
  const recorded = await store.readByRunId(original.agentRunId)
  const trans = recorded.find(e => e.type === 'fsm.transition'
    && (e.payload as import('../trace/types').FsmTransitionPayload).guardEvaluations)
  expect(trans).toBeDefined()

  // replay
  const replayGateway = new SequentialGateway([text('this would be wrong')])
  const replayMilkie = new Milkie({ stateStore: new MemoryStore(), gateway: replayGateway, eventStore: store, tools: [guardTool] })
  replayMilkie.registerAgent(agentWithGuard)
  const replayed = await replayMilkie.replay(original.agentRunId)

  expect(replayed.status).toBe(original.status)
  expect(replayed.output).toBe(original.output)
  expect(replayGateway.callCount).toBe(0)   // cache served everything → determinism intact
})
```

- [ ] **Step 2: 运行,确认通过**

Run: `npx vitest run src/__tests__/Replay.test.ts -t "#31"`
Expected: PASS。原理:guardEvaluations 仅是 `fsm.transition`(绕过 IOPort)的额外 payload,不新增任何 `ioPort.uuid()/now()` 调用,缓存序列不变,replay 命中、gateway 调用为 0。
若 FAIL 且 `callCount > 0` 或 divergence:说明 guard 写入误用了 IOPort —— 回查 Task 2 Step 4,确认仍在 `setupFSMCallbacks` 的 `uuidv4()/Date.now()` 直接生成分支内,未引入 `this.ioPort.*`。

- [ ] **Step 3: 提交**

```bash
git add src/__tests__/Replay.test.ts
git commit -m "test(#31): replay determinism preserved with guardEvaluations"
```

---

## Task 4: s-011 真实 producer 接线 + live 断言

**Files:**
- Modify: `tests/e2e/s-011-multi-state-fsm-intent-routing-and-slot-filling.e2e.test.ts`（`classify_intent` `:42-62`、`collect_slot` `:80` 附近、`confirm_action`）

- [ ] **Step 1: classify_intent 两个 emit 点补上报**

```ts
// confidence < 0.75 分支
ctx.emit('ESCALATE', undefined, {
  guardId: 'intent-threshold', result: 'ESCALATE',
  contextSlice: { intent, confidence, threshold: 0.75 },
})
// 正常分支
ctx.emit(eventMap[intent] ?? 'ESCALATE', undefined, {
  guardId: 'intent-threshold', result: eventMap[intent] ?? 'ESCALATE',
  contextSlice: { intent, confidence, threshold: 0.75 },
})
```

- [ ] **Step 2: collect_slot 补上报**

```ts
if (allFilled) ctx.emit('SLOTS_COMPLETE', undefined, {
  guardId: 'slots-complete', result: 'SLOTS_COMPLETE',
  contextSlice: { filled: Object.keys(slots), required: ['orderId', 'reason', 'preferRefund'] },
})
```

- [ ] **Step 3: confirm_action 补上报**

```ts
ctx.emit(confirmed ? 'USER_CONFIRMED' : 'USER_REJECTED', undefined, {
  guardId: 'user-confirm', result: confirmed ? 'USER_CONFIRMED' : 'USER_REJECTED',
  contextSlice: { confirmed },
})
```

- [ ] **Step 4: 加 live 断言**

在 s-011 已有的 `live('classify_intent 调用意图为 cancellation...')` 用例附近追加一条(复用文件读取 trace 的方式;trace 来自 e2e 的 `Trajectory`/事件,断言对应 `fsm.transition` payload):

```ts
live('classify 的硬转移带 intent-threshold guard 依据', () => {
  const t = trace.events.find(e => e.type === 'fsm.transition'
    && String((e.payload as { trigger: { name: string } }).trigger.name).startsWith('INTENT_'))
  const ge = (t?.payload as import('../../src/trace/types').FsmTransitionPayload | undefined)?.guardEvaluations
  expect(ge?.[0]?.guardId).toBe('intent-threshold')
  expect((ge?.[0]?.contextSlice as { confidence?: number })?.confidence).toBeGreaterThan(0)
})
```

> 若 s-011 现有断言读的是 `Trajectory` span 而非事件日志,改为按该文件既有取事件的方式获取 `fsm.transition` 事件(文件顶部已 import 相关类型);断言内容不变。

- [ ] **Step 5: 运行(需 live token,否则 skip 属正常)**

Run: `VOLCENGINE_TOKEN=... VOLCENGINE_API_BASE=... npx vitest run tests/e2e/s-011-*.e2e.test.ts`
Expected: 有 token → 新断言 PASS;无 token → 整组 skip(预期,确定性验证由 Task 2/3 保证)。

- [ ] **Step 6: 提交**

```bash
git add tests/e2e/s-011-multi-state-fsm-intent-routing-and-slot-filling.e2e.test.ts
git commit -m "test(#31): wire s-011 tools to report guard evaluations + live assertion"
```

---

## Task 5(可选,建议延后): HTML 时间线渲染 fsm.transition + guard

> **决策提示**:`fsm.transition` 当前未在 `tree.ts` 渲染,本任务等于"给时间线新增转移条目"——属 #21 渲染缺口,非 guard 特有。guard 数据已在事件 payload(Task 2 已满足 #31 验收)。**建议延后到 #33(why-this-transition)或单独 issue**;若现在就要在 HTML 里看到,按下列执行。

**Files:**
- Modify: `src/trace/render/tree.ts`(`TimelineEntry` 联合 `:37`;`buildTimelineTree` 分支 `:125` 之后)
- Modify: `src/trace/render/html.ts`(`summaryFor` `:17`、details `:26`、icon `:41`、filter chips `:104`)
- Test: `src/__tests__/render-tree.test.ts`、`src/__tests__/render-html.test.ts`

- [ ] **Step 1: 写失败测试**(`src/__tests__/render-tree.test.ts`,新增用例)

```ts
it('renders an fsm.transition entry carrying guardEvaluations', () => {
  const events: Event[] = [
    { id: 's', runId: 'r1', actor: 'a', type: 'agent.run.started', timestamp: 1,
      payload: { agentId: 'x', goal: 'g', input: 'i', contextId: 'c' } },
    { id: 't', runId: 'r1', actor: 'a', type: 'fsm.transition', timestamp: 2,
      payload: { from: 'classify', to: 'handle_b',
        trigger: { domain: 'business', name: 'INTENT_B' },
        guardEvaluations: [{ guardId: 'intent-threshold', result: 'INTENT_B', contextSlice: { confidence: 0.7 } }] } },
  ]
  const tree = buildTimelineTree(events)
  const entry = tree[0]!.entries.find(e => e.kind === 'fsm')
  expect(entry).toBeDefined()
  expect((entry as { summary: string }).summary).toContain('intent-threshold')
})
```

- [ ] **Step 2: 运行,确认失败**

Run: `npx vitest run src/__tests__/render-tree.test.ts -t "fsm.transition entry"`
Expected: FAIL —— 无 `kind === 'fsm'` 条目。

- [ ] **Step 3: tree.ts 加 FsmEntry 类型与分支**

`TimelineEntry` 联合前新增:

```ts
export interface FsmEntry {
  kind:      'fsm'
  eventId:   string
  eventType: 'fsm.transition'
  timestamp: number
  summary:   string
  payload:   unknown
}
```

并入联合:

```ts
export type TimelineEntry = LlmEntry | ToolEntry | LifecycleEntry | RegionEntry | FsmEntry
```

`buildTimelineTree` 中 `context.boundary.applied` 分支(`:125`)之后追加 `else if`:

```ts
} else if (evt.type === 'fsm.transition') {
  const p = evt.payload as import('../types.js').FsmTransitionPayload
  const guards = p.guardEvaluations?.length
    ? ' [' + p.guardEvaluations.map(g => `${g.guardId}→${String(g.result)}`).join(', ') + ']'
    : ''
  entries.push({
    kind:      'fsm',
    eventId:   evt.id,
    eventType: 'fsm.transition',
    timestamp: evt.timestamp,
    summary:   `${p.from} → ${p.to} (${p.trigger.name})${guards}`,
    payload:   evt.payload,
  })
}
```

> 注:`import('../types.js')` 路径以 `tree.ts` 实际相对位置为准(同目录其他分支若已 import `Event` from `'../types.js'`,复用同一 import,不要重复内联)。

- [ ] **Step 4: html.ts 处理 'fsm' kind**

`summaryFor`(`:17-21`)加一行(`region` 分支后):

```ts
if (entry.kind === 'fsm')       return esc(entry.summary)
```

details 段(`:26-30`)的 `region` 分支并列加:

```ts
} else if (entry.kind === 'fsm') {
  sections.push(JSON.stringify(entry.payload, null, 2))
```

icon(`:41-45`)加:

```ts
: entry.kind === 'fsm' ? '⇒'
```

filter chips(`:104` 附近)加:

```ts
<span class="chip" data-kind="fsm">fsm</span>
```

- [ ] **Step 5: 写 html 渲染测试**(`src/__tests__/render-html.test.ts`,新增用例)

```ts
it('renders fsm.transition with guard summary in html', () => {
  const events: Event[] = [
    e({ id: 's', runId: 'r1', type: 'agent.run.started', timestamp: 1,
        payload: { agentId: 'x', goal: 'g', input: 'i', contextId: 'c' } }),
    e({ id: 't', runId: 'r1', type: 'fsm.transition', timestamp: 2,
        payload: { from: 'classify', to: 'handle_b', trigger: { domain: 'business', name: 'INTENT_B' },
          guardEvaluations: [{ guardId: 'intent-threshold', result: 'INTENT_B', contextSlice: {} }] } }),
  ]
  const html = renderHtml(events)
  expect(html).toContain('intent-threshold')
  expect(html).toContain('data-kind="fsm"')
})
```

- [ ] **Step 6: 运行,确认通过**

Run: `npx vitest run src/__tests__/render-tree.test.ts src/__tests__/render-html.test.ts`
Expected: PASS（含原有用例,无回归）。

- [ ] **Step 7: 提交**

```bash
git add src/trace/render/tree.ts src/trace/render/html.ts src/__tests__/render-tree.test.ts src/__tests__/render-html.test.ts
git commit -m "feat(#31): render fsm.transition + guardEvaluations in HTML timeline"
```

---

## 收尾

- [ ] **全量测试 + 构建**

Run: `npx vitest run && npm run build`
Expected: 全绿。

- [ ] **更新 issue / PR**:开 PR,body 带 `Closes #31`,引用 spec 与本 plan。

---

## Self-Review(plan 作者已核对)

**Spec 覆盖:**
- §4 形状 → Task 1 ✓ §5 上报 API(甲) → Task 2 ✓ §6 捕获 → Task 1+2 ✓ §7 replay → Task 3 ✓ §8 producer → Task 4 ✓ §9 渲染 → Task 5(标注延后)✓ §10 测试 → Task 1–4 测试 ✓ §12 验收 → Task 2(还原)/Task 4(最小化示例)/Task 3(byte-identical)✓

**占位扫描:** 无 TBD/TODO;每个改代码步骤均含完整代码。唯二"以实际为准"提示(tree.ts import 路径、s-011 取事件方式)是因这两处依赖目标文件既有写法,已给出明确判定方法,非内容缺失。

**类型一致性:** `GuardEvaluation{guardId,result,contextSlice}` 在 Task 1 定义,Task 2/3/4/5 引用一致;`FsmTransitionPayload.guardEvaluations?: GuardEvaluation[]` 全程一致;`emitEvent`/`ctx.emit` 第三参 `guard?: GuardEvaluation | GuardEvaluation[]`、归一化为数组,前后一致。
