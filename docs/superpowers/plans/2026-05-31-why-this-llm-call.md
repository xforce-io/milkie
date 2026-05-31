# why-this-llm-call explainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 点 trace 报告里的 `llm.requested` 能读懂"这次 LLM 调用为什么发生"——触发它的 turn 终结事件 + 当时 FSM state + 因果链,纯 event-log 投影,与 #33 平行。

**Architecture:** 新建 `src/trace/diagnostics/`:`fsmStateAt`(折叠 transition 求某时刻 state)+ `explainLlmCall`(核心投影,复用 #33 的 `walkCausedBy`/`summarizeEvent` + #23 的 `contextRefsAt`)。HTML 渲染层给 llm 条目加紧凑 "Why?" 块(复用 #33 的 `.why` 样式 + 事件锚点),与 #26 已有的 "Assembled by" 块并存。零 LLM、零存储、零 schema 变更。

**Tech Stack:** TypeScript;Jest(`npx jest <file>`);现有 `src/trace/diagnostics/{walkCausedBy,summarizeEvent}.ts`、`src/trace/RegionContextView.ts`、`src/trace/render/html.ts`。

参考 spec:`docs/superpowers/specs/2026-05-31-why-this-llm-call-design.md`。**测试运行器是 JEST。**

---

## File Structure

| 文件 | 责任 | 改动 |
|---|---|---|
| `src/trace/diagnostics/fsmStateAt.ts` | 折叠 transition 求某事件时刻的 FSM state | 新建 |
| `src/trace/diagnostics/explainLlmCall.ts` | 核心投影 + `LlmCallExplanation` 类型 | 新建 |
| `src/trace/render/html.ts` | llm 条目渲 "Why?" 块(内联调 explainLlmCall) | 修改 |
| `src/__tests__/fsmStateAt.test.ts` | 单测 | 新建 |
| `src/__tests__/explainLlmCall.test.ts` | 单测 | 新建 |
| `src/__tests__/render-html.test.ts` | llm Why? + 并存断言 | 修改 |

> 测试用手工构造的 `Event[]`(纯投影,合成输入最聚焦、确定性最强、天然无 LLM 调用)。

---

## Task 1: fsmStateAt — 某事件时刻的 FSM state

**Files:**
- Create: `src/trace/diagnostics/fsmStateAt.ts`
- Test: `src/__tests__/fsmStateAt.test.ts`

- [ ] **Step 1: 写失败测试**

```ts
import { fsmStateAt } from '../trace/diagnostics/fsmStateAt'
import type { Event } from '../trace/types'

const ev = (id: string, type: string, payload: unknown): Event =>
  ({ id, runId: 'r1', actor: 'a', type: type as Event['type'], timestamp: 0, payload })
const trans = (id: string, from: string, to: string) =>
  ev(id, 'fsm.transition', { from, to, trigger: { domain: 'business', name: 'E' } })

describe('fsmStateAt', () => {
  it('returns the last transition target before the event', () => {
    const events: Event[] = [
      trans('t1', 's0', 's1'),
      ev('llm', 'llm.requested', {}),
      trans('t2', 's1', 's2'),
    ]
    expect(fsmStateAt(events, 'llm')).toBe('s1')
  })

  it('returns the initial state (first transition from) when no transition precedes the event', () => {
    const events: Event[] = [
      ev('llm', 'llm.requested', {}),
      trans('t1', 's0', 's1'),
    ]
    expect(fsmStateAt(events, 'llm')).toBe('s0')
  })

  it('returns null when there are no transitions at all', () => {
    const events: Event[] = [ev('llm', 'llm.requested', {})]
    expect(fsmStateAt(events, 'llm')).toBeNull()
  })
})
```

- [ ] **Step 2: 运行,确认失败**

Run: `npx jest src/__tests__/fsmStateAt.test.ts`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现**

```ts
import type { Event, FsmTransitionPayload } from '../types.js'

/**
 * The FSM state in effect at `eventId`: fold fsm.transition events strictly
 * before it and take the last `to`. If none precede it, the state is the
 * initial state — revealed by the `from` of the run's first transition.
 * Returns null when the run has no transitions at all. Pure.
 */
export function fsmStateAt(events: Event[], eventId: string): string | null {
  const index = events.findIndex(e => e.id === eventId)
  const cut = index < 0 ? events.length : index
  let state: string | null = null
  for (let i = 0; i < cut; i++) {
    const e = events[i]!
    if (e.type === 'fsm.transition') state = (e.payload as FsmTransitionPayload).to
  }
  if (state !== null) return state
  const first = events.find(e => e.type === 'fsm.transition')
  return first ? (first.payload as FsmTransitionPayload).from : null
}
```

- [ ] **Step 4: 运行,确认通过**

Run: `npx jest src/__tests__/fsmStateAt.test.ts`
Expected: PASS（3 用例）。Also `npx tsc --noEmit` clean.

- [ ] **Step 5: 提交**

```bash
git add src/trace/diagnostics/fsmStateAt.ts src/__tests__/fsmStateAt.test.ts
git commit -m "feat(#34): fsmStateAt — FSM state in effect at a given event"
```

---

## Task 2: explainLlmCall — 核心投影

**Files:**
- Create: `src/trace/diagnostics/explainLlmCall.ts`
- Test: `src/__tests__/explainLlmCall.test.ts`

- [ ] **Step 1: 写失败测试**

```ts
import { explainLlmCall } from '../trace/diagnostics/explainLlmCall'
import type { Event } from '../trace/types'

const ev = (id: string, type: string, payload: unknown, causedBy?: string): Event =>
  ({ id, runId: 'r1', actor: 'a', type: type as Event['type'], timestamp: 0, payload, ...(causedBy ? { causedBy } : {}) })
const region = (id: string, contentHash?: string) =>
  ev(`add-${id}`, 'region.added', { id, target: 'message', section: 's', stability: 'volatile', reason: 'r', ...(contentHash ? { contentHash } : {}) })

function scenario(): Event[] {
  return [
    ev('start', 'agent.run.started', { agentId: 'a', goal: 'g', input: 'i', contextId: 'c' }),
    region('header', 'H1'),
    region('hist', 'H2'),
    ev('treq', 'tool.requested', { toolName: 'search', input: {} }, 'start'),
    ev('tres', 'tool.responded', { toolName: 'search', output: {} }, 'treq'),
    ev('fsm', 'fsm.transition', { from: 'plan', to: 'reflect', trigger: { domain: 'business', name: 'NEXT' } }, 'tres'),
    ev('llm', 'llm.requested', { model: 'm' }, 'tres'),
  ]
}

describe('explainLlmCall', () => {
  it('projects trigger, fsm state, region count, causal chain and summary (no LLM)', () => {
    const exp = explainLlmCall(scenario(), 'llm')
    expect(exp.trigger.causedByEventId).toBe('tres')
    expect(exp.trigger.causedBySummary).toBe('tool.responded(search)')
    expect(exp.fsmState).toBe('reflect')          // after fsm plan→reflect, before the llm call
    expect(exp.regionCount).toBe(2)                // header + hist active
    expect(exp.causalChain.map(c => c.eventId)).toEqual(['llm', 'tres', 'treq', 'start'])
    expect(exp.summary).toContain('reflect')
    expect(exp.summary).toContain('tool.responded(search)')
    expect(exp.summary).toContain('2')
  })

  it('falls back when there is no upstream trigger and no transitions', () => {
    const events: Event[] = [ev('llm', 'llm.requested', { model: 'm' })]
    const exp = explainLlmCall(events, 'llm')
    expect(exp.trigger.causedByEventId).toBeUndefined()
    expect(exp.fsmState).toBeNull()
    expect(exp.regionCount).toBe(0)
    expect(exp.summary).toContain('(无上游)')
  })

  it('throws on a non-llm.requested event id', () => {
    expect(() => explainLlmCall(scenario(), 'fsm')).toThrow(/llm\.requested/)
  })

  it('throws on an unknown event id', () => {
    expect(() => explainLlmCall(scenario(), 'nope')).toThrow(/nope/)
  })
})
```

- [ ] **Step 2: 运行,确认失败**

Run: `npx jest src/__tests__/explainLlmCall.test.ts`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现**

```ts
import type { Event, EventKind } from '../types.js'
import { walkCausedBy } from './walkCausedBy.js'
import { summarizeEvent } from './summarizeEvent.js'
import { fsmStateAt } from './fsmStateAt.js'
import { contextRefsAt } from '../RegionContextView.js'

export interface LlmCallExplanation {
  llmRequestedEventId: string
  trigger: {
    causedByEventId?: string
    causedBySummary?: string
  }
  fsmState: string | null
  regionCount: number
  causalChain: Array<{ eventId: string; type: EventKind; summary: string }>
  summary: string
}

/**
 * Pure event-log projection: explain why a given llm.requested fired — the
 * turn-terminator that triggered it, the FSM state at the time, how many
 * regions composed the prompt, and the causal chain. No LLM, no I/O, no
 * stored snapshot. Returns a plain serializable object (the JSON shape the
 * CLI explainer #36 will emit). Region details live in #26's "Assembled by".
 */
export function explainLlmCall(events: Event[], llmRequestedEventId: string): LlmCallExplanation {
  const byId = new Map<string, Event>()
  for (const e of events) byId.set(e.id, e)

  const evt = byId.get(llmRequestedEventId)
  if (!evt) throw new Error(`explainLlmCall: unknown event id "${llmRequestedEventId}"`)
  if (evt.type !== 'llm.requested') {
    throw new Error(`explainLlmCall: event "${llmRequestedEventId}" is "${evt.type}", expected "llm.requested"`)
  }

  const causeEvt = evt.causedBy ? byId.get(evt.causedBy) : undefined
  const causeSummary = causeEvt ? summarizeEvent(causeEvt) : undefined
  const fsmState = fsmStateAt(events, llmRequestedEventId)
  const regionCount = contextRefsAt(events, llmRequestedEventId, 'at').size

  const causalChain = walkCausedBy(events, llmRequestedEventId).map(e => ({
    eventId: e.id,
    type:    e.type,
    summary: summarizeEvent(e),
  }))

  const triggerSource = causeSummary ?? '(无上游)'
  const summary = `LLM 调用 @ state ${fsmState ?? '?'},由 ${triggerSource} 触发;prompt 由 ${regionCount} 个 region 拼成`

  return {
    llmRequestedEventId,
    trigger: {
      ...(evt.causedBy ? { causedByEventId: evt.causedBy } : {}),
      ...(causeSummary ? { causedBySummary: causeSummary } : {}),
    },
    fsmState,
    regionCount,
    causalChain,
    summary,
  }
}
```

- [ ] **Step 4: 运行,确认通过**

Run: `npx jest src/__tests__/explainLlmCall.test.ts`
Expected: PASS（4 用例）。Also `npx tsc --noEmit` clean.

- [ ] **Step 5: 提交**

```bash
git add src/trace/diagnostics/explainLlmCall.ts src/__tests__/explainLlmCall.test.ts
git commit -m "feat(#34): explainLlmCall — pure event-log projection of why-this-llm-call"
```

---

## Task 3: HTML "Why?" 块(llm 条目)

**Files:**
- Modify: `src/trace/render/html.ts`
- Test: `src/__tests__/render-html.test.ts`

- [ ] **Step 1: 写失败测试**(`src/__tests__/render-html.test.ts`,复用文件已有 `e({...})` 助手 + `renderHtml`/`Event` 导入)

```ts
describe('#34 llm Why?', () => {
  it('renders a Why? block on llm.requested with trigger link, state and causal chain', () => {
    const events: Event[] = [
      e({ id: 'start', runId: 'r1', type: 'agent.run.started', timestamp: 1,
          payload: { agentId: 'x', goal: 'g', input: 'i', contextId: 'c' } }),
      e({ id: 'treq', runId: 'r1', type: 'tool.requested', timestamp: 2, causedBy: 'start',
          payload: { toolName: 'search', input: {}, requestHash: 'h' } }),
      e({ id: 'tres', runId: 'r1', type: 'tool.responded', timestamp: 3, causedBy: 'treq',
          payload: { toolName: 'search', output: {}, requestHash: 'h' } }),
      e({ id: 'fsm', runId: 'r1', type: 'fsm.transition', timestamp: 4, causedBy: 'tres',
          payload: { from: 'plan', to: 'reflect', trigger: { domain: 'business', name: 'NEXT' } } }),
      e({ id: 'llm', runId: 'r1', type: 'llm.requested', timestamp: 5, causedBy: 'tres', payload: { model: 'm' } }),
    ]
    const html = renderHtml(events)
    expect(html).toContain('class="why"')
    expect(html).toContain('reflect')                       // fsm state in the why summary
    expect(html).toContain('tool.responded(search)')        // trigger
    expect(html).toContain('href="#ev-tres"')               // jump link to the trigger
  })

  it('shows (未知) state when the run has no transitions', () => {
    const events: Event[] = [
      e({ id: 'start', runId: 'r1', type: 'agent.run.started', timestamp: 1,
          payload: { agentId: 'x', goal: 'g', input: 'i', contextId: 'c' } }),
      e({ id: 'llm', runId: 'r1', type: 'llm.requested', timestamp: 2, causedBy: 'start', payload: { model: 'm' } }),
    ]
    const html = renderHtml(events)
    expect(html).toContain('class="why"')
    expect(html).toContain('(未知)')
  })
})
```

- [ ] **Step 2: 运行,确认失败**

Run: `npx jest src/__tests__/render-html.test.ts -t "llm Why?"`
Expected: FAIL。

- [ ] **Step 3: html.ts — 导入 explainLlmCall**

在顶部 import 区(`explainTransition` import 旁)加:

```ts
import { explainLlmCall, type LlmCallExplanation } from '../diagnostics/explainLlmCall.js'
```

- [ ] **Step 4: html.ts — renderWhyLlm 助手**

在 `renderEntry` 之前(可放在 `renderWhy` 旁)加。`fsmState` 为 null 时显示 `(未知)`:

```ts
function renderWhyLlm(exp: LlmCallExplanation): string {
  const trigger = exp.trigger.causedByEventId
    ? `<div class="why-trigger">触发: <a href="#ev-${esc(exp.trigger.causedByEventId)}">${esc(exp.trigger.causedBySummary ?? exp.trigger.causedByEventId)}</a></div>`
    : ''
  const state = `<div class="why-guards">state: ${esc(exp.fsmState ?? '(未知)')}</div>`
  const chain = `<div class="why-chain">`
    + exp.causalChain.map(c => `<a href="#ev-${esc(c.eventId)}">${esc(c.summary)}</a>`).join(' → ')
    + `</div>`
  return `<div class="why">`
       + `<div class="why-summary">${esc(exp.summary)}</div>`
       + trigger + state + chain
       + `</div>`
}
```

- [ ] **Step 5: html.ts — renderEntry 内联渲染 llm Why?**

在 `renderEntry` 里,`assembled` 那行之后加一行(复用 `regionCtx.events`,无需新参数):

```ts
  const assembled = entry.kind === 'llm' ? renderAssembled(entry.requestedId, regionCtx) : ''
  const whyLlm = entry.kind === 'llm' ? renderWhyLlm(explainLlmCall(regionCtx.events, entry.requestedId)) : ''
```

并在 return 串里把 `whyLlm` 插在 `assembled` 之后、`<pre class="payload">` 之前:

```ts
       + why
       + assembled
       + whyLlm
       + `<pre class="payload">${payloadFor(entry, eventById)}</pre>`
```

- [ ] **Step 6: 运行,确认通过**

Run: `npx jest src/__tests__/render-html.test.ts && npx jest src/__tests__/render-tree.test.ts`
Expected: PASS（2 新 #34 用例 + 原有 #26/#33 用例无回归)。Also `npx tsc --noEmit` clean。

- [ ] **Step 7: 提交**

```bash
git add src/trace/render/html.ts src/__tests__/render-html.test.ts
git commit -m "feat(#34): Why? block on llm.requested — trigger, fsm state, causal chain"
```

---

## 收尾

- [ ] **全量测试 + 构建**

Run: `npx jest && npm run build`
Expected: 全绿。

- [ ] **浏览器实测(交付前必做,用户要求)**:用 dist 生成一份含 llm.requested + region + transition 的报告,浏览器打开,确认 llm 条目上 **#26 Assembled-by 与 #34 Why? 两块并存**的实际观感;**据此决定是否需要后续合并两块**(spec §1 留的问题),截图给用户看。

- [ ] **开 PR**:body 带 `Closes #34`;说明与 #33/#26 的复用关系、region 详单归 #26、两块是否合并待实测决定。

---

## Self-Review(plan 作者已核对)

**Spec 覆盖:** §4.1 fsmStateAt → Task 1 ✓ §4.2 explainLlmCall → Task 2 ✓ §5 llm Why? 渲染 → Task 3 ✓ §6 错误处理(抛错/无上游/无 transition)→ Task 2 测试二三四 + Task 3 测试二 ✓ §7 测试 → 各任务 ✓ §9 验收 → Task 2+3 ✓。region 详单复用 #26、不重复 → Task 3 只渲 trigger+state+chain ✓。

**占位扫描:** 无 TBD/TODO;改代码步骤均含完整代码。

**类型一致性:** `LlmCallExplanation { llmRequestedEventId, trigger:{causedByEventId?,causedBySummary?}, fsmState, regionCount, causalChain, summary }`(Task 2 定义)在 Task 3 `renderWhyLlm` 引用一致;`fsmStateAt(events,id): string|null`(Task 1)在 Task 2 调用一致;`walkCausedBy`/`summarizeEvent`/`contextRefsAt` 取自现有模块;`explainLlmCall(events,id)` 在 Task 3 内联调用一致。
