# why-this-transition explainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 点 trace 报告里的 `fsm.transition` 能读懂"为什么这次跳转"——纯 event-log 投影出触发事件 + guard + 因果链,并能一键跳到上游事件。

**Architecture:** 新建 `src/trace/diagnostics/`(#36 CLI 也将复用):`walkCausedBy`(沿 causedBy 上溯)+ `summarizeEvent`(事件→人类标签)+ `explainTransition`(核心投影,返回普通可序列化对象)。HTML 渲染层(`html.ts`)给 fsm 条目加可读 "Why?" 块,给所有事件条目加 `id="ev-<eventId>"` 锚点。零 LLM、零存储、零事件 schema 变更。

**Tech Stack:** TypeScript;Jest(`npx jest <file>`);现有 `src/trace/render/html.ts`、`Event`/`FsmTransitionPayload`/`GuardEvaluation`/`EventKind`/`FsmEventDomain` 类型。

参考 spec:`docs/superpowers/specs/2026-05-30-why-this-transition-design.md`

**测试运行器是 JEST**(不是 vitest)。

---

## File Structure

| 文件 | 责任 | 改动 |
|---|---|---|
| `src/trace/diagnostics/walkCausedBy.ts` | 沿 causedBy 上溯遍历(通用) | 新建 |
| `src/trace/diagnostics/summarizeEvent.ts` | 事件 → 人类可读标签 | 新建 |
| `src/trace/diagnostics/explainTransition.ts` | 核心投影 + `TransitionExplanation` 类型 | 新建 |
| `src/trace/render/html.ts` | fsm 条目 "Why?" 块 + 事件锚点 id | 修改 |
| `src/__tests__/walkCausedBy.test.ts` | 单测 | 新建 |
| `src/__tests__/summarizeEvent.test.ts` | 单测 | 新建 |
| `src/__tests__/explainTransition.test.ts` | 单测 | 新建 |
| `src/__tests__/render-html.test.ts` | "Why?" + 锚点断言 | 修改 |

> 测试用**手工构造的 `Event[]`**(纯投影函数,合成输入比跑 runtime 更聚焦、确定性更强,且天然满足"无 LLM 调用")。

---

## Task 1: walkCausedBy — 沿 causedBy 上溯

**Files:**
- Create: `src/trace/diagnostics/walkCausedBy.ts`
- Test: `src/__tests__/walkCausedBy.test.ts`

- [ ] **Step 1: 写失败测试**

```ts
import { walkCausedBy } from '../trace/diagnostics/walkCausedBy'
import type { Event } from '../trace/types'

const ev = (id: string, type: string, causedBy?: string): Event =>
  ({ id, runId: 'r1', actor: 'a', type: type as Event['type'], timestamp: 0, payload: {}, ...(causedBy ? { causedBy } : {}) })

describe('walkCausedBy', () => {
  it('walks the causedBy chain from an event up to the root', () => {
    const events: Event[] = [
      ev('start', 'agent.run.started'),
      ev('llm1', 'llm.responded', 'start'),
      ev('treq', 'tool.requested', 'llm1'),
      ev('tres', 'tool.responded', 'treq'),
      ev('fsm', 'fsm.transition', 'tres'),
    ]
    const chain = walkCausedBy(events, 'fsm')
    expect(chain.map(e => e.id)).toEqual(['fsm', 'tres', 'treq', 'llm1', 'start'])
  })

  it('stops gracefully when a causedBy points to a missing event', () => {
    const events: Event[] = [ev('fsm', 'fsm.transition', 'gone')]
    const chain = walkCausedBy(events, 'fsm')
    expect(chain.map(e => e.id)).toEqual(['fsm'])
  })

  it('does not loop forever on a cycle', () => {
    const events: Event[] = [ev('a', 'x', 'b'), ev('b', 'x', 'a')]
    const chain = walkCausedBy(events, 'a')
    expect(chain.map(e => e.id)).toEqual(['a', 'b'])  // stops when 'a' seen again
  })

  it('returns empty when the start id is unknown', () => {
    expect(walkCausedBy([], 'nope')).toEqual([])
  })
})
```

- [ ] **Step 2: 运行,确认失败**

Run: `npx jest src/__tests__/walkCausedBy.test.ts`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现**

```ts
import type { Event } from '../types.js'

/**
 * Walk the causedBy chain upstream from `startId`, nearest-first, until an
 * event has no causedBy, its causedBy points outside `events`, or a cycle is
 * detected. Pure: reads only the provided events. Reusable by #32/#36.
 */
export function walkCausedBy(events: Event[], startId: string): Event[] {
  const byId = new Map<string, Event>()
  for (const e of events) byId.set(e.id, e)

  const chain: Event[] = []
  const seen = new Set<string>()
  let current = byId.get(startId)
  while (current && !seen.has(current.id)) {
    seen.add(current.id)
    chain.push(current)
    current = current.causedBy ? byId.get(current.causedBy) : undefined
  }
  return chain
}
```

- [ ] **Step 4: 运行,确认通过**

Run: `npx jest src/__tests__/walkCausedBy.test.ts`
Expected: PASS（4 用例）。

- [ ] **Step 5: 提交**

```bash
git add src/trace/diagnostics/walkCausedBy.ts src/__tests__/walkCausedBy.test.ts
git commit -m "feat(#33): walkCausedBy — upstream causal chain traversal"
```

---

## Task 2: summarizeEvent — 事件 → 人类标签

**Files:**
- Create: `src/trace/diagnostics/summarizeEvent.ts`
- Test: `src/__tests__/summarizeEvent.test.ts`

- [ ] **Step 1: 写失败测试**

```ts
import { summarizeEvent } from '../trace/diagnostics/summarizeEvent'
import type { Event } from '../trace/types'

const ev = (type: string, payload: unknown): Event =>
  ({ id: 'x', runId: 'r1', actor: 'a', type: type as Event['type'], timestamp: 0, payload })

describe('summarizeEvent', () => {
  it('labels tool events with the tool name', () => {
    expect(summarizeEvent(ev('tool.responded', { toolName: 'classify_intent', output: {} })))
      .toBe('tool.responded(classify_intent)')
    expect(summarizeEvent(ev('tool.requested', { toolName: 'search', input: {} })))
      .toBe('tool.requested(search)')
  })

  it('falls back to the bare type for events without a tool name', () => {
    expect(summarizeEvent(ev('llm.responded', {}))).toBe('llm.responded')
    expect(summarizeEvent(ev('agent.run.started', { agentId: 'x' }))).toBe('agent.run.started')
  })
})
```

- [ ] **Step 2: 运行,确认失败**

Run: `npx jest src/__tests__/summarizeEvent.test.ts`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现**

```ts
import type { Event } from '../types.js'

/**
 * Human-readable one-line label for an event. Shared by the HTML "Why?" block
 * and (future) the CLI explainer (#36), so both speak the same language.
 */
export function summarizeEvent(event: Event): string {
  const p = event.payload as { toolName?: unknown }
  if ((event.type === 'tool.requested' || event.type === 'tool.responded')
      && typeof p?.toolName === 'string') {
    return `${event.type}(${p.toolName})`
  }
  return event.type
}
```

- [ ] **Step 4: 运行,确认通过**

Run: `npx jest src/__tests__/summarizeEvent.test.ts`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/trace/diagnostics/summarizeEvent.ts src/__tests__/summarizeEvent.test.ts
git commit -m "feat(#33): summarizeEvent — human-readable event label"
```

---

## Task 3: explainTransition — 核心投影

**Files:**
- Create: `src/trace/diagnostics/explainTransition.ts`
- Test: `src/__tests__/explainTransition.test.ts`

- [ ] **Step 1: 写失败测试**

```ts
import { explainTransition } from '../trace/diagnostics/explainTransition'
import type { Event } from '../trace/types'

const ev = (id: string, type: string, payload: unknown, causedBy?: string): Event =>
  ({ id, runId: 'r1', actor: 'a', type: type as Event['type'], timestamp: 0, payload, ...(causedBy ? { causedBy } : {}) })

function routingEvents(): Event[] {
  return [
    ev('start', 'agent.run.started', { agentId: 'a', goal: 'g', input: 'i', contextId: 'c' }),
    ev('llm1', 'llm.responded', {}, 'start'),
    ev('treq', 'tool.requested', { toolName: 'classify_intent', input: {} }, 'llm1'),
    ev('tres', 'tool.responded', { toolName: 'classify_intent', output: {} }, 'treq'),
    ev('fsm', 'fsm.transition', {
      from: 'classify', to: 'handle_b',
      trigger: { domain: 'business', name: 'INTENT_B' },
      guardEvaluations: [{ guardId: 'intent-threshold', result: 'INTENT_B', contextSlice: { confidence: 0.9, threshold: 0.75 } }],
    }, 'tres'),
  ]
}

describe('explainTransition', () => {
  it('projects a readable explanation from the event log (no LLM)', () => {
    const exp = explainTransition(routingEvents(), 'fsm')
    expect(exp.from).toBe('classify')
    expect(exp.to).toBe('handle_b')
    expect(exp.trigger.name).toBe('INTENT_B')
    expect(exp.trigger.causedByEventId).toBe('tres')
    expect(exp.trigger.causedBySummary).toBe('tool.responded(classify_intent)')
    expect(exp.guards).toEqual([{ guardId: 'intent-threshold', result: 'INTENT_B', contextSlice: { confidence: 0.9, threshold: 0.75 } }])
    expect(exp.causalChain.map(c => c.eventId)).toEqual(['fsm', 'tres', 'treq', 'llm1', 'start'])
    expect(exp.summary).toContain('classify → handle_b')
    expect(exp.summary).toContain('INTENT_B')
    expect(exp.summary).toContain('intent-threshold')
  })

  it('every causalChain eventId resolves to a real event', () => {
    const events = routingEvents()
    const ids = new Set(events.map(e => e.id))
    const exp = explainTransition(events, 'fsm')
    for (const c of exp.causalChain) expect(ids.has(c.eventId)).toBe(true)
  })

  it('omits guards when the transition carries none', () => {
    const events: Event[] = [
      ev('start', 'agent.run.started', {}),
      ev('fsm', 'fsm.transition', { from: 's0', to: 'end', trigger: { domain: 'lifecycle', name: 'DONE' } }, 'start'),
    ]
    const exp = explainTransition(events, 'fsm')
    expect(exp.guards).toEqual([])
    expect(exp.summary).toContain('s0 → end')
  })

  it('throws on a non-fsm.transition event id', () => {
    expect(() => explainTransition(routingEvents(), 'llm1')).toThrow(/fsm\.transition/)
  })

  it('throws on an unknown event id', () => {
    expect(() => explainTransition(routingEvents(), 'nope')).toThrow(/nope/)
  })
})
```

- [ ] **Step 2: 运行,确认失败**

Run: `npx jest src/__tests__/explainTransition.test.ts`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现**

```ts
import type { Event, EventKind, FsmEventDomain, FsmTransitionPayload, GuardEvaluation } from '../types.js'
import { walkCausedBy } from './walkCausedBy.js'
import { summarizeEvent } from './summarizeEvent.js'

export interface TransitionExplanation {
  transitionEventId: string
  from: string
  to:   string
  trigger: {
    name:             string
    domain:           FsmEventDomain
    causedByEventId?: string
    causedBySummary?: string
  }
  guards: GuardEvaluation[]
  causalChain: Array<{ eventId: string; type: EventKind; summary: string }>
  summary: string
}

/**
 * Pure event-log projection: explain why a given fsm.transition fired.
 * Reads only `events` — no LLM, no I/O, no stored snapshot. Returns a plain
 * serializable object (also the JSON shape the CLI explainer #36 will emit).
 */
export function explainTransition(events: Event[], transitionEventId: string): TransitionExplanation {
  const byId = new Map<string, Event>()
  for (const e of events) byId.set(e.id, e)

  const evt = byId.get(transitionEventId)
  if (!evt) throw new Error(`explainTransition: unknown event id "${transitionEventId}"`)
  if (evt.type !== 'fsm.transition') {
    throw new Error(`explainTransition: event "${transitionEventId}" is "${evt.type}", expected "fsm.transition"`)
  }

  const p = evt.payload as FsmTransitionPayload
  const causeEvt = evt.causedBy ? byId.get(evt.causedBy) : undefined
  const guards = p.guardEvaluations ?? []

  const causalChain = walkCausedBy(events, transitionEventId).map(e => ({
    eventId: e.id,
    type:    e.type,
    summary: summarizeEvent(e),
  }))

  const guardPart = guards.length
    ? `;guard ${guards.map(g => `${g.guardId} 判定 ${String(g.result)}`).join('、')}`
    : ''
  const triggerSource = causeEvt ? summarizeEvent(causeEvt) : '(无上游记录)'
  const summary = `${p.from} → ${p.to}:由 ${triggerSource} 发出的 ${p.trigger.name} 触发${guardPart}`

  return {
    transitionEventId,
    from: p.from,
    to:   p.to,
    trigger: {
      name:   p.trigger.name,
      domain: p.trigger.domain,
      ...(evt.causedBy ? { causedByEventId: evt.causedBy } : {}),
      ...(causeEvt ? { causedBySummary: summarizeEvent(causeEvt) } : {}),
    },
    guards,
    causalChain,
    summary,
  }
}
```

- [ ] **Step 4: 运行,确认通过**

Run: `npx jest src/__tests__/explainTransition.test.ts`
Expected: PASS（5 用例）。Also `npx tsc --noEmit` clean.

- [ ] **Step 5: 提交**

```bash
git add src/trace/diagnostics/explainTransition.ts src/__tests__/explainTransition.test.ts
git commit -m "feat(#33): explainTransition — pure event-log projection of why-this-transition"
```

---

## Task 4: HTML "Why?" 渲染 + 事件锚点

**Files:**
- Modify: `src/trace/render/html.ts`
- Test: `src/__tests__/render-html.test.ts`

- [ ] **Step 1: 写失败测试**(`src/__tests__/render-html.test.ts`,复用文件已有的 `e({...})` 助手与 `renderHtml`/`Event` 导入)

```ts
it('renders a Why? block on fsm.transition with anchor links to upstream events', () => {
  const events: Event[] = [
    e({ id: 'start', runId: 'r1', type: 'agent.run.started', timestamp: 1,
        payload: { agentId: 'x', goal: 'g', input: 'i', contextId: 'c' } }),
    e({ id: 'treq', runId: 'r1', type: 'tool.requested', timestamp: 2, causedBy: 'start',
        payload: { toolName: 'classify_intent', input: {}, requestHash: 'h' } }),
    e({ id: 'tres', runId: 'r1', type: 'tool.responded', timestamp: 3, causedBy: 'treq',
        payload: { toolName: 'classify_intent', output: {}, requestHash: 'h' } }),
    e({ id: 'fsm', runId: 'r1', type: 'fsm.transition', timestamp: 4, causedBy: 'tres',
        payload: { from: 'classify', to: 'handle_b', trigger: { domain: 'business', name: 'INTENT_B' },
          guardEvaluations: [{ guardId: 'intent-threshold', result: 'INTENT_B', contextSlice: { confidence: 0.9 } }] } }),
  ]
  const html = renderHtml(events)
  // readable Why summary present
  expect(html).toContain('class="why"')
  expect(html).toContain('classify → handle_b')
  expect(html).toContain('intent-threshold')
  // a jump link to the upstream tool.responded, and that target id exists
  expect(html).toContain('href="#ev-tres"')
  expect(html).toContain('id="ev-tres"')
})

it('renders a Why? block for an fsm.transition without guards', () => {
  const events: Event[] = [
    e({ id: 'start', runId: 'r1', type: 'agent.run.started', timestamp: 1,
        payload: { agentId: 'x', goal: 'g', input: 'i', contextId: 'c' } }),
    e({ id: 'fsm', runId: 'r1', type: 'fsm.transition', timestamp: 2, causedBy: 'start',
        payload: { from: 's0', to: 'end', trigger: { domain: 'lifecycle', name: 'DONE' } } }),
  ]
  const html = renderHtml(events)
  expect(html).toContain('class="why"')
  expect(html).toContain('s0 → end')
})
```

- [ ] **Step 2: 运行,确认失败**

Run: `npx jest src/__tests__/render-html.test.ts -t "Why?"`
Expected: FAIL（无 `class="why"` / 无 `id="ev-..."`）。

- [ ] **Step 3: 改 html.ts — 锚点 id**

在 `src/trace/render/html.ts` 顶部 import 处加:

```ts
import { explainTransition, type TransitionExplanation } from '../diagnostics/explainTransition.js'
```

新增一个助手(放在 `renderEntry` 之前):返回某条目覆盖的所有事件 id(合并条目覆盖 requested+responded 两个 id):

```ts
function entryEventIds(entry: TimelineEntry): string[] {
  if (entry.kind === 'llm' || entry.kind === 'tool') {
    return entry.respondedId ? [entry.requestedId, entry.respondedId] : [entry.requestedId]
  }
  return [entry.eventId]
}
```

- [ ] **Step 4: 改 html.ts — Why? 块渲染**

新增 Why 块助手(放在 `renderEntry` 之前):

```ts
function renderWhy(exp: TransitionExplanation): string {
  const trigger = exp.trigger.causedByEventId
    ? `<div class="why-trigger">触发: <a href="#ev-${esc(exp.trigger.causedByEventId)}">${esc(exp.trigger.causedBySummary ?? exp.trigger.causedByEventId)}</a></div>`
    : ''
  const guards = exp.guards.length
    ? `<div class="why-guards">guard: ${esc(exp.guards.map(g => `${g.guardId}=${String(g.result)}`).join(', '))}</div>`
    : ''
  const chain = `<div class="why-chain">`
    + exp.causalChain.map(c => `<a href="#ev-${esc(c.eventId)}">${esc(c.summary)}</a>`).join(' → ')
    + `</div>`
  return `<div class="why">`
       + `<div class="why-summary">${esc(exp.summary)}</div>`
       + trigger + guards + chain
       + `</div>`
}
```

- [ ] **Step 5: 改 html.ts — renderEntry 接入锚点 + Why 块**

`renderEntry` 改签名,接收 `explanations` map,渲染锚点 id 与(fsm 条目的)Why 块:

```ts
function renderEntry(
  entry: TimelineEntry,
  eventById: Map<string, Event>,
  explanations: Map<string, TransitionExplanation>,
): string {
  const icon = entry.kind === 'llm' ? '◆'
             : entry.kind === 'tool' ? '▣'
             : entry.kind === 'region'
               ? (entry.eventType === 'context.boundary.applied' ? '⌖'
                 : entry.eventType === 'region.added' ? '＋' : '－')
             : entry.kind === 'fsm' ? '⇒'
             : '●'
  const ids = entryEventIds(entry)
  const primaryId = ids[0]!
  const extraAnchors = ids.slice(1).map(id => `<span class="anchor" id="ev-${esc(id)}"></span>`).join('')
  const why = entry.kind === 'fsm' && explanations.has(entry.eventId)
    ? renderWhy(explanations.get(entry.eventId)!)
    : ''
  return `<div class="entry ${entry.kind}" data-kind="${entry.kind}" id="ev-${esc(primaryId)}">`
       + extraAnchors
       + `<div class="entry-head">`
       + `<span class="icon">${icon}</span>`
       + `<span class="summary">${summaryFor(entry)}</span>`
       + `<span class="ts">${entry.timestamp}</span>`
       + `</div>`
       + why
       + `<pre class="payload">${payloadFor(entry, eventById)}</pre>`
       + `</div>`
}
```

`renderNode` 把 `explanations` 透传给 `renderEntry`:

```ts
function renderNode(
  node: TimelineNode,
  eventById: Map<string, Event>,
  explanations: Map<string, TransitionExplanation>,
): string {
  const status = node.status ?? 'in-flight'
  const badgeClass = BADGE_STATUS_CLASSES.has(status) ? ' ' + status : ''
  return `<section class="run" data-run-id="${esc(node.runId)}">`
       + `<div class="run-head">`
       + `<strong>${esc(node.agentId ?? '(unknown)')}</strong>`
       + `<span class="run-id">${esc(node.runId)}</span>`
       + `<span class="badge${badgeClass}">${esc(status)}</span>`
       + `</div>`
       + node.entries.map(en => renderEntry(en, eventById, explanations)).join('')
       + (node.children.length > 0
           ? `<div class="child-run">${node.children.map(n => renderNode(n, eventById, explanations)).join('')}</div>`
           : '')
       + `</section>`
}
```

- [ ] **Step 6: 改 html.ts — renderHtml 预算解释并下传**

在 `renderHtml` 里,`eventById` 构建后、`return` 之前加:

```ts
const explanations = new Map<string, TransitionExplanation>()
for (const evt of events) {
  if (evt.type === 'fsm.transition') explanations.set(evt.id, explainTransition(events, evt.id))
}
```

并把 `tree.map(n => renderNode(n, eventById)).join('')` 改为 `tree.map(n => renderNode(n, eventById, explanations)).join('')`。

- [ ] **Step 7: 运行,确认通过**

Run: `npx jest src/__tests__/render-html.test.ts && npx jest src/__tests__/render-tree.test.ts`
Expected: PASS（含原有用例,无回归)。Also `npx tsc --noEmit` clean.

- [ ] **Step 8: 提交**

```bash
git add src/trace/render/html.ts src/__tests__/render-html.test.ts
git commit -m "feat(#33): HTML Why? block on fsm.transition + event anchors for jump links"
```

---

## 收尾

- [ ] **全量测试 + 构建**

Run: `npx jest && npm run build`
Expected: 全绿。

- [ ] **开 PR**:body 带 `Closes #33`,引用 spec 与本 plan;说明 region 部分按设计暂缓(等 #26),CLINK 投影(`walkCausedBy`/`explainTransition`)已为 #36 CLI 备好。

---

## Self-Review(plan 作者已核对)

**Spec 覆盖:** §4.1 walkCausedBy → Task 1 ✓ §4.2 explainTransition + summarizeEvent → Task 2/3 ✓ §5 HTML Why? + 锚点 → Task 4 ✓ §6 错误处理 → Task 3 抛错用例 + walkCausedBy 断链/环用例 ✓ §7 测试 → 各任务测试 ✓ §9 验收(可读解释/无 LLM/跳转链接/纯 event-log)→ Task 3+4 ✓。region 按 Non-goals 不做 ✓。

**占位扫描:** 无 TBD/TODO;每个改代码步骤含完整代码。

**类型一致性:** `TransitionExplanation`(Task 3 定义)字段 `from/to/trigger.{name,domain,causedByEventId,causedBySummary}/guards/causalChain/summary` 在 Task 4 `renderWhy` 中引用一致;`walkCausedBy(events, id)`(Task 1)签名在 Task 3 调用一致;`summarizeEvent(event)`(Task 2)在 Task 3 调用一致;`GuardEvaluation`/`Event`/`EventKind`/`FsmEventDomain`/`FsmTransitionPayload` 均取自 `src/trace/types`。
