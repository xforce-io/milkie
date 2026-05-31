# 因果下钻 trace viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `milkie trace report` 从平铺时间线重组成"决策脊柱 + Why 面板 + 点击下钻"的诊断 viewer,纯静态自包含 HTML,"两次点击到根因"。

**Architecture:** 纯投影 `buildDecisionSpine`(events→决策脊柱,含"最近决策祖先"`causeDecisionId`)+ `explainToolCall`;复用 explainTransition/explainLlmCall/contextRefsAt/walkCausedBy。`viewer.ts` 出两栏 HTML + 内嵌全部 explanation JSON,vanilla JS 纯查表做选中/下钻/高亮;raw 全时间线作次要 tab(复用从 html.ts 抽出的 `renderTimelineSections`)。

**Tech Stack:** TypeScript;Jest(`npx jest`);dev-browser(交互验收);现有 `src/trace/diagnostics/*`、`src/trace/RegionContextView.ts`、`src/trace/render/html.ts`、`src/cli/main.ts`。

参考 spec:`docs/superpowers/specs/2026-05-31-trace-viewer-design.md`。**测试运行器是 JEST。**

---

## File Structure

| 文件 | 责任 | 改动 |
|---|---|---|
| `src/trace/diagnostics/buildDecisionSpine.ts` | events→决策脊柱(+causeDecisionId) | 新建 |
| `src/trace/diagnostics/explainToolCall.ts` | 工具节点投影 | 新建 |
| `src/trace/render/html.ts` | 抽出 `renderTimelineSections`,`renderHtml` 改为薄包装 | 重构 |
| `src/trace/render/viewer.ts` | 两栏 viewer HTML + 内嵌数据 + raw tab | 新建 |
| `src/trace/render/viewer-template.ts` | viewer CSS + vanilla JS | 新建 |
| `src/cli/main.ts` | `trace report` 改用 `renderViewer` | 改 |
| `src/__tests__/buildDecisionSpine.test.ts` `explainToolCall.test.ts` `render-viewer.test.ts` | 单测 | 新建 |

---

## Task 1: buildDecisionSpine — 决策脊柱投影

**Files:** Create `src/trace/diagnostics/buildDecisionSpine.ts`; Test `src/__tests__/buildDecisionSpine.test.ts`

- [ ] **Step 1: 写失败测试**

```ts
import { buildDecisionSpine } from '../trace/diagnostics/buildDecisionSpine'
import type { Event } from '../trace/types'

const ev = (id: string, type: string, payload: unknown, causedBy?: string, ts = 0): Event =>
  ({ id, runId: 'r1', actor: 'a', type: type as Event['type'], timestamp: ts, payload, ...(causedBy ? { causedBy } : {}) })

function scenario(): Event[] {
  return [
    ev('start', 'agent.run.started', { agentId: 'a', goal: 'g', input: 'i', contextId: 'c' }, undefined, 1),
    ev('llm1', 'llm.requested', { model: 'm' }, 'start', 2),
    ev('lr1', 'llm.responded', {}, 'llm1', 3),
    ev('treq', 'tool.requested', { toolName: 'classify_intent', input: {} }, 'lr1', 4),
    ev('tres', 'tool.responded', { toolName: 'classify_intent', output: {} }, 'treq', 5),
    ev('fsm', 'fsm.transition', { from: 'classify', to: 'handle', trigger: { domain: 'business', name: 'GO' } }, 'tres', 6),
    ev('done', 'agent.run.completed', { status: 'completed', lastTextOutput: 'ok' }, 'fsm', 7),
  ]
}

describe('buildDecisionSpine', () => {
  it('keeps only decision nodes in timestamp order with labels', () => {
    const spine = buildDecisionSpine(scenario())
    expect(spine.nodes.map(n => [n.kind, n.eventId])).toEqual([
      ['llm', 'llm1'], ['tool', 'treq'], ['transition', 'fsm'], ['output', 'done'],
    ])
    expect(spine.nodes.find(n => n.eventId === 'treq')!.label).toBe('tool: classify_intent')
    expect(spine.nodes.find(n => n.eventId === 'fsm')!.label).toBe('classify → handle')
  })

  it('resolves causeDecisionId to the nearest decision ancestor (skipping non-decision causes)', () => {
    const spine = buildDecisionSpine(scenario())
    // fsm.causedBy = tres (tool.responded, NOT a spine node) → nearest decision ancestor = treq (tool.requested)
    expect(spine.nodes.find(n => n.eventId === 'fsm')!.causeDecisionId).toBe('treq')
    // done.causedBy = fsm (a decision) → causeDecisionId = fsm
    expect(spine.nodes.find(n => n.eventId === 'done')!.causeDecisionId).toBe('fsm')
    // llm1.causedBy = start (run.started, not a spine node, no decision ancestor) → undefined
    expect(spine.nodes.find(n => n.eventId === 'llm1')!.causeDecisionId).toBeUndefined()
  })

  it('returns empty spine for a run with no decisions', () => {
    expect(buildDecisionSpine([ev('s', 'agent.run.started', {})]).nodes).toEqual([])
  })
})
```

- [ ] **Step 2: 运行,确认失败** — `npx jest src/__tests__/buildDecisionSpine.test.ts` → FAIL

- [ ] **Step 3: 实现**

```ts
import type { Event } from '../types.js'
import { walkCausedBy } from './walkCausedBy.js'

export type DecisionKind = 'llm' | 'tool' | 'transition' | 'output'

export interface DecisionNode {
  eventId:          string
  kind:             DecisionKind
  label:            string
  timestamp:        number
  causedByEventId?: string
  causeDecisionId?: string
}

export interface DecisionSpine {
  nodes: DecisionNode[]
}

const SPINE_TYPES = new Set(['llm.requested', 'tool.requested', 'fsm.transition', 'agent.run.completed'])

function kindOf(type: string): DecisionKind {
  if (type === 'llm.requested')  return 'llm'
  if (type === 'tool.requested') return 'tool'
  if (type === 'fsm.transition') return 'transition'
  return 'output'
}

function labelOf(e: Event): string {
  if (e.type === 'llm.requested')  return 'LLM call'
  if (e.type === 'tool.requested') return `tool: ${String((e.payload as { toolName?: unknown }).toolName ?? '?')}`
  if (e.type === 'fsm.transition') { const p = e.payload as { from: string; to: string }; return `${p.from} → ${p.to}` }
  return '输出'
}

/**
 * Project the event log down to the decision spine: only LLM calls, tool
 * calls, transitions, and the final output, in timestamp order. For each node,
 * causeDecisionId is the nearest decision ancestor reached by walking causedBy
 * (skipping non-decision causes like tool.responded / run.started). Pure.
 */
export function buildDecisionSpine(events: Event[]): DecisionSpine {
  const spineIds = new Set(events.filter(e => SPINE_TYPES.has(e.type)).map(e => e.id))
  const nodes: DecisionNode[] = events
    .filter(e => SPINE_TYPES.has(e.type))
    .map(e => {
      // walkCausedBy returns [e, cause, ...]; the first ancestor (excluding e) that is itself a spine node.
      const ancestor = walkCausedBy(events, e.id).slice(1).find(a => spineIds.has(a.id))
      return {
        eventId:   e.id,
        kind:      kindOf(e.type),
        label:     labelOf(e),
        timestamp: e.timestamp,
        ...(e.causedBy ? { causedByEventId: e.causedBy } : {}),
        ...(ancestor ? { causeDecisionId: ancestor.id } : {}),
      }
    })
    .sort((a, b) => a.timestamp - b.timestamp)
  return { nodes }
}
```

- [ ] **Step 4: 运行,确认通过** — `npx jest src/__tests__/buildDecisionSpine.test.ts` → PASS(3);`npx tsc --noEmit` clean

- [ ] **Step 5: 提交**

```bash
git add src/trace/diagnostics/buildDecisionSpine.ts src/__tests__/buildDecisionSpine.test.ts
git commit -m "feat(#64): buildDecisionSpine — project events to the decision spine with nearest-decision-ancestor"
```

---

## Task 2: explainToolCall — 工具节点投影

**Files:** Create `src/trace/diagnostics/explainToolCall.ts`; Test `src/__tests__/explainToolCall.test.ts`

- [ ] **Step 1: 写失败测试**

```ts
import { explainToolCall } from '../trace/diagnostics/explainToolCall'
import type { Event } from '../trace/types'

const ev = (id: string, type: string, payload: unknown, causedBy?: string): Event =>
  ({ id, runId: 'r1', actor: 'a', type: type as Event['type'], timestamp: 0, payload, ...(causedBy ? { causedBy } : {}) })

function scenario(): Event[] {
  return [
    ev('start', 'agent.run.started', {}),
    ev('lr1', 'llm.responded', {}, 'start'),
    ev('treq', 'tool.requested', { toolName: 'search', input: { q: 'x' }, requestHash: 'h' }, 'lr1'),
    ev('tres', 'tool.responded', { toolName: 'search', output: { hits: 3 }, requestHash: 'h' }, 'treq'),
  ]
}

describe('explainToolCall', () => {
  it('projects toolName, input, paired output, trigger and causal chain', () => {
    const exp = explainToolCall(scenario(), 'treq')
    expect(exp.toolName).toBe('search')
    expect(exp.input).toEqual({ q: 'x' })
    expect(exp.output).toEqual({ hits: 3 })
    expect(exp.trigger.causedByEventId).toBe('lr1')
    expect(exp.trigger.causedBySummary).toBe('llm.responded')
    expect(exp.causalChain.map(c => c.eventId)).toEqual(['treq', 'lr1', 'start'])
    expect(exp.summary).toContain('search')
  })

  it('omits output when there is no paired tool.responded', () => {
    const events: Event[] = [ev('treq', 'tool.requested', { toolName: 'search', input: {} })]
    expect(explainToolCall(events, 'treq').output).toBeUndefined()
  })

  it('throws on a non-tool.requested event id', () => {
    expect(() => explainToolCall(scenario(), 'tres')).toThrow(/tool\.requested/)
  })
})
```

- [ ] **Step 2: 运行,确认失败** — FAIL

- [ ] **Step 3: 实现**

```ts
import type { Event, EventKind } from '../types.js'
import { walkCausedBy } from './walkCausedBy.js'
import { summarizeEvent } from './summarizeEvent.js'

export interface ToolCallExplanation {
  toolRequestedEventId: string
  toolName: string
  input: unknown
  output?: unknown
  trigger: { causedByEventId?: string; causedBySummary?: string }
  causalChain: Array<{ eventId: string; type: EventKind; summary: string }>
  summary: string
}

/**
 * Pure projection: explain a tool call — its name/input, the paired output,
 * which llm.responded decided to call it, and the causal chain. No LLM/I/O.
 * Region/CLI #36 reuse the serializable result. Parallel to explainLlmCall.
 */
export function explainToolCall(events: Event[], toolRequestedEventId: string): ToolCallExplanation {
  const byId = new Map<string, Event>()
  for (const e of events) byId.set(e.id, e)

  const evt = byId.get(toolRequestedEventId)
  if (!evt) throw new Error(`explainToolCall: unknown event id "${toolRequestedEventId}"`)
  if (evt.type !== 'tool.requested') {
    throw new Error(`explainToolCall: event "${toolRequestedEventId}" is "${evt.type}", expected "tool.requested"`)
  }

  const p = evt.payload as { toolName?: unknown; input?: unknown; requestHash?: unknown }
  const toolName = String(p.toolName ?? '?')
  const responded = events.find(e =>
    e.type === 'tool.responded' && (e.payload as { requestHash?: unknown }).requestHash === p.requestHash)
  const output = responded ? (responded.payload as { output?: unknown }).output : undefined

  const causeEvt = evt.causedBy ? byId.get(evt.causedBy) : undefined
  const causeSummary = causeEvt ? summarizeEvent(causeEvt) : undefined

  const causalChain = walkCausedBy(events, toolRequestedEventId).map(e => ({
    eventId: e.id, type: e.type, summary: summarizeEvent(e),
  }))

  const summary = `工具 ${toolName} 被 ${causeSummary ?? '(无上游记录)'} 调用`

  return {
    toolRequestedEventId,
    toolName,
    input: p.input,
    ...(output !== undefined ? { output } : {}),
    trigger: {
      ...(evt.causedBy ? { causedByEventId: evt.causedBy } : {}),
      ...(causeSummary ? { causedBySummary: causeSummary } : {}),
    },
    causalChain,
    summary,
  }
}
```

- [ ] **Step 4: 运行,确认通过** — PASS(3);tsc clean

- [ ] **Step 5: 提交**

```bash
git add src/trace/diagnostics/explainToolCall.ts src/__tests__/explainToolCall.test.ts
git commit -m "feat(#64): explainToolCall — pure projection of a tool call's why"
```

---

## Task 3: 抽出 renderTimelineSections(为 raw tab 复用)

**Files:** Modify `src/trace/render/html.ts`; Test `src/__tests__/render-html.test.ts`

- [ ] **Step 1: 写失败测试**(`render-html.test.ts` 追加)

```ts
it('exposes renderTimelineSections returning the timeline body (filters + sections, no <html>)', async () => {
  const { renderTimelineSections } = await import('../trace/render/html')
  const events: Event[] = [
    e({ id: 's', runId: 'r1', type: 'agent.run.started', timestamp: 1,
        payload: { agentId: 'x', goal: 'g', input: 'i', contextId: 'c' } }),
  ]
  const body = renderTimelineSections(events)
  expect(body).toContain('class="filters"')
  expect(body).toContain('data-run-id="r1"')
  expect(body).not.toContain('<!doctype html>')   // body fragment, not a full document
})
```

- [ ] **Step 2: 运行,确认失败** — FAIL

- [ ] **Step 3: 重构 html.ts**

READ the current `renderHtml`. Extract everything from the `<div class="filters">` through the `${tree.map(...)}` sections AND the two embedded `<script type="application/json">` registries (region-content + trace-data) into a new exported function `renderTimelineSections(events, opts?)`. `renderHtml` keeps the `<!doctype html>` + `<head>`/`<style>` + `<body><h1>` + `${renderTimelineSections(events, opts)}` + `<script>${SCRIPT}</script>` wrapper. Concretely:

```ts
export function renderTimelineSections(events: Event[], opts: { regionContent?: Map<string, string> } = {}): string {
  const tree = buildTimelineTree(events)
  const eventById = new Map<string, Event>()
  for (const evt of events) eventById.set(evt.id, evt)
  const explanations = new Map<string, TransitionExplanation>()
  for (const evt of events) {
    if (evt.type === 'fsm.transition') explanations.set(evt.id, explainTransition(events, evt.id))
  }
  const regionContent = opts.regionContent ?? new Map<string, string>()
  const regionCtx: RegionCtx = { events, reuseCounts: regionReuseCounts(events), regionContent }
  const registryJson = JSON.stringify(Object.fromEntries(regionContent)).replace(/<\/script/gi, '<\\/script')
  const dataJson = JSON.stringify(events).replace(/<\/script/gi, '<\\/script')
  return `<div class="filters">
  <span class="chip" data-kind="llm">LLM</span>
  <span class="chip" data-kind="tool">tool</span>
  <span class="chip" data-kind="lifecycle">lifecycle</span>
  <span class="chip" data-kind="region">region</span>
  <span class="chip" data-kind="fsm">fsm</span>
</div>
${tree.map(n => renderNode(n, eventById, explanations, regionCtx)).join('')}
<script type="application/json" id="region-content">${registryJson}</script>
<script type="application/json" id="trace-data">${dataJson}</script>`
}

export function renderHtml(events: Event[], opts: { regionContent?: Map<string, string> } = {}): string {
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>milkie trace report</title>
<style>${STYLES}</style>
</head>
<body>
<h1>milkie trace report</h1>
${renderTimelineSections(events, opts)}
<script>${SCRIPT}</script>
</body>
</html>`
}
```
(Match the EXACT current chip list / script ids in the file — copy them verbatim from the current renderHtml. Behavior must be identical: the new renderHtml output equals the old one.)

- [ ] **Step 4: 运行,确认通过** — `npx jest src/__tests__/render-html.test.ts` → PASS（new test + ALL existing #26/#33/#34 tests, output unchanged）;tsc clean

- [ ] **Step 5: 提交**

```bash
git add src/trace/render/html.ts src/__tests__/render-html.test.ts
git commit -m "refactor(#64): extract renderTimelineSections from renderHtml for viewer raw tab reuse"
```

---

## Task 4: viewer.ts + viewer-template.ts — 决策 viewer

**Files:** Create `src/trace/render/viewer-template.ts`, `src/trace/render/viewer.ts`; Test `src/__tests__/render-viewer.test.ts`

- [ ] **Step 1: 写失败测试**(`src/__tests__/render-viewer.test.ts`)

```ts
import { renderViewer } from '../trace/render/viewer'
import type { Event } from '../trace/types'

const e = (over: Partial<Event> & { id: string; runId: string; type: Event['type'] }): Event =>
  ({ actor: 'a', timestamp: 0, payload: {}, ...over })

function scenario(): Event[] {
  return [
    e({ id: 'start', runId: 'r1', type: 'agent.run.started', timestamp: 1, payload: { agentId: 'x', goal: 'g', input: 'i', contextId: 'c' } }),
    e({ id: 'llm1', runId: 'r1', type: 'llm.requested', timestamp: 2, causedBy: 'start', payload: { model: 'm' } }),
    e({ id: 'lr1', runId: 'r1', type: 'llm.responded', timestamp: 3, causedBy: 'llm1', payload: {} }),
    e({ id: 'treq', runId: 'r1', type: 'tool.requested', timestamp: 4, causedBy: 'lr1', payload: { toolName: 'classify_intent', input: {}, requestHash: 'h' } }),
    e({ id: 'tres', runId: 'r1', type: 'tool.responded', timestamp: 5, causedBy: 'treq', payload: { toolName: 'classify_intent', output: {}, requestHash: 'h' } }),
    e({ id: 'fsm', runId: 'r1', type: 'fsm.transition', timestamp: 6, causedBy: 'tres', payload: { from: 'classify', to: 'handle', trigger: { domain: 'business', name: 'GO' }, guardEvaluations: [{ guardId: 'g1', result: 'GO', contextSlice: {} }] } }),
    e({ id: 'done', runId: 'r1', type: 'agent.run.completed', timestamp: 7, causedBy: 'fsm', payload: { status: 'completed', lastTextOutput: 'ok' } }),
  ]
}

describe('renderViewer', () => {
  it('produces a self-contained document with a decision spine and embedded explanations', () => {
    const html = renderViewer(scenario())
    expect(html.startsWith('<!doctype html>')).toBe(true)
    // spine has nodes with data-id for each decision
    expect(html).toContain('data-id="llm1"')
    expect(html).toContain('data-id="treq"')
    expect(html).toContain('data-id="fsm"')
    expect(html).toContain('data-id="done"')
    // output node carries the Why entry
    expect(html).toContain('class="spine-output"')
    // embedded data the JS reads
    expect(html).toContain('id="spine-data"')
    expect(html).toContain('id="explanations-data"')
    // why panel container + the decision/raw tabs
    expect(html).toContain('id="why-panel"')
    expect(html).toContain('data-tab="decision"')
    expect(html).toContain('data-tab="raw"')
    // raw tab reuses the timeline (filters present)
    expect(html).toContain('class="filters"')
  })

  it('renders without crashing for a run with no decisions', () => {
    const html = renderViewer([{ id: 's', runId: 'r1', actor: 'a', type: 'agent.run.started', timestamp: 1, payload: {} } as Event])
    expect(html.startsWith('<!doctype html>')).toBe(true)
    expect(html).toContain('id="why-panel"')
  })
})
```

- [ ] **Step 2: 运行,确认失败** — FAIL

- [ ] **Step 3: viewer-template.ts**(CSS + JS 常量)

```ts
export const VIEWER_STYLES = `
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f7f7f8; color: #1c1c1e; }
  h1 { font-size: 16px; margin: 0; padding: 12px 16px; border-bottom: 1px solid #e5e5e7; background: #fff; }
  .tabs { display: flex; gap: 8px; padding: 8px 16px; background: #fff; border-bottom: 1px solid #e5e5e7; }
  .tab { font-size: 12px; padding: 4px 12px; border-radius: 999px; border: 1px solid #d1d1d6; cursor: pointer; background: #fff; user-select: none; }
  .tab.active { background: #1c1c1e; color: #fff; border-color: #1c1c1e; }
  .pane { display: none; }
  .pane.active { display: block; }
  #pane-decision { display: flex; height: calc(100vh - 90px); }
  #pane-decision.active { display: flex; }
  .spine { width: 42%; border-right: 1px solid #e5e5e7; overflow: auto; padding: 8px; background: #fafafa; }
  .node { font-family: ui-monospace, SFMono-Regular, monospace; font-size: 12px; padding: 5px 8px; margin: 3px 0; border-left: 3px solid transparent; border-radius: 3px; cursor: pointer; background: #fff; }
  .node.k-llm { border-left-color: #5b3ec9; } .node.k-tool { border-left-color: #2563eb; }
  .node.k-transition { border-left-color: #8a5a00; } .node.k-output { border-left-color: #1c1c1e; font-weight: 600; }
  .node.selected { outline: 2px solid #f0a; }
  .node.cause { background: #eef6ff; }
  .spine-output .why-entry { color: #c026a6; font-size: 11px; margin-left: 6px; }
  #why-panel { width: 58%; overflow: auto; padding: 14px; background: #fff; font-size: 13px; line-height: 1.6; }
  #why-panel .ph { color: #888; }
  #why-panel h3 { font-size: 14px; margin: 0 0 8px; }
  .why-block { background: #f7f7f8; padding: 10px; border-radius: 6px; margin-bottom: 10px; }
  .nav-link { display: block; padding: 6px 10px; border-radius: 5px; margin: 4px 0; cursor: pointer; }
  .nav-cause { background: #eef6ff; color: #2563eb; } .nav-effect { background: #fef0f5; color: #a13; }
  .rawpre { font-family: ui-monospace, monospace; font-size: 11px; white-space: pre-wrap; background: #fafafa; padding: 8px; border-radius: 4px; max-height: 240px; overflow: auto; }
  #pane-raw { padding: 12px 16px; }
`

export const VIEWER_SCRIPT = `
(function () {
  var spine = JSON.parse(document.getElementById('spine-data').textContent || '{"nodes":[]}');
  var exps  = JSON.parse(document.getElementById('explanations-data').textContent || '{}');
  function esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
  function nodeEl(id){ return document.querySelector('.node[data-id="'+CSS.escape(id)+'"]'); }
  function clearMarks(){ document.querySelectorAll('.node.selected,.node.cause').forEach(function(n){ n.classList.remove('selected','cause'); }); }
  function chainHtml(chain){ return (chain||[]).map(function(c){ return '<a class="xlink" data-go="'+esc(c.eventId)+'">'+esc(c.summary)+'</a>'; }).join(' → '); }
  function navHtml(causeId){
    var out='';
    if(causeId){ out += '<a class="nav-link nav-cause" data-go="'+esc(causeId)+'">← 谁导致的:'+esc((exps[causeId]&&exps[causeId].title)||causeId)+'</a>'; }
    return out;
  }
  function panelHtml(id){
    var x = exps[id]; if(!x){ return '<p class="ph">选一个决策节点看 why</p>'; }
    var h = '<h3>'+esc(x.title)+'</h3>';
    h += '<div class="why-block">'+x.bodyHtml+'</div>';
    h += '<div class="why-block">'+navHtml(x.causeDecisionId)+'<div style="color:#888;font-size:11px;margin-top:6px">因果链: '+chainHtml(x.chain)+'</div></div>';
    h += '<details><summary style="cursor:pointer;color:#888;font-size:11px">原始 payload</summary><pre class="rawpre">'+esc(x.rawJson)+'</pre></details>';
    return h;
  }
  function selectNode(id){
    clearMarks();
    var n = nodeEl(id); if(n){ n.classList.add('selected'); }
    var cid = exps[id] && exps[id].causeDecisionId;
    if(cid){ var cn = nodeEl(cid); if(cn){ cn.classList.add('cause'); } }
    document.getElementById('why-panel').innerHTML = panelHtml(id);
  }
  document.addEventListener('click', function(ev){
    var go = ev.target.closest('[data-go]'); if(go){ selectNode(go.getAttribute('data-go')); return; }
    var node = ev.target.closest('.node[data-id]'); if(node){ selectNode(node.getAttribute('data-id')); return; }
    var tab = ev.target.closest('.tab[data-tab]');
    if(tab){
      document.querySelectorAll('.tab').forEach(function(t){ t.classList.remove('active'); });
      tab.classList.add('active');
      document.querySelectorAll('.pane').forEach(function(p){ p.classList.remove('active'); });
      document.getElementById('pane-'+tab.getAttribute('data-tab')).classList.add('active');
    }
  });
})();
`
```

- [ ] **Step 4: viewer.ts**(组装 explanation 对象 + 两栏 + raw tab)

```ts
import type { Event } from '../types.js'
import { buildDecisionSpine, type DecisionNode } from '../diagnostics/buildDecisionSpine.js'
import { explainTransition } from '../diagnostics/explainTransition.js'
import { explainLlmCall } from '../diagnostics/explainLlmCall.js'
import { explainToolCall } from '../diagnostics/explainToolCall.js'
import { regionReuseCounts } from '../RegionContextView.js'
import { renderTimelineSections } from './html.js'
import { VIEWER_STYLES, VIEWER_SCRIPT } from './viewer-template.js'

function esc(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;')
}

// One panel "explanation" record per node, consumed verbatim by the client JS.
function panelRecord(events: Event[], node: DecisionNode, eventById: Map<string, Event>): unknown {
  const base = {
    causeDecisionId: node.causeDecisionId,
    chain: [] as Array<{ eventId: string; type: string; summary: string }>,
    rawJson: JSON.stringify(eventById.get(node.eventId)?.payload ?? {}, null, 2),
  }
  if (node.kind === 'transition') {
    const x = explainTransition(events, node.eventId)
    const guards = x.guards.map(g => `${esc(g.guardId)} 判定 ${esc(String(g.result))}`).join('、')
    return { ...base, chain: x.causalChain, title: `为什么 ${esc(x.from)} → ${esc(x.to)}?`,
      bodyHtml: `触发: ${esc(x.trigger.causedBySummary ?? x.trigger.name)}<br>guard: ${guards || '(无)'}` }
  }
  if (node.kind === 'llm') {
    const x = explainLlmCall(events, node.eventId)
    return { ...base, chain: x.causalChain, title: `为什么这次 LLM 调用?`,
      bodyHtml: `state: ${esc(x.fsmState ?? '(未知)')}<br>触发: ${esc(x.trigger.causedBySummary ?? '(无上游)')}<br>prompt 由 ${x.regionCount} 个 region 拼成` }
  }
  if (node.kind === 'tool') {
    const x = explainToolCall(events, node.eventId)
    return { ...base, chain: x.causalChain, title: `工具 ${esc(x.toolName)}`,
      bodyHtml: `调用方: ${esc(x.trigger.causedBySummary ?? '(无上游)')}<br>入参: ${esc(JSON.stringify(x.input))}<br>出参: ${esc(JSON.stringify(x.output ?? null))}` }
  }
  // output
  const evt = eventById.get(node.eventId)
  const out = (evt?.payload as { lastTextOutput?: string; status?: string } | undefined)
  return { ...base, chain: [], title: '为什么是这个结果?',
    bodyHtml: `输出: ${esc(out?.lastTextOutput ?? out?.status ?? '')}<br>由上游决策产生(点 ← 谁导致的 下钻)` }
}

/**
 * Causal drill-down trace viewer: decision spine + Why panel + a raw timeline
 * tab. Self-contained static HTML; all explanations precomputed and embedded,
 * client JS only looks them up. Pure projection (region content hydration is
 * the caller's job, passed via opts, same as renderHtml).
 */
export function renderViewer(events: Event[], opts: { regionContent?: Map<string, string> } = {}): string {
  const spine = buildDecisionSpine(events)
  const eventById = new Map<string, Event>()
  for (const e of events) eventById.set(e.id, e)

  const explanations: Record<string, unknown> = {}
  for (const node of spine.nodes) explanations[node.eventId] = panelRecord(events, node, eventById)

  const spineHtml = spine.nodes.map(n => {
    const cls = `node k-${n.kind}${n.kind === 'output' ? ' spine-output' : ''}`
    const why = n.kind === 'output' ? `<span class="why-entry">❓ 为什么是这个结果</span>` : ''
    return `<div class="${cls}" data-id="${esc(n.eventId)}">${esc(n.label)}${why}</div>`
  }).join('')

  const spineJson = JSON.stringify(spine).replace(/<\/script/gi, '<\\/script')
  const expsJson = JSON.stringify(explanations).replace(/<\/script/gi, '<\\/script')

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>milkie trace viewer</title>
<style>${VIEWER_STYLES}</style>
</head>
<body>
<h1>milkie trace viewer</h1>
<div class="tabs">
  <span class="tab active" data-tab="decision">决策视图</span>
  <span class="tab" data-tab="raw">原始时间线</span>
</div>
<div id="pane-decision" class="pane active">
  <div class="spine">${spineHtml || '<p style="color:#888;font-size:12px">无决策事件</p>'}</div>
  <div id="why-panel"><p class="ph">从底部输出 ❓ 或任一决策点开始</p></div>
</div>
<div id="pane-raw" class="pane">${renderTimelineSections(events, opts)}</div>
<script type="application/json" id="spine-data">${spineJson}</script>
<script type="application/json" id="explanations-data">${expsJson}</script>
<script>${VIEWER_SCRIPT}</script>
</body>
</html>`
}
```

注:raw tab 复用 `renderTimelineSections`,但它内部也嵌了 `id="region-content"`/`id="trace-data"` 脚本 + `.entry` 等 class,而 viewer 自己的 CSS/JS 用 `.node`/`.tab`/`#why-panel` 等不冲突的选择器。**raw tab 的折叠/筛选交互需要 html.ts 的 SCRIPT**——本 MVP 的 raw tab 仅保证"可见全时间线 + payload 已内联",其 `.entry` 折叠 JS 暂不引入(viewer JS 不含);若实测要 raw 也能折叠,作为收尾微调把 `SCRIPT` 一并内联(class 不冲突)。

- [ ] **Step 5: 运行,确认通过** — `npx jest src/__tests__/render-viewer.test.ts` → PASS(2);`npx jest`(全量)无回归;tsc clean

- [ ] **Step 6: 提交**

```bash
git add src/trace/render/viewer.ts src/trace/render/viewer-template.ts src/__tests__/render-viewer.test.ts
git commit -m "feat(#64): renderViewer — decision spine + why panel + drill JS + raw tab"
```

---

## Task 5: CLI `trace report` → renderViewer

**Files:** Modify `src/cli/main.ts`

- [ ] **Step 1: 改命令** — READ the `report <runId>` action. Change the final `stdout.push(renderHtml(events, { regionContent }))` to use the viewer, and swap the import:
```ts
// import line: replace renderHtml import with renderViewer
const { renderViewer } = await import('../trace/render/viewer.js')
// ... unchanged events + regionContent hydration ...
stdout.push(renderViewer(events, { regionContent }))
```
Keep everything else (milkieDir/runsDir/findDescendantRuns/regionContent hydration) identical. Leave the `render-html` command UNCHANGED (it still emits the flat renderHtml — a no-objectStore path).

- [ ] **Step 2: 验证** — `npx tsc --noEmit` clean; `npm run build` succeeds.

- [ ] **Step 3: 提交**

```bash
git add src/cli/main.ts
git commit -m "feat(#64): trace report emits the decision viewer"
```

---

## 收尾

- [ ] **全量测试 + 构建** — `npx jest && npm run build` 全绿。

- [ ] **浏览器实测(验收,用户要求)**:用 dist + 一份真实/合成 run 生成 viewer HTML,dev-browser 打开:① 点底部输出 ❓ → Why 面板出"因"的解释 + 脊柱该因节点 `.cause` 高亮;② 点"← 谁导致的" → `.selected` 上移、面板刷新(两次点击到一个决策);③ 切"原始时间线" tab 可见全事件。截图给用户。**据实测决定是否把 html.ts SCRIPT 引入让 raw tab 也可折叠(Task 4 Step 4 注)**。

- [ ] **开 PR**:`Closes #64`;说明复用 #26/#30/#31/#33/#34、viewer 主输出 + raw tab、MVP 砍 DAG/diff。

---

## Self-Review(plan 作者已核对)

**Spec 覆盖:** §3.1 buildDecisionSpine(含 causeDecisionId)→ Task 1 ✓ §3.2 explainToolCall → Task 2 ✓ §4 viewer 渲染 + raw tab → Task 3(抽 sections)+ Task 4 ✓ §5 下钻数据流(渲染期预算 + JS 查表)→ Task 4 ✓ §5 CLI → Task 5 ✓ §7 测试(投影/渲染/浏览器)→ Task 1/2/4 + 收尾浏览器 ✓ §9 验收 → 收尾浏览器 ✓。

**占位扫描:** 无 TBD/TODO;每步含完整代码。Task 4 注明 raw-tab 折叠 JS 为实测后微调点(非占位,是明确的条件性收尾)。

**类型一致性:** `DecisionNode`/`DecisionSpine`(Task1)在 Task4 viewer.ts 引用一致;`ToolCallExplanation`(Task2)在 Task4 引用一致;`renderTimelineSections(events,opts)`(Task3)在 Task4 viewer.ts 调用一致;`buildDecisionSpine`/`explainToolCall`/`explainTransition`/`explainLlmCall` 签名前后一致;客户端内嵌结构 `{title,bodyHtml,causeDecisionId,chain,rawJson}` 在 viewer.ts 产出与 viewer-template.ts JS 消费一致。
