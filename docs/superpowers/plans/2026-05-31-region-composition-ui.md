# region composition + 内容预览 UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 核心静态报告里点 `llm.requested` 能看到 "Assembled by N regions"——参与 region 的 metadata + stability 配色 + 内容预览(折叠/全文)+ 同 hash 报告级去重标注复用次数。

**Architecture:** 复用 #23 的 `RegionContextView.contextRefsAt`(某时刻活跃 region 集,纯)。新增纯函数 `regionReuseCounts`。`renderHtml` 加可选 `opts.regionContent`(`Map<contentHash, content>`),内容由 CLI `trace report` 从 objectStore hydrate 后传入——renderHtml 保持纯函数。内容按 hash 内联到一个 `<script id="region-content">` 注册表(去重),行按 `data-hash` 展开时从注册表取。

**Tech Stack:** TypeScript;Jest(`npx jest <file>`);现有 `src/trace/render/{html,template,tree}.ts`、`src/trace/RegionContextView.ts`、`src/cli/main.ts`、`FileTraceObjectStore`。

参考 spec:`docs/superpowers/specs/2026-05-31-region-composition-ui-design.md`。**测试运行器是 JEST**。

---

## File Structure

| 文件 | 责任 | 改动 |
|---|---|---|
| `src/trace/RegionContextView.ts` | region-context 投影 | +`regionReuseCounts(events)` |
| `src/trace/render/html.ts` | 静态报告渲染 | `renderHtml` 加 `opts.regionContent`;llm 条目渲 "Assembled by" 块;嵌入内容注册表 |
| `src/trace/render/template.ts` | CSS/JS | stability 配色 + assembled/preview 样式 + 行展开 JS |
| `src/cli/main.ts` | CLI | `trace report` hydrate region 内容并传入 renderHtml |
| `src/__tests__/regionReuseCounts.test.ts` | 单测 | 新建 |
| `src/__tests__/render-html.test.ts` | 单测 | Assembled-by + 去重 + 降级断言 |

---

## Task 1: regionReuseCounts — 报告级 contentHash 引用计数

**Files:**
- Modify: `src/trace/RegionContextView.ts`
- Test: `src/__tests__/regionReuseCounts.test.ts`

- [ ] **Step 1: 写失败测试**

```ts
import { regionReuseCounts } from '../trace/RegionContextView'
import type { Event } from '../trace/types'

const ev = (id: string, type: string, payload: unknown): Event =>
  ({ id, runId: 'r1', actor: 'a', type: type as Event['type'], timestamp: 0, payload })

const regionAdded = (id: string, contentHash?: string) =>
  ev(`add-${id}`, 'region.added', { id, target: 'message', section: 's', stability: 'volatile', reason: 'r', ...(contentHash ? { contentHash } : {}) })

describe('regionReuseCounts', () => {
  it('counts how many llm.requested active-sets reference each contentHash', () => {
    const events: Event[] = [
      regionAdded('h', 'HASH'),                 // region h, content HASH
      ev('llm1', 'llm.requested', {}),          // active set: {h} → HASH ×1 so far
      ev('llm2', 'llm.requested', {}),          // h still active → HASH ×2
    ]
    const counts = regionReuseCounts(events)
    expect(counts.get('HASH')).toBe(2)
  })

  it('single reference counts as 1', () => {
    const events: Event[] = [regionAdded('h', 'HASH'), ev('llm1', 'llm.requested', {})]
    expect(regionReuseCounts(events).get('HASH')).toBe(1)
  })

  it('ignores regions without a contentHash and llm calls with no active regions', () => {
    const events: Event[] = [
      ev('llm0', 'llm.requested', {}),          // no regions yet
      regionAdded('h'),                         // no contentHash
      ev('llm1', 'llm.requested', {}),
    ]
    expect(regionReuseCounts(events).size).toBe(0)
  })
})
```

- [ ] **Step 2: 运行,确认失败**

Run: `npx jest src/__tests__/regionReuseCounts.test.ts`
Expected: FAIL（`regionReuseCounts` 未导出）。

- [ ] **Step 3: 实现**(在 `src/trace/RegionContextView.ts` 末尾追加)

```ts
/**
 * Report-wide reuse count: for every llm.requested, fold the active region set
 * (contextRefsAt 'at') and tally each region's contentHash. The value is how
 * many (llm.requested × region) references share that content — used by the UI
 * to dedup identical content and annotate "复用 ×N". Pure.
 */
export function regionReuseCounts(events: Event[]): Map<string, number> {
  const counts = new Map<string, number>()
  for (const event of events) {
    if (event.type !== 'llm.requested') continue
    for (const ref of contextRefsAt(events, event.id, 'at').values()) {
      if (!ref.contentHash) continue
      counts.set(ref.contentHash, (counts.get(ref.contentHash) ?? 0) + 1)
    }
  }
  return counts
}
```

- [ ] **Step 4: 运行,确认通过**

Run: `npx jest src/__tests__/regionReuseCounts.test.ts`
Expected: PASS（3 用例）。Also `npx tsc --noEmit` clean.

- [ ] **Step 5: 提交**

```bash
git add src/trace/RegionContextView.ts src/__tests__/regionReuseCounts.test.ts
git commit -m "feat(#26): regionReuseCounts — report-wide contentHash reference tally"
```

---

## Task 2: renderHtml "Assembled by" 块 + 内容注册表 + 样式

**Files:**
- Modify: `src/trace/render/html.ts`
- Modify: `src/trace/render/template.ts`
- Test: `src/__tests__/render-html.test.ts`

- [ ] **Step 1: 写失败测试**(`src/__tests__/render-html.test.ts`,复用文件已有 `e({...})` 助手与 `renderHtml`/`Event` 导入)

```ts
describe('#26 Assembled by', () => {
  const region = (id: string, stability: string, contentHash?: string) =>
    e({ id: `add-${id}`, runId: 'r1', type: 'region.added', timestamp: 1,
        payload: { id, target: 'message', section: 'history', stability, reason: 'turn-archived',
          ...(contentHash ? { contentHash } : {}) } })

  it('renders an Assembled by block on llm.requested with metadata + stability class', () => {
    const events: Event[] = [
      region('header', 'immutable', 'H1'),
      e({ id: 'llm', runId: 'r1', type: 'llm.requested', timestamp: 2, payload: { model: 'm' } }),
    ]
    const html = renderHtml(events, { regionContent: new Map([['H1', 'SYSTEM PROMPT TEXT']]) })
    expect(html).toContain('Assembled by')
    expect(html).toContain('header')
    expect(html).toContain('stab-immutable')
    expect(html).toContain('data-hash="H1"')
    // content embedded once in the registry
    expect(html).toContain('SYSTEM PROMPT TEXT')
  })

  it('dedups identical content across prompts and annotates reuse count', () => {
    const events: Event[] = [
      region('header', 'immutable', 'H1'),
      e({ id: 'llm1', runId: 'r1', type: 'llm.requested', timestamp: 2, payload: { model: 'm' } }),
      e({ id: 'llm2', runId: 'r1', type: 'llm.requested', timestamp: 3, payload: { model: 'm' } }),
    ]
    const html = renderHtml(events, { regionContent: new Map([['H1', 'SHARED-CONTENT-XYZ']]) })
    // embedded once in the registry, not per-prompt
    expect(html.split('SHARED-CONTENT-XYZ').length - 1).toBe(1)
    expect(html).toContain('复用 ×2')
  })

  it('degrades gracefully without region content (metadata only)', () => {
    const events: Event[] = [
      region('header', 'immutable', 'H1'),
      e({ id: 'llm', runId: 'r1', type: 'llm.requested', timestamp: 2, payload: { model: 'm' } }),
    ]
    const html = renderHtml(events)   // no opts
    expect(html).toContain('Assembled by')
    expect(html).toContain('header')
    expect(html).toContain('(内容不可用)')
  })

  it('renders no Assembled by block for an llm.requested with no active regions', () => {
    const events: Event[] = [
      e({ id: 'llm', runId: 'r1', type: 'llm.requested', timestamp: 2, payload: { model: 'm' } }),
    ]
    const html = renderHtml(events)
    expect(html).not.toContain('Assembled by')
  })
})
```

- [ ] **Step 2: 运行,确认失败**

Run: `npx jest src/__tests__/render-html.test.ts -t "Assembled by"`
Expected: FAIL。

- [ ] **Step 3: html.ts — 导入 + renderHtml 签名与注册表**

顶部 import 加(与现有 `.js` 风格一致):
```ts
import { contextRefsAt, regionReuseCounts, type RegionContentRef } from '../RegionContextView.js'
```

把 `renderHtml` 签名改为接收 opts,并构建 region 渲染上下文 + 内容注册表。替换现有 `renderHtml`:
```ts
export function renderHtml(events: Event[], opts: { regionContent?: Map<string, string> } = {}): string {
  const tree = buildTimelineTree(events)
  const eventById = new Map<string, Event>()
  for (const evt of events) eventById.set(evt.id, evt)

  const explanations = new Map<string, TransitionExplanation>()
  for (const evt of events) {
    if (evt.type === 'fsm.transition') explanations.set(evt.id, explainTransition(events, evt.id))
  }

  const regionContent = opts.regionContent ?? new Map<string, string>()
  const regionCtx: RegionCtx = { events, reuseCounts: regionReuseCounts(events), regionContent }

  const registryJson = JSON.stringify(Object.fromEntries(regionContent))
    .replace(/<\/script/gi, '<\\/script')
  const dataJson = JSON.stringify(events).replace(/<\/script/gi, '<\\/script')

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
  <span class="chip" data-kind="region">region</span>
  <span class="chip" data-kind="fsm">fsm</span>
</div>
${tree.map(n => renderNode(n, eventById, explanations, regionCtx)).join('')}
<script type="application/json" id="region-content">${registryJson}</script>
<script type="application/json" id="trace-data">${dataJson}</script>
<script>${SCRIPT}</script>
</body>
</html>`
}
```

Add the `RegionCtx` type (near the top of the file, after imports):
```ts
interface RegionCtx {
  events:        Event[]
  reuseCounts:   Map<string, number>
  regionContent: Map<string, string>
}
```

- [ ] **Step 4: html.ts — Assembled by 渲染助手**

在 `renderEntry` 之前加:
```ts
function renderAssembled(requestedId: string, ctx: RegionCtx): string {
  const refs: RegionContentRef[] = [...contextRefsAt(ctx.events, requestedId, 'at').values()]
  if (refs.length === 0) return ''
  const rows = refs.map(ref => {
    const hash    = ref.contentHash
    const reuse   = hash ? (ctx.reuseCounts.get(hash) ?? 1) : 1
    const reuseB  = reuse > 1 ? ` <span class="reuse">复用 ×${reuse}</span>` : ''
    const avail   = !!(hash && ctx.regionContent.has(hash))
    const note    = hash && !avail ? ` <span class="ar-note">(内容不可用)</span>` : ''
    const meta    = `<span class="ar-id">${esc(ref.id)}</span> · ${esc(ref.section)} · ${esc(ref.target)} · ${esc(ref.reason)}`
    const preview = avail ? `<pre class="region-preview"></pre>` : ''
    const dataH   = hash ? ` data-hash="${esc(hash)}"` : ''
    return `<div class="ar-row stab-${esc(ref.stability)}"${dataH}><div class="ar-meta">${meta}${reuseB}${note}</div>${preview}</div>`
  }).join('')
  return `<div class="assembled"><div class="ar-head">Assembled by ${refs.length} regions</div>${rows}</div>`
}
```

- [ ] **Step 5: html.ts — renderEntry/renderNode 接 RegionCtx,llm 条目渲 Assembled**

`renderEntry` 改签名,llm 条目加 Assembled 块(其它分支不变;保留 #33 的 `explanations`/`entryEventIds`/`renderWhy`/锚点逻辑):
```ts
function renderEntry(
  entry: TimelineEntry,
  eventById: Map<string, Event>,
  explanations: Map<string, TransitionExplanation>,
  regionCtx: RegionCtx,
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
  const assembled = entry.kind === 'llm' ? renderAssembled(entry.requestedId, regionCtx) : ''
  return `<div class="entry ${entry.kind}" data-kind="${entry.kind}" id="ev-${esc(primaryId)}">`
       + extraAnchors
       + `<div class="entry-head">`
       + `<span class="icon">${icon}</span>`
       + `<span class="summary">${summaryFor(entry)}</span>`
       + `<span class="ts">${entry.timestamp}</span>`
       + `</div>`
       + why
       + assembled
       + `<pre class="payload">${payloadFor(entry, eventById)}</pre>`
       + `</div>`
}
```

`renderNode` 透传 `regionCtx`:
```ts
function renderNode(
  node: TimelineNode,
  eventById: Map<string, Event>,
  explanations: Map<string, TransitionExplanation>,
  regionCtx: RegionCtx,
): string {
  const status = node.status ?? 'in-flight'
  const badgeClass = BADGE_STATUS_CLASSES.has(status) ? ' ' + status : ''
  return `<section class="run" data-run-id="${esc(node.runId)}">`
       + `<div class="run-head">`
       + `<strong>${esc(node.agentId ?? '(unknown)')}</strong>`
       + `<span class="run-id">${esc(node.runId)}</span>`
       + `<span class="badge${badgeClass}">${esc(status)}</span>`
       + `</div>`
       + node.entries.map(en => renderEntry(en, eventById, explanations, regionCtx)).join('')
       + (node.children.length > 0
           ? `<div class="child-run">${node.children.map(n => renderNode(n, eventById, explanations, regionCtx)).join('')}</div>`
           : '')
       + `</section>`
}
```

- [ ] **Step 6: template.ts — CSS + 行展开 JS**

在 `STYLES` 模板末尾(反引号前)追加:
```css
  .assembled { margin: 4px 0 4px 22px; padding: 6px 10px; border-left: 3px solid #b58;
               background: rgba(0,0,0,0.03); font-size: 12px; line-height: 1.5; }
  .ar-head { font-weight: 600; margin-bottom: 4px; }
  .ar-row { padding: 3px 0 3px 8px; border-left: 3px solid transparent; cursor: pointer; }
  .ar-row[data-hash]:hover { background: rgba(0,0,0,0.04); }
  .ar-id { font-family: ui-monospace, SFMono-Regular, monospace; }
  .ar-note { color: #a13; }
  .reuse { color: #8a5a00; font-size: 11px; }
  .region-preview { display: none; margin-top: 4px; background: #fafafa; padding: 8px;
                    font-family: ui-monospace, monospace; font-size: 11px; white-space: pre-wrap;
                    border-radius: 4px; max-height: 280px; overflow: auto; }
  .ar-row.open .region-preview { display: block; }
  .stab-immutable      { border-left-color: #2d6a2d; }
  .stab-session-stable { border-left-color: #1a56db; }
  .stab-turn-stable    { border-left-color: #8a5a00; }
  .stab-volatile       { border-left-color: #a13; }
```

在 `SCRIPT` 的 click 处理器里,**`.why a` 早退之后、entry toggle 之前**插入 region-row 展开逻辑(从注册表按 data-hash 取内容,首次展开时填充):
```js
      var arRow = ev.target.closest('.ar-row');
      if (arRow && arRow.dataset.hash) {
        ev.stopPropagation();
        arRow.classList.toggle('open');
        var pre = arRow.querySelector('.region-preview');
        if (pre && !pre.dataset.loaded) {
          var reg = JSON.parse((document.getElementById('region-content') || {}).textContent || '{}');
          var c = reg[arRow.dataset.hash];
          pre.textContent = (c != null) ? c : '(内容不可用)';
          pre.dataset.loaded = '1';
        }
        return;
      }
```
(放在 `if (ev.target.closest && ev.target.closest('.why a')) return;` 之后,`var entry = ev.target.closest('.entry');` 之前。)

- [ ] **Step 7: 运行,确认通过**

Run: `npx jest src/__tests__/render-html.test.ts && npx jest src/__tests__/render-tree.test.ts`
Expected: PASS（含 #26 四用例 + 原有用例无回归)。Also `npx tsc --noEmit` clean.

- [ ] **Step 8: 提交**

```bash
git add src/trace/render/html.ts src/trace/render/template.ts src/__tests__/render-html.test.ts
git commit -m "feat(#26): Assembled by block on llm.requested — regions, stability, content preview, hash dedup"
```

---

## Task 3: CLI `trace report` —— hydrate region 内容并传入 renderHtml

**Files:**
- Modify: `src/cli/main.ts`(`report <runId>` 命令,约 `:177-192`)

- [ ] **Step 1: 改 `trace report` 命令**

先 READ `src/cli/main.ts` 的 `report` 命令,确认 `runsDir`/`milkieDir` 的推导方式(`:33-34` 与 `:203-204` 用 `path.join(milkieDir, 'objects')` 建 `FileTraceObjectStore`;`FileTraceObjectStore` 已在 `:6` 导入)。把 `report` 命令体改为:在读完 events 后,构造 objectStore、收集所有 `region.added` 的 `contentHash`、`getCanonical` 批量取、组装 `Map<hash,content>`,传给 `renderHtml`:

```ts
// 在 events 收集完之后、renderHtml 之前:
const traceObjectStore = new FileTraceObjectStore(path.join(milkieDir, 'objects'))
const hashes = new Set<string>()
for (const ev of events) {
  if (ev.type === 'region.added') {
    const h = (ev.payload as { contentHash?: string }).contentHash
    if (h) hashes.add(h)
  }
}
const regionContent = new Map<string, string>()
for (const h of hashes) {
  const c = await traceObjectStore.getCanonical(h)
  if (c !== undefined) regionContent.set(h, c)
}
stdout.push(renderHtml(events, { regionContent }))
```
说明:`milkieDir` 是 `report` 命令里推导 `runsDir` 用到的同一个目录基(读代码确认它的变量名——若该命令只算了 `runsDir`,用 `path.dirname(runsDir)` 取回 milkieDir,或仿照 `:33` 重新算 `milkieDir`)。`render-html`(从 `--input` JSONL,无 objectStore)**保持不变**,继续 `renderHtml(events)`,自动降级 metadata-only。

- [ ] **Step 2: 类型检查 + 构建**

Run: `npx tsc --noEmit`
Expected: 干净。(CLI 无现成单测;此命令是 I/O 装配,靠 tsc + Task 2 的 renderHtml 测试保证。)

- [ ] **Step 3: 手验(可选,有 run 数据时)**

Run: `node dist/cli/main.js trace report <某 runId> > /tmp/report.html` 后浏览器打开,确认 llm.requested 出 "Assembled by"、点 region 行展开看到内容、复用标注。无 run 数据则跳过。

- [ ] **Step 4: 提交**

```bash
git add src/cli/main.ts
git commit -m "feat(#26): trace report hydrates region content from objectStore into the HTML report"
```

---

## 收尾

- [ ] **全量测试 + 构建**

Run: `npx jest && npm run build`
Expected: 全绿。

- [ ] **开 PR**:body 带 `Closes #26`;说明范围(核心能力 + 核心静态报告;示例 app audit panel 复用为 follow-up),复用 #23 的 RegionContextView。

---

## Self-Review(plan 作者已核对)

**Spec 覆盖:** §4.1 regionReuseCounts → Task 1 ✓ §4.2 renderHtml opts → Task 2 Step 3 ✓ §4.3 CLI hydrate → Task 3 ✓ §5 Assembled-by(metadata/stability/预览/去重)→ Task 2 Step 4-6 ✓ §6 降级(无内容/无 contentHash/无 region)→ Task 2 测试三、四 ✓ §7 测试 → Task 1/2 ✓ §9 验收(清单/内容/长 prompt 去重)→ Task 2 ✓。

**占位扫描:** 无 TBD/TODO;改代码步骤均含完整代码。Task 3 Step 1 的 `milkieDir` 变量名以实际 `report` 命令为准——已给出明确推导方法,非内容缺失。

**类型一致性:** `RegionCtx { events, reuseCounts, regionContent }`(Task 2 Step 3 定义)在 renderAssembled/renderEntry/renderNode(Step 4-5)一致;`regionReuseCounts(events): Map<string,number>`(Task 1)在 Step 3 调用一致;`contextRefsAt`/`RegionContentRef` 取自 `RegionContextView`;`renderHtml(events, opts?)` 在 Task 3 调用一致;`opts.regionContent: Map<contentHash,string>` 全程一致。
