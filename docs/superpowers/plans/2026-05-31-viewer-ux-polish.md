# #71 决策 viewer UX polish — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 output ❓ 在真实 run 能下钻、输出按 markdown 渲染、因果链精简到决策跳且标签带入参。

**Architecture:** 三处独立改动。A 在 runtime（`RecordingIOPort.detach` 给 `agent.run.completed` 填 `causedBy = 最后 llm.responded`，replay 安全的裸 uuid 元数据）。B 新增纯函数 `renderMarkdown`（先转义后渲染最小 markdown 子集），viewer 的 output 面板改用它。C 在 `summarizeEvent` 给工具标签加截断入参、在 `viewer.ts` 把每个面板的因果链按脊柱决策 id 过滤、output 无因时显示诚实兜底文案。

**Tech Stack:** TypeScript、ts-jest（从仓库根 `npx jest <file>` 运行）、纯投影/渲染（无 I/O）。

**Spec:** `docs/superpowers/specs/2026-05-31-viewer-ux-polish-design.md`

---

## File Structure

| 文件 | 职责 | 改动 |
|---|---|---|
| `src/trace/RecordingIOPort.ts` | 录制 IO → 事件 | `detach` 填 causedBy |
| `src/trace/render/markdown.ts` | 极简 markdown→HTML 纯函数 | 新建 |
| `src/trace/render/viewer.ts` | 决策 viewer 渲染 | output 用 markdown；chain 按 spineIds 过滤；兜底文案 |
| `src/trace/diagnostics/summarizeEvent.ts` | 事件单行标签 | 工具标签加截断入参 |

测试：`CausedByGraph.test.ts`、`markdown.test.ts`（新）、`summarizeEvent.test.ts`、`render-viewer.test.ts`。

---

## Task 1: A — `agent.run.completed.causedBy`（output ❓ 可下钻）

**Files:**
- Modify: `src/trace/RecordingIOPort.ts`（`detach`，约 line 134-144）
- Test: `src/__tests__/CausedByGraph.test.ts`

- [ ] **Step 1: Write the failing tests**

加到 `src/__tests__/CausedByGraph.test.ts` 的 `describe('causedBy densify …')` 内（紧接 edge 2 测试之后）：

```typescript
  it('edge 3: agent.run.completed.causedBy is the final llm.responded', async () => {
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(new ExecInnerPort(), store, 'r1', 'runtime', undefined, new CausalCursor())

    await port.attach(START)
    await port.invokeLLM(req())
    await port.detach({ status: 'completed' })

    const evs = await events(store)
    const llmResponded = firstOf(evs, 'llm.responded')
    const completed    = firstOf(evs, 'agent.run.completed')
    expect(completed.causedBy).toBe(llmResponded.id)
  })

  it('agent.run.completed has no causedBy when the run made no LLM call', async () => {
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(new ExecInnerPort(), store, 'r1', 'runtime', undefined, new CausalCursor())

    await port.attach(START)
    await port.detach({ status: 'completed' })

    const completed = firstOf(await events(store), 'agent.run.completed')
    expect(completed.causedBy).toBeUndefined()
  })
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx jest src/__tests__/CausedByGraph.test.ts -t "edge 3"`
Expected: FAIL — `completed.causedBy` is `undefined`（detach 当前不设 causedBy）。

- [ ] **Step 3: Implement — set causedBy in detach**

把 `src/trace/RecordingIOPort.ts` 的 `detach`（当前 line 134-144）改为：

```typescript
  async detach(payload: AgentRunCompletedPayload): Promise<void> {
    await this.flushPendingNondet()
    await this.store.append({
      id:        this.inner.uuid(),
      runId:     this.runId,
      type:      'agent.run.completed',
      actor:     this.actor,
      // The final output is produced by the last LLM response; link to it so the
      // output node can drill to the final decision (nearest-decision-ancestor).
      // causedBy is trace metadata (a bare uuid) — replay never compares it.
      ...(this.cursor?.lastLlmRespondedId ? { causedBy: this.cursor.lastLlmRespondedId } : {}),
      timestamp: this.inner.now(),
      payload,
    })
  }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx jest src/__tests__/CausedByGraph.test.ts`
Expected: PASS（含新 2 条 + 既有全绿）。

- [ ] **Step 5: Run buildDecisionSpine tests (projection already handles it)**

Run: `npx jest src/__tests__/buildDecisionSpine.test.ts`
Expected: PASS — 既有用例 `done.causeDecisionId === 'fsm'` 已证明：completed 一旦带 causedBy，output 节点的 causeDecisionId 正确解析。无需改 buildDecisionSpine。

- [ ] **Step 6: Commit**

```bash
git add src/trace/RecordingIOPort.ts src/__tests__/CausedByGraph.test.ts
git commit -m "feat(#71): agent.run.completed.causedBy = final llm.responded so output drills"
```

---

## Task 2: B — `renderMarkdown` 纯函数（新模块）

**Files:**
- Create: `src/trace/render/markdown.ts`
- Test: `src/__tests__/markdown.test.ts`

- [ ] **Step 1: Write the failing tests**

新建 `src/__tests__/markdown.test.ts`：

```typescript
import { renderMarkdown } from '../trace/render/markdown'

describe('renderMarkdown', () => {
  it('renders headers # ## ### as h3 h4 h5', () => {
    expect(renderMarkdown('# A')).toBe('<h3>A</h3>')
    expect(renderMarkdown('## B')).toBe('<h4>B</h4>')
    expect(renderMarkdown('### C')).toBe('<h5>C</h5>')
  })

  it('renders bold and inline code', () => {
    expect(renderMarkdown('**x**')).toBe('<p><strong>x</strong></p>')
    expect(renderMarkdown('`y`')).toBe('<p><code>y</code></p>')
  })

  it('renders unordered and ordered lists', () => {
    expect(renderMarkdown('- a\n- b')).toBe('<ul><li>a</li><li>b</li></ul>')
    expect(renderMarkdown('1. a\n2. b')).toBe('<ol><li>a</li><li>b</li></ol>')
  })

  it('groups consecutive non-empty lines into a paragraph with <br>', () => {
    expect(renderMarkdown('l1\nl2')).toBe('<p>l1<br>l2</p>')
  })

  it('escapes HTML before applying markdown (no injection)', () => {
    expect(renderMarkdown('<script>alert(1)</script>')).toBe('<p>&lt;script&gt;alert(1)&lt;/script&gt;</p>')
  })

  it('returns empty string for empty input', () => {
    expect(renderMarkdown('')).toBe('')
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx jest src/__tests__/markdown.test.ts`
Expected: FAIL — `Cannot find module '../trace/render/markdown'`。

- [ ] **Step 3: Implement `renderMarkdown`**

新建 `src/trace/render/markdown.ts`：

```typescript
/**
 * Minimal, dependency-free markdown → HTML for the trace viewer's final-output
 * panel. Escapes HTML FIRST, then renders a small subset (headers, bold, inline
 * code, lists, paragraphs). All emitted tags are ours — no raw HTML passes
 * through, so the result is safe to inject via innerHTML.
 */
function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

function inline(s: string): string {
  // s is already HTML-escaped. Bold first, then inline code.
  return s
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
}

export function renderMarkdown(text: string): string {
  const lines = escapeHtml(text).split(/\r?\n/)
  const out: string[] = []
  let para: string[] = []
  let list: { type: 'ul' | 'ol'; items: string[] } | null = null

  const flushPara = () => {
    if (para.length) { out.push(`<p>${para.map(inline).join('<br>')}</p>`); para = [] }
  }
  const flushList = () => {
    if (list) {
      out.push(`<${list.type}>${list.items.map(i => `<li>${inline(i)}</li>`).join('')}</${list.type}>`)
      list = null
    }
  }
  const flushAll = () => { flushPara(); flushList() }

  for (const line of lines) {
    const h  = line.match(/^(#{1,3})\s+(.*)$/)
    const ul = line.match(/^[-*]\s+(.*)$/)
    const ol = line.match(/^\d+\.\s+(.*)$/)
    if (h) {
      flushAll()
      const level = h[1]!.length + 2  // # -> h3, ## -> h4, ### -> h5
      out.push(`<h${level}>${inline(h[2]!)}</h${level}>`)
    } else if (ul) {
      flushPara()
      if (!list || list.type !== 'ul') { flushList(); list = { type: 'ul', items: [] } }
      list.items.push(ul[1]!)
    } else if (ol) {
      flushPara()
      if (!list || list.type !== 'ol') { flushList(); list = { type: 'ol', items: [] } }
      list.items.push(ol[1]!)
    } else if (line.trim() === '') {
      flushAll()
    } else {
      flushList()
      para.push(line)
    }
  }
  flushAll()
  return out.join('')
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx jest src/__tests__/markdown.test.ts`
Expected: PASS（6 条）。

- [ ] **Step 5: Commit**

```bash
git add src/trace/render/markdown.ts src/__tests__/markdown.test.ts
git commit -m "feat(#71): renderMarkdown — minimal safe markdown→HTML for the viewer"
```

---

## Task 3: C1 — `summarizeEvent` 工具标签带截断入参

**Files:**
- Modify: `src/trace/diagnostics/summarizeEvent.ts`
- Test: `src/__tests__/summarizeEvent.test.ts`

- [ ] **Step 1: Write the failing test**

加到 `src/__tests__/summarizeEvent.test.ts` 的 describe 内：

```typescript
  it('appends a short input summary to tool.requested with non-empty input', () => {
    expect(summarizeEvent(ev('tool.requested', { toolName: 'grep', input: { pattern: '孔明' } })))
      .toBe('tool.requested(grep · {"pattern":"孔明"})')
  })

  it('truncates a long input summary to 24 chars with an ellipsis', () => {
    const long = summarizeEvent(ev('tool.requested', { toolName: 'grep', input: { pattern: 'x'.repeat(100) } }))
    expect(long.startsWith('tool.requested(grep · ')).toBe(true)
    expect(long.endsWith('…)')).toBe(true)
    // 24-char input summary inside the parens
    expect(long).toContain(' · ' + '{"pattern":"' + 'x'.repeat(12) + '…)')
  })

  it('omits the input summary for empty or absent input (tool.requested / tool.responded)', () => {
    expect(summarizeEvent(ev('tool.requested', { toolName: 'search', input: {} })))
      .toBe('tool.requested(search)')
    expect(summarizeEvent(ev('tool.responded', { toolName: 'classify_intent', output: {} })))
      .toBe('tool.responded(classify_intent)')
  })
```

注：第 3 条与现有 `'labels tool events with the tool name'` 用例的断言（`input: {}` → `tool.requested(search)`）一致——空 input 不加摘要，故现有用例不破。

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx jest src/__tests__/summarizeEvent.test.ts -t "appends a short input"`
Expected: FAIL — 当前输出 `tool.requested(grep)`（无入参）。

- [ ] **Step 3: Implement — append truncated input summary**

把 `src/trace/diagnostics/summarizeEvent.ts` 改为：

```typescript
import type { Event } from '../types.js'

/**
 * Human-readable one-line label for an event. Shared by the HTML "Why?" block
 * and the decision viewer's causal chain, so both speak the same language.
 * tool.requested labels carry a short, truncated input summary so repeated
 * calls to the same tool (e.g. grep "孔明" vs grep "诸葛亮") are distinguishable.
 */
function shortInput(input: unknown): string | undefined {
  if (input === undefined || input === null) return undefined
  const json = JSON.stringify(input)
  if (json === undefined || json === '{}' || json === 'null' || json === '""') return undefined
  return json.length > 24 ? json.slice(0, 24) + '…' : json
}

export function summarizeEvent(event: Event): string {
  const p = event.payload as { toolName?: unknown; input?: unknown }
  if ((event.type === 'tool.requested' || event.type === 'tool.responded')
      && typeof p?.toolName === 'string') {
    const arg = event.type === 'tool.requested' ? shortInput(p.input) : undefined
    return arg ? `${event.type}(${p.toolName} · ${arg})` : `${event.type}(${p.toolName})`
  }
  return event.type
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx jest src/__tests__/summarizeEvent.test.ts`
Expected: PASS（新 3 条 + 既有 2 条）。

- [ ] **Step 5: Run connected suites (label change blast radius)**

Run: `npx jest src/__tests__/explainLlmCall.test.ts src/__tests__/explainTransition.test.ts src/__tests__/render-html.test.ts`
Expected: PASS — 这些断言的是 `tool.responded(...)`（responded 无 input → 不加摘要），故不受影响。若任何 `tool.requested(...)` 断言出现且其事件含非空 input 而变红，更新该断言为带入参的新文案（预期连带，非 bug）。

- [ ] **Step 6: Commit**

```bash
git add src/trace/diagnostics/summarizeEvent.ts src/__tests__/summarizeEvent.test.ts
git commit -m "feat(#71): summarizeEvent — disambiguate tool.requested with truncated input"
```

---

## Task 4: C2+C3+B-wiring — viewer.ts：markdown 输出 / 链精简 / 兜底

**Files:**
- Modify: `src/trace/render/viewer.ts`（`panelRecord` + `renderViewer`）
- Test: `src/__tests__/render-viewer.test.ts`

- [ ] **Step 1: Write the failing tests**

加到 `src/__tests__/render-viewer.test.ts` 的 `describe('renderViewer', …)` 内：

```typescript
  it('renders the output answer as markdown (not escaped literal)', () => {
    const events: Event[] = [
      e({ id: 'start', runId: 'r1', type: 'agent.run.started', timestamp: 1, payload: { agentId: 'x', goal: 'g', input: 'i', contextId: 'c' } }),
      e({ id: 'llm1', runId: 'r1', type: 'llm.requested', timestamp: 2, causedBy: 'start', payload: { model: 'm' } }),
      e({ id: 'lr1', runId: 'r1', type: 'llm.responded', timestamp: 3, causedBy: 'llm1', payload: {} }),
      e({ id: 'done', runId: 'r1', type: 'agent.run.completed', timestamp: 4, causedBy: 'lr1', payload: { status: 'completed', lastTextOutput: '## 标题\n**重点**' } }),
    ]
    const html = renderViewer(events)
    // explanations JSON is embedded; markdown must be rendered, not literal "##"
    expect(html).toContain('<h4>标题</h4>')
    expect(html).toContain('<strong>重点</strong>')
  })

  it('trims a panel causal chain to spine decisions only', () => {
    // scenario(): done <- fsm <- tres <- treq <- lr1 <- llm1 <- start.
    // The fsm transition panel's chain must drop non-decision events
    // (llm.responded 'lr1', tool.responded 'tres') and keep decisions only.
    const html = renderViewer(scenario())
    const exps = JSON.parse(html.match(/id="explanations-data">(.*?)<\/script>/s)![1]!)
    const fsmChain = exps['fsm'].chain.map((c: { eventId: string }) => c.eventId)
    expect(fsmChain).not.toContain('lr1')   // llm.responded — non-decision
    expect(fsmChain).not.toContain('tres')  // tool.responded — non-decision
    for (const id of fsmChain) expect(['llm1', 'treq', 'fsm', 'done', 'start']).toContain(id)
  })

  it('shows an honest fallback when the output node has no upstream decision', () => {
    const events: Event[] = [
      e({ id: 'start', runId: 'r1', type: 'agent.run.started', timestamp: 1, payload: { agentId: 'x', goal: 'g', input: 'i', contextId: 'c' } }),
      e({ id: 'done', runId: 'r1', type: 'agent.run.completed', timestamp: 2, payload: { status: 'completed', lastTextOutput: 'ok' } }),
    ]
    const html = renderViewer(events)
    const exps = JSON.parse(html.match(/id="explanations-data">(.*?)<\/script>/s)![1]!)
    expect(exps['done'].bodyHtml).toContain('无上游决策记录')
    expect(exps['done'].bodyHtml).not.toContain('点 ← 谁导致的')
  })
```

注：`scenario()` 与 `e()` helper 已存在于该测试文件顶部（见现有用例）。

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx jest src/__tests__/render-viewer.test.ts -t "markdown"`
Expected: FAIL — output 当前 `esc()` → `## 标题` 字面，无 `<h4>`。

- [ ] **Step 3: Implement — viewer.ts panelRecord + renderViewer**

a) 顶部 import 加：

```typescript
import { renderMarkdown } from './markdown.js'
```

b) `panelRecord` 签名加 `spineIds`，并加 `trim` helper、改 output 分支。把 `panelRecord(events, node, eventById, regionContent)` 改为接受 `spineIds: Set<string>`，整段函数替换为：

```typescript
function panelRecord(events: Event[], node: DecisionNode, eventById: Map<string, Event>, regionContent: Map<string, string>, spineIds: Set<string>): PanelRecord {
  const trim = (chain: Array<{ eventId: string; type: string; summary: string }>) =>
    chain.filter(c => spineIds.has(c.eventId))
  const base = {
    causeDecisionId: node.causeDecisionId,
    chain: [] as Array<{ eventId: string; type: string; summary: string }>,
    rawJson: JSON.stringify(eventById.get(node.eventId)?.payload ?? {}, null, 2),
  }
  if (node.kind === 'transition') {
    const x = explainTransition(events, node.eventId)
    const guards = x.guards.map(g => `${esc(g.guardId)} 判定 ${esc(String(g.result))}`).join('、')
    return { ...base, chain: trim(x.causalChain), title: `为什么 ${x.from} → ${x.to}?`,
      bodyHtml: `触发: ${esc(x.trigger.causedBySummary ?? x.trigger.name)}<br>guard: ${guards || '(无)'}` }
  }
  if (node.kind === 'llm') {
    const x = explainLlmCall(events, node.eventId)
    const refs: RegionContentRef[] = [...contextRefsAt(events, node.eventId, 'at').values()]
    const comp = refs.map(r => {
      const content = r.contentHash && regionContent.has(r.contentHash) ? regionContent.get(r.contentHash)! : undefined
      const meta = `${esc(r.id)} · ${esc(r.section)} · ${esc(r.stability)}`
      return content !== undefined
        ? `<details class="rdetail"><summary>${meta}</summary><pre class="rpre">${esc(content)}</pre></details>`
        : `<div class="rdetail">${meta} <span class="ar-na">(内容不可用)</span></div>`
    }).join('')
    return { ...base, chain: trim(x.causalChain), title: `为什么这次 LLM 调用?`,
      bodyHtml: `state: ${esc(x.fsmState ?? '(未知)')}<br>触发: ${esc(x.trigger.causedBySummary ?? '(无上游)')}`
        + `<div style="margin-top:8px;font-weight:600">Assembled by ${refs.length} regions</div>${comp}` }
  }
  if (node.kind === 'tool') {
    const x = explainToolCall(events, node.eventId)
    return { ...base, chain: trim(x.causalChain), title: `工具 ${x.toolName}`,
      bodyHtml: `调用方: ${esc(x.trigger.causedBySummary ?? '(无上游)')}<br>入参: ${esc(JSON.stringify(x.input))}<br>出参: ${esc(JSON.stringify(x.output ?? null))}` }
  }
  // output
  const evt = eventById.get(node.eventId)
  const out = (evt?.payload as { lastTextOutput?: string; status?: string } | undefined)
  const drill = node.causeDecisionId ? '由上游决策产生(点 ← 谁导致的 下钻)' : '（无上游决策记录）'
  return { ...base, chain: [], title: '为什么是这个结果?',
    bodyHtml: `<div>${renderMarkdown(out?.lastTextOutput ?? out?.status ?? '')}</div><div style="color:#888;font-size:11px;margin-top:8px">${drill}</div>` }
}
```

c) 在 `renderViewer` 里构造 `spineIds` 并传入。把现有

```typescript
  for (const node of spine.nodes) explanations[node.eventId] = panelRecord(events, node, eventById, regionContent)
```

改为：

```typescript
  const spineIds = new Set(spine.nodes.map(n => n.eventId))
  for (const node of spine.nodes) explanations[node.eventId] = panelRecord(events, node, eventById, regionContent, spineIds)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx jest src/__tests__/render-viewer.test.ts`
Expected: PASS（新 3 条 + 既有全绿）。

- [ ] **Step 5: Commit**

```bash
git add src/trace/render/viewer.ts src/__tests__/render-viewer.test.ts
git commit -m "feat(#71): viewer — markdown output, decision-only causal chain, honest output fallback"
```

---

## Task 5: 全量回归 + dev-browser 真实验证

**Files:** 无新代码改动（纯验证）。

- [ ] **Step 1: 全量单测 + 类型检查**

Run（仓库根）:
```bash
npx jest src/__tests__/ examples/agent-docs-qa/__tests__/server.test.ts
npx tsc --noEmit
```
Expected: 全绿。若有断言旧 `tool.requested(<name>)` 文案、且事件含非空 input 的测试变红，更新为带入参新文案（预期连带）。

- [ ] **Step 2: dev-browser 真实 doubao run 验证（覆盖 #71 三条验收）**

起 server（`cd examples/agent-docs-qa && PORT=7891 npx tsx server.ts`），发一条触发多次工具的问题，打开 Why tab：
1. 点输出 ❓ → 断言选中其上游决策（终轮 LLM）、面板出 why（**output ❓ 真能下钻**）。
2. output 面板的答案文本按 markdown 渲染（标题/粗体/列表，非字面 `##`）。
3. 因果链只剩决策跳；两次 grep 标签带不同入参可区分。

- [ ] **Step 3: 记录验证结论入 PR 描述。**

---

## Self-Review

**Spec coverage:**
- §3 A（detach causedBy）→ Task 1 ✓（buildDecisionSpine 投影已被既有测试覆盖，无需改）
- §4 B（renderMarkdown + 接入 output）→ Task 2（模块）+ Task 4 Step 3b（接入）✓
- §5 C1（summarizeEvent 入参）→ Task 3 ✓
- §5 C2（链按 spineIds 过滤）→ Task 4（trim + spineIds 传参）✓
- §5 C3（output 兜底文案）→ Task 4（drill 条件）✓
- §7 连带回归（summarizeEvent blast radius）→ Task 3 Step 5 + Task 5 Step 1 ✓
- 验收三条 → Task 5 Step 2 ✓

**Placeholder scan:** 无 TBD/TODO；每个 code step 有完整代码。

**Type consistency:** `panelRecord` 新增 `spineIds: Set<string>` 在 Task 4 定义并由 `renderViewer` 传入；`renderMarkdown(text: string): string` 在 Task 2 定义、Task 4 import 使用；`shortInput`/`summarizeEvent` 签名一致；`PanelRecord.chain` 元素形状 `{ eventId; type; summary }` 与 `trim` 过滤一致。

**已知连带:** `summarizeEvent` label 变更只影响含**非空 input 的 tool.requested** 断言；`tool.responded` 断言不变（无 input）。Task 3 Step 5 + Task 5 Step 1 已覆盖。
