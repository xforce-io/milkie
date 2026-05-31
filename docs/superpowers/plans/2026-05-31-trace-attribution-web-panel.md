# #68 决策因果归因接进 agent-docs-qa audit panel — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 #64 的决策脊柱+Why+因果下钻 viewer，作为一个 iframe tab 接进 agent-docs-qa 的 live audit panel，并接通 region 内容持久化让 #26 预览可用。

**Architecture:** 整块复用核心 `renderViewer`（零 `src/trace/*` 改动）。后端给 example 的 `Milkie` 接 `FileTraceObjectStore`（让 region 内容落盘），新增 `GET /run/:runId/viewer` endpoint（`readByRunId` + hydrate regionContent + `renderViewer` → 完整 HTML 文档）；前端把它作为 lazy `<iframe>` 渲染在新「Why」tab 里，#64 自带 JS 在 iframe 内自跑。

**Tech Stack:** Node `http` server（vanilla，无框架）、vanilla JS 前端、Vitest + StubGateway 注入、相对路径 import 核心 `../../src/`。

**Spec:** `docs/superpowers/specs/2026-05-31-trace-attribution-web-panel-design.md`

---

## File Structure

| 文件 | 职责 | 改动 |
|---|---|---|
| `examples/agent-docs-qa/server.ts` | HTTP server + Milkie 装配 | 接 `FileTraceObjectStore`；新增 viewer endpoint + handler |
| `examples/agent-docs-qa/public/index.html` | 单文件前端 | 新增「Why」tab（iframe）+ CSS；Steps→Execution |
| `examples/agent-docs-qa/__tests__/server.test.ts` | server 测试 | objectStore 持久化测试 + viewer endpoint 测试 |

不改任何 `src/trace/*`。

---

## Task 1: 给 example 的 Milkie 接 FileTraceObjectStore（region 内容落盘）

**Files:**
- Modify: `examples/agent-docs-qa/server.ts`（imports；`ServerState`；`startServer`）
- Test: `examples/agent-docs-qa/__tests__/server.test.ts`

- [ ] **Step 1: Write the failing test**

加到 `describe('server — REST endpoints', …)` 内（紧跟现有 `POST /chat` 测试之后）：

```typescript
  it('persists region content to .milkie/objects after a run', async () => {
    await postJson(`${baseUrl}/chat`, { input: 'hi' })
    const objectsDir = path.join(exampleDir, '.milkie', 'objects')
    expect(fs.existsSync(objectsDir)).toBe(true)
    // A run composes context regions; their canonical content is written to
    // the trace object store. Non-empty objects dir proves the wiring.
    const entries = fs.readdirSync(objectsDir)
    expect(entries.length).toBeGreaterThan(0)
  })
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd examples/agent-docs-qa && npx vitest run __tests__/server.test.ts -t "persists region content"`
Expected: FAIL — `objectsDir` 不存在或为空（当前 Milkie 未挂 traceObjectStore）。

- [ ] **Step 3: Add imports**

在 `examples/agent-docs-qa/server.ts` 顶部 import 区（现有 `import { JsonlEventStore } …` 一带）追加：

```typescript
import { FileTraceObjectStore } from '../../src/trace/TraceObjectStore.js'
```

- [ ] **Step 4: Extend ServerState**

把 `interface ServerState`（现 server.ts:24-30）改为含 traceObjectStore：

```typescript
interface ServerState {
  milkie:           Milkie
  eventStore:       BroadcastingEventStore
  traceObjectStore: FileTraceObjectStore
  runsDir:          string
  publicDir:        string
  corpusRoot:       string
}
```

- [ ] **Step 5: Wire it in startServer**

在 `startServer` 里，把 runsDir 创建之后、`new Milkie(...)` 之前插入 objects 目录与 store；并把 traceObjectStore 传进 Milkie 和 state。替换现有 server.ts:221-243 这段为：

```typescript
  const runsDir = path.join(config.exampleDir, '.milkie', 'runs')
  if (!existsSync(runsDir)) mkdirSync(runsDir, { recursive: true })

  const objectsDir = path.join(config.exampleDir, '.milkie', 'objects')
  if (!existsSync(objectsDir)) mkdirSync(objectsDir, { recursive: true })
  const traceObjectStore = new FileTraceObjectStore(objectsDir)

  const eventStore = new BroadcastingEventStore(new JsonlEventStore(runsDir))
  const milkie     = new Milkie({
    stateStore: new MemoryStore(),
    gateway:    config.gateway,   // when omitted, Milkie falls back to createGateway(agent.model) per-invoke
    eventStore,
    traceObjectStore,
  })

  for (const tool of makeCorpusToolDefinitions(config.corpusRoot)) {
    milkie.registerTool(tool)
  }
  milkie.loadAgentFile(config.agentFile)

  // Public dir resolution: prefer co-located public/ when present (production
  // mode using the real example dir), else fall back to this file's
  // sibling public/ (test mode where exampleDir is a tmpDir).
  const colocatedPublic = path.join(config.exampleDir, 'public')
  const fallbackPublic  = path.resolve(__dirname, 'public')
  const publicDir = existsSync(colocatedPublic) ? colocatedPublic : fallbackPublic

  const state: ServerState = { milkie, eventStore, traceObjectStore, runsDir, publicDir, corpusRoot: config.corpusRoot }
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd examples/agent-docs-qa && npx vitest run __tests__/server.test.ts -t "persists region content"`
Expected: PASS。

- [ ] **Step 7: Commit**

```bash
git add examples/agent-docs-qa/server.ts examples/agent-docs-qa/__tests__/server.test.ts
git commit -m "feat(#68): wire FileTraceObjectStore into agent-docs-qa so region content persists"
```

---

## Task 2: 新增 GET /run/:runId/viewer endpoint

**Files:**
- Modify: `examples/agent-docs-qa/server.ts`（imports；新 handler；路由）
- Test: `examples/agent-docs-qa/__tests__/server.test.ts`

- [ ] **Step 1: Write the failing tests**

加到 server 测试 describe 内：

```typescript
  it('GET /run/:runId/viewer returns the decision viewer HTML', async () => {
    const chat = await postJson(`${baseUrl}/chat`, { input: 'hi' })
    const { runId } = JSON.parse(chat.body) as { runId: string }

    const r = await get(`${baseUrl}/run/${runId}/viewer`)
    expect(r.status).toBe(200)
    // renderViewer emits a self-contained document with the decision spine.
    expect(r.body).toContain('<!doctype html>')
    expect(r.body).toContain('milkie trace viewer')
    expect(r.body).toContain('data-id=')        // spine nodes
    expect(r.body).toContain('spine-output')     // the output node with ❓ entry
  })

  it('GET /run/:runId/viewer 404s on an unknown run', async () => {
    const r = await get(`${baseUrl}/run/does-not-exist/viewer`)
    expect(r.status).toBe(404)
  })
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd examples/agent-docs-qa && npx vitest run __tests__/server.test.ts -t "viewer"`
Expected: FAIL — 路由未命中，返回 404 给第一个测试（"返回 viewer HTML" 断言失败），404 测试可能恰好过（默认 404）但第一个必失。

- [ ] **Step 3: Add imports**

在 server.ts 顶部 import 区追加（接 Task 1 的 import 之后）：

```typescript
import { regionReuseCounts } from '../../src/trace/RegionContextView.js'
import { renderViewer } from '../../src/trace/render/viewer.js'
```

- [ ] **Step 4: Add the handler**

在 `handleReplay` 之后（server.ts:174 之后）新增：

```typescript
async function handleViewer(
  res: ServerResponse, s: ServerState, runId: string,
): Promise<void> {
  const events = await s.eventStore.readByRunId(runId)
  if (events.length === 0) {
    sendJson(res, 404, { error: 'run not found' })
    return
  }
  // Hydrate region content the same way `milkie trace report` does
  // (cli/main.ts): look up each region's canonical content by hash.
  const regionContent = new Map<string, string>()
  for (const h of regionReuseCounts(events).keys()) {
    const c = await s.traceObjectStore.getCanonical(h)
    if (c !== undefined) regionContent.set(h, c)
  }
  const html = renderViewer(events, { regionContent })
  res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' })
  res.end(html)
}
```

- [ ] **Step 5: Add the route**

在路由块里，`replayMatch` 分支之后（server.ts:268 之后）插入：

```typescript
      const viewerMatch = route.match(/^\/run\/([^/]+)\/viewer$/)
      if (req.method === 'GET' && viewerMatch) {
        return handleViewer(res, state, decodeURIComponent(viewerMatch[1]!))
      }
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd examples/agent-docs-qa && npx vitest run __tests__/server.test.ts -t "viewer"`
Expected: PASS（两个）。

- [ ] **Step 7: Commit**

```bash
git add examples/agent-docs-qa/server.ts examples/agent-docs-qa/__tests__/server.test.ts
git commit -m "feat(#68): GET /run/:runId/viewer renders the #64 decision viewer (reuses renderViewer)"
```

---

## Task 3: 前端「Why」tab（iframe）+ Steps 改名 Execution

**Files:**
- Modify: `examples/agent-docs-qa/public/index.html`（tab 按钮、CSS、`renderAuditBody`）
- Test: `examples/agent-docs-qa/__tests__/server.test.ts`（GET / 回归断言新 tab）

- [ ] **Step 1: Write the failing test**

加到 server 测试 describe 内：

```typescript
  it('serves the audit panel with a Why tab', async () => {
    const r = await get(`${baseUrl}/`)
    expect(r.status).toBe(200)
    expect(r.body).toContain('data-tab="why"')
    expect(r.body).toContain('>Why<')
  })
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd examples/agent-docs-qa && npx vitest run __tests__/server.test.ts -t "Why tab"`
Expected: FAIL — index.html 尚无 `data-tab="why"`。

- [ ] **Step 3: Add the tab button + rename Steps**

把 index.html:460-464 的 tab 栏：

```html
    <div class="ap-tabs">
      <div class="ap-tab active" data-tab="sources">Sources</div>
      <div class="ap-tab" data-tab="steps">Steps</div>
      <div class="ap-tab" data-tab="provenance">Provenance</div>
    </div>
```

改为：

```html
    <div class="ap-tabs">
      <div class="ap-tab active" data-tab="sources">Sources</div>
      <div class="ap-tab" data-tab="steps">Execution</div>
      <div class="ap-tab" data-tab="provenance">Provenance</div>
      <div class="ap-tab" data-tab="why">Why</div>
    </div>
```

（`data-tab="steps"` 不变 → `renderStepsTab` 逻辑零改动，只改可见文案。）

- [ ] **Step 4: Add iframe CSS**

在 index.html 的 `#audit-panel .ap-body { … }` 规则（约 index.html:191）之后追加：

```css
    #audit-panel .ap-body.viewer { padding: 0 }
    #audit-panel .ap-viewer-frame { display: block; width: 100%; height: 100%; border: 0 }
```

- [ ] **Step 5: Render the iframe in renderAuditBody**

把 `renderAuditBody`（index.html:1241-1261）改为：在取到 `msgEl` 后先按是否 why 切换 `.viewer` class，再加 why 分支。替换整个函数体为：

```javascript
    function renderAuditBody() {
      if (!auditAnchorRunId) { $auditBody.innerHTML = ''; return }
      const msgEl = $log.querySelector(`.msg.assistant[data-run-id="${auditAnchorRunId}"]`)
      $auditBody.classList.toggle('viewer', auditActiveTab === 'why')
      if (auditActiveTab === 'sources') {
        $auditBody.innerHTML = renderSourcesTab(auditAnchorRunId, msgEl)
      } else if (auditActiveTab === 'steps') {
        $auditBody.innerHTML = renderStepsTab(auditAnchorRunId)
      } else if (auditActiveTab === 'why') {
        // Reuse the core #64 decision viewer wholesale: it is a self-contained
        // HTML document (own CSS + drill-down JS), so an iframe isolates it and
        // its "two clicks to root cause" interactions run inside the frame.
        $auditBody.innerHTML =
          `<iframe class="ap-viewer-frame" src="/run/${encodeURIComponent(auditAnchorRunId)}/viewer"></iframe>`
      } else if (auditActiveTab === 'provenance') {
        $auditBody.innerHTML = `<div class="ap-empty">Analyzing answer against sources…</div>`
        // Async render: classifies each segment after fetching its source.
        // Capture runId so a fast tab switch doesn't paint stale results
        // over a panel that has since changed anchor.
        const pendingRunId = auditAnchorRunId
        const pendingTab   = auditActiveTab
        renderProvenanceTab(auditAnchorRunId).then(html => {
          if (auditAnchorRunId === pendingRunId && auditActiveTab === pendingTab) {
            $auditBody.innerHTML = html
          }
        })
      }
    }
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd examples/agent-docs-qa && npx vitest run __tests__/server.test.ts -t "Why tab"`
Expected: PASS。

- [ ] **Step 7: Commit**

```bash
git add examples/agent-docs-qa/public/index.html examples/agent-docs-qa/__tests__/server.test.ts
git commit -m "feat(#68): add Why tab (iframe decision viewer) to audit panel; rename Steps→Execution"
```

---

## Task 4: Live 验证（dev-browser）— 两次点击到根因 + 真实 region 预览

> 这是手工/自动化浏览器验证（覆盖 #68 验收第 1、3 条），不产出 committed 单测（与 #64 的 dev-browser 验证一致）。

**Files:** 无代码改动（纯验证）。如发现 bug，回到 Task 1-3 修。

- [ ] **Step 1: 起 server（真实 doubao，#66 修复后）**

```bash
cd examples/agent-docs-qa && PORT=7878 npx tsx server.ts
```
打开 `http://localhost:7878`，发一条真实问题（如"刘备和诸葛亮的关系？"），等回答完成。

- [ ] **Step 2: dev-browser 验证「两次点击到根因」（验收第 1 条）**

用 dev-browser skill：
1. 点 assistant 消息下的 "View audit ▾" 打开 panel
2. 点 **Why** tab → 断言 `iframe.ap-viewer-frame` 出现、加载完成
3. 进 iframe：点输出节点的 "❓ 为什么是这个结果" → 断言右侧 Why 面板渲染出"因"的解释、且该因脊柱节点高亮（`.cause`）
4. 点面板里 "← 谁导致的" → 断言选中（`.selected`）上移一层、面板内容刷新 = **两次点击到根因**

- [ ] **Step 3: 验证真实 region composition 内容预览（验收第 3 条）**

在 Why tab 的 iframe 里选中一个 LLM 节点 → 断言其 composition 区出现真实 region 内容预览（非"(内容不可用)"占位）。这依赖 Task 1 的 objectStore 落盘 + doubao 真实 run。

- [ ] **Step 4: 记录验证结果**

把 dev-browser 的关键断言/截图结论记到 PR 描述里（这是 #68 验收证据）。

---

## Task 5: 收尾 — 全量测试、提交、PR、follow-up issue

**Files:** 无新代码改动。

- [ ] **Step 1: 全量测试 + 类型检查**

Run（仓库根）:
```bash
npm test
npx tsc --noEmit
```
Expected: 全绿（含 example 的 server.test.ts 新增 4 个用例、既有用例不破）。

- [ ] **Step 2: 确认 roadmap.md 不混入**

```bash
git status --short
```
`roadmap.md` 若仍是用户未提交 WIP，**不要** stage 它。只确认本 PR 的 commit 都是 #68 相关。

- [ ] **Step 3: Push 分支**

```bash
git push -u origin feat/68-trace-attribution-in-web-panel
```

- [ ] **Step 4: 开 PR（body 默认带 Closes #68）**

```bash
gh pr create --title "feat(#68): decision attribution in agent-docs-qa audit panel" --body "$(cat <<'EOF'
把 #64 的决策脊柱+Why+因果下钻 viewer 接进 agent-docs-qa 的 live audit panel。

## 做了什么
- 给 example 的 Milkie 接 `FileTraceObjectStore` → region 内容落盘（#26 预览可用）
- 新增 `GET /run/:runId/viewer`：readByRunId + hydrate regionContent + 整块复用核心 `renderViewer`（零 `src/trace/*` 改动）
- audit panel 新增 **Why** tab（lazy iframe，#64 自带下钻 JS 在 frame 内自跑）；Steps 改名 Execution，与 Why 形成「成本/上下文」vs「因果/为什么」两个镜头

## 验收（#68）
- [x] localhost web UI 两次点击到根因 — dev-browser 验证（见下）
- [x] 复用核心投影、前端不自带归因逻辑 — 整块 renderViewer iframe
- [x] 真实 doubao run 能看到真实 region composition 内容预览

<verification screenshots / notes>

Closes #68

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 5: 开 follow-up issue（Steps 投影化重构，本 PR 不做）**

```bash
gh issue create --title "[diagnosable] 把 agent-docs-qa 的 Execution(原 Steps) tab 改成投影驱动" --body "$(cat <<'EOF'
Parent: #20 ;follow-up of #68

## Why
#68 给 audit panel 加了 Why tab（整块复用核心 renderViewer），但 Execution(原 Steps) tab 仍是**前端自己解析事件重写归因**（`renderStepsTab`），违反 ARCHITECTURE.md「UI = projection over CLI/SDK，前端不自带归因逻辑」。

## What
把 Steps 的 LLM/tool/region/cache 视图从前端重写改成消费核心投影（server 出投影 JSON 或复用核心渲染），消除前端归因逻辑。保留其 cache health 分级 / region by stability 的 UX 价值。

## Related
- follow-up of #68;依赖 #26 / Phase 4a cache
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- §4.1 接 FileTraceObjectStore → Task 1 ✓
- §4.2 GET /run/:runId/viewer（readByRunId + hydrate + renderViewer + 404）→ Task 2 ✓
- §4.3 新 tab（lazy iframe）+ Steps 重命名 + renderAuditBody → Task 3 ✓
- §4.4 镜头分工（命名）→ Task 3 文案 ✓
- §7 测试（200+脊柱标记、404、objectStore 落盘、live 两次点击、真实预览）→ Task 1/2/3 单测 + Task 4 dev-browser ✓
- §9 follow-up issue → Task 5 Step 5 ✓
- 验收三条 → Task 2/3 单测 + Task 4 验证 ✓

**Placeholder scan:** 无 TBD/TODO；唯一 `<verification ...>` 占位是 PR body 里待填的人工验证证据（Task 4 Step 4 产出），非代码占位。

**Type consistency:** `traceObjectStore: FileTraceObjectStore` 在 ServerState（Task 1）定义、handler（Task 2）用 `s.traceObjectStore.getCanonical`；`renderViewer(events, { regionContent })` 签名与核心一致；`regionReuseCounts(events)` 返回 `Map<string,number>`，`.keys()` 为 hash 串，与 `getCanonical(hash)` 入参一致；`data-tab="why"` 与 `renderAuditBody` 的 `auditActiveTab === 'why'` 一致。

**已知偏离 spec:** tab 标签用英文（Execution/Why）而非 spec 的中文（执行·成本/决策·为什么），为与现有英文 UI 一致；功能等价，可一行调整。

**作用域:** 单 PR、聚焦 example，不碰核心。
