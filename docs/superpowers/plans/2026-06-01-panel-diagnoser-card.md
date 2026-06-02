# Panel 集成 diagnoser — 答案正确性诊断卡片 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `examples/agent-docs-qa` 的 panel 上，加一个手动「诊断」按钮，调内置 diagnoser agent 对当前被审计 run 做答案正确性诊断，渲染 verdict/firstBreak/explanation 卡片。

**Architecture:** 纯增量。后端新增 `POST /run/:runId/diagnose`，内部 `milkie.invoke('diagnoser', { input: runId })`，把 diagnoser 输出的 JSON 文本 parse 成结构化结果返回；前端在审计区加按钮 + banner 式结果卡，复用现有 `.ap-replay-banner` 视觉体系。不碰核心、不碰 #89 已内置的 diagnoser/读-Trace 工具。

**Tech Stack:** TypeScript、Node `http`、Jest（example 自带 `StubGateway` 注入确定性 LLM 响应）、原生 DOM（index.html 内联脚本，无前端框架）。

参考 issue：#94（设计权威）/ #88（diagnoser 能力）/ #89（内置化）。

---

## File Structure

- `examples/agent-docs-qa/server.ts` — 加 `handleDiagnose` + 路由（修改）
- `examples/agent-docs-qa/__tests__/server.test.ts` — 加诊断端点测试（修改）
- `examples/agent-docs-qa/public/index.html` — 加诊断按钮 + 结果卡 CSS/JS（修改）

诊断逻辑全部由内置 diagnoser 承载，server 端只做 invoke + parse + 兜底，故无需新文件。

---

## Task 1: 后端 — `POST /run/:runId/diagnose` 端点

**Files:**
- Modify: `examples/agent-docs-qa/server.ts`
- Test: `examples/agent-docs-qa/__tests__/server.test.ts`

### 背景：diagnoser 链路与 stub 序列

diagnoser 是单 llm state，带 `tools: [get_run_io, get_execution]`，按 systemPrompt 先后调两个工具，最后输出**一段 JSON 文本**（无 markdown 围栏）。用 `StubGateway` 注入时，`complete` 被依次调用，序列为：

1. 第 1 次 `complete` → 返回带 `toolCalls: [{ name: 'get_run_io', ... }]` 的响应；
2. 第 2 次 → `toolCalls: [{ name: 'get_execution', ... }]`；
3. 第 3 次 → 纯 `text` 的最终 JSON。

被诊断 run 必须先存在于 eventStore（diagnoser 的工具靠 `readByRunId(runId)` 读它）。测试里先用一次 `/chat`（喂 sanguo-researcher 的 stub 响应）产生一个真实 run，拿到它的 `runId`，再诊断它。

> 注意：现有 `server.test.ts` 顶部已有 `StubGateway`/`text()` helper 和 `postJson`/`get`。先阅读该文件已有的 `/chat` 测试，照搬它构造被诊断 run 的方式（包括 sanguo-researcher 需要的工具调用 stub 序列），不要另起炉灶。

- [ ] **Step 1: 写失败测试 — 诊断一个 suspect run**

在 `examples/agent-docs-qa/__tests__/server.test.ts` 末尾新增（沿用文件已有的 `text()`、`postJson`、`startServer`/`stopServer`、tmpDir/StubGateway 套路；下面的 `toolCall` helper 若文件已有则复用、没有则在文件顶部 helper 区加）：

```ts
// helper（若文件尚无）：构造一次带 toolCall 的模型响应
const toolCall = (name: string, args: Record<string, unknown>): ModelResponse => ({
  content: [], finishReason: 'tool_use',
  toolCalls: [{ id: `tc-${name}`, name, input: args }],
})

describe('POST /run/:runId/diagnose', () => {
  it('returns structured verdict/firstBreak for a suspect run', async () => {
    // 1) 先产生一个被诊断 run：用最少的 sanguo-researcher stub 响应跑一轮 /chat。
    //    （照搬本文件已有 /chat 测试里 sanguo-researcher 的 stub 序列。）
    const diagnoseJson = JSON.stringify({
      verdict: 'suspect',
      firstBreak: { step: 'evt-tool-1', what: "grep('赤壁')", why: '工具 query 与「曹操爸爸」无关' },
      explanation: '工具检索跑偏到赤壁之战，答案未回答父亲是谁。',
    })
    const gateway = new StubGateway([
      // —— sanguo-researcher 这一轮：照搬已有 /chat 测试的序列 ——
      text('曹操爸爸是曹嵩。'),          // 占位；以本文件已有 /chat 用例为准替换
      // —— diagnoser 这一轮 ——
      toolCall('get_run_io', { runId: '__RUNID__' }),
      toolCall('get_execution', { runId: '__RUNID__' }),
      text(diagnoseJson),
    ])
    const server = await startServer({
      port: 0, exampleDir: tmpDir, gateway,
      agentFile: path.join(__dirname, '..', 'agents', 'sanguo-researcher.md'),
      corpusRoot: path.join(__dirname, '..', 'corpus'),
    })
    const { port } = server.address() as { port: number }

    const chat = await postJson(`http://localhost:${port}/chat`, { input: '曹操爸爸是谁' })
    const runId = JSON.parse(chat.body).runId as string

    const res = await postJson(`http://localhost:${port}/run/${runId}/diagnose`, {})
    await stopServer(server)

    expect(res.status).toBe(200)
    const body = JSON.parse(res.body)
    expect(body.verdict).toBe('suspect')
    expect(body.firstBreak.why).toMatch(/赤壁|无关/)
    expect(body.diagnoseRunId).toBeTruthy()
    expect(body.diagnoseRunId).not.toBe(runId)   // 被诊断 run 与诊断 run 是两个 runId
  })
})
```

> 实现者注意：`toolCalls` 的字段名（`input` vs `args`、是否需要 `id`）以本仓 `ModelResponse`/`ToolCall` 类型定义为准——打开 `src/types/model.ts` 和现有带工具调用的测试（如 `/chat` 用例）核对后再定稿，不要照抄上面的猜测字段。`__RUNID__` 仅示意工具入参；diagnoser 的 systemPrompt 会自己把真实 runId 填进工具调用，stub 只需返回"发起了这次工具调用"，工具入参由 runtime 据 LLM 输出决定——若 StubGateway 模式下工具入参由 stub 决定，则在 invoke 时 diagnoser 收到的 `input` 即 runId，照其 prompt 透传。

- [ ] **Step 2: 跑测试确认失败**

Run: `cd examples/agent-docs-qa && npx jest server.test.ts -t "suspect run"`
Expected: FAIL —— 404 或路由不存在（端点未实现）。

- [ ] **Step 3: 实现 `handleDiagnose` + 路由**

在 `server.ts` 加函数（放在 `handleExecution` 之后）：

```ts
import { v4 as uuidv4 } from 'uuid'   // 文件已 import，确认即可

async function handleDiagnose(
  res: ServerResponse, s: ServerState, runId: string,
): Promise<void> {
  // 被诊断 run 必须存在，否则白白烧一次 LLM。
  const events = await s.eventStore.readByRunId(runId)
  if (events.length === 0) {
    sendJson(res, 404, { error: 'run not found' })
    return
  }
  // diagnoser 跑在独立 contextId 上 —— 被诊断 run 与诊断 run 是两个 runId，
  // 互不污染对话流（#88 边界）。
  const diagnoseCtx = `diagnose:${runId}`
  const result = await s.milkie.invoke({
    agentId:   'diagnoser',
    goal:      runId,
    input:     runId,
    contextId: diagnoseCtx,
  })
  // diagnoser 按契约只输出一段 JSON 文本。parse 失败时降级返回，不 500。
  const raw = String(result.output ?? '')
  try {
    const parsed = JSON.parse(raw) as {
      verdict: 'ok' | 'suspect'
      firstBreak: { step: string; what: string; why: string } | null
      explanation: string
    }
    sendJson(res, 200, { ...parsed, diagnoseRunId: result.agentRunId })
  } catch {
    sendJson(res, 200, { error: 'unparseable', raw, diagnoseRunId: result.agentRunId })
  }
}
```

在 `server.ts` 路由区（`executionMatch` 之后、`sourceMatch` 之前）加：

```ts
const diagnoseMatch = route.match(/^\/run\/([^/]+)\/diagnose$/)
if (req.method === 'POST' && diagnoseMatch) {
  return handleDiagnose(res, state, decodeURIComponent(diagnoseMatch[1]!))
}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd examples/agent-docs-qa && npx jest server.test.ts -t "suspect run"`
Expected: PASS。

- [ ] **Step 5: 加兜底用例 — parse 失败 + 404**

在同一 `describe` 里追加两个用例：

```ts
it('returns 404 when the run does not exist', async () => {
  const gateway = new StubGateway([])
  const server = await startServer({
    port: 0, exampleDir: tmpDir, gateway,
    agentFile: path.join(__dirname, '..', 'agents', 'sanguo-researcher.md'),
    corpusRoot: path.join(__dirname, '..', 'corpus'),
  })
  const { port } = server.address() as { port: number }
  const res = await postJson(`http://localhost:${port}/run/nope/diagnose`, {})
  await stopServer(server)
  expect(res.status).toBe(404)
})

it('degrades gracefully when diagnoser output is not JSON', async () => {
  const gateway = new StubGateway([
    text('曹操爸爸是曹嵩。'),                 // 占位；以已有 /chat 序列为准
    toolCall('get_run_io', { runId: 'x' }),
    toolCall('get_execution', { runId: 'x' }),
    text('这不是 JSON'),                       // diagnoser 违约输出
  ])
  const server = await startServer({
    port: 0, exampleDir: tmpDir, gateway,
    agentFile: path.join(__dirname, '..', 'agents', 'sanguo-researcher.md'),
    corpusRoot: path.join(__dirname, '..', 'corpus'),
  })
  const { port } = server.address() as { port: number }
  const chat = await postJson(`http://localhost:${port}/chat`, { input: '曹操爸爸是谁' })
  const runId = JSON.parse(chat.body).runId as string
  const res = await postJson(`http://localhost:${port}/run/${runId}/diagnose`, {})
  await stopServer(server)
  expect(res.status).toBe(200)
  expect(JSON.parse(res.body).error).toBe('unparseable')
})
```

- [ ] **Step 6: 跑全 server 测试确认绿**

Run: `cd examples/agent-docs-qa && npx jest server.test.ts`
Expected: 全 PASS（含原有用例不回归）。

- [ ] **Step 7: 提交**

```bash
git add examples/agent-docs-qa/server.ts examples/agent-docs-qa/__tests__/server.test.ts
git commit -m "feat(#94): POST /run/:runId/diagnose — invoke built-in diagnoser, parse verdict JSON"
```

---

## Task 2: 前端 — 诊断按钮 + 结果卡

**Files:**
- Modify: `examples/agent-docs-qa/public/index.html`

前端无自动化测试框架（index.html 历来靠手验）。本任务以手动验证收尾。每步给出确切代码与插入位置。

- [ ] **Step 1: 加按钮（DOM）**

在 `public/index.html` 审计区 header，`↻ Replay` 按钮（约 line 463）之后、`.ap-close` 之前插入：

```html
      <button class="ap-diagnose" aria-label="Diagnose answer correctness" title="诊断这次回答是否答到问题上、第一处跑偏在哪">⚖ 诊断</button>
```

- [ ] **Step 2: 加结果卡容器（DOM）**

在 `.ap-replay-banner`（约 line 466）之后插入独立 banner（与 replay banner 并存，互不顶替）：

```html
    <div class="ap-diagnose-banner" role="status"></div>
```

- [ ] **Step 3: 加 CSS（复用 replay banner 体系 + verdict 配色）**

在 `<style>` 内 replay-banner 规则附近（约 line 182 之后）追加：

```css
    #audit-panel .ap-diagnose {
      margin-left: 8px; padding: 4px 10px; font-size: 12px; cursor: pointer;
      border: 1px solid var(--border); border-radius: 6px; background: var(--bg);
      color: var(--text);
    }
    #audit-panel .ap-diagnose:hover { opacity: 0.9 }
    #audit-panel .ap-diagnose:disabled { opacity: 0.5; cursor: progress }
    #audit-panel .ap-diagnose-banner {
      display: none; margin: 8px 16px 0; padding: 10px 12px;
      border-radius: 6px; font-size: 13px; position: relative;
    }
    #audit-panel .ap-diagnose-banner.shown { display: block }
    #audit-panel .ap-diagnose-banner.ok      { background: #f0faf3; border-left: 4px solid #2da14b }
    #audit-panel .ap-diagnose-banner.suspect { background: #fdf2e6; border-left: 4px solid #e07a00 }
    #audit-panel .ap-diagnose-banner.error   { background: #fdf0f0; border-left: 4px solid #d93030 }
    #audit-panel .ap-diagnose-banner .db-title { font-weight: 600; margin-bottom: 4px }
    #audit-panel .ap-diagnose-banner.ok      .db-title { color: #2da14b }
    #audit-panel .ap-diagnose-banner.suspect .db-title { color: #c66800 }
    #audit-panel .ap-diagnose-banner.error   .db-title { color: #c02020 }
    #audit-panel .ap-diagnose-banner .db-break { color: var(--text); margin-top: 4px }
    #audit-panel .ap-diagnose-banner .db-break b { color: var(--muted); font-weight: 600 }
    #audit-panel .ap-diagnose-banner .db-expl { color: var(--muted); font-size: 12px; margin-top: 6px }
    #audit-panel .ap-diagnose-banner .db-close {
      position: absolute; top: 6px; right: 8px; cursor: pointer; color: var(--muted);
      background: none; border: none; font-size: 16px; line-height: 1;
    }
    #audit-panel .ap-diagnose-banner .db-close:hover { color: var(--text) }
```

- [ ] **Step 4: 抓 DOM 句柄**

在脚本 DOM 句柄区（约 line 518-519，`$auditReplay`/`$auditBanner` 旁）追加：

```js
    const $auditDiagnose = $audit.querySelector('.ap-diagnose')
    const $auditDiagBanner = $audit.querySelector('.ap-diagnose-banner')
```

- [ ] **Step 5: 加 show/hide/render + 触发逻辑**

在 replay 逻辑块（`runReplay` 定义之后，约 line 1287）追加：

```js
    // ─── Diagnose (#94) ───────────────────────────────────────────────
    function hideDiagnoseBanner() {
      $auditDiagBanner.className = 'ap-diagnose-banner'
      $auditDiagBanner.innerHTML = ''
    }
    function showDiagnoseBanner(state, html) {
      $auditDiagBanner.className = `ap-diagnose-banner shown ${state}`
      $auditDiagBanner.innerHTML = html + `<button class="db-close" aria-label="Dismiss">×</button>`
    }
    function renderDiagnoseResult(data) {
      if (data.error === 'unparseable') {
        showDiagnoseBanner('error',
          `<div class="db-title">诊断输出无法解析</div>` +
          `<div class="db-expl">${esc(data.raw || '')}</div>`)
        return
      }
      if (data.error) {
        showDiagnoseBanner('error',
          `<div class="db-title">诊断失败</div>` +
          `<div class="db-expl">${esc(data.error)}</div>`)
        return
      }
      if (data.verdict === 'ok') {
        showDiagnoseBanner('ok',
          `<div class="db-title">✓ 答案与问题相关</div>` +
          `<div class="db-expl">${esc(data.explanation || '')}</div>`)
      } else {
        const fb = data.firstBreak || {}
        showDiagnoseBanner('suspect',
          `<div class="db-title">⚠ 检出跑偏（suspect）</div>` +
          `<div class="db-break"><b>断点</b> ${esc(fb.step || '?')} · ${esc(fb.what || '')}</div>` +
          `<div class="db-break"><b>原因</b> ${esc(fb.why || '')}</div>` +
          `<div class="db-expl">${esc(data.explanation || '')}</div>`)
      }
    }
    async function runDiagnose() {
      if (!auditAnchorRunId) return
      $auditDiagnose.disabled = true
      const orig = $auditDiagnose.textContent
      $auditDiagnose.textContent = '诊断中…'
      hideDiagnoseBanner()
      try {
        const resp = await fetch(`/run/${encodeURIComponent(auditAnchorRunId)}/diagnose`, { method: 'POST' })
        renderDiagnoseResult(await resp.json())
      } catch (err) {
        renderDiagnoseResult({ error: (err && err.message) || String(err) })
      } finally {
        $auditDiagnose.disabled = false
        $auditDiagnose.textContent = orig
      }
    }
    $auditDiagnose.addEventListener('click', runDiagnose)
    $auditDiagBanner.addEventListener('click', (ev) => {
      const t = ev.target
      if (t instanceof HTMLElement && t.classList.contains('db-close')) hideDiagnoseBanner()
    })
```

- [ ] **Step 6: 关闭 panel 时清结果卡**

在 `closeAudit()`（约 line 1218-1221，已调用 `hideReplayBanner()`）那一行旁追加：

```js
      hideDiagnoseBanner()
```

- [ ] **Step 7: 手动验证**

```bash
cd examples/agent-docs-qa && PORT=7878 npx tsx server.ts
```

打开 http://localhost:7878 ，发一条问题 → 点 assistant 气泡打开审计 panel → 点「⚖ 诊断」。
Expected：
- 按钮进入「诊断中…」禁用态；
- 返回后出现 banner：ok=绿色「答案与问题相关」，或 suspect=橙色含「断点/原因/解释」；
- 点 banner 右上 × 可关闭；关 panel 再开后 banner 不残留。

- [ ] **Step 8: 提交**

```bash
git add examples/agent-docs-qa/public/index.html
git commit -m "feat(#94): panel 诊断按钮 + verdict 结果卡（手动触发，复用 banner 体系）"
```

---

## Task 3: README 同步 + 收尾

**Files:**
- Modify: `examples/agent-docs-qa/README.md`

- [ ] **Step 1: README 标注诊断入口**

在 README 描述 panel/审计能力处，补一句：审计 panel 现支持「⚖ 诊断」——手动调内置 diagnoser 对该 run 做答案正确性诊断，输出 verdict/firstBreak/explanation。先读 README 现有结构，插在 Replay/Execution 介绍邻近，风格对齐。

- [ ] **Step 2: 跑 example 全测试**

Run: `cd examples/agent-docs-qa && npx jest`
Expected: 全 PASS。

- [ ] **Step 3: 提交**

```bash
git add examples/agent-docs-qa/README.md
git commit -m "docs(#94): README — 审计 panel 新增 ⚖ 诊断入口"
```

---

## 已知取舍（写入 #94，非 v1 阻塞）

- diagnoser 运行在 `diagnose:${runId}` 这个独立 contextId 上，会作为一条独立 run 落进 `.milkie/runs/`，可能出现在对话 picker 中。对 example 可接受；若要从 picker 隐藏，属后续增量（`scanConversations` 过滤 `diagnose:` 前缀）。
- firstBreak 不下钻高亮到 Execution tab（按 #94 v1 决策剥离）。

## Self-Review 记录

- **Spec 覆盖**：#94 后端端点(Task1)/前端按钮+卡片(Task2)/测试(Task1 三用例)/README(Task3) 全部有对应任务；非目标（下钻、自动触发）未实现，符合 v1 边界。
- **占位符**：测试代码里 sanguo-researcher 的 stub 序列与 `toolCalls` 字段名标注为"以本仓现有 /chat 用例与 `src/types/model.ts` 为准"——这是有意的核对指令而非占位 TODO，因该序列依赖实现者读现有测试定稿，不宜在计划里臆造。
- **类型一致**：端点返回 `{ verdict, firstBreak, explanation, diagnoseRunId }` 或 `{ error, raw, diagnoseRunId }`，前端 `renderDiagnoseResult` 三分支（ok/suspect/error+unparseable）与之逐一对齐；DOM 句柄名 `$auditDiagnose`/`$auditDiagBanner`、CSS 类 `ap-diagnose`/`ap-diagnose-banner`/`db-*` 前后一致。
