# #71 决策 viewer UX polish — 设计 spec

**Issue:** #71（diagnosable；follow-up of #64 / #68；Parent #20）
**日期:** 2026-05-31
**定位:** #68 把 #64 的 viewer 接进 agent-docs-qa panel 后，近距离 review 暴露的三个 viewer 自身 UX/正确性问题，一并修复。

三项放在一个 PR（用户决定不拆）：
- **A（runtime）**：output ❓ 在真实 run 上能下钻。
- **B（viewer）**：输出渲染 markdown。
- **C（viewer）**：因果链精简到决策跳 + 标签带入参 + 诚实兜底文案。

经 brainstorm 定稿：链处理选「精简到决策跳 + 入参消歧」（分支 run 下决策路径是脊柱子集，有保留价值）。

---

## 1. 背景

| 问题 | 现状 | 证据 |
|---|---|---|
| output ❓ 无法下钻 | `agent.run.completed` 不带 `causedBy` → output 节点 `causeDecisionId` 空 → 无"← 谁导致的"链接，但文案写死叫人点 | run d27e62da 的 completed 事件 `causedBy: None` |
| 输出不渲染 markdown | `viewer.ts` 对 `lastTextOutput` 做 `esc()`，`###`/`**` 按字面显示 | 截图 |
| 因果链太长/重复/丢入参 | `causalChain = walkCausedBy(...)` 回溯到 run.started，线性 run ≈ 整条时间线；`summarizeEvent` 只显示工具名 | grep 被调两次（`pattern:"诸葛亮"`/`"孔明"`）显示成两个相同的 `tool.requested(grep)` |

## 2. Non-goals

- 不改 replay 语义、不改事件 schema 字段定义（`causedBy` 已是可选字段，本次只是给 completed 事件**填**它）。
- markdown 不做完整 CommonMark：只覆盖 doubao 输出常见子集（见 §4.B）。
- 不动脊柱节点的 label 格式（`tool: grep` 等保持原样）；入参消歧只加在 `summarizeEvent`（链/html Why 块用）。

## 3. A — Runtime：`agent.run.completed.causedBy`

### 改动
`src/trace/RecordingIOPort.ts` 的 `detach()`（约 line 134）追加 causedBy，照搬现有 causal-edge 模式：

```ts
async detach(payload: AgentRunCompletedPayload): Promise<void> {
  await this.flushPendingNondet()
  await this.store.append({
    id:        this.inner.uuid(),
    runId:     this.runId,
    type:      'agent.run.completed',
    actor:     this.actor,
    // The final output is produced by the last LLM response; link to it so the
    // output node can drill to the final decision (nearest-decision-ancestor).
    ...(this.cursor?.lastLlmRespondedId ? { causedBy: this.cursor.lastLlmRespondedId } : {}),
    timestamp: this.inner.now(),
    payload,
  })
}
```

### 为什么指向 `lastLlmRespondedId`
最终答案文本来自最后一个 `llm.responded`。`buildDecisionSpine` 从 output 沿 causedBy 找最近决策祖先：completed → last llm.responded →（其 causedBy）last llm.requested（脊柱决策）→ `causeDecisionId` = 终轮 llm.requested。于是 output ❓ 下钻到终轮 LLM 调用。

### 安全性 / 边界
- **replay 安全**：`causedBy` 是裸 uuid 的 trace 元数据，replay 从不比对 trace event id（#64 设计已明确）。不影响 requestHash / 决定性。
- 无 LLM 的退化 run（`lastLlmRespondedId` 空）→ 不设 causedBy，行为同今天（降级，不报错）。
- `CausalCursor.lastLlmRespondedId` 已存在（`invokeLLM` 在 llm.responded 后设置，line ~179），detach 时即终轮响应 id。

## 4. B — Viewer：输出 markdown 渲染

### 新文件 `src/trace/render/markdown.ts`
纯函数 `renderMarkdown(text: string): string`，零依赖。**先整段 HTML 转义**（防注入），再按行/正则应用最小子集，返回 HTML 串：

- 标题：`# ` / `## ` / `### ` → `<h3>`/`<h4>`/`<h5>`（viewer 内层级，避免抢 h1/h3 既有样式）
- 粗体：`**x**` → `<strong>x</strong>`
- 行内代码：`` `x` `` → `<code>x</code>`
- 无序列表：以 `- ` / `* ` 起的连续行 → `<ul><li>…</li></ul>`
- 有序列表：以 `1. ` 起的连续行 → `<ol><li>…</li></ol>`
- 段落 / 换行：空行分段 `<p>`，段内换行 `<br>`

安全：转义在前，所有标记由我们生成的标签，无 raw HTML 透传。

### 接入 `viewer.ts`
output 节点 `panelRecord` 的 bodyHtml：`输出:` 后的 `lastTextOutput` 由 `esc(...)` 改为 `renderMarkdown(...)`。仅 output 节点；LLM/tool 面板仍结构化 `esc`。

### 测试 `src/__tests__/markdown.test.ts`
覆盖每种子集 + HTML 转义（`<script>` → 文本）+ 空串。

## 5. C — Viewer：因果链精简 + 入参消歧 + 兜底

### C1. `summarizeEvent` 工具标签带入参
`src/trace/diagnostics/summarizeEvent.ts`：tool.requested/responded 在工具名后追加简短入参摘要：

```
tool.requested(grep · {"pattern":"孔明"})
```

- 入参摘要 = `JSON.stringify(input)` 单行，截断到 24 字符（超出加 `…`）。`tool.responded` 无 input 则只显示工具名（保持原样）。
- 影响面：链 + html.ts 的 Why 块共享此函数 → 一处改、两处一致受益。脊柱 label 不走此函数，不变。

### C2. 链精简到决策跳（viewer.ts）
`panelRecord` 的 `chain` 现取 `explainX.causalChain`（全量 walkCausedBy）。改为**过滤成只剩脊柱决策事件**：

- `renderViewer` 已有 `spine.nodes`；构造 `spineIds = new Set(spine.nodes.map(n => n.eventId))`，传入 `panelRecord`。
- `chain = explainX.causalChain.filter(c => spineIds.has(c.eventId))`。
- 效果：去掉 `llm.responded`/中间非决策事件，链变成"决策→决策"路径；线性 run 大幅变短；分支 run 显示本节点的决策子路径。
- `chainHtml`（viewer-template）不变——过滤后元素都是决策（`exps[id]` 命中）→ 全部可点蓝链。

### C3. output 兜底文案（viewer.ts）
output 节点 bodyHtml 的"由上游决策产生(点 ← 谁导致的 下钻)"改为条件渲染：有 `causeDecisionId` 才显示该提示；否则"（无上游决策记录）"。（A 落地后 output 正常有因；此为防御。）

## 6. 改动文件清单

| 文件 | 改动 |
|---|---|
| `src/trace/RecordingIOPort.ts` | detach 填 `causedBy = lastLlmRespondedId` |
| `src/trace/render/markdown.ts` | 新建：`renderMarkdown` 纯函数 |
| `src/trace/render/viewer.ts` | output 用 renderMarkdown；chain 按 spineIds 过滤；output 兜底文案；panelRecord 接 spineIds |
| `src/trace/diagnostics/summarizeEvent.ts` | tool 标签带截断入参 |
| `src/__tests__/markdown.test.ts` | 新建 |
| `src/__tests__/summarizeEvent.test.ts` | 补入参断言（若无则新建） |
| `src/__tests__/render-viewer.test.ts` | 补：output markdown 渲染、chain 只含决策 |
| `src/__tests__/RecordingIOPort.*` 或 `Trace.test.ts` | completed 带 causedBy=last llm.responded |
| `src/__tests__/buildDecisionSpine.test.ts` | output 节点有 causeDecisionId（当 completed 有 causedBy 链到决策） |

## 7. 测试要点

- **A**：detach 后 completed.causedBy === 最后 llm.responded id；无 LLM 的 run 不设 causedBy；buildDecisionSpine 对带因的 run 给 output `causeDecisionId`。
- **B**：renderMarkdown 各子集 + 转义；viewer output 面板含渲染后标签（如 `<strong>`/`<h3>`/`<li>`）。
- **C**：summarizeEvent 两次不同入参 grep → 两个不同 label；viewer chain 不含 `llm.responded`（非决策）；output 无因时显示兜底文案。
- **回归**：现有 render-viewer / buildDecisionSpine / html / Trace 单测全绿；`tsc --noEmit`。
- **C1 连带回归**：`summarizeEvent` 改 label 会波及任何断言旧 `tool.requested(<name>)` 文本的测试（explainTransition/explainLlmCall/explainToolCall、render-html 等）。计划阶段需 grep 全仓这些断言并相应更新（这是 label 变更的预期连带，不是 bug）。
- **dev-browser 真实 run**：output ❓ → 下钻到终轮 LLM；输出渲染 markdown；链短且两次 grep 可区分。

## 8. 验收对照（#71）

- [ ] output ❓ 在真实 run 能下钻（§3 + dev-browser）
- [ ] 输出按 markdown 渲染（§4 + 单测）
- [ ] 因果链只剩决策跳、两次 grep 标签可区分、output 无因不骗点击（§5 + 单测）

## 9. 分支

`feat/71-viewer-ux-polish`，基于 main（含 #68）。
