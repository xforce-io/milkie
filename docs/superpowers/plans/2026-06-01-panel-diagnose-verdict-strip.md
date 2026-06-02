# Panel 诊断布局重构（#98）Implementation Plan

> 方向 B：顶部结论条 + 证据 tabs + 断点跳 Execution。重构 #94 的诊断前端。靠步序下标，零核心投影改动。

**Goal:** 把诊断从「右上按钮 + banner」重构为 header 下方一条结论条；firstBreak 断点可点 → 跳到 Execution tab 高亮对应步。

**Tech Stack:** 原生 DOM 内联脚本（无框架、无前端测试）；diagnoser 为框架内置 agent（markdown prompt）；server 端无改动。

文件：
- `agents/diagnoser.md` — 收紧 firstBreak.step 契约为 0-based 整数下标
- `src/tools/trace.ts` — get_execution 描述点明 steps 0 起
- `examples/agent-docs-qa/public/index.html` — 结论条 + 断点跳转（主改动）
- `examples/agent-docs-qa/__tests__/server.test.ts` — diagnose 用例 step 改整数
- `examples/agent-docs-qa/README.md` — 同步

---

## Task 1：契约 — firstBreak.step = 步序下标

- `agents/diagnoser.md`：JSON 契约里 `firstBreak.step` 从「<eventId 或步序>」改为「出问题那一步在 get_execution 返回的 steps 数组中的 **0-based 整数下标**」；步骤说明里点明按下标定位。
- `src/tools/trace.ts`：get_execution 的 description 末尾补「steps 按执行顺序排列，下标从 0 起」。
- `examples/agent-docs-qa/__tests__/server.test.ts`：把 diagnose 用例 A 的 fixture `firstBreak.step` 从 `'evt-tool-1'` 改为整数（如 `1`）；断言端点原样透传该整数。
- 跑：`npx jest examples/agent-docs-qa/__tests__/server.test.ts` 全绿。
- commit：`feat(#98): diagnoser firstBreak.step 契约收紧为 0-based 步序下标`

## Task 2：前端 — 结论条 + 断点跳转（index.html）

### 2a 删除 #94 旧件
- 删 header 里 `<button class="ap-diagnose" ...>`（491）。
- 把 `<div class="ap-diagnose-banner">`（495）改名为 `<div class="ap-verdict" role="status"></div>`（结论条容器，常驻 header 与 tabs 之间）。
- 删 `$auditDiagnose` 句柄（549）；保留/改名 `$auditDiagBanner` → `$auditVerdict`（550）。
- 删旧 CSS（187–213 的 `.ap-diagnose` / `.ap-diagnose-banner` 整块），换成结论条 CSS（见 2c）。
- 删旧 JS（1327–1382：hide/show/renderDiagnoseResult/runDiagnose + 两个 listener），换成 2d。

### 2b 结论条结构（render 产出）
结论条是一行 flex：`⚖ 诊断` 标签 | 中部 summary（未跑=提示语；跑后=verdict 徽章 + suspect 时断点 chip） | 右侧触发按钮（未跑「诊断」/ 跑后「重新诊断」）。下方 `vd-detail`：explanation（ok 默认折叠、suspect 默认展开），由 summary 上的 caret 切换。

### 2c CSS（替换旧块）
```css
    #audit-panel .ap-verdict { display: none; margin: 8px 16px 0; }
    #audit-panel .ap-verdict.shown { display: block; }
    #audit-panel .vd-row { display: flex; align-items: center; gap: 10px;
      padding: 8px 12px; border-radius: 6px; border-left: 4px solid var(--border); background: var(--bg); font-size: 13px; }
    #audit-panel .ap-verdict.ok      .vd-row { background: #f0faf3; border-left-color: #2da14b; }
    #audit-panel .ap-verdict.suspect .vd-row { background: #fdf2e6; border-left-color: #e07a00; }
    #audit-panel .ap-verdict.error   .vd-row { background: #fdf0f0; border-left-color: #d93030; }
    #audit-panel .vd-label { font-weight: 600; color: var(--muted); white-space: nowrap; }
    #audit-panel .vd-summary { flex: 1; cursor: pointer; }
    #audit-panel .ap-verdict.ok      .vd-summary { color: #2da14b; font-weight: 600; }
    #audit-panel .ap-verdict.suspect .vd-summary { color: #c66800; font-weight: 600; }
    #audit-panel .ap-verdict.error   .vd-summary { color: #c02020; font-weight: 600; }
    #audit-panel .vd-run { padding: 4px 10px; font-size: 12px; cursor: pointer;
      border: 1px solid var(--border); border-radius: 6px; background: var(--bg); color: var(--text); white-space: nowrap; }
    #audit-panel .vd-run:hover { opacity: .9 } 
    #audit-panel .vd-run:disabled { opacity: .5; cursor: progress }
    #audit-panel .vd-jump { margin-left: 6px; padding: 1px 8px; font-size: 12px; cursor: pointer;
      border: 1px solid #e0a060; border-radius: 10px; background: #fff7ee; color: #c66800; }
    #audit-panel .vd-jump:hover { background: #ffe9d2 }
    #audit-panel .vd-detail { margin: 6px 0 0 2px; font-size: 12px; color: var(--muted); }
    #audit-panel .vd-detail.collapsed { display: none }
    #audit-panel .vd-detail .vd-break b { color: var(--text); font-weight: 600; }
    @keyframes stepFlash { 0%{background:#fff3d6} 100%{background:transparent} }
    #audit-panel .ap-step.flash { animation: stepFlash 1.6s ease-out }
```

### 2d JS（替换旧 Diagnose 块）
- 状态：`const diagnosisByRun = new Map()`（缓存每个 runId 的结果，切回不丢）；`let pendingStepFlash = null`。
- `renderVerdict()`：根据 `diagnosisByRun.get(auditAnchorRunId)` 渲染结论条 innerHTML 到 `$auditVerdict`；无结果时显示「⚖ 诊断 · 判断答案是否答到问题上 · [诊断]」；有结果时按 ok/suspect/error 渲染 summary + 重新诊断按钮 + detail。suspect 且 `Number.isInteger(fb.step)` 时，断点渲染为 `<button class="vd-jump" data-step-idx="${fb.step}">步 ${fb.step}</button>`，否则纯文本。`$auditVerdict.className = 'ap-verdict shown ' + state`。空 anchor 时 `className='ap-verdict'`、清空。
- `async function runDiagnose()`：禁用按钮、文案「诊断中…」；fetch POST `/run/:id/diagnose`；结果存 `diagnosisByRun.set(runId, data)`；`renderVerdict()`；catch 存 `{error}`。
- 事件委托（绑到 `$auditVerdict`）：点 `.vd-run` → runDiagnose；点 `.vd-summary` → 切换 `.vd-detail.collapsed`；点 `.vd-jump` → `jumpToStep(+dataset.stepIdx)`。
- `function jumpToStep(idx)`：`pendingStepFlash = idx; setAuditTab('steps')`。
- 在 `renderAuditBody` 的 steps `.then` 块里，`$auditBody.innerHTML = html` 之后追加：若 `pendingStepFlash != null`，`const el = $auditBody.querySelector('.ap-step[data-step-idx="'+pendingStepFlash+'"]')`，有则 `el.scrollIntoView({block:'center'})` + `el.classList.add('flash')`；`pendingStepFlash = null`。
- `openAudit`（1238）：`if (auditAnchorRunId !== runId) { hideReplayBanner() }` —— 删掉这里的 `hideDiagnoseBanner()`（改为切 run 后调 `renderVerdict()` 重画结论条；结果靠 map 持久，不再"清空"）。在设置完 `auditAnchorRunId` 后调用 `renderVerdict()`。
- `closeAudit`（1252）：删 `hideDiagnoseBanner()`（结论条随 panel 关闭隐藏即可；下次开由 renderVerdict 重画）。closeAudit 里 `$auditVerdict.className='ap-verdict'` 清隐藏。

### 2e 验证
- `node` 抽 `<script>` 块 `new Function` 语法校验。
- 起 server 手验：ok 结论条一行 + 点开解释；suspect 断点 chip 点击跳 Execution 高亮；切 run 结论条随之变化、旧结果不串；解析失败降级。
- commit：`feat(#98): 诊断结论条 + 断点跳 Execution（替换 #94 按钮+banner）`

## Task 3：README + 收尾
- README：把 #94 写的「⚖ 诊断按钮」描述改为「header 下方结论条 + 断点可跳 Execution」。
- 全测试 `npx jest examples/agent-docs-qa` 绿。
- commit：`docs(#98): README 同步诊断结论条布局`

## 自检要点
- esc() 覆盖所有模型可控插值（explanation/why/what/step/raw/error）。
- `vd-jump` 仅在 step 为整数时出现；越界（querySelector 找不到）静默不跳。
- diagnosisByRun 用 runId 键，切 run 不串。
