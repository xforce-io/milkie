# #68 把决策因果归因接进 agent-docs-qa 的 web audit panel — 设计 spec

**Issue:** #68（diagnosable；Parent #20；依赖 #64 决策 viewer，及 #26/#30/#31/#33/#34，均已落地于 main）
**日期:** 2026-05-31
**定位:** 把 #64 已聚合好的「决策脊柱 + Why 面板 + 因果下钻」搬进 `examples/agent-docs-qa` 这个**参考 web 应用**的 live audit panel——**整块复用核心 `renderViewer`，零核心代码改动**，守住 ARCHITECTURE.md「UI = projection over CLI/SDK，前端不自带归因逻辑」。

经 brainstorm 定稿：集成形态 = 方案 A（复用 renderViewer，iframe 嵌入为新 tab）；Steps 处理 = 保留并重定位（分工镜头），其投影化重构踢成独立 follow-up。

---

## 1. 目标

用户在运行的 localhost web UI 里（不再只能开独立 HTML），对一次真实对话能**因果下钻**——从输出两次点击钻到根因。把 #64 的全部归因能力（FSM 转移 / #30 因果 / #33 转移why / #34 llm-why+#26 composition / #31 guard / #64 工具why + 脊柱 + 下钻）呈现在 panel 里，和 panel 已有的「源溯源」(Sources/Provenance) 并存。

## 2. 背景与现状

agent-docs-qa 是 milkie 的**参考应用**——演示能力 + 示范正确架构。其右侧 audit panel 现有三个 tab：

- **Sources** — 答案引用的源文档（源归因）
- **Steps** — region 列表 + cache health（#26 + Phase 4a）；**前端自己解析事件重写**（`renderStepsTab`）
- **Provenance** — 答案段落 → 源文档验证（源归因）

缺口：决策→成因（为什么这么跳/这么调/prompt 怎么拼）= 零（grep `explainTransition`/`explainLlmCall`/`fsm.transition`/`causedBy` 全无命中）。且 Steps 的前端重写正是 ARCHITECTURE.md 不让做的反模式。

核心侧 #64 的 `renderViewer(events, { regionContent })`（`src/trace/render/viewer.ts`）已把全部决策归因聚合在一个**自包含两栏 HTML 文档**里，但它是无 server 的静态 post-hoc 输出（`trace report` 的主输出）。

## 3. Non-goals

- **零核心改动**：不改 `src/trace/*`，整块复用 `renderViewer`。
- **不重构 Steps**：Steps 改投影驱动（消除前端重写）= 独立 follow-up issue，不进本 PR。"加能力"与"重构旧代码"分开走，避免 PR 膨胀。
- **MVP 不做**：viewer tab 的 live 自动刷新、descendant 子 run 合并、DAG/diff/搜索/跨 run 导航。

## 4. 架构与组件

```
examples/agent-docs-qa/
  server.ts        // 接 FileTraceObjectStore；新增 GET /run/:runId/viewer
  public/index.html// 新增「决策·为什么」tab（iframe）；Steps 改名「执行·成本」
  __tests__/server.test.ts // viewer endpoint 测试
```

复用（不改）：`src/trace/render/viewer.ts` 的 `renderViewer`、`src/trace/TraceObjectStore.ts` 的 `FileTraceObjectStore`。

### 4.1 后端 — `FileTraceObjectStore` 装配（必须项）

`server.ts:startServer` 现以 `new Milkie({ stateStore, gateway, eventStore })` 构造，**未挂 traceObjectStore** → region 内容不落盘 → #26 预览只能降级"(内容不可用)"，过不了验收第 3 条。

改动：
- `const traceObjectStore = new FileTraceObjectStore(path.join(config.exampleDir, '.milkie', 'objects'))`
- 传进 `new Milkie({ …, traceObjectStore })`，使 region 内容在 invoke 期间落盘
- 存进 `ServerState`，供 viewer endpoint hydrate

### 4.2 后端 — `GET /run/:runId/viewer` → HTML

平行现有 `POST /run/:runId/replay`（`server.ts:126` 已用 `eventStore.readByRunId(runId)`）：

1. `events = await eventStore.readByRunId(runId)`；空 → 404
2. hydrate regionContent：照搬 `cli/main.ts:195-201` 模式——遍历事件里 region 的内容 hash，`traceObjectStore.getCanonical(hash)` 命中则填入 `Map<hash, content>`
3. `return renderViewer(events, { regionContent })`，`Content-Type: text/html`

作用域 = 单 runId（MVP）。example 的 `sanguo-researcher` 是单 agent ReAct，不 spawn 子 run；descendant 合并留作扩展。

### 4.3 前端 — 新 tab + 重命名（`public/index.html`）

- tab 栏（现 `data-tab="sources|steps|provenance"`，行 ~460）新增 `<div class="ap-tab" data-tab="why">决策 · 为什么</div>`
- Steps 按钮可见文案改 `执行 · 成本`（**仅改文案**；`data-tab="steps"` 与 `renderStepsTab` 不动，保留其 cache health/region 分组 UX）
- `renderAuditBody` 对 `why` tab 渲染 `<iframe class="ap-viewer-frame" src="/run/${runId}/viewer">`，撑满 `ap-body`、border:none；**lazy 设 src**（切到该 tab 才加载）
- #64 viewer 的脊柱/Why/下钻 vanilla JS 在 iframe 内自跑，样式/JS 天然隔离（renderViewer 出完整 `<!doctype html>` 文档）

最终四 tab：`Sources` · `执行·成本` · `Provenance` · `决策·为什么`

### 4.4 镜头分工（产品决策）

Steps 与新 tab **非冗余，是两个镜头**，命名上明确分工：

| tab | 回答 | 能力 |
|---|---|---|
| 执行·成本（原 Steps） | 做了什么 / 花多少 | cache 健康、region by stability（#26 + Phase 4a） |
| 决策·为什么（新） | 为什么这么做 | 因果下钻、FSM 转移、why、guard（#30/#31/#33/#34/#64） |

## 5. 数据流 / 下钻

- viewer endpoint 在请求期算好全部 explanation 并内嵌（#64 既有机制，纯投影、客户端不重算）
- "两次点击到根因" = iframe 内：点输出 ❓ → 选中其因（最近决策祖先）+ 脊柱高亮 → 点"← 谁导致的" → 再上钻一层
- live：MVP 打开 tab 时按当前已落盘事件渲染；run 进行中可重开/刷新（#64 本就 post-hoc，验收针对已完成 run）

## 6. 错误处理 / 边界

- 未知 runId → endpoint 404；前端 iframe 显示错误页（沿用 server 默认）
- region 无内容 hash / objectStore 未命中 → composition 降级"(内容不可用)"（沿用 #26/#64）
- 无决策事件的 run → 脊柱空、面板提示（#64 既有行为）
- 旧 run（接 objectStore 之前产生的，objects 目录无内容）→ 预览降级，不报错

## 7. 测试

- **`server.test.ts`**：
  - `GET /run/:id/viewer` → 200 + `Content-Type: text/html`；body 含脊柱标记（`data-id`、`spine-output`、内嵌 spine JSON）
  - 未知 runId → 404
  - 真实 stub run（产生 region.added）后，viewer body 含真实 region 内容预览（覆盖**验收第 3 条**；前置 = objectStore 已接）
  - 现有测试全绿（含 Steps tab 文案改动不破坏断言）
- **dev-browser（live 验证，覆盖验收第 1 条）**：起 server → 发一次 chat → 开 audit panel → 点「决策·为什么」→ iframe 加载 → 点输出 ❓ → 断言 Why 面板出"因" + 该因脊柱节点高亮 → 点"← 谁导致的" → 断言选中上移、面板刷新 = **两次点击到根因**

## 8. 改动文件清单

| 文件 | 改动 |
|---|---|
| `examples/agent-docs-qa/server.ts` | 接 `FileTraceObjectStore` 进 Milkie + ServerState；新增 `GET /run/:runId/viewer` handler（readByRunId + hydrate regionContent + renderViewer） |
| `examples/agent-docs-qa/public/index.html` | 新增「决策·为什么」tab（lazy iframe）；Steps 文案改「执行·成本」；`renderAuditBody` 处理 `why` |
| `examples/agent-docs-qa/__tests__/server.test.ts` | viewer endpoint 测试（200+脊柱标记、404、regionContent 预览） |
| dev-browser 自动化 | "两次点击到根因" live 验证 |

不改任何 `src/trace/*`（整块复用）。

## 9. Follow-up（不进本 PR）

- 新开 issue：把 Steps（`renderStepsTab`）从"前端解析事件重写归因"改成消费核心投影（即 brainstorm 中被否的方案 C，独立重构）。
- 可选：viewer tab live 自动刷新；descendant 子 run 合并。

## 10. 验收对照（#68）

- [ ] localhost web UI 对一次真实对话两次点击到根因 → §4.3/§5 + dev-browser 测
- [ ] 复用核心投影、前端不自带归因逻辑 → §3/§4（整块 renderViewer，iframe）
- [ ] 真实 run（doubao，#66 修复后）能看到真实 region composition 内容预览 → §4.1 接 objectStore + §7 测

## 11. 分支

`feat/68-trace-attribution-in-web-panel`，基于 main（含 #64）。
