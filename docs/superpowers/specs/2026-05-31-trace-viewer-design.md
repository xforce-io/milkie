# #64 因果下钻 trace viewer(决策脊柱 + Why 面板)— 设计 spec

**Issue:** #64（diagnosable;依赖 #26/#30/#31/#33/#34,均已落地;Parent #20）
**日期:** 2026-05-31
**定位:** 把 `trace report` 从"时间平铺数据堆"重组成"**决策脊柱 + Why 面板 + 点击下钻**"的诊断工具。仍是**纯静态 HTML projection**(自包含、无 server),让前几轮建的能力(causedBy / guard / explain* / composition)第一次连成"从输出钻到根因"的动作。

经 brainstorm + 可视化 mockup 定稿(范式 A:决策脊柱+Why面板;时间序;viewer 主输出)。mockup 存 `.superpowers/brainstorm/51358-1780194589/content/`。

---

## 1. 目标

诊断按**因果**进行,不按时间浏览。本 viewer:从一个输出/可疑步,**两次点击钻到"哪步决策错了、为什么"**。

## 2. Non-goals

- **零 LLM、零存储、零新事件类型、零 schema 变更**:纯只读 event-log 投影(+ region 内容经 objectStore hydrate,沿用 #26 CLI 模式)。
- **无 server**:自包含静态 HTML + vanilla JS。守 ARCHITECTURE.md「UI = CLI/SDK projection,不自带查询逻辑」。
- **MVP 不做**:DAG 图、diff、失败路径自动高亮、搜索、跨 run 导航。
- **不重写已有投影**:复用 explainTransition/explainLlmCall/contextRefsAt/walkCausedBy。

## 3. 架构与组件

```
src/trace/diagnostics/
  buildDecisionSpine(events) → DecisionSpine     // 新:events → 决策脊柱
  explainToolCall(events, id) → ToolCallExplanation  // 新:工具节点 why
  (复用) explainTransition / explainLlmCall / contextRefsAt / walkCausedBy / summarizeEvent

src/trace/render/
  viewer.ts            // 新:两栏 HTML + 内嵌 explanation JSON + 节点数据
  viewer-template.ts   // 新:viewer 的 CSS + vanilla JS(选中/下钻/高亮)
```

### 3.1 `buildDecisionSpine(events): DecisionSpine`(纯)

```ts
type DecisionKind = 'llm' | 'tool' | 'transition' | 'output'
interface DecisionNode {
  eventId:          string
  kind:             DecisionKind
  label:            string          // summarizeEvent 风格的人类标签
  timestamp:        number
  causedByEventId?: string          // 原始因(该事件的 causedBy,可能指向非决策事件)
  causeDecisionId?: string          // 最近决策祖先:沿 causedBy 上溯找到的第一个脊柱节点 id(渲染期用 walkCausedBy 算)。下钻目标。
}
interface DecisionSpine {
  nodes: DecisionNode[]            // 时间序(timestamp 升序)
}
```

- **入选节点**:`llm.requested`(label "LLM call")、`tool.requested`(label `tool: <name>`)、`fsm.transition`(label `from → to`)、`agent.run.completed`(label "输出",kind `output`)。其余(clock/uuid/region.*/llm.responded/tool.responded/run.started/boundary)**不进脊柱**。
- `causedByEventId` = 该事件的 `causedBy`(#30)。注意:它可能指向一个**非决策事件**(如 transition.causedBy → tool.responded)。脊柱节点的"下钻目标"在渲染/JS 层解析为"最近的决策祖先"(见 §5)。
- 纯函数,可单测。

### 3.2 `explainToolCall(events, toolRequestedId): ToolCallExplanation`(纯)

```ts
interface ToolCallExplanation {
  toolRequestedEventId: string
  toolName: string
  input: unknown                  // tool.requested.payload.input
  output?: unknown                // 配对 tool.responded.payload.output(若有)
  trigger: { causedByEventId?: string; causedBySummary?: string }  // 谁调的(llm.responded)
  causalChain: Array<{ eventId: string; type: EventKind; summary: string }>
  summary: string
}
```
- 平行 explainTransition/explainLlmCall;复用 walkCausedBy/summarizeEvent。抛错语义同兄弟件(未知 id / 非 tool.requested)。

## 4. 渲染(`viewer.ts` / `viewer-template.ts`)

- 出**两栏自包含 HTML**:左 = 脊柱(节点列表,时间序,输出在底带 ❓ 入口);右 = Why 面板容器(初始空/提示)。
- **内嵌数据**(`<script type="application/json">`):
  - `spine`:`DecisionNode[]`
  - `explanations`:`{ [eventId]: <该节点的 explanation 对象> }`——渲染期对每个脊柱节点按 kind 调对应 explain*(transition→explainTransition,llm→explainLlmCall,tool→explainToolCall,output→见下)预算好。
  - `regionContent`:`{ [hash]: content }`——CLI hydrate(沿用 #26),供 LLM 节点 composition 内容预览。
- **output 节点**的 explanation:`{ kind:'output', causedByEventId, causedBySummary, summary }`——指向产生它的决策(其 causedBy 的最近决策祖先)。
- Why 面板按 explanation.kind 渲染:transition=触发/guard/链;llm=触发/state/composition(region 列表+内容预览,复用 #26 注册表机制)/链;tool=入参/出参/谁调的/链;output=由哪个决策产生 + ❓。
- **raw 全时间线**:作为次要视图保留——顶部一个 tab 切换"决策视图 ⇄ 原始时间线",原始时间线复用现有 `renderHtml` 的 timeline 渲染(#26/#33/#34 内联块)。

## 5. 下钻机制 / 数据流

- **渲染期**:全部 explanation 预算 + 内嵌(纯,无客户端重算)。
- **客户端 vanilla JS**:
  - `selectNode(eventId)`:从内嵌 `explanations[eventId]` 渲染 Why 面板;给脊柱里 `eventId` 节点加 `.selected`;解析其"因"= `causedByEventId` 的**最近决策祖先**(沿 causedBy 上溯到第一个在 `spine` 里的事件;用内嵌的 node→causedBy + spine id 集合),给该因节点加 `.cause`。
  - 点脊柱节点 → `selectNode`;点输出 ❓ → `selectNode(output 的因)`;点面板"← 谁导致的" → `selectNode(因 id)`。
  - **"两次点击到根因"** = 输出❓ → 选中其因 → "← 谁导致的" → 再上钻一层。
- **"最近决策祖先"在渲染期算好**(§3.1 的 `causeDecisionId`):`buildDecisionSpine` 对每个节点沿 `causedBy`(walkCausedBy)上溯,取第一个落在脊柱里的事件 id,内嵌备用。客户端 JS **不做图遍历,纯查表**——`selectNode` 直接读 `causeDecisionId` 定位"因"节点。

## 6. 错误处理 / 边界

- 无 transition 的 run(ReAct 单状态):LLM 节点 fsmState 显示"(未知)"(沿用 #34);脊柱仍有 llm/tool/output 节点。
- 节点无 `causedBy` 或因不在脊柱:"← 谁导致的"不渲染/置灰,`causeDecisionId` 为空。
- 空 run(无任何决策):脊柱空,面板提示"无决策事件"。
- region 无 contentHash / objectStore 无内容:composition 降级"(内容不可用)"(沿用 #26)。

## 7. 测试

- **`buildDecisionSpine`**:只留 4 类决策、时间序、`causedByEventId`/`causeDecisionId` 正确(含"因是非决策事件→解析到最近决策祖先"、"因不在脊柱→空");output 节点存在。
- **`explainToolCall`**:input/output/trigger/causalChain/summary 正确;抛错(未知/非 tool.requested)。
- **`viewer.ts` 渲染单测**:HTML 含脊柱节点(data-id)、内嵌 spine/explanations/regionContent JSON、Why 面板容器、tab 切换;output 节点带 ❓。
- **浏览器自动化(dev-browser)**:真实/合成 run → 点输出 ❓ → 断言面板出"因"的 why + 该因节点 `.cause` 高亮;点"← 谁导致的" → 断言 `.selected` 上移、面板内容刷新 = **"两次点击到根因"验收**;tab 切到 raw 时间线可见。

## 8. 改动文件清单

| 文件 | 改动 |
|---|---|
| `src/trace/diagnostics/buildDecisionSpine.ts` | 新建:脊柱投影 + `DecisionSpine`/`DecisionNode` 类型 |
| `src/trace/diagnostics/explainToolCall.ts` | 新建:工具节点投影 + `ToolCallExplanation` |
| `src/trace/render/viewer.ts` | 新建:两栏 HTML + 内嵌数据 + raw tab |
| `src/trace/render/viewer-template.ts` | 新建:CSS + vanilla JS(select/下钻/高亮/tab) |
| `src/cli/main.ts` | `trace report` 改用 `renderViewer`(仍 hydrate regionContent;保留 raw 复用 renderHtml) |
| `src/__tests__/buildDecisionSpine.test.ts` | 新建 |
| `src/__tests__/explainToolCall.test.ts` | 新建 |
| `src/__tests__/render-viewer.test.ts` | 新建 |

## 9. 验收对照(#64)

- [x] 输出 ❓ → Why 面板出因的解释 + 脊柱高亮 → §4/§5 + 浏览器测
- [x] "← 谁导致的"两次点击可达决策 → `causeDecisionId` 下钻 + 浏览器测
- [x] 纯 projection、不调 LLM、不依赖运行时副本 → §2/§3 纯函数
- [x] 投影 + 渲染 + 浏览器三层测 → §7
