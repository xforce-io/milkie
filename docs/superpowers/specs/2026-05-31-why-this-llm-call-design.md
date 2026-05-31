# #34 why-this-llm-call explainer — 设计 spec

**Issue:** #34（diagnosable P1;依赖 #26 / #30,均已落地;Parent #20;Stories s-003 平行)
**日期:** 2026-05-31
**定位:** 点 `llm.requested` 解释"这次 LLM 调用为什么发生"——触发它的 turn 终结事件 + 当时 FSM state + 因果链。**纯 event-log 投影,零 LLM、零存储、零 schema 变更。** 与 #33 平行、共享 explainer 层。

---

## 1. 目标与范围

#33 解决了"为什么这次 transition";本 issue 平行解决"为什么这次 LLM 调用"。点 `llm.requested`,Why? 块展开:
- **触发它的 turn 终结事件**(上一个 tool.responded,或 run 起点)——来自 `causedBy`(#30 edge 2);
- **当时 FSM state**——折叠该调用之前的 `fsm.transition`;
- **因果链**——`walkCausedBy`。

**交付面**(已确认):**互补两块**——region composition 详单仍由 **#26 已有的 "Assembled by" 块**承担,本 issue 只加紧凑 Why? 块(触发 + state + 因果链 + 一句"N regions")。两块叠在同一 llm 条目上,**实现后浏览器实测,若显碎再小 follow-up 合并**。

## 2. Non-goals

- **不重复 region 详单**:Assembled-by(#26)已渲染;Why? 只说"prompt 由 N region 拼成",详见上方。
- **不调 LLM、零存储、零新事件类型、零 schema 变更**:纯只读投影。
- **不做 CLI**:`explainLlmCall` 返回可序列化对象,供 #36 复用;CLI 本身是 #36。
- **不动 #26 的 Assembled-by 渲染**(除非实测后决定合并,届时单独 follow-up)。

## 3. 数据齐备性(全部已落地)

| 解释要素 | 来源 | 谁存的 |
|---|---|---|
| 触发的 turn 终结事件 | `llm.requested` 的 `causedBy` | #30(edge 2:`lastTerminatorId` = 上一个 tool.responded,或 seed 的 agent.run.started) |
| 当时 FSM state | 折叠 `fsm.transition` 事件(新 helper) | #21(transition 事件) |
| 参与 region 数 | `contextRefsAt(events, id, 'at').size` | #23/#26 |
| 因果链 | `walkCausedBy` | #30 + #33 |

## 4. 架构:`src/trace/diagnostics/`(复用 #33 层)

两个纯函数,只吃内存 `Event[]`,零 I/O。

### 4.1 `fsmStateAt(events, eventId): string | null`

- 文件:`src/trace/diagnostics/fsmStateAt.ts`
- 折叠 `eventId` **之前**(不含)的 `fsm.transition`,返回最后一次的 `to`(= 当时所处 state)。
- 若之前无 transition:返回**全局第一个** `fsm.transition` 的 `from`(= 初始态,即该调用所处态);若全程无任何 transition → `null`(未知)。
- 纯函数。可复用(#36 / 未来 swimlane #28)。

### 4.2 `explainLlmCall(events, llmRequestedEventId): LlmCallExplanation`

- 文件:`src/trace/diagnostics/explainLlmCall.ts`
- 输出(普通可序列化对象,亦为 #36 CLI 的 JSON 形状):

```ts
export interface LlmCallExplanation {
  llmRequestedEventId: string
  trigger: {
    causedByEventId?: string                  // = 该 llm.requested 的 causedBy
    causedBySummary?: string                  // summarizeEvent(那条上游事件)
  }
  fsmState: string | null                     // fsmStateAt 结果
  regionCount: number                         // contextRefsAt(events, id, 'at').size
  causalChain: Array<{ eventId: string; type: EventKind; summary: string }>
  summary: string                             // 模板拼接,无 LLM
}
```

- **行为**:
  - 找到 id 对应事件;不存在或 `type !== 'llm.requested'` → 抛 `Error`(消息含 id 与实际 type)。
  - `trigger.causedByEventId` = 事件 `causedBy`;`causedBySummary` = 该上游事件经 `summarizeEvent`。
  - `fsmState` = `fsmStateAt(events, id)`;`regionCount` = `contextRefsAt(events, id, 'at').size`。
  - `causalChain` = `walkCausedBy(events, id).map(...)`(复用 #33)。
  - `summary` 模板(纯数据,示例):`"LLM 调用 @ state {fsmState|'?'},由 {causedBySummary|'(无上游)'} 触发;prompt 由 {regionCount} 个 region 拼成"`。
- 复用 `summarizeEvent`(#33);无重复造轮子。

## 5. UI:llm 条目加紧凑 "Why?" 块(`src/trace/render/html.ts`)

- 对 `kind:'llm'` 条目,在现有 #26 "Assembled by" 块**之后**加一个 Why? 块(由 `explainLlmCall` 投影):摘要、触发(带 `#ev-` 锚点跳转)、state、因果链(每跳一个链接)。
- **复用 #33 的 `.why` 样式与事件锚点**(`id="ev-..."` 已由 #33 落地)。region 详单不重复(只在摘要里说 "N region",详见上方 Assembled-by)。
- renderHtml 已持有全部 `Event[]` 与现有 explanations 机制;为 llm 条目预算 `explainLlmCall`(类比 #33 fsm 的 explanations map)。
- `fsmState` 为 null → state 行显示 "(未知)"(不省略,保持块结构一致)。

## 6. 错误处理 / 边界

- `explainLlmCall`:未知 id / 非 llm.requested → 抛 `Error`。
- 无 causedBy(理论上首个 llm 也 seed 到 run.started;防御)→ trigger 字段省略,摘要降级 "(无上游)"。
- 无任何 transition → `fsmState: null` → UI "(未知)"。
- 渲染层只对 `kind:'llm'` 调用,不触发抛错路径。

## 7. 测试

- **`fsmStateAt`**:① 之前有 transition → last `to`;② 之前无、之后有 → 初始态(首个 transition 的 `from`);③ 全程无 transition → null。
- **`explainLlmCall`**:trigger(causedBy + summary)、fsmState、regionCount、causalChain(每 id 可解析)、summary 含关键字段正确;非 llm.requested id → 抛错;无 LLM 调用(合成事件)。
- **渲染**:llm 条目出 Why? 块、摘要、`#ev-` 锚点链接、state 行;与 #26 Assembled-by 并存(同条目两块都在);fsmState null → "(未知)"。

## 8. 改动文件清单

| 文件 | 改动 |
|---|---|
| `src/trace/diagnostics/fsmStateAt.ts` | 新建:折叠 transition 求某时刻 state |
| `src/trace/diagnostics/explainLlmCall.ts` | 新建:核心投影 + `LlmCallExplanation` 类型 |
| `src/trace/render/html.ts` | llm 条目渲 Why? 块(复用 .why 样式 + 锚点);renderHtml 预算 explainLlmCall |
| `src/__tests__/fsmStateAt.test.ts` | 新建 |
| `src/__tests__/explainLlmCall.test.ts` | 新建 |
| `src/__tests__/render-html.test.ts` | llm Why? 块 + 并存断言 |

## 9. 验收对照(#34)

- [x] 任一 llm.requested 的 "Why?" 给出 "composition + 触发上下文" → Why?(触发+state+因果链+N regions)+ 共存的 #26 Assembled-by(composition 详单)
- [x] 不依赖运行时副本 → 纯 `Event[]` 投影,零存储
- [x] 与 #33 风格一致 → 复用 walkCausedBy/summarizeEvent/.why 样式/锚点
- 附:region 详单复用 #26;两块是否合并 → 实现后浏览器实测再定
