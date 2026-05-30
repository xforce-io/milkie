# #33 why-this-transition explainer — 设计 spec

**Issue:** #33（diagnosable P1;依赖 #21/#30/#31 已落地;Stories s-003 / s-011）
**日期:** 2026-05-30
**定位:** 把 diagnosable 做出可见效果——点 `fsm.transition` 能读懂"为什么这次跳转"。**纯 event-log 投影,零 LLM、零存储。**

---

## 1. 目标与范围

今天点 trace 里的 transition 什么解释都没有,s-003「explain a decision」无交付物。本 issue:
- 抽一个**共享投影函数** `explainTransition(events, eventId)`,把"为什么这次跳转"从 event log 折叠成结构化解释;
- 在静态 HTML 报告的 `fsm.transition` 条目上加**可读的 "Why?" 展开**(摘要 + 触发事件 + guard + 因果链),并能**一键跳到上游事件**。

**交付面**(已与用户确认):增强现有静态 HTML 报告 + 抽共享投影(给 #36 CLI 复用),**不**新建 playground/web app。

## 2. Non-goals(明确不做)

- **不做 region**:`region.added` 无 transition 标记、#26 region-link 未完成。本 issue 只交付 trigger + guard + 因果链;region 关联等 #26 落地再补。
- **不调 LLM**:`summary` 是模板拼接的确定性字符串。
- **零存储**:不新增事件类型、不写快照/缓存、不改任何事件 schema。每次现算(见 §4)。
- **不做 CLI**:`trace why/explain/path` 是 #36;本 issue 只提供它要复用的投影函数(返回可序列化对象)。
- **不做其它投影**:`explainLlmCall` / `failurePath`(#34/#35/#36)不在本 issue。

## 3. 数据齐备性(全部来自已落地的存储)

| 解释要素 | 来源字段 | 谁存的 |
|---|---|---|
| from / to / trigger.name / domain | `fsm.transition` payload | #21 |
| guard 评估 | `fsm.transition` payload `.guardEvaluations` | **#31** |
| 触发它的上游事件 | `fsm.transition` 的 `causedBy` | **#30** |
| 因果链 | 沿 `causedBy` 遍历 Event[] | 现算 |

`fsm.transition` 事件结构**一字段不加**——#33 只读。

## 4. 架构:`src/trace/diagnostics/`(#36 也复用的共享投影之家)

两个纯函数,只吃内存里的 `Event[]`(由调用方 `eventStore.readByRunId(runId)` 读出),**不做任何 I/O、不写任何东西**。符合 ARCHITECTURE.md「event-sourced view:从 append-only log 折叠,不作 canonical 存储」。

### 4.1 `walkCausedBy(events, eventId): Event[]`

- 文件:`src/trace/diagnostics/walkCausedBy.ts`
- 从 `eventId` 沿 `causedBy` 上溯,返回 `[该事件, 其 cause, …, agent.run.started]`(含端点,顺序由近到远)。
- **断链/缺失**:若某事件的 `causedBy` 指向的 id 不在 `events` 里(或无 `causedBy`),在该点优雅停止,返回已收集的链,不抛错。
- **环防护**:用 `Set<eventId>` 去重,遇已访问 id 即停(防御性,正常不会成环)。
- 通用:#32(causedBy 可视化)/#36(`trace path`)亦可复用。

### 4.2 `explainTransition(events, transitionEventId): TransitionExplanation`

- 文件:`src/trace/diagnostics/explainTransition.ts`
- 输出(普通可序列化对象,既是 UI 渲染源,也是 #36 CLI 的 JSON 输出源):

```ts
export interface TransitionExplanation {
  transitionEventId: string
  from: string
  to:   string
  trigger: {
    name:            string                    // 'INTENT_B' / 'DONE' / 'interrupt'
    domain:          FsmEventDomain
    causedByEventId?: string                   // 发出它的 tool.responded/llm.responded(= 事件的 causedBy)
    causedBySummary?: string                   // 'tool.responded(classify_intent)' 之类的人类标签
  }
  guards: GuardEvaluation[]                     // 直接取自 payload.guardEvaluations(无则空数组)
  causalChain: Array<{ eventId: string; type: EventKind; summary: string }>  // 由 walkCausedBy 投影
  summary: string                               // 模板拼接的可读一句话,无 LLM
}
```

- **行为**:
  - 找到 id 对应事件;若不存在或 `type !== 'fsm.transition'` → 抛 `Error`(#36 CLI 据此非零退出;UI 只对 fsm 条目调用,安全)。
  - `from/to/trigger.{name,domain}` 取自 payload;`trigger.causedByEventId` = 该事件 `causedBy`;`causedBySummary` 由那条上游事件经 `summarizeEvent` 生成。
  - `guards` = `payload.guardEvaluations ?? []`。
  - `causalChain` = `walkCausedBy(events, transitionEventId).map(e => ({ eventId, type, summary: summarizeEvent(e) }))`。
  - `summary` 模板(纯数据,示例):
    `"{from} → {to}:由 {causedBySummary} 发出的 {trigger.name} 触发"` +(若有 guard)`";guard {g.guardId} 判定 {g.result}({contextSlice 紧凑串})"`。
- 复用一个小工具 `summarizeEvent(e): string`(`src/trace/diagnostics/summarizeEvent.ts`),把事件转人类标签(如 `llm.responded`、`tool.responded(classify_intent)`、`agent.run.started`)。渲染层与 CLI 共用,避免两处各写一套。

## 5. HTML "Why?" 渲染(`src/trace/render/`)

- 对 `kind:'fsm'` 条目,在现有可展开 JSON 之外,增加一个**人类可读的 "Why?" 块**,内容由 `explainTransition` 投影而来:摘要、触发(带跳转链接)、guard 列表、因果链(每跳一个链接)。
- **事件锚点**:为支持"一键跳上游",渲染每个事件条目时加 `id="ev-<eventId>"`;"Why?" 里的链接用 `href="#ev-<eventId>"`。这是纯加属性,不改现有点击展开行为。
- 不带 guard 的 transition 也正常渲染(无 guard 行)。
- 渲染层调用 `explainTransition(allEvents, entry.eventId)`;`renderHtml` 已持有全部 `Event[]`,直接传入。

## 6. 错误处理

- `explainTransition`:未知 id / 非 transition → 抛 `Error`(消息含 eventId 与实际 type)。
- `walkCausedBy`:断链或缺失 cause → 优雅停止;成环 → Set 去重即停。
- 渲染层只对 `kind:'fsm'` 调用,故不触发上面的抛错路径。

## 7. 测试

- **`walkCausedBy`**:① 正常链走到 `agent.run.started`;② `causedBy` 指向不存在 id → 在该点停、返回部分链;③ 人造环 → 不死循环。
- **`explainTransition`**:用真实运行(routing FSM:`classify`→tool emit `INTENT_*`)产出的事件,断言 `from/to/trigger.name/causedByEventId/guards` 正确、`causalChain` 每个 `eventId` 都能在 events 里找到、`summary` 含关键字段;断言**全程无 gateway/LLM 调用**(用已录制事件,不跑 live)。③ 非 fsm.transition id → 抛错。
- **渲染**:fsm 条目 HTML 含 "Why?" 块、摘要文本、`href="#ev-..."` 锚点链接、对应 `id="ev-..."` 存在;不带 guard 的转移也出 "Why?"(无 guard 行)。

## 8. 改动文件清单

| 文件 | 改动 |
|---|---|
| `src/trace/diagnostics/walkCausedBy.ts` | 新建:causedBy 上溯遍历 |
| `src/trace/diagnostics/summarizeEvent.ts` | 新建:事件 → 人类标签 |
| `src/trace/diagnostics/explainTransition.ts` | 新建:核心投影 + `TransitionExplanation` 类型(与函数同文件,它是投影结果、非事件 payload) |
| `src/trace/render/html.ts` | fsm 条目渲染 "Why?" 块(调 `explainTransition`,`renderHtml` 已持有全部 events);`renderEntry` 给每个事件条目加 `id="ev-..."` 锚点。`tree.ts` 不动 |
| `src/__tests__/walkCausedBy.test.ts` | 新建 |
| `src/__tests__/explainTransition.test.ts` | 新建 |
| `src/__tests__/render-*.test.ts` | 加 "Why?" 渲染 + 锚点断言 |

## 9. 验收对照(#33)

- [x] ReAct/routing 的每次 transition 点 "Why?" 都给可读解释 → `explainTransition` + HTML 块
- [x] 解释生成不调用 LLM → 纯模板投影(测试断言无 gateway 调用)
- [x] 解释链接可一键跳上游事件 → 事件锚点 `id="ev-..."` + `href="#ev-..."`
- [x] 只用 event log、不依赖运行时副本 → 函数只吃 `Event[]`、零存储
- 备注:region 部分按 Non-goals 暂缓(等 #26)
