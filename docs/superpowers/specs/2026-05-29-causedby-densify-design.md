# causedBy 加密（4 条边）（#30）

**Issue:** #30 [diagnosable P0] causedBy 加密（4 条边）
**Parent:** #20（Trace substrate gap — 6-capability surface）
**Date:** 2026-05-29
**依赖:** #21 fsm.transition 事件（已合）
**Blocks:** #32 / #33 / #34 / #35 / #36（diagnosable 整层）

## 背景

今天 `causedBy` 只在 `RecordingIOPort.ts` 三处写入（`llm.responded` / 成功 `tool.responded` /
错误 `tool.responded`），**全部是 `*.requested → *.responded` 的配对边（length=2）**。所谓
“causedBy 链”实际是**边集合,不是图**：给定任一 `tool.requested` 追不回触发它的 `llm.responded`，
更回溯不到 `agent.run.started`。diagnosable 整层（#32 图可视化 / #33-34 explainer / #35 失败路径）
都要一张**稠密、可顺向回溯**的 causedBy 图。

本 issue 再织入 4 条边，把边集合补成图：

1. `tool.requested.causedBy` = 决定该调用的 `llm.responded`（含 toolCalls 的那一帧）
2. `llm.requested.causedBy` = 上一个 turn 终结事件
3. `fsm.transition.causedBy` = 触发该转移的事件
4. `region.added.causedBy` = 触发它的 `context.boundary.applied` 或 `agent-set`

“加密”= **加密度**（densify），与加解密无关。

## 关键事实（决定设计形态）

### F1. emit 走两条路径，append 顺序 ≠ 因果顺序

- `RecordingIOPort` 的 llm/tool 事件是 **`await store.append` 内联**写。
- `AgentRuntime` 的 `fsm.transition` / `region.*` / `agent.*` 事件走 `enqueueTraceWrite`
  （`AgentRuntime.ts:566`）——只把写操作 `.then` 进一条 promise 链,**立即返回不 await**,
  实际落盘被推迟（常排在后续内联事件之后）。

**推论：** 任何“从存储读上一条 append”的游标都会读到错乱值；`causedBy` 必须在**事件构造的同步
时刻**就盖上，绝不能放进延迟的 `enqueueTraceWrite` 闭包里、也不能做成 `store.append` 拦截器。

### F2. trace 事件 id 是裸 uuid，不进 nondet 缓存 → 加 causedBy 不影响 replay

trace 事件 id 全是裸 uuid：`RecordingIOPort` 用 `this.inner.uuid()`（绕过 nondet pending buffer），
`AgentRuntime` 用 `uuidv4()` 直发（见 `AgentRuntime.ts:170-173` 的注释论证）。它们**不被 replay 的
nondet 缓存消费**，每次 replay 重新生成。`causedBy` 只是引用这些 id，既不消费
`ioPort.uuid()/now()`，也不改变 agent 产出。**byte-identical replay 断言的是 agent 产出（走 nondet
缓存），不比 trace 事件 id**——故本 issue 对 replay 零影响（验收第 4 条）。

副推论：并行工具批（`executeTools` 的 `Promise.allSettled`）里多条 `tool.responded` 完成次序不定 →
“下一个 `llm.requested` 的终结者是哪条”有竞态，但 (a) replay 不比 trace id、(b) issue 只要单条
causedBy，取“最近一条终结者”即可，竞态不影响正确性。

> **实现注释要求（强制）：** 在 `RecordingIOPort.invokeTool` 写 `cursor.lastTerminatorId` 那处必须留
> 一行注释，说明“并行批下取最后完成的 tool.responded 是有意的、对正确性与 replay 无害”——否则读者会
> 把这个良性竞态当 bug 去“修”。这是本 issue 唯一一处“看着像竞态、实则有意”的非显然点，值得注释。

### F3. 4 条边里 3 条的上游就在本地手边，只有 1 条跨文件

| 边 | 上游 id 实际来源 | 跨文件? |
|---|---|---|
| `tool.requested ← llm.responded` | RecordingIOPort 自己刚 emit 的 llm.responded | 否（本地） |
| `llm.requested ← 上个终结者` | 终结者 = RecordingIOPort 自己 emit 的 tool.responded（子 agent 也以工具结果 tool.responded 回来，覆盖 issue 列的 agent.spawned/returned）；首帧 seed 成 `attach` 的 agent.run.started | 否（本地） |
| `region.added ← boundary.applied` | 同在 AgentRuntime,本地存 boundary id | 否（本地） |
| `fsm.transition ← 触发事件` | 触发者是工具 `ctx.emit`（→tool.responded）或文本 DONE（→llm.responded），**id 只在 RecordingIOPort 手里**；且 `ctx.emit` 时 tool.responded 尚未生成（`invokeTool` 在 handler 返回后才 emit），只能事后取“最近一条” | **是** |

所以**不需要**一个统一 emit helper、也**不需要** ambient 全局——3 条本地边各用本地字段即可；唯一跨文件
的 `fsm.transition` 用一个 per-run 共享小对象接（下文 §CausalCursor）。

> **显式不做（避免过度工程）：** 不把 5、6 处 `store.append` 收口成统一 `emitTraceEvent` helper。
> 理由仅是“#32-36 以后可能要”——为假想未来提前上机制。#30 只需 4 条边；收口待真有第二个消费者再做。

## 设计

### CausalCursor（新文件 `src/trace/CausalCursor.ts`）

一个 **per-run** 的纯状态小对象（无逻辑、无 IO），所有权显式，不污染 `IIOPort`，不依赖全局：

```ts
export class CausalCursor {
  /** 最近一条由 RecordingIOPort emit 的 llm/tool 事件 id。fsm.transition 读它。 */
  lastIoEventId?: string
  /** 最近一条 llm.responded 的事件 id。tool.requested 读它。 */
  lastLlmRespondedId?: string
  /** 最近一条 turn 终结事件 id（tool.responded / agent.run.started seed）。llm.requested 读它。 */
  lastTerminatorId?: string
}
```

**铁律（呼应 F1）：** 上述字段的读 / 写**全部发生在同步的 emit / callback 时刻**，写在事件被构造、
入队（或内联 append）之前；fsm.transition 在其同步 callback 里读 `lastIoEventId`。

### 各边落地

**RecordingIOPort（构造函数新增可选 `cursor?: CausalCursor`）**

- `attach(payload)`：emit `agent.run.started` 后,`cursor.lastTerminatorId = <该事件 id>`
  （为首个 `llm.requested` 提供回溯根）。
- `invokeLLM`：
  - 构造 `llm.requested` 时盖 `causedBy = cursor.lastTerminatorId`（边 2）；emit 后
    `cursor.lastIoEventId = <reqId>`。
  - emit `llm.responded` 后：`cursor.lastLlmRespondedId = <该事件 id>`；`cursor.lastIoEventId = <该事件 id>`。
  - （`llm.responded` 自身 causedBy 仍 = reqId，既有配对边不变。）
- `invokeTool`：
  - 构造 `tool.requested` 时盖 `causedBy = cursor.lastLlmRespondedId`（边 1）；emit 后
    `cursor.lastIoEventId = <reqId>`。
  - 成功 / 错误 `tool.responded` emit 后：`cursor.lastTerminatorId = <该事件 id>`；
    `cursor.lastIoEventId = <该事件 id>`。（配对边 causedBy=reqId 不变。）

> cursor 缺省（未注入）时全部降级为不盖新边——既有行为与测试不变。

**AgentRuntime（构造函数 / opts 新增可选 `causalCursor?: CausalCursor`）**

- `setupFSMCallbacks` 的 `onTransitionCallback`（`AgentRuntime.ts:160`）：在**同步 callback 内**
  读 `this.causalCursor?.lastIoEventId`，作为 `fsm.transition` 的 `causedBy`（边 3），再 enqueue
  延迟写（值已捕获，落盘晚无妨）。
- boundary：`emitBoundaryApplied`（`:590`）emit `context.boundary.applied` 时，把该事件 id 记到
  本地字段 `this.lastBoundaryId`。
- region delta：`emitRegionDelta`（`:436/447`）构造 `region.added` 时盖
  `causedBy = this.lastBoundaryId`（边 4，仅 `added`；`removed` 不盖）。turn-end crystallization
  产出的 region delta 紧跟 boundary，故 `lastBoundaryId` 此刻有效；`agent-set` 来源（非 boundary
  触发，如 header / 初始 set）`lastBoundaryId` 可能为空 → causedBy 缺省（可接受，见验收）。

### Wiring（cursor 注入，root + child）

cursor 须 per-run，且同一 run 的 RecordingIOPort 与 AgentRuntime 共享同一实例。

- **root（`Milkie.run` / `resume`）**：在创建 ioPort 处 `const cursor = new CausalCursor()`，
  - `wrapIOPort(gateway, runId, cursor)` 新增参数,透传给 `new RecordingIOPort(..., cursor)`；
  - 同一 cursor 传入 AgentRuntime opts `causalCursor`。
- **child（`Milkie.buildMakeChildPort`）**：`makeChildPort` 内 `const cursor = new CausalCursor()`，
  注入子 `RecordingIOPort`，并把 cursor 加进返回值 `{ port, finish, cursor }`；
  `AgentRuntime.makeSubAgentTool`（`:240-262`）建子 runtime 时 `causalCursor: built.cursor`。
  → 每个子 run 自己一份 cursor，父子互不串。

## 测试（`src/__tests__/`，新增 `CausedByGraph.test.ts` + 扩展既有 harness）

1. **边 1**：一次 run 含 `llm.responded(toolCalls) → tool.requested`，断言
   `tool.requested.causedBy === <那条 llm.responded.id>`。
2. **边 2**：第二轮 `llm.requested.causedBy === <上一轮 tool.responded.id>`；首个
   `llm.requested.causedBy === <agent.run.started.id>`。
3. **边 3**：工具 `ctx.emit` 触发的 `fsm.transition.causedBy === <触发它的 tool.responded.id>`。
4. **边 4**：turn-end 后 `region.added.causedBy === <对应 context.boundary.applied.id>`。
5. **图连通（验收 1）**：任取一条 `tool.requested`，顺 `causedBy` 能走到 `agent.run.started`。
6. **无孤儿（验收 3）**：除 `agent.run.started`（及 `agent-set` 非 boundary 触发的 region.added）外，
   关键节点均有 causedBy。
7. **cursor 缺省**：不注入 cursor 时不盖新边、不抛错、既有事件不变。
8. **replay 回归（验收 4）**：`Replay.test.ts` / `s-005` 全过；byte-identical 不变。
9. 子 agent：父子 cursor 隔离，子 run 内部 4 条边自洽（基于 s-007 harness）。

## 验收（issue 原文 + 一处偏离）

- [ ] 给定任一 `tool.requested` 可顺 causedBy 回溯到 `agent.run.started`
- [ ] 给定任一 **boundary 触发的** `region.added` 可定位到对应 `context.boundary.applied`
- [ ] causedBy 图无孤儿节点，**两类例外允许 causedBy 缺省**：
  - `agent.run.started`（回溯根，issue 原文）
  - **`agent-set` 来源的 `region.added`**（初始 header / 非 boundary 触发的 region，无上游 boundary）
- [ ] replay byte-identical 不变

> **偏离说明（已与用户确认）：** issue 原文写「仅 `agent.run.started` 允许缺省」。实现中
> `agent-set` 来源的 region（初始 header、非 turn-end 触发的 set）没有上游 `boundary.applied` 可挂，
> 强行造一条边会是假因果。故接受其 causedBy 缺省，作为第二类显式例外。后续若需要，可统一挂到
> `agent.run.started` 作根——但当前不做（无消费者要求）。

## 显式 deferral

- 统一 `emitTraceEvent` helper 收口（见 §F3 注）→ 待第二个因果消费者出现再单开。
- `causedBy` 图可视化 / explainer / 失败路径 → #32 / #33 / #34 / #35。
- 并行工具批的“多终结者”精确归因（目前取最近一条）→ 若 #35 失败路径需要再细化。

## Related

- ARCH.md 6-capability surface（diagnosable）
- 依赖 #21 fsm.transition（`AgentRuntime.ts:159-196` 的 `onTransitionCallback` + event 落盘）
- Blocks #32 / #33 / #34 / #35 / #36
