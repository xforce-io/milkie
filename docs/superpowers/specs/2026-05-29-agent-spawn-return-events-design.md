# agent.spawned / agent.returned 事件化（#24）

**Issue:** #24 [observable P0] agent.spawned / agent.returned 事件化
**Parent:** #20（Trace substrate gap — 6-capability surface）
**Date:** 2026-05-29

## 背景

Supervisor tree 已实现（`src/runtime/AgentRuntime.ts` 的 `makeSubAgentTool` / `recordChild`
+ `types/store.ts` 的 `ChildAgentRecord`），但 sub-agent 边界**没有显式事件**：

- UI 的子 agent 嵌套只能从运行时数据结构（`ChildAgentRecord`，存在 state store）反推；
- replay / lineage / 跨 trace 导航缺 spawn 链接；
- in-flight 场景（s-015：子 agent 实时读父 trace）缺起点信号。

只要还有一类状态「事件里看不到、必须回读运行时态」，event log 就还不是
source of truth，只是 partial log。本 issue 关掉「supervisor tree 只能从
`ChildAgentRecord` 反推」这个 observable 的洞。

## Scope 决策（重要）

ARCH invariant #11 的目标态是：

> A sub-agent appears in the parent's Agent Trace as **one event plus a nested sub-trace**。

完整目标态 = （1）父 run emit `agent.spawned`/`agent.returned` 锚点 **+**
（2）子拥有独立 runId、自己的 `<childRunId>.jsonl` sub-trace、emit 自己的
`agent.run.started/completed`。

**本 issue 只做 (1)。** 原因：

- 现状下子 agent 复用父的 `agentRunId` 和父的 `RecordingIOPort`，子的所有
  I/O 事件进父的同一条 jsonl。
- `Milkie.replay(runId)` 读单条 runId 建一个 CacheIndex，父 replay 跑到 spawn
  时，**子的 LLM/tool 调用是从父的同一个 CacheIndex 喂回去的**。
- 一旦做 (2)（子独立 runId、I/O 进 `<childRunId>.jsonl`），父的 CacheIndex 里
  就没有子的 I/O，父 replay 一 spawn 子立刻 over-consume / divergence。
- 即 **(2) 与「父 replay 递归下钻子 run」是焊死的、不可分**。把 (2) 留在本
  issue 而 replay 留到下个 issue，会留下 replay 是坏的的 main。

因此干净的切割线在 (1) 与 (2) 之间：

```
(1) 父锚点                          ← 本 issue，replay-safe、可独立 ship
─────────────────────────────────  ← 切割线
(2) 子独立 runId / sub-trace
    + 子 emit agent.run.started/completed
    + Milkie.replay 递归下钻子 run   ← 新 issue（replay/fork 域），(2) 与递归 replay 同一单元
```

## 设计

### 1. 事件 taxonomy & payload（`src/trace/types.ts`）

`EventKind` 增加：

```ts
| 'agent.spawned'
| 'agent.returned'
```

```ts
export interface AgentSpawnedPayload {
  parentRunId: string   // 父 AgentRuntime.agentRunId
  childRunId:  string   // 子的稳定身份（见下）
  agentId:     string   // 子 agent 的 agentId
  goal:        string   // spawn 时传入的 goal
}

export interface AgentReturnedPayload {
  childRunId:  string
  status:      'completed' | 'interrupted' | 'error'
}
```

并补对应的 typed event 别名（与现有 `RegionAddedEvent` 等一致）：

```ts
export type AgentSpawnedEvent  = Event<AgentSpawnedPayload>  & { type: 'agent.spawned' }
export type AgentReturnedEvent = Event<AgentReturnedPayload> & { type: 'agent.returned' }
```

**`childRunId` 命名取舍（采用 a）：** 今天子复用父 runId，不存在真正独立的
child runId；子的稳定身份是 `childContextId`。本 issue 用字段名 `childRunId`、
当前值填 `childContextId`。理由：这是给 agent consumer 的协议面（invariant #12
「stable ids」），字段名稳定优先于暂时的语义精确；(2) 落地时只把 `childRunId`
的**值**换成子的独立 runId，**不改 schema**。types.ts 注释里写明这层语义。

### 2. emit 点 & 机制（`src/runtime/AgentRuntime.ts` · `makeSubAgentTool`）

在 sub-agent tool handler 里：

- `recordChild({ status: 'running' })` 之后，emit **`agent.spawned`**，payload
  `{ parentRunId: this.agentRunId, childRunId: childContextId, agentId, goal }`；
- 成功分支 `recordChild` 之后，emit **`agent.returned`**，
  `{ childRunId: childContextId, status: result.status }`——直接用
  `AgentResult.status`（`'completed' | 'interrupted' | 'error'`），**不**走
  `ChildAgentRecord` 的 `success` 重映射；
- catch 分支 emit **`agent.returned`** `{ childRunId: childContextId, status: 'error' }`。

机制：

- 走 **`enqueueTraceWrite`**（有序 + best-effort 吞错 + LLM 调用前 flush），与
  region / fsm.transition 事件一致；
- `if (this.eventStore)` 守卫；
- `id: uuidv4()` / `timestamp: Date.now()` 直取，**绕过 IOPort**——和 region /
  fsm 同款理由：这些是信息性事件，不进 replay 的 nondet cache，不能消耗
  ioPort.uuid()/now() 否则 record/replay 路径不对称。replay 路径不传 eventStore，
  自动跳过 emit。

**附带收口（review #6）：** 把已有的 `emitSkillLifecycle`（#45 引入，仍用
`void this.eventStore.append(...)` fire-and-forget）也迁到 `enqueueTraceWrite`，
让全部 trace 事件路径一致（有序、best-effort、flush 前落盘）。

### 3. causedBy

**本期不做。** `agent.spawned` 理应 `causedBy` 指向触发它的 `tool.requested`
（sub-agent 本身是 named tool），但 tool handler 当前拿不到那条 `tool.requested`
的 event id，接线属于 #30（causedBy 加密）范畴。本 issue 不设 `causedBy`。

### 4. 错误处理

- emit 失败：由 `enqueueTraceWrite` + `tryFlushTraceWrites` 的 best-effort 语义
  兜底——trace 写入失败不改变 agent 业务结果（与本分支既有约定一致）。
- 子 agent 抛错：catch 分支照常 `recordChild({status:'error'})` + emit
  `agent.returned` `'error'` + 重抛（不吞业务异常）。

## 测试（`src/__tests__/`，沿用 StubGateway + sub-agent harness）

- 父 spawn 子 → 父 run 出现 `agent.spawned`，`parentRunId` == 父 runId、
  `childRunId` == 子 contextId、`agentId` / `goal` 正确；
- 子完成 → `agent.returned` 的 `status` 与子 `AgentResult.status` 一致；
- 子报错 → `agent.returned` `status: 'error'`，且业务异常照常向上抛；
- 现有 supervisor 中断/恢复测试仍绿（interrupted → `agent.returned`
  `'interrupted'`）；
- `emitSkillLifecycle` 迁移后，现有 skill.loaded/unloaded 测试仍绿。

## 显式 deferral（不在本 issue）

- 子继续复用父 runId，**不**获独立 run 身份；
- 子**不** emit 自己的 `agent.run.started/completed`；
- `causedBy` → #30；
- 子一类公民化（独立 runId / sub-trace）+ `Milkie.replay` 递归下钻 → **新 issue**
  （(2) 与递归 replay 为同一不可分单元）；
- 原 issue 验收「子 run 第一帧 `agent.run.started.parentId` 与 `agent.spawned`
  一致」本期**放宽**——它描述的是目标态 (2)，待新 issue 满足。

## 验收（本 issue 范围内）

- [ ] `agent.spawned` / `agent.returned` 进 `EventKind`，payload 类型 + event
      别名落地；
- [ ] 父 spawn 子时父 run 出现 `agent.spawned`，字段正确；
- [ ] 子结束后父 run 出现 `agent.returned`，`status` 与子 `AgentResult.status`
      一致（含 completed / interrupted / error 三态）；
- [ ] 新事件经 `enqueueTraceWrite`，trace 写入失败不改变 run 结果；
- [ ] `emitSkillLifecycle` 迁至 `enqueueTraceWrite`；
- [ ] 现有 supervisor 中断/恢复 + skill 生命周期测试全绿。

## Related

- ARCH.md §Implemented today（supervisor tree）、invariant #11
- Stories: s-007 / s-015
- Blocks: #27（sub-agent nav）、lineage 跨 run 追踪
- 待开新 issue：sub-agent 一类公民化 + 递归 replay（(2)）
