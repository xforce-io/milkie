---
id: s-002
title: Inspect a completed agent run
status: active
kind: scenario
subsystems:
  - agent-trace
capability: observability
requires:
  - Trajectory observability
  - Agent Trace event log (basic)
owner: "@xupeng"
created: 2026-05-23
tests:
  - tests/e2e/s-002-inspect-a-completed-run.e2e.test.ts
related:
  - ARCHITECTURE.md#agent-trace
---

## 场景叙事

开发者跑完一个 agent run（成功或失败），手里只有一个 `agentRunId`，想**前向浏览**这次运行里发生的一切——agent 生命周期事件、FSM 状态转移、LLM 请求/响应、tool 调用——按时间顺序呈现，并能按事件类型或时间窗过滤。

不重跑、不分叉、不沿因果链反向追溯，**纯阅读**。这是大多数用户第一次接触 Agent Trace 的入口，校验 ARCHITECTURE.md 里"observable"这条 capability：给定 `runId`，timeline 可读、有序、可过滤。

这个 story 不依赖 LLM；测试用 stub gateway 注入固定响应，目的是验证记录/查询基础设施本身正确，不是 LLM 行为。

## 关键交互流

```
test
  ├─ 构造 milkie，挂上 TrajectoryStore + MemoryEventStore + 一个 echo 工具
  ├─ 注册一个 2-state FSM（greet → finalize），第一个状态调用一次 echo 工具
  └─ milkie.invoke({...}) → 得到 result.agentRunId

inspect (无 LLM 调用)
  ├─ trajectoryStore.getByRunId(runId) → Trajectory{ status: 'completed', spans }
  ├─ eventStore.readByRunId(runId)    → Event[] 按 append 顺序
  ├─ filter by type 'llm.requested'   → 子集
  ├─ filter by time window            → 子集
  └─ readRange(runId, 0, 2)           → 前两条
```

## 验收准则

- [ ] `milkie.invoke(...)` 返回 `status: 'completed'`，含可用 `agentRunId`
- [ ] `trajectoryStore.getByRunId(runId).status === 'completed'`
- [ ] Trajectory 中包含至少 1 个 `llm.call` span、1 个 `tool.call` span、1 个 `fsm.transition` span
- [ ] Trajectory 中所有 span 的 `endTime` 都 `>= startTime`，整体按 startTime 升序可排
- [ ] `eventStore.readByRunId(runId)` 返回的事件数 `>= 6`（start + 2×LLM 配对 + 1×tool 配对 + completed）
- [ ] 首条事件 `type === 'agent.run.started'`，末条 `type === 'agent.run.completed'`
- [ ] 每个 `llm.requested` 都有恰好一个 `llm.responded` 与之配对（同 `requestHash`），`responded.causedBy === requested.id`
- [ ] 每个 `tool.requested` 都有恰好一个 `tool.responded` 与之配对（同 `requestHash`），`responded.causedBy === requested.id`
- [ ] 事件按 `timestamp` 单调不降
- [ ] 按 `type` 字段过滤 `events.filter(e => e.type === 'llm.requested')` 与 LLM 调用次数一致
- [ ] 按时间窗 `events.filter(e => e.timestamp <= cutoff)` 返回事件子集（数量在 `[1, total]` 内、且每条都满足 `timestamp <= cutoff`）。注：hermetic 跑得很快时整个 timeline 可能落在同一毫秒，故只断"子集"语义而非"严格前缀"
- [ ] `eventStore.readRange(runId, 0, 2)` 返回前两条事件且与 `readByRunId` 切片一致
- [ ] 整个 inspection 流程不触发任何额外 LLM 调用（stub gateway 的 callCount 与 invoke 后相同）

## 不在此 story 范围

- **解释单点决策的完整推理材料** → 见 [s-003](./s-003-explain-a-decision-with-context.md)
- **沿因果链反向追溯 artifact 来源** → 见 [s-004](./s-004-lineage-from-artifact-to-source.md)
- **确定性回放** → 见 [s-005](./s-005-deterministic-replay.md)
- **跨多个 run 的批量查询 / suite replay** → 见 [s-012](./s-012-batch-replay-suite-and-classify-divergences.md)
- **in-flight 查询语义**（运行中的 run） → 见 [s-015](./s-015-subagent-reads-parent-trace-runtime.md)
