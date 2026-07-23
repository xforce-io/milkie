---
id: s-016
title: Record and query task outcome after a run
status: active
kind: scenario
subsystems:
  - agent-trace
capability: task-outcome
requires:
  - Agent Trace event log (basic)
  - Task outcome record
owner: "@xupeng"
created: 2026-07-23
tests:
  - tests/e2e/s-016-record-and-query-task-outcome.e2e.test.ts
related:
  - ARCHITECTURE.md#outcome-task-outcome
  - ARCHITECTURE.md#cross-cutting-decisions-invariants
  - docs/stories/s-002-inspect-a-completed-run.md
  - docs/stories/s-003-explain-a-decision-with-context.md
  - docs/stories/s-006-fork-at-event-for-what-if.md
---

## 场景叙事

一次 agent run 已正常跑完：runtime **execution status** 为 `completed`，调用方也拿到了最终 output。但业务或评测事后发现**任务并没有做对**（例如推荐了错误供应商、答案未通过规则校验）。

操作者（人或 eval 脚本）对这次 run **后置**写入一条 **task outcome**（如 `failure`），可选附带 score。之后任意时刻仅凭 `runId` 能查回该 outcome，且 **不覆盖、不改写** 原有的 execution status。

本 story 固化 `ARCHITECTURE.md` 不变量 14：**Execution status ≠ Task outcome**。  
`completed` 只表示运行时结束；任务成败是挂在 run 上的独立判定记录。

不在此做自动「猜业务是否成功」、不写 Attribution 引擎、不触发 Evolution promote。

## 关键交互流

```
record path
  ├─ milkie.invoke({ agentId, goal, input }) → result
  │    · result.status === 'completed'
  │    · result.agentRunId (or runId) 可用
  ├─ （可选）inspect timeline — 见 s-002；此时尚无 outcome 或 outcome 仍为 unknown/absent
  └─ recordTaskOutcome({
       runId: result.agentRunId,
       value: 'failure',           // success | failure | partial | unknown
       source: 'eval' | 'human' | 'rule' | 'business',
       note?: '...',
       scores?: [{ name, value }],
     })

query path（无额外 LLM）
  ├─ getTaskOutcome(runId) → { value: 'failure', source, at, ... }
  ├─ 同一 run 的 execution status 仍为 'completed'（invoke 结果 / run 记录）
  └─ 未写过 outcome 的 run：查询返回 absent 或 value: 'unknown'（实现二选一，契约稳定即可）
```

> 具体 SDK/CLI 动词名以实现 design 为准；本 story 约束的是**可写、可查、与 status 分离**的用户可见能力。

## 验收准则

- [x] 存在一次 `invoke` 返回 `status: 'completed'` 且带可用 `runId` / `agentRunId`
- [x] 对该 `runId` **后置**写入 task outcome，`value: 'failure'`（或等价枚举），`source` 非空
- [x] 按同一 `runId` 查询，读回的 outcome `value` 与写入一致
- [x] 写入 outcome **之后**，该 run 的 execution status 仍为 `completed`（未被改成 error/failed 等）
- [x] 允许 `status: 'completed'` 与 outcome `failure` **同时成立**（核心：跑完 ≠ 做对）
- [x] （可选）写入时附带至少一条 score；查询时可读到，且不改变 outcome `value` 语义
- [x] 从未写入 outcome 的另一 `runId`：查询为 absent 或 `unknown`，且不抛成「run 不存在」（与非法 runId 错误可区分）
- [x] 记录 / 查询路径不要求再次调用真实 LLM（hermetic 测试可用 stub gateway）

> 实现：`Milkie.recordTaskOutcome` / `getTaskOutcome`；事件 `task.outcome.recorded`；e2e `tests/e2e/s-016-record-and-query-task-outcome.e2e.test.ts`（#217）。

## 不在此 story 范围

- **Timeline 浏览 / 过滤** → 见 [s-002](./s-002-inspect-a-completed-run.md)
- **单点决策证据材料（explain）** → 见 [s-003](./s-003-explain-a-decision-with-context.md)；explain 是证据，不是 task outcome
- **结构化失败归因（Attribution 引擎或必填归因）** → 延后；Attribution 为可选外部写入，非本 story
- **Fork 修复当前任务** → 见 [s-006](./s-006-fork-at-event-for-what-if.md)
- **确定性 Replay** → 见 [s-005](./s-005-deterministic-replay.md)
- **Retry 瞬时工具失败** → 见 [s-009](./s-009-multi-turn-with-tool-error-recovery.md)
- **自动从 output 猜测业务成功 / 失败** → 非目标
- **Evolution 流量切分、显著性、promote/rollback** → 非本 story；Collector 日后可消费 outcome，本 story 只保证 outcome 可挂 run
- **Learnable Delta / 控制面 / 内嵌反思 LLM** → ARCH Non-goals
